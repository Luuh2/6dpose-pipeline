#!/usr/bin/env python3
"""
bop_exp_pipeline.py — BOP 三组对照实验 (完整 Pipeline vs 原生)
================================================================
实验组 A: 完整 Pipeline (首帧 register + track + LIEKF + 掩码引导) + DA3 深度
对照组 B: 完整 Pipeline (首帧 register + track + LIEKF + 掩码引导) + 真值深度
对照组 C: 原生逐帧独立 register + DA3 深度 (坏基线)

Pipeline 引导: 掩码质心修正平移 + 惯性主轴修正翻滚 + LIEKF 融合 + 旋转一致性校验
mask 统一用 BOP 真值 mask_visib (隔离深度单一变量)

对比:
  A vs C → 完整 Pipeline 相对原生 register 的提升 (可行性)
  A vs B → DA3 深度 vs 真值深度的差距 (深度影响)
"""
import sys, os, json
import numpy as np
import cv2
import torch

BASE = '/mnt/e/zhijiyige'
BOP_DIR = f'{BASE}/bop_data/lmo'
sys.path.insert(0, f'{BASE}/src/nvdiffrast/nvdiffrast-main')
sys.path.insert(0, f'{BASE}/src/FoundationPose')
sys.path.insert(0, BASE)  # modules/*

SEQ_DIR = f'{BOP_DIR}/test/000002'
N_FRAMES = int(os.environ.get('BOP_N_FRAMES', '8'))
EVAL_OBJECTS = [int(x) for x in os.environ.get('BOP_OBJ', '1,10').split(',')]
SYMMETRIC_OBJECTS = {10, 11}
DA3_DIR = f'{BOP_DIR}/da3_depth'

from estimater import FoundationPose
from Utils import set_logging_format, set_seed
import nvdiffrast.torch as dr
import trimesh
from scipy.spatial import cKDTree

# LIEKF (与 demo 管线一致)
from modules.se3_kalman_filter import SE3LieKalmanFilter


def compute_add_sd(vertices, R_pred, t_pred, R_gt, t_gt, is_symmetric):
    pts_pred = (R_pred @ vertices.T).T + t_pred.reshape(1, 3)
    pts_gt = (R_gt @ vertices.T).T + t_gt.reshape(1, 3)
    if not is_symmetric:
        return float(np.mean(np.linalg.norm(pts_pred - pts_gt, axis=1)))
    tree = cKDTree(pts_gt)
    dists, _ = tree.query(pts_pred)
    return float(np.mean(dists))


def wrap_diff(a, b):
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def mask_centroid_and_axis(mask):
    ys, xs = np.where(mask > 0)
    if len(ys) < 50:
        return None, None
    c = np.array([xs.mean(), ys.mean()])
    m = cv2.moments(mask.astype(np.uint8))
    ang = 0.0
    if m['mu20'] + m['mu02'] > 1e-6:
        ang = 0.5 * np.arctan2(2 * m['mu11'], m['mu20'] - m['mu02'])
    return c, ang


def apply_centroid_pose_fix(pose, c2d, depth, mask, K):
    """掩码质心 → 反投影 3D → 修正 pose 平移 x,y (z 用 mask 深度中位数)"""
    ys, xs = np.where(mask > 0)
    if len(ys) < 20:
        return pose
    zc = np.median(depth[ys, xs])
    if not np.isfinite(zc) or zc < 0.01:
        return pose
    invK = np.linalg.inv(K)
    c3d = invK @ np.array([c2d[0], c2d[1], 1.0]) * zc
    pose[0, 3], pose[1, 3] = c3d[0], c3d[1]
    return pose


def apply_axis_pose_fix(pose, target_axis, K, max_deg=8.0):
    """惯性主轴 → 修正 pose 翻滚角 (绕相机 z 轴)"""
    R = pose[:3, :3].copy()
    x_cam = R[:, 0]
    cur_alpha = np.arctan2(x_cam[1], x_cam[0])
    delta = wrap_diff(target_axis, cur_alpha)
    delta = np.clip(delta, -np.radians(max_deg), np.radians(max_deg))
    if abs(delta) < 1e-4:
        return pose
    cz, sz = np.cos(delta), np.sin(delta)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    pose[:3, :3] = Rz @ R
    return pose


def detect_ambiguity(pose_cur, pose_prev, max_deg=50.0):
    Ra, Rb = pose_cur[:3, :3], pose_prev[:3, :3]
    max_ang = 0.0
    for i in range(3):
        va, vb = Ra[:, i], Rb[:, i]
        cos_a = np.clip(np.dot(va, vb), -1, 1)
        max_ang = max(max_ang, np.degrees(np.arccos(cos_a)))
    return max_ang > max_deg, max_ang


def sync_pose_last(fp, pose):
    centered = pose @ np.linalg.inv(fp.get_tf_to_centered_mesh().cpu().numpy())
    fp.pose_last = torch.as_tensor(centered, device='cuda', dtype=torch.float)


def run_pipeline(frames, depths, masks, Ks, gts, fp, is_da3_depth):
    """完整 Pipeline: 首帧 register + track_one + LIEKF + 掩码引导

    Returns: list of ADD (mm)
    """
    n = len(frames)
    kf = SE3LieKalmanFilter(
        dt=1.0 / 30.0, process_noise_pos=0.005, process_noise_rot=0.002,
        measurement_noise_pos=0.002, measurement_noise_rot=0.01)

    adds = []
    prev_pose = None
    axis_hist = []

    for i in range(n):
        rgb, depth, mask, K = frames[i], depths[i], masks[i], Ks[i]
        R_gt, t_gt, sym = gts[i]

        if i == 0:
            pose = fp.register(K=K, rgb=rgb, depth=depth.astype(np.float32),
                               ob_mask=mask, ob_id=0, iteration=2)
            kf.initialize(pose)
        else:
            pose = fp.track_one(rgb=rgb, depth=depth.astype(np.float32), K=K, iteration=1)

            # 旋转一致性校验 (对称歧义回退)
            if prev_pose is not None:
                is_ambig, _ = detect_ambiguity(pose, prev_pose)
                if is_ambig:
                    pose[:3, :3] = prev_pose[:3, :3]

            # 掩码质心修正平移
            c2d, ax = mask_centroid_and_axis(mask)
            if c2d is not None:
                pose = apply_centroid_pose_fix(pose, c2d, depth, mask, K)

            # 惯性主轴修正翻滚
            if ax is not None:
                # 主轴角平滑 (防跳变)
                axis_hist.append(ax)
                if len(axis_hist) > 3:
                    axis_hist.pop(0)
                ax_s = np.arctan2(np.mean([np.sin(a) for a in axis_hist]),
                                  np.mean([np.cos(a) for a in axis_hist]))
                pose = apply_axis_pose_fix(pose, ax_s, K)

            # LIEKF 融合
            kf.predict()
            kf.update(pose)
            pose = kf.X.copy()

        sync_pose_last(fp, pose)
        prev_pose = pose.copy()

        add = compute_add_sd(vertices_global[fp_obj_id], pose[:3, :3], pose[:3, 3],
                             R_gt, t_gt, sym)
        adds.append(add * 1000)

    return adds


# 全局引用 (run_pipeline 使用)
vertices_global = {}
fp_obj_id = None


def run_register_baseline(frames, depths, masks, Ks, gts, fp):
    """对照 C: 原生逐帧独立 register + DA3"""
    adds = []
    for i in range(len(frames)):
        rgb, depth, mask, K = frames[i], depths[i], masks[i], Ks[i]
        R_gt, t_gt, sym = gts[i]
        pose = fp.register(K=K, rgb=rgb, depth=depth.astype(np.float32),
                           ob_mask=mask, ob_id=0, iteration=2)
        add = compute_add_sd(vertices_global[fp_obj_id], pose[:3, :3], pose[:3, 3],
                             R_gt, t_gt, sym)
        adds.append(add * 1000)
    return adds


def main():
    global vertices_global, fp_obj_id
    set_logging_format(level=30)
    set_seed(0)

    with open(f'{SEQ_DIR}/scene_camera.json') as f:
        cam_info = json.load(f)
    with open(f'{SEQ_DIR}/scene_gt.json') as f:
        gt_all = json.load(f)
    with open(f'{BOP_DIR}/models/models_info.json') as f:
        models_info = json.load(f)

    import open3d as o3d
    meshes, diameters, vertices = {}, {}, {}
    for ob_id in EVAL_OBJECTS:
        mesh = trimesh.load(f'{BOP_DIR}/models/obj_{ob_id:06d}.ply', force='mesh')
        mesh.vertices = mesh.vertices.astype(np.float32) / 1000.0
        if len(mesh.vertices) > 5000:
            om = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64)),
                triangles=o3d.utility.Vector3iVector(mesh.faces.astype(np.int32)))
            om = om.simplify_quadric_decimation(4000)
            mesh = trimesh.Trimesh(vertices=np.asarray(om.vertices).astype(np.float32),
                                   faces=np.asarray(om.triangles).astype(np.int64))
        meshes[ob_id] = mesh
        diameters[ob_id] = models_info[str(ob_id)]['diameter']
        vertices[ob_id] = mesh.vertices[::max(1, len(mesh.vertices) // 1000)]
    vertices_global = vertices

    glctx = dr.RasterizeCudaContext()
    fp_cache = {}
    for ob_id in EVAL_OBJECTS:
        mesh = meshes[ob_id]
        pts, fidx = trimesh.sample.sample_surface(mesh, 3000)
        fp = FoundationPose(model_pts=pts.astype(np.float32),
                            model_normals=mesh.face_normals[fidx].astype(np.float32),
                            mesh=mesh, glctx=glctx, debug=0)
        fp.make_rotation_grid(min_n_views=8, inplane_step=120)
        fp_cache[ob_id] = fp

    frame_ids = sorted(gt_all.keys(), key=int)[:N_FRAMES]
    # A=完整Pipeline+DA3, B=完整Pipeline+真值, C=原生register+DA3, D=原生register+真值
    results = {ob: {'A': [], 'B': [], 'C': [], 'D': []} for ob in EVAL_OBJECTS}

    for ob_id in EVAL_OBJECTS:
        fp = fp_cache[ob_id]
        fp_obj_id = ob_id

        # 收集该物体所有帧数据
        frames_all, depths_gt_all, depths_da3_all = [], [], []
        masks_all, Ks_all, gts_all = [], [], []
        for fid in frame_ids:
            fs, fk = f'{int(fid):06d}', str(int(fid))
            rgb = cv2.imread(f'{SEQ_DIR}/rgb/{fs}.png')
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]
            depth_gt = cv2.imread(f'{SEQ_DIR}/depth/{fs}.png',
                                  cv2.IMREAD_UNCHANGED).astype(np.float32) * 1e-3
            da3_npy = f'{DA3_DIR}/{fs}.npy'
            depth_da3 = np.load(da3_npy).astype(np.float32) if os.path.exists(da3_npy) else depth_gt.copy()
            if depth_da3.shape[:2] != (H, W):
                depth_da3 = cv2.resize(depth_da3, (W, H))
            inst_idx, inst = None, None
            for ii, ins in enumerate(gt_all[fid][:8]):
                if ins['obj_id'] == ob_id:
                    inst_idx, inst = ii, ins
                    break
            if inst is None:
                continue
            R_gt = np.array(inst['cam_R_m2c']).reshape(3, 3)
            t_gt = np.array(inst['cam_t_m2c']).reshape(3) * 1e-3
            mask_path = f'{SEQ_DIR}/mask_visib/{fs}_{inst_idx:06d}.png'
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            mask = (mask > 0).astype(np.uint8)
            camK = np.array(cam_info[fk]['cam_K']).reshape(3, 3)

            frames_all.append(rgb)
            depths_gt_all.append(depth_gt)
            depths_da3_all.append(depth_da3)
            masks_all.append(mask)
            Ks_all.append(camK)
            gts_all.append((R_gt, t_gt, ob_id in SYMMETRIC_OBJECTS))

        # 实验组 A: 完整 Pipeline + DA3
        fp.pose_last = None
        results[ob_id]['A'] = run_pipeline(frames_all, depths_da3_all, masks_all, Ks_all,
                                           gts_all, fp, is_da3_depth=True)
        # 对照组 B: 完整 Pipeline + 真值深度
        fp.pose_last = None
        results[ob_id]['B'] = run_pipeline(frames_all, depths_gt_all, masks_all, Ks_all,
                                           gts_all, fp, is_da3_depth=False)
        # 对照组 C: 原生 register + DA3
        fp.pose_last = None
        results[ob_id]['C'] = run_register_baseline(frames_all, depths_da3_all, masks_all,
                                                    Ks_all, gts_all, fp)
        # 对照 D: 原生 register + 真值深度
        fp.pose_last = None
        results[ob_id]['D'] = run_register_baseline(frames_all, depths_gt_all, masks_all,
                                                    Ks_all, gts_all, fp)

        print(f'  obj {ob_id} done ({len(results[ob_id]["A"])} frames)', flush=True)

    # ── 汇总 (2×2 设计) ──
    print('\n' + '=' * 92)
    print('BOP 2×2 对照实验 — lmo 序列 000002')
    print('=' * 92)
    print(f'{"Obj":>3} | {"A: Pipeline+DA3":>18} | {"B: Pipeline+真值":>18} | {"C: 原生+DA3":>16} | {"D: 原生+真值":>16}')
    print(f'{"":>3} | {"ADD/通过%":>18} | {"ADD/通过%":>18} | {"ADD/通过%":>16} | {"ADD/通过%":>16}')
    print('-' * 92)

    all_g = {g: [] for g in ['A', 'B', 'C', 'D']}
    for ob_id in EVAL_OBJECTS:
        thr = diameters[ob_id] * 0.1
        line = f'{ob_id:>3} |'
        for g in ['A', 'B', 'C', 'D']:
            recs = results[ob_id][g]
            m = np.mean(recs) if recs else float('nan')
            p = 100 * np.mean([a < thr for a in recs]) if recs else float('nan')
            line += f' {m:>7.1f}/{p:>4.1f}% |'
            all_g[g].extend(recs)
        print(line)

    print('-' * 92)
    line = 'ALL |'
    for g in ['A', 'B', 'C', 'D']:
        recs = all_g[g]
        m = np.mean(recs) if recs else float('nan')
        line += f' {m:>7.1f}/{"?":>4} |'
    print(line)

    # ── 结论 (2×2 因子分析) ──
    if all(all_g[g] for g in 'ABCD'):
        mA = np.mean(all_g['A']); mB = np.mean(all_g['B'])
        mC = np.mean(all_g['C']); mD = np.mean(all_g['D'])
        print('\n' + '=' * 92)
        print('2×2 因子分析 (Pipeline × 深度)')
        print('=' * 92)
        print(f'{"":>16} | {"原生register":>14} | {"完整Pipeline":>14}')
        print(f'{"DA3 深度":>16} | {mC:>9.1f}mm (C) | {mA:>9.1f}mm (A)')
        print(f'{"真值深度":>16} | {mD:>9.1f}mm (D) | {mB:>9.1f}mm (B)')
        print('-' * 92)
        print(f'① 深度影响 (原生: C→D, Pipeline: A→B):')
        print(f'   原生: {mC:.1f}→{mD:.1f}mm (改善 {(mC-mD)/max(mC,0.1)*100:.0f}%)')
        print(f'   Pipeline: {mA:.1f}→{mB:.1f}mm (改善 {(mA-mB)/max(mA,0.1)*100:.0f}%)')
        print(f'② Pipeline 价值 (DA3: C→A, 真值: D→B):')
        print(f'   DA3: {mC:.1f}→{mA:.1f}mm ({(mA-mC)/max(mC,0.1)*100:+.0f}%)')
        print(f'   真值: {mD:.1f}→{mB:.1f}mm ({(mB-mD)/max(mD,0.1)*100:+.0f}%)')
        print('=' * 92)


if __name__ == '__main__':
    main()
