#!/usr/bin/env python3
"""
bop_eval_detection.py — BOP 单帧 6D 检测评估
==============================================
评估单帧姿态检测效果 (非跟踪), 2×2 设计:

D1: 原生 register + DA3 深度
D2: 原生 register + 真值深度
D3: 单帧 Pipeline (仅掩码质心引导) + DA3 深度
D4: 单帧 Pipeline (仅掩码质心引导) + 真值深度

单帧 Pipeline = 每帧独立 register + 掩码质心反投影修正平移 (仅此引导, 无追踪组件)
对比:
  D1 vs D3 / D2 vs D4 → 掩码引导的价值
  D1 vs D2 / D3 vs D4 → 深度影响
"""
import sys, os, json
import numpy as np
import cv2
import torch

BASE = '/mnt/e/zhijiyige'
BOP_DIR = f'{BASE}/bop_data/lmo'
sys.path.insert(0, f'{BASE}/src/nvdiffrast/nvdiffrast-main')
sys.path.insert(0, f'{BASE}/src/FoundationPose')

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


def compute_add_sd(vertices, R_pred, t_pred, R_gt, t_gt, is_symmetric):
    pts_pred = (R_pred @ vertices.T).T + t_pred.reshape(1, 3)
    pts_gt = (R_gt @ vertices.T).T + t_gt.reshape(1, 3)
    if not is_symmetric:
        return float(np.mean(np.linalg.norm(pts_pred - pts_gt, axis=1)))
    tree = cKDTree(pts_gt)
    dists, _ = tree.query(pts_pred)
    return float(np.mean(dists))


def apply_mask_centroid_guide(pose, mask, depth, K):
    """仅掩码引导: 掩码质心 + 深度反投影 → 修正 pose 平移 x,y

    用 mask 区域深度中位数反投影质心, 得到物体中心 3D, 修正 pose 平移.
    """
    ys, xs = np.where(mask > 0)
    if len(ys) < 20:
        return pose
    zc = np.median(depth[ys, xs])
    if not np.isfinite(zc) or zc < 0.01:
        return pose
    uc, vc = xs.mean(), ys.mean()
    invK = np.linalg.inv(K)
    c3d = invK @ np.array([uc, vc, 1.0]) * zc
    pose[0, 3], pose[1, 3] = c3d[0], c3d[1]
    return pose


def main():
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
    # D1=原生+DA3, D2=原生+真值, D3=引导+DA3, D4=引导+真值
    results = {ob: {'D1': [], 'D2': [], 'D3': [], 'D4': []} for ob in EVAL_OBJECTS}

    for ob_id in EVAL_OBJECTS:
        fp = fp_cache[ob_id]
        for fi, fid in enumerate(frame_ids):
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
            sym = ob_id in SYMMETRIC_OBJECTS

            # 每帧独立 register (4 组)
            for group, depth_in, use_guide in [
                    ('D1', depth_da3, False),  # 原生 + DA3
                    ('D2', depth_gt, False),   # 原生 + 真值
                    ('D3', depth_da3, True),   # 引导 + DA3
                    ('D4', depth_gt, True)]:   # 引导 + 真值
                try:
                    pose = fp.register(K=camK, rgb=rgb, depth=depth_in.astype(np.float32),
                                       ob_mask=mask, ob_id=0, iteration=2)
                    if use_guide:
                        pose = apply_mask_centroid_guide(pose, mask, depth_in, camK)
                    add = compute_add_sd(vertices[ob_id], pose[:3, :3], pose[:3, 3],
                                         R_gt, t_gt, sym)
                    results[ob_id][group].append(add * 1000)
                except Exception:
                    pass
        print(f'  obj {ob_id} done ({len(results[ob_id]["D1"])} frames)', flush=True)

    # ── 汇总 ──
    print('\n' + '=' * 96)
    print('BOP 单帧 6D 检测 — 2×2 (方法 × 深度)')
    print('=' * 96)
    print(f'{"Obj":>3} | {"D1:原生+DA3":>16} | {"D2:原生+真值":>16} | {"D3:引导+DA3":>16} | {"D4:引导+真值":>16}')
    print(f'{"":>3} | {"ADD/通过%":>16} | {"ADD/通过%":>16} | {"ADD/通过%":>16} | {"ADD/通过%":>16}')
    print('-' * 96)

    all_g = {g: [] for g in ['D1', 'D2', 'D3', 'D4']}
    for ob_id in EVAL_OBJECTS:
        thr = diameters[ob_id] * 0.1
        line = f'{ob_id:>3} |'
        for g in ['D1', 'D2', 'D3', 'D4']:
            recs = results[ob_id][g]
            m = np.mean(recs) if recs else float('nan')
            p = 100 * np.mean([a < thr for a in recs]) if recs else float('nan')
            line += f' {m:>7.1f}/{p:>4.1f}% |'
            all_g[g].extend(recs)
        print(line)

    print('-' * 96)
    line = 'ALL |'
    for g in ['D1', 'D2', 'D3', 'D4']:
        recs = all_g[g]
        m = np.mean(recs) if recs else float('nan')
        line += f' {m:>7.1f}/{"?":>4} |'
    print(line)

    # ── 分析 ──
    if all(all_g[g] for g in ['D1', 'D2', 'D3', 'D4']):
        m1, m2, m3, m4 = (np.mean(all_g[g]) for g in ['D1', 'D2', 'D3', 'D4'])
        print('\n' + '=' * 96)
        print('分析')
        print('=' * 96)
        print(f'① 掩码引导价值:')
        print(f'   DA3: D1 {m1:.1f} → D3 {m3:.1f}mm ({(m3-m1)/max(m1,0.1)*100:+.0f}%)')
        print(f'   真值: D2 {m2:.1f} → D4 {m4:.1f}mm ({(m4-m2)/max(m2,0.1)*100:+.0f}%)')
        print(f'② 深度影响:')
        print(f'   原生: D1 {m1:.1f} → D2 {m2:.1f}mm ({(m2-m1)/max(m1,0.1)*100:+.0f}%)')
        print(f'   引导: D3 {m3:.1f} → D4 {m4:.1f}mm ({(m4-m3)/max(m3,0.1)*100:+.0f}%)')
        print('=' * 96)


if __name__ == '__main__':
    main()
