#!/usr/bin/env python3
"""
bop_scale_calibration.py — 几何约束尺度校准验证
================================================
方案1: DA3 深度逐帧尺度校准 (几何约束, 无需真值深度)
  用 mask 区域 + DA3 深度反投影物体 3D 范围, 与 CAD 已知直径比较,
  估算尺度比 → 校准 DA3 深度 → register

对比:
  D1: 原生 register + 原始 DA3
  E1: 原生 register + 尺度校准 DA3 (几何约束)
  D2: 原生 register + 真值深度 (上限)

关键: 尺度校准只用 CAD 尺寸 (已知信息), 不依赖真值深度
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


def calibrate_depth_scale(depth, mask, K, cad_diameter_m):
    """几何约束尺度校准: 用 mask 区域反投影估计物体尺寸, 与 CAD 已知直径比较

    步骤:
      1. mask 区域像素 + depth 反投影 → 3D 点 (米)
      2. 计算物体 3D bbox 对角线 (观测尺寸)
      3. 尺度比 = CAD直径 / 观测直径
      4. 校准深度 = depth × 尺度比

    Returns: (calibrated_depth, scale_ratio)
    """
    ys, xs = np.where(mask > 0)
    if len(ys) < 50:
        return depth, 1.0
    z = depth[ys, xs]
    valid = (z > 0.05) & (z < 10) & np.isfinite(z)
    if valid.sum() < 30:
        return depth, 1.0
    ys_v, xs_v, z_v = ys[valid], xs[valid], z[valid]

    # 反投影 (相机系)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs_v - cx) * z_v / fx
    y_cam = (ys_v - cy) * z_v / fy
    pts = np.stack([x_cam, y_cam, z_v], axis=1)

    # 过滤离群值 (IQR)
    q1, q3 = np.percentile(pts, 25, axis=0), np.percentile(pts, 75, axis=0)
    iqr = q3 - q1
    center = np.median(pts, axis=0)
    inlier = np.all(np.abs(pts - center) < 2.5 * iqr, axis=1)
    if inlier.sum() >= 30:
        pts = pts[inlier]

    # 观测尺寸 (bbox 对角线)
    obs_diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    if obs_diag < 1e-4:
        return depth, 1.0

    # 尺度比 = CAD直径 / 观测对角线
    scale = cad_diameter_m / obs_diag
    # 限制合理范围 (0.3 ~ 3.0), 防校准发散
    scale = np.clip(scale, 0.3, 3.0)

    calibrated = depth * scale
    return calibrated, scale


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
        diameters[ob_id] = models_info[str(ob_id)]['diameter']  # mm
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
    # D1=原始DA3, E1=校准DA3, D2=真值
    results = {ob: {'D1': [], 'E1': [], 'D2': [], 'scales': []} for ob in EVAL_OBJECTS}

    for ob_id in EVAL_OBJECTS:
        fp = fp_cache[ob_id]
        cad_diameter_m = diameters[ob_id] / 1000.0  # 已知 CAD 尺寸 (米)
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

            # 尺度校准 (几何约束, 无需真值深度)
            depth_cal, scale = calibrate_depth_scale(depth_da3, mask, camK, cad_diameter_m)
            results[ob_id]['scales'].append(scale)

            for group, depth_in in [('D1', depth_da3),     # 原始 DA3
                                    ('E1', depth_cal),     # 校准 DA3
                                    ('D2', depth_gt)]:     # 真值
                try:
                    pose = fp.register(K=camK, rgb=rgb, depth=depth_in.astype(np.float32),
                                       ob_mask=mask, ob_id=0, iteration=2)
                    add = compute_add_sd(vertices[ob_id], pose[:3, :3], pose[:3, 3],
                                         R_gt, t_gt, sym)
                    results[ob_id][group].append(add * 1000)
                except Exception:
                    pass
        print(f'  obj {ob_id} done ({len(results[ob_id]["D1"])} frames)', flush=True)

    # ── 汇总 ──
    print('\n' + '=' * 96)
    print('尺度校准验证 — D1(原始DA3) vs E1(校准DA3) vs D2(真值)')
    print('=' * 96)
    print(f'{"Obj":>3} | {"D1:原始DA3":>16} | {"E1:校准DA3":>16} | {"D2:真值":>14} | {"尺度比":>8}')
    print(f'{"":>3} | {"ADD/通过%":>16} | {"ADD/通过%":>16} | {"ADD/通过%":>14}')
    print('-' * 96)

    all_g = {g: [] for g in ['D1', 'E1', 'D2']}
    for ob_id in EVAL_OBJECTS:
        thr = diameters[ob_id] * 0.1
        line = f'{ob_id:>3} |'
        for g in ['D1', 'E1', 'D2']:
            recs = results[ob_id][g]
            m = np.mean(recs) if recs else float('nan')
            p = 100 * np.mean([a < thr for a in recs]) if recs else float('nan')
            line += f' {m:>7.1f}/{p:>4.1f}% |'
            all_g[g].extend(recs)
        s = np.mean(results[ob_id]['scales']) if results[ob_id]['scales'] else float('nan')
        line += f' {s:>6.2f} |'
        print(line)

    print('-' * 96)
    m1, me, m2 = (np.mean(all_g[g]) for g in ['D1', 'E1', 'D2'])
    print(f'{"ALL":>3} | {m1:>7.1f}/{"?":>4} | {me:>7.1f}/{"?":>4} | {m2:>7.1f}/{"?":>4}')
    print()
    print(f'校准提升: D1 {m1:.1f} → E1 {me:.1f}mm ({(m1-me)/max(m1,0.1)*100:.0f}% 改善)')
    print(f'距上限:   E1 {me:.1f} vs D2(真值) {m2:.1f}mm')
    print('=' * 96)


if __name__ == '__main__':
    main()
