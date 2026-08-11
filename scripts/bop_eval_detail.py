#!/usr/bin/env python3
"""
bop_eval_detail.py — BOP lmo 详细评估 (真实深度 vs DA3 深度)
==============================================================
输出: 逐帧 ADD(-S) 分布, 多阈值精度, 旋转/平移误差分解.
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
N_FRAMES = int(os.environ.get('BOP_N_FRAMES', '5'))
EVAL_OBJECTS = [int(x) for x in os.environ.get('BOP_OBJ', '1').split(',')]
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
        dists = np.linalg.norm(pts_pred - pts_gt, axis=1)
        return float(np.mean(dists))
    else:
        tree = cKDTree(pts_gt)
        dists, _ = tree.query(pts_pred)
        return float(np.mean(dists))


def main():
    set_logging_format(level=30)
    set_seed(0)
    device = 'cuda:0'

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
        v = mesh.vertices
        vertices[ob_id] = v[::max(1, len(v) // 1000)]

    glctx = dr.RasterizeCudaContext()
    fp_cache = {}
    for ob_id in EVAL_OBJECTS:
        mesh = meshes[ob_id]
        pts, fidx = trimesh.sample.sample_surface(mesh, 3000)
        fp_obj = FoundationPose(model_pts=pts.astype(np.float32),
                                model_normals=mesh.face_normals[fidx].astype(np.float32),
                                mesh=mesh, glctx=glctx, debug=0)
        fp_obj.make_rotation_grid(min_n_views=8, inplane_step=120)
        fp_cache[ob_id] = fp_obj

    # 结果收集: 每个深度记录 (add, trans_err, rot_err)
    records = {'real': [], 'da3': []}
    frame_ids = sorted(gt_all.keys(), key=int)[:N_FRAMES]

    for fi, fid in enumerate(frame_ids):
        fid_int, fid_str, fid_key = int(fid), f'{int(fid):06d}', str(int(fid))
        rgb = cv2.imread(f'{SEQ_DIR}/rgb/{fid_str}.png')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        depth_real = cv2.imread(f'{SEQ_DIR}/depth/{fid_str}.png', cv2.IMREAD_UNCHANGED).astype(np.float32) * 1e-3
        da3_npy = f'{DA3_DIR}/{fid_str}.npy'
        depth_da3 = np.load(da3_npy).astype(np.float32) if os.path.exists(da3_npy) else depth_real.copy()
        if depth_da3.shape[:2] != (H, W):
            depth_da3 = cv2.resize(depth_da3, (W, H))
        camK = np.array(cam_info[fid_key]['cam_K']).reshape(3, 3)

        for inst_idx, inst in enumerate(gt_all[fid][:8]):
            ob_id = inst['obj_id']
            if ob_id not in EVAL_OBJECTS:
                continue
            R_gt = np.array(inst['cam_R_m2c']).reshape(3, 3)
            t_gt = np.array(inst['cam_t_m2c']).reshape(3) * 1e-3
            mask_path = f'{SEQ_DIR}/mask_visib/{fid_str}_{inst_idx:06d}.png'
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            mask = (mask > 0).astype(np.uint8)

            fp = fp_cache[ob_id]
            for dname, depth_in in [('real', depth_real), ('da3', depth_da3)]:
                try:
                    pose = fp.register(K=camK, rgb=rgb, depth=depth_in.astype(np.float32),
                                       ob_mask=mask, ob_id=0, iteration=2)
                    R_pred, t_pred = pose[:3, :3], pose[:3, 3]
                    add = compute_add_sd(vertices[ob_id], R_pred, t_pred, R_gt, t_gt,
                                         is_symmetric=(ob_id in SYMMETRIC_OBJECTS))
                    # 旋转误差 (度)
                    cos_r = np.clip((np.trace(R_pred.T @ R_gt) - 1) / 2, -1, 1)
                    rot_err = np.degrees(np.arccos(cos_r))
                    trans_err = np.linalg.norm(t_pred - t_gt) * 1000  # mm
                    records[dname].append({
                        'add': add * 1000, 'rot': rot_err, 'trans': trans_err,
                        'obj': ob_id, 'frame': fid_int})
                except Exception as e:
                    print(f'  [FAIL obj {ob_id} {dname}] {str(e)[:50]}')

    # ── 汇总 ──
    print('\n' + '=' * 70)
    print('BOP LINEMOD 序列 000002 — 真实深度 vs DA3 深度 (详细)')
    print('=' * 70)
    for dname in ['real', 'da3']:
        recs = records[dname]
        if not recs:
            continue
        adds = np.array([r['add'] for r in recs])
        rots = np.array([r['rot'] for r in recs])
        trans = np.array([r['trans'] for r in recs])
        print(f'\n=== {dname.upper()} 深度 ({len(recs)} 样本) ===')
        print(f'  ADD(-S): mean={adds.mean():.2f}mm  median={np.median(adds):.2f}mm  max={adds.max():.2f}mm')
        print(f'  旋转误差: mean={rots.mean():.2f}°  max={rots.max():.2f}°')
        print(f'  平移误差: mean={trans.mean():.2f}mm  max={trans.max():.2f}mm')
        # 多阈值精度
        thr = diameters[EVAL_OBJECTS[0]] * 0.1
        print(f'  精度 (<{thr:.0f}mm 10%直径): {100*(adds<thr).mean():.1f}%')
        for thr_mm in [5, 10, 20, 50]:
            print(f'    ADD < {thr_mm}mm: {100*(adds<thr_mm).mean():.1f}%')

    # 对比
    if records['real'] and records['da3']:
        print('\n' + '-' * 70)
        r_add = np.array([r['add'] for r in records['real']])
        d_add = np.array([r['add'] for r in records['da3']])
        r_rot = np.array([r['rot'] for r in records['real']])
        d_rot = np.array([r['rot'] for r in records['da3']])
        print(f'ADD 差距: DA3 vs 真实 = {d_add.mean()-r_add.mean():+.2f}mm ({d_add.mean()/max(r_add.mean(),0.01):.1f}x)')
        print(f'旋转差距: DA3 vs 真实 = {d_rot.mean()-r_rot.mean():+.2f}°')
        print('-' * 70)


if __name__ == '__main__':
    main()
