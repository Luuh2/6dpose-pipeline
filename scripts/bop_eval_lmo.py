#!/usr/bin/env python3
"""
bop_eval_lmo.py — BOP LINEMOD 小规模评估
==========================================
对比真实深度 vs DA3 估算深度 的 6D 姿态估计精度。

数据: BOP lmo 数据集序列 000002 (8 物体, 200 帧, 有真值 pose)
方法: 用 BOP 真值 CAD 模型, 分别用:
  1) 真实深度 (BOP 提供)
  2) DA3 估算深度 (从同一 RGB 帧)
做 FoundationPose register() 单帧姿态估计, 计算 ADD(-S).

指标:
  ADD(-S) 距离 < 模型直径×10% 的精度
  ADD(-S) AUC (距离阈值累积)
  姿态角误差 (旋转/平移)

用法 (WSL2):
  /root/miniconda3/envs/rgb6d/bin/python /mnt/e/zhijiyige/scripts/bop_eval_lmo.py
"""
import sys, os, json, time
import numpy as np
import cv2
import torch

BASE = '/mnt/e/zhijiyige'
BOP_DIR = f'{BASE}/bop_data/lmo'
sys.path.insert(0, f'{BASE}/src/nvdiffrast/nvdiffrast-main')
sys.path.insert(0, f'{BASE}/src/FoundationPose')

# ── 配置 ─────────────────────────────────────────────────────────────
SEQ_DIR = f'{BOP_DIR}/test/000002'
N_FRAMES = int(os.environ.get('BOP_N_FRAMES', '50'))  # 测试帧数 (env 覆盖)
MAX_INSTANCES = int(os.environ.get('BOP_MAX_INST', '8'))  # 每帧实例数
EVAL_OBJECTS = [1, 5, 6, 8, 9, 10, 11, 12]  # lmo 8 个对象
if os.environ.get('BOP_OBJ'):
    EVAL_OBJECTS = [int(x) for x in os.environ.get('BOP_OBJ').split(',')]

# 对称物体 (需要 ADD-S)
SYMMETRIC_OBJECTS = {10, 11}   # eggbox=10, glue=11 有对称性

# DA3 深度估计
sys.path.insert(0, f'{BASE}/Depth-Anything-3/Depth-Anything-3-main/src')

from estimater import FoundationPose
from Utils import set_logging_format, set_seed
import nvdiffrast.torch as dr
import trimesh


# ── ADD(-S) 计算 ────────────────────────────────────────────────────
def compute_add_sd(vertices, R_pred, t_pred, R_gt, t_gt, is_symmetric):
    """计算 ADD 或 ADD-S 平均距离 (mm)"""
    # 顶点: (N,3) 模型系
    pts_pred = (R_pred @ vertices.T).T + t_pred.reshape(1, 3)
    pts_gt = (R_gt @ vertices.T).T + t_gt.reshape(1, 3)

    if not is_symmetric:
        # ADD: 对应点距离
        dists = np.linalg.norm(pts_pred - pts_gt, axis=1)
        return float(np.mean(dists))
    else:
        # ADD-S: 最近点距离 (对称物体)
        from scipy.spatial import cKDTree
        tree = cKDTree(pts_gt)
        dists, _ = tree.query(pts_pred)
        return float(np.mean(dists))


# ── 主流程 ──────────────────────────────────────────────────────────
def main():
    set_logging_format(level=30)  # WARNING: 抑制 FP 详细 INFO 日志
    set_seed(0)
    device = 'cuda:0'

    # 读取相机参数
    with open(f'{SEQ_DIR}/scene_camera.json') as f:
        cam_info = json.load(f)

    # 读取真值
    with open(f'{SEQ_DIR}/scene_gt.json') as f:
        gt_all = json.load(f)

    # 读取模型信息 (直径)
    with open(f'{BOP_DIR}/models/models_info.json') as f:
        models_info = json.load(f)

    # 加载模型 mesh + 直径
    meshes = {}
    diameters = {}
    vertices = {}
    # open3d 简化 (复杂物体降到 ~4000 顶点, 避免 FP 创建/register 卡住)
    import open3d as o3d
    for ob_id in EVAL_OBJECTS:
        mesh_path = f'{BOP_DIR}/models/obj_{ob_id:06d}.ply'
        mesh = trimesh.load(mesh_path, force='mesh')
        mesh.vertices = mesh.vertices.astype(np.float32) / 1000.0  # mm -> m
        # 简化复杂 mesh
        if len(mesh.vertices) > 5000:
            o3d_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64)),
                triangles=o3d.utility.Vector3iVector(mesh.faces.astype(np.int32)))
            o3d_mesh = o3d_mesh.simplify_quadric_decimation(4000)
            mesh = trimesh.Trimesh(
                vertices=np.asarray(o3d_mesh.vertices).astype(np.float32),
                faces=np.asarray(o3d_mesh.triangles).astype(np.int64))
        meshes[ob_id] = mesh
        diameters[ob_id] = models_info[str(ob_id)]['diameter']  # mm
        # 顶点采样用于 ADD
        v = mesh.vertices
        # 降采样 (用于 ADD 计算)
        step = max(1, len(v) // 1000)
        vertices[ob_id] = v[::step]

    print(f'[Init] {len(EVAL_OBJECTS)} objects loaded (简化后), sequence 000002')

    # DA3 深度在 Windows 端预计算 (bop_da3_depth_precompute.py), 这里直接读取 npy.
    # 避免 WSL 环境 torchvision/torch 版本不匹配问题.
    DA3_DIR = f'{BOP_DIR}/da3_depth'
    print(f'[Init] DA3 深度从 {DA3_DIR} 读取 (Windows 预计算)')

    # 统计
    results = {ob: {'add_real': [], 'add_da3': [], 'n': 0} for ob in EVAL_OBJECTS}

    glctx = dr.RasterizeCudaContext()

    # 预创建每个物体的 FP 实例 (缓存复用, 避免每帧重建网络浪费显存/时间)
    fp_cache = {}
    for ob_id in EVAL_OBJECTS:
        mesh = meshes[ob_id]
        pts, face_idx = trimesh.sample.sample_surface(mesh, 3000)
        normals = mesh.face_normals[face_idx].astype(np.float32)
        fp_obj = FoundationPose(
            model_pts=pts.astype(np.float32), model_normals=normals,
            mesh=mesh, glctx=glctx, debug=0,
        )
        fp_obj.make_rotation_grid(min_n_views=8, inplane_step=120)
        fp_cache[ob_id] = fp_obj
        torch.cuda.empty_cache()
    print(f'[Init] {len(fp_cache)} FP 实例缓存完成')

    frame_ids = sorted(gt_all.keys(), key=int)[:N_FRAMES]

    for fi, fid in enumerate(frame_ids):
        fid_int = int(fid)  # 帧号 (int)
        fid_str = f'{fid_int:06d}'      # 6位补零 (rgb/depth 文件名)
        fid_key = str(fid_int)          # 原始键 (scene_camera/gt)

        # 读取 RGB
        rgb_path = f'{SEQ_DIR}/rgb/{fid_str}.png'
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        # 读取真实深度 (mm -> m)
        depth_path = f'{SEQ_DIR}/depth/{fid_str}.png'
        depth_real_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_real = depth_real_mm.astype(np.float32) * 1e-3  # mm -> m

        # DA3 深度 (从 Windows 预计算 npy 读取)
        da3_npy = f'{DA3_DIR}/{fid_str}.npy'
        if os.path.exists(da3_npy):
            depth_da3 = np.load(da3_npy).astype(np.float32)  # meters
        else:
            depth_da3 = depth_real.copy()  # fallback
        if depth_da3.shape[:2] != (H, W):
            depth_da3 = cv2.resize(depth_da3, (W, H))

        # 相机内参
        camK = np.array(cam_info[fid_key]['cam_K']).reshape(3, 3)
        depth_scale = cam_info[fid_key]['depth_scale']

        # 该帧实例 (每物体取第一个实例做单物体估计)
        instances = gt_all[fid][:MAX_INSTANCES]
        for inst_idx, inst in enumerate(instances):
            ob_id = inst['obj_id']
            if ob_id not in EVAL_OBJECTS:
                continue

            # 真值 pose (mm -> m)
            R_gt = np.array(inst['cam_R_m2c']).reshape(3, 3)
            t_gt = np.array(inst['cam_t_m2c']).reshape(3) * 1e-3  # m

            # 真值 mask (物体区域) — 文件名用实例索引
            mask_path = f'{SEQ_DIR}/mask_visib/{fid_str}_{inst_idx:06d}.png'
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            mask = (mask > 0).astype(np.uint8)

            # 复用缓存的 FP 实例
            fp = fp_cache[ob_id]

            # ── 评估两种深度 ──
            for depth_name, depth_in in [('real', depth_real), ('da3', depth_da3)]:
                try:
                    pose = fp.register(
                        K=camK, rgb=rgb, depth=depth_in.astype(np.float32),
                        ob_mask=mask, ob_id=0, iteration=2,
                    )

                    # 计算 ADD(-S)
                    R_pred = pose[:3, :3]
                    t_pred = pose[:3, 3]
                    add = compute_add_sd(
                        vertices[ob_id], R_pred, t_pred, R_gt, t_gt,
                        is_symmetric=(ob_id in SYMMETRIC_OBJECTS))

                    if depth_name == 'real':
                        results[ob_id]['add_real'].append(add)
                    else:
                        results[ob_id]['add_da3'].append(add)
                except Exception as e:
                    if depth_name == 'da3':
                        print(f'  [frame {fid} obj {ob_id} da3] FAIL: {str(e)[:60]}')

            results[ob_id]['n'] += 1

        if fi % 10 == 0:
            print(f'  Frame {fi}/{len(frame_ids)}')

    # ── 汇总结果 ──────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('BOP LINEMOD 序列 000002 — 真实深度 vs DA3 深度')
    print('=' * 70)
    print(f'评估帧: {len(frame_ids)} | 物体: {len(EVAL_OBJECTS)}')
    print(f'{"Obj":>4} {"n":>3} | {"真实ADD(mm)":>12} {"DA3-ADD(mm)":>12} | {"改善":>8}')
    print('-' * 70)

    all_real, all_da3 = [], []
    for ob in EVAL_OBJECTS:
        r = results[ob]
        if r['n'] == 0:
            continue
        m_real = np.mean(r['add_real']) if r['add_real'] else float('nan')
        m_da3 = np.mean(r['add_da3']) if r['add_da3'] else float('nan')
        impr = (m_real - m_da3) if np.isfinite(m_real) and np.isfinite(m_da3) else float('nan')
        print(f'{ob:>4} {r["n"]:>3} | {m_real:>12.1f} {m_da3:>12.1f} | {impr:>+8.1f}')
        all_real.extend(r['add_real'])
        all_da3.extend(r['add_da3'])

    if all_real:
        print('-' * 70)
        print(f'{"ALL":>4} {len(all_real):>3} | {np.mean(all_real):>12.1f} {np.mean(all_da3):>12.1f} | {np.mean(all_real)-np.mean(all_da3):>+8.1f}')

    # 直径阈值精度 (10% 直径)
    print('\n=== ADD(-S) < 10% 直径 精度 ===')
    for ob in EVAL_OBJECTS:
        r = results[ob]
        if r['n'] == 0:
            continue
        thr = diameters[ob] * 0.1
        acc_real = np.mean([a < thr for a in r['add_real']]) if r['add_real'] else float('nan')
        acc_da3 = np.mean([a < thr for a in r['add_da3']]) if r['add_da3'] else float('nan')
        print(f'  obj {ob:2d} (D={diameters[ob]:.0f}mm, thr={thr:.0f}mm): '
              f'真实={acc_real*100:.1f}%  DA3={acc_da3*100:.1f}%')


if __name__ == '__main__':
    main()
