#!/usr/bin/env python3
"""
wsl_track_debug.py — 调试: 每帧深度下 track_one 能否跟上物体移动
只跑前 120 帧, 对比 track 投影 vs 实际物体位置
"""
import sys, os, time, cv2, numpy as np, torch

BASE = '/mnt/e/zhijiyige'
sys.path.insert(0, f'{BASE}/src/nvdiffrast/nvdiffrast-main')
sys.path.insert(0, f'{BASE}/src/FoundationPose')
sys.path.insert(0, BASE)

OUT = f'{BASE}/output'
VIDEO = f'{BASE}/demo/test_mustard.mp4'
DEPTHS = f'{OUT}/intermediate/depths_metric.dat'
MASKS = f'{OUT}/intermediate/masks.dat'
MESH = f'{OUT}/meshes/proxy_mesh_aligned.glb'

# 只跑前 120 帧
N_TEST = 120

cap = cv2.VideoCapture(VIDEO)
frames = []
fps = cap.get(cv2.CAP_PROP_FPS)
while len(frames) < N_TEST:
    ret, f = cap.read()
    if not ret: break
    h, w = f.shape[:2]; s = 360/min(h,w)
    frames.append(cv2.resize(f, (int(w*s), int(h*s))))
cap.release()
n = len(frames); hp, wp = frames[0].shape[:2]

depths_raw = np.memmap(DEPTHS, dtype=np.float16, mode='r', shape=(737, hp, wp))
depths_m = depths_raw[:n].astype(np.float32)
masks = np.memmap(MASKS, dtype=np.uint8, mode='r', shape=(737, hp, wp))
K = np.load(f'{OUT}/K.npy')
print(f'{n}f, depth {depths_m.min():.2f}-{depths_m.max():.2f}m')

import nvdiffrast.torch as dr
import trimesh
from estimater import FoundationPose
from Utils import set_logging_format, set_seed
set_logging_format(); set_seed(0)
glctx = dr.RasterizeCudaContext()

mesh = trimesh.load(MESH, force='mesh')
mesh.vertices = mesh.vertices.astype(np.float32) / 1000.0
if len(mesh.vertices) > 5000:
    import open3d as o3d
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64)),
        triangles=o3d.utility.Vector3iVector(mesh.faces.astype(np.int32)))
    o3d_mesh = o3d_mesh.simplify_quadric_decimation(4000)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices).astype(np.float32),
        faces=np.asarray(o3d_mesh.triangles).astype(np.int64))

pts, face_idx = trimesh.sample.sample_surface(mesh, 3000)
normals = mesh.face_normals[face_idx].astype(np.float32)
pts = pts.astype(np.float32)
fp = FoundationPose(model_pts=pts, model_normals=normals, mesh=mesh, glctx=glctx, debug=0)
fp.make_rotation_grid(min_n_views=8, inplane_step=120)
print(f'FP ready, {len(fp.rot_grid)} rots')

# 追踪
poses = []
t0 = time.time()
for i in range(n):
    rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
    depth = depths_m[i]
    mask = masks[i].astype(np.uint8)
    if i == 0:
        pose = fp.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, ob_id=0, iteration=2)
    else:
        pose = fp.track_one(rgb=rgb, depth=depth, K=K, iteration=1)
    poses.append(pose)
print(f'tracked {n}f in {time.time()-t0:.1f}s')

# 保存 pose 便于分析
np.save(f'{OUT}/debug_poses_{n}.npy', np.array(poses))
print('saved', f'{OUT}/debug_poses_{n}.npy')

# 追踪 bbox 中心 (2D)
from modules.output_writer import VisualizationRenderer
renderer = VisualizationRenderer(MESH, model_scale=0.001)
print('\nframe | track_bbox_center | 说明')
for fi in [0, 30, 60, 90, 119]:
    T = poses[fi]
    pts2d = renderer.project_points(renderer.bbox_3d, T, K)
    cx, cy = pts2d[:,0].mean(), pts2d[:,1].mean()
    print(f'  {fi:4d} | ({cx:.0f}, {cy:.0f})')
