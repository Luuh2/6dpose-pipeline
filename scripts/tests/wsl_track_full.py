"""FoundationPose full 737-frame tracking — heuristic K, RGB-only."""
import sys
sys.path.insert(0,'/mnt/e/zhijiyige/src/nvdiffrast/nvdiffrast-main')
sys.path.insert(0,'/mnt/e/zhijiyige/src/FoundationPose')

import cv2, numpy as np, torch, time, os

BASE = '/mnt/e/zhijiyige'
OUT = f'{BASE}/output_full'
VIDEO = f'{BASE}/demo/test_mustard.mp4'

# 1. Load frames
cap = cv2.VideoCapture(VIDEO)
frames, fps = [], cap.get(cv2.CAP_PROP_FPS)
while True:
    ret, f = cap.read()
    if not ret: break
    h,w = f.shape[:2]; s = 360/min(h,w)
    frames.append(cv2.resize(f, (int(w*s), int(h*s))))
cap.release()
n = len(frames)
h_proc, w_proc = frames[0].shape[:2]
print(f"[Load] {n}f {w_proc}x{h_proc} @{fps:.0f}fps")

# 2. Load pre-computed data
depths_raw = np.memmap(f'{OUT}/depths_metric.dat', dtype=np.float16, mode='r', shape=(n,h_proc,w_proc))
masks = np.memmap(f'{OUT}/masks.dat', dtype=np.uint8, mode='r', shape=(n,h_proc,w_proc))
depths = (depths_raw[:].astype(np.float32) / 1000.0)
print(f"[Data] depth [{depths.min():.2f},{depths.max():.2f}]m mask={masks[0].sum()}px")

# 3. Heuristic K — NO GROUND TRUTH
f_est = max(w_proc, h_proc) * 1.1
K = np.array([[f_est, 0, w_proc/2], [0, f_est, h_proc/2], [0, 0, 1]], dtype=np.float64)
print(f"[Cam] Heuristic K (f={f_est:.0f}):\n{K}")

# 4. nvdiffrast context FIRST
import nvdiffrast.torch as dr
glctx = dr.RasterizeCudaContext()

# 5. FoundationPose
from estimater import FoundationPose
import trimesh
mesh = trimesh.load(f'{OUT}/meshes/proxy_mesh.glb', force='mesh')
mesh.vertices = mesh.vertices.astype(np.float32) * 0.12
pts, fidx = trimesh.sample.sample_surface(mesh, 3000)
normals = mesh.face_normals[fidx]

fp = FoundationPose(model_pts=pts, model_normals=normals, mesh=mesh, glctx=glctx, debug=0)
fp.make_rotation_grid(min_n_views=8, inplane_step=120)
print(f"[FP] {len(fp.rot_grid)} rots VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")

# 6. Track ALL 737 frames
poses, confidences = [], []
t0 = time.time()
for i in range(n):
    rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
    depth = depths[i]; mask = masks[i].astype(np.uint8)
    if i == 0:
        pose = fp.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, ob_id=0, iteration=2)
        conf = 1.0
    else:
        pose = fp.track_one(rgb=rgb, depth=depth, K=K, iteration=1)
        conf = 0.8
    poses.append(pose); confidences.append(conf)
    if i % 100 == 0:
        dt = time.time()-t0
        print(f"  F{i}/{n} | {i/dt:.1f}fps" if dt>0 else f"  F{i}/{n}")

dt = time.time()-t0
print(f"[FP] {n}f in {dt:.1f}s ({n/dt:.1f}fps) VRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB")

# 7. Kalman + Output
sys.path.insert(0, BASE)
from modules.se3_kalman_filter import SE3LieKalmanFilter
kf = SE3LieKalmanFilter(dt=1.0/fps)
smoothed = kf.smooth(poses, confidences)

from modules.output_writer import PoseOutputWriter, VisualizationRenderer
timestamps = [i/fps for i in range(n)]
PoseOutputWriter.write_csv(smoothed, confidences, timestamps, f'{OUT}/poses.csv')
renderer = VisualizationRenderer(f'{OUT}/meshes/proxy_mesh.glb')
renderer.render_video(frames, smoothed, K, f'{OUT}/output_vis.avi', fps=fps)
print(f"\nDONE! CSV: {OUT}/poses.csv  Video: {OUT}/output_vis.avi")
