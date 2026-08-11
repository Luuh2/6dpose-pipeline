"""FoundationPose 5-frame tracking + Kalman + output test."""
import sys
sys.path.insert(0,'/mnt/e/zhijiyige/src/nvdiffrast/nvdiffrast-main')
sys.path.insert(0,'/mnt/e/zhijiyige/src/FoundationPose')

import cv2, numpy as np, torch, time, os

BASE = '/mnt/e/zhijiyige'
OUT = f'{BASE}/output'

# 1. Load data
cap = cv2.VideoCapture(f'{BASE}/demo/test_mustard_clip.mp4')
frames, fps = [], cap.get(cv2.CAP_PROP_FPS)
while True:
    ret, f = cap.read()
    if not ret: break
    h,w = f.shape[:2]; s = 360/min(h,w)
    frames.append(cv2.resize(f, (int(w*s), int(h*s))))
cap.release()
n = len(frames)
print(f"[Load] {n}f {frames[0].shape[1]}x{frames[0].shape[0]} @{fps:.0f}fps")

depths_raw = np.memmap(f'{OUT}/depths_metric_all.dat', dtype=np.float16, mode='r', shape=(n,360,480))
masks = np.memmap(f'{OUT}/masks_all.dat', dtype=np.uint8, mode='r', shape=(n,360,480))
depths = (depths_raw[:].astype(np.float32) / 1000.0)
print(f"[Data] depth range [{depths.min():.2f},{depths.max():.2f}]m, mask={masks[0].sum()}px")

# Heuristic K estimation — NO ground truth!
# f = max(w,h) * 1.1 (as specified in the plan)
h_proc, w_proc = frames[0].shape[:2]
f_est = max(w_proc, h_proc) * 1.1
K = np.array([[f_est, 0, w_proc/2], [0, f_est, h_proc/2], [0, 0, 1]], dtype=np.float64)
print(f"[Cam] Heuristic K (f={f_est:.0f}px, from plan spec, NO GT):\n{K}")

# 2. Create nvdiffrast context FIRST
import nvdiffrast.torch as dr
glctx = dr.RasterizeCudaContext()

# 3. Load FoundationPose
from estimater import FoundationPose
import trimesh
mesh = trimesh.load(f'{OUT}/meshes/proxy_mesh.glb', force='mesh')
# TripoSR mesh in [-0.5,0.5]3 normalized. FoundationPose uses METERS.
# Mustard bottle ~15cm → scale to 0.12m approximate
mesh.vertices = mesh.vertices.astype(np.float32) * 0.12
pts, fidx = trimesh.sample.sample_surface(mesh, 3000)
normals = mesh.face_normals[fidx]

fp = FoundationPose(model_pts=pts, model_normals=normals, mesh=mesh, glctx=glctx, debug=0)
fp.make_rotation_grid(min_n_views=8, inplane_step=120)
print(f"[FP] {len(fp.rot_grid)} rots, VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")

# 4. Track all 76 frames
poses, confidences = [], []
t0 = time.time()
n_frames = n  # Full video: all 76 frames

for i in range(n_frames):
    rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
    depth = depths[i]; mask = masks[i].astype(np.uint8)

    if i == 0:
        pose = fp.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, ob_id=0, iteration=2)
        conf = 1.0
    else:
        pose = fp.track_one(rgb=rgb, depth=depth, K=K, iteration=1)
        conf = 0.8

    poses.append(pose); confidences.append(conf)
    dt = time.time() - t0
    print(f"  Frame {i}/{n_frames} | conf={conf:.3f} | {i/dt:.1f} FPS" if dt>0 else f"  Frame {i}/{n_frames}")

dt = time.time() - t0
print(f"[FP] Done: {n_frames}f in {dt:.1f}s ({n_frames/dt:.1f} FPS), VRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB peak")

# 5. Kalman filter
sys.path.insert(0, BASE)
from modules.se3_kalman_filter import SE3LieKalmanFilter
kf = SE3LieKalmanFilter(dt=1.0/fps)
smoothed = kf.smooth(poses, confidences)
print("[KF] Done")

# 6. CSV output
from modules.output_writer import PoseOutputWriter, VisualizationRenderer
timestamps = [i/fps for i in range(n_frames)]
PoseOutputWriter.write_csv(smoothed, confidences, timestamps, f'{OUT}/poses_final.csv')

# 7. Visualization
renderer = VisualizationRenderer(f'{OUT}/meshes/proxy_mesh.glb')
renderer.render_video(frames[:n_frames], smoothed, K, f'{OUT}/output_final.avi', fps=fps)

print(f"\nDONE! CSV: {OUT}/poses_final.csv, Video: {OUT}/output_final.mp4")
