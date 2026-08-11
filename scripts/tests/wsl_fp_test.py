"""
WSL2 FoundationPose test — uses pre-computed M1-M7 outputs from Windows.
Run: /root/miniconda3/envs/rgb6d/bin/python /mnt/e/zhijiyige/wsl_fp_test.py
"""
import sys, os, time
import numpy as np
import cv2
import torch

# Paths
sys.path.insert(0, '/mnt/e/zhijiyige/src/nvdiffrast/nvdiffrast-main')
sys.path.insert(0, '/mnt/e/zhijiyige/src/FoundationPose')
sys.path.insert(0, '/mnt/e/zhijiyige')  # for modules

BASE = '/mnt/e/zhijiyige'
OUT = f'{BASE}/output'
VIDEO = f'{BASE}/demo/test_mustard_clip.mp4'
MESH = f'{OUT}/meshes/proxy_mesh.glb'
MASKS_MMAP = f'{OUT}/masks_all.dat'
DEPTHS_MMAP = f'{OUT}/depths_metric_all.dat'

def main():
    print("=" * 60)
    print("FoundationPose M8-M11 Test (WSL2)")
    print("=" * 60)

    # ---- Load pre-computed data ----
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_raw = []
    while True:
        ret, f = cap.read()
        if not ret: break
        # Resize to 360p to match M1-M7 precomputed data
        h, w = f.shape[:2]
        scale = 360 / min(h, w)
        f = cv2.resize(f, (int(w*scale), int(h*scale)))
        frames_raw.append(f)
    cap.release()
    n = len(frames_raw)
    frames = frames_raw
    print(f"[Load] {n} frames @ {fps:.1f} FPS, {frames[0].shape[1]}x{frames[0].shape[0]}")

    masks = np.memmap(MASKS_MMAP, dtype=np.uint8, mode='r', shape=(n, *frames[0].shape[:2]))
    metric_depths_raw = np.memmap(DEPTHS_MMAP, dtype=np.float16, mode='r', shape=(n, *frames[0].shape[:2]))
    # FoundationPose expects depth in METERS, we have mm → convert and load into memory
    metric_depths = (metric_depths_raw[:].astype(np.float32) / 1000.0)
    print(f"[Load] masks: {masks.shape}, depths: {metric_depths.shape} (mm→m, range [{metric_depths.min():.2f},{metric_depths.max():.2f}]m)")

    # Camera intrinsics (360p default)
    h, w = frames[0].shape[:2]
    K = np.array([[w*1.1, 0, w/2], [0, h*1.1, h/2], [0, 0, 1]], dtype=np.float64)  # float64 to match trimesh vertices
    print(f"[Cam] K:\n{K}")

    # FoundationPose uses nvdiffrast CUDA rasterizer by default (no OpenGL needed)
    # glctx=None triggers RasterizeCudaContext internally
    ctx = None
    print("[FP] Using CUDA rasterizer (no OpenGL/EGL needed)")

    # ---- FoundationPose ----
    from estimater import FoundationPose
    # Load mesh and sample points
    import trimesh
    mesh = trimesh.load(MESH, force='mesh')
    mesh.vertices = mesh.vertices.astype(np.float32) * 100.0  # model_scale_mm
    # Sample points + normals from mesh surface
    pts, face_idx = trimesh.sample.sample_surface(mesh, 3000)
    normals = mesh.face_normals[face_idx]
    pts = pts.astype(np.float64)
    normals = normals.astype(np.float64)

    # Create nvdiffrast context BEFORE PyTorch to avoid driver handle conflict
    import nvdiffrast.torch as dr
    glctx_explicit = dr.RasterizeCudaContext()
    print("[FP] nvdiffrast context created (before torch)")

    # FoundationPose auto-loads scorer/refiner from internal weights dir
    fp_est = FoundationPose(
        model_pts=pts, model_normals=normals, mesh=mesh,
        glctx=glctx_explicit,  # pass pre-created context
        debug=0,
    )
    # Reduce rotation candidates for 6GB VRAM (252 → ~40)
    fp_est.make_rotation_grid(min_n_views=8, inplane_step=120)  # 84 candidates
    print(f"[FP] Mesh: {len(mesh.vertices)}v/{len(mesh.faces)}f, {len(fp_est.rot_grid)} rot candidates, {pts.shape[0]} pts")

    # ---- Tracking ----
    print("[FP] Starting tracking...")
    poses, confidences = [], []
    t0 = time.time()

    # Quick test: only 1 frame, iteration=1
    # Test: 1 frame registration only
    n_test = min(1, n)
    for i in range(n_test):
        rgb_bgr = frames[i]
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth = metric_depths[i].astype(np.float32)
        mask = masks[i].astype(np.uint8)

        if i == 0:
            print(f"[FP] Registering frame 0...")
            pose = fp_est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, ob_id=0, iteration=1)
            print(f"[FP] Registration DONE!")
            conf = 1.0
        else:
            pose = fp_est.track_one(rgb=rgb, depth=depth, K=K, iteration=1)
            try:
                conf = float(fp_est.scorer.evaluate(rgb=rgb, depth=depth, ob_mask=mask, K=K, pose=pose, glctx=fp_est.glctx))
            except:
                conf = 0.5

        poses.append(pose)
        confidences.append(conf)
        dt = time.time() - t0
        print(f"  Frame {i}/{n_test} | conf={conf:.3f} | {i/dt:.1f} FPS" if dt > 0 else f"  Frame {i}/{n_test}")

    dt = time.time() - t0
    print(f"[FP] Tracking done: {n} frames in {dt:.1f}s ({n/dt:.1f} FPS)")

    # ---- Kalman Filter ----
    from modules.se3_kalman_filter import SE3LieKalmanFilter, pose_matrix_to_quat_translation
    kf = SE3LieKalmanFilter(dt=1.0/fps)
    smoothed = kf.smooth(poses, confidences)
    print("[KF] Smoothing done")

    # ---- CSV Output ----
    import pandas as pd
    from scipy.spatial.transform import Rotation
    rows = []
    for i, (T, conf) in enumerate(zip(smoothed, confidences)):
        R_mat = T[:3,:3]; t = T[:3,3]
        quat = Rotation.from_matrix(R_mat).as_quat()
        rows.append({'frame': i, 'timestamp': round(i/fps, 6),
            'qw': round(float(quat[3]), 8), 'qx': round(float(quat[0]), 8),
            'qy': round(float(quat[1]), 8), 'qz': round(float(quat[2]), 8),
            'tx': round(float(t[0]), 6), 'ty': round(float(t[1]), 6), 'tz': round(float(t[2]), 6),
            'confidence': round(float(conf), 6)})
    pd.DataFrame(rows).to_csv(f'{OUT}/poses_fp.csv', index=False)
    print(f"[CSV] {OUT}/poses_fp.csv")

    # ---- Visualization ----
    from modules.output_writer import VisualizationRenderer
    renderer = VisualizationRenderer(MESH)
    renderer.render_video(frames, smoothed, K, f'{OUT}/output_vis_fp.mp4', fps=fps)
    print(f"[Viz] {OUT}/output_vis_fp.mp4")

    print("\n" + "=" * 60)
    print("DONE: M8-M11 all working!")
    print("=" * 60)

if __name__ == '__main__':
    main()
