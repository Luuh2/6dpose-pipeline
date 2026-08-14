#!/usr/bin/env python3
"""480p 公平对比: Stream3D vs TripoSR 网格追踪 (同参数 N_PTS=8000)"""
import os
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial.transform import Rotation

BASE = r'e:/zhijiyige'
H, W = 480, 640
K = np.load(f'{BASE}/output/stream3d/intermediate_480/K.npy').astype(np.float32)
masks = np.memmap(f'{BASE}/output/stream3d/intermediate_480/masks_xmem_full.dat',
                  dtype=np.uint8, mode='r', shape=(737, H, W))

RUNS = {
    'TripoSR@480': (f'{BASE}/output/stream3d/poses_triposr_480.csv',
                    f'{BASE}/output/stream3d/proxy_mesh_aligned_triposr.glb'),
    'Stream3D@480': (f'{BASE}/output/stream3d/poses_stream3d_480.csv',
                     f'{BASE}/output/stream3d/proxy_mesh_stream3d_mm.glb'),
}


def compute(csv, mesh_path, label):
    df = pd.read_csv(csv)
    n = len(df)
    mesh = trimesh.load(mesh_path, force='mesh')
    mesh.vertices = mesh.vertices.astype(np.float32) / 1000.0
    mn, mx = mesh.vertices.min(0), mesh.vertices.max(0)
    diag = np.linalg.norm(mx - mn) * 1000
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ], dtype=np.float32)
    Rs, ts = [], []
    for _, r in df.iterrows():
        Rs.append(Rotation.from_quat([r.qx, r.qy, r.qz, r.qw]).as_matrix())
        ts.append([r.tx, r.ty, r.tz])
    Rs = np.stack(Rs)
    ts = np.stack(ts)

    cerr, ious, rdel, tdel = [], [], [], []
    for i in range(n):
        pose = np.eye(4)
        pose[:3, :3] = Rs[i]
        pose[:3, 3] = ts[i]
        ch = np.hstack([corners, np.ones((8, 1))])
        cam = (pose @ ch.T).T[:, :3]
        im = (K @ cam.T).T
        im = im[:, :2] / im[:, 2:3]
        m = masks[i]
        ys, xs = np.where(m > 0)
        if len(ys) < 20:
            continue
        bb = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2])
        cerr.append(np.hypot(im[:, 0].mean() - bb[0], im[:, 1].mean() - bb[1]))
        x1, y1 = int(im[:, 0].min()), int(im[:, 1].min())
        x2, y2 = int(im[:, 0].max()), int(im[:, 1].max())
        tr = np.zeros((H, W), np.uint8)
        tr[max(0, y1):min(H - 1, y2) + 1, max(0, x1):min(W - 1, x2) + 1] = 1
        ious.append(np.logical_and(tr, m).sum() / max(np.logical_or(tr, m).sum(), 1))
        if i > 0:
            d = np.trace(Rs[i] @ Rs[i - 1].T) - 1
            rdel.append(np.degrees(np.arccos(np.clip(d / 2, -1, 1))))
            tdel.append(np.linalg.norm(ts[i] - ts[i - 1]) * 1000)
    cerr, ious = np.array(cerr), np.array(ious)
    rdel, tdel = np.array(rdel), np.array(tdel)
    print(f'{label}:')
    print(f'  mesh: {len(mesh.vertices)}v, diag={diag:.0f}mm, '
          f'bbox={((mx - mn) * 1000).round(1)}mm')
    print(f'  掩码贴合: mean={cerr.mean():.2f} max={cerr.max():.2f}px')
    print(f'  追踪IoU:  mean={ious.mean():.3f} min={ious.min():.3f} '
          f'<0.3占{(ious < 0.3).mean() * 100:.1f}%')
    print(f'  旋转平滑: mean={rdel.mean():.2f} max={rdel.max():.1f}°/帧')
    print(f'  平移:     mean={tdel.mean():.1f} max={tdel.max():.1f}mm')
    return cerr.mean(), ious.mean()


print('===== 480p 公平对比 (同参数 N_PTS=8000, 仅网格不同) =====')
for name, (csv, mesh) in RUNS.items():
    if not os.path.exists(csv):
        print(f'[跳过] {csv} 不存在')
        continue
    compute(csv, mesh, name)
