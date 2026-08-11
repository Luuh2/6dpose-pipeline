#!/usr/bin/env python3
"""
viz_stream3d_compare.py — 生成 Stream3D vs TripoSR 追踪对比可视化
=================================================================
1) 逐帧曲线对比图 (bbox 中心差 / IoU / 旋转增量) — 英文标签
2) 关键帧叠加: 视频帧 + XMem mask 轮廓 + 两套追踪 bbox 投影

用法: python scripts/viz_stream3d_compare.py
"""
import os
import numpy as np
import cv2
import pandas as pd
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = f'{BASE}/demo/test_mustard.mp4'
MASKS = f'{BASE}/output/intermediate/masks_xmem_full.dat'
K_PATH = f'{BASE}/output/K.npy'
H, W = 360, 480

RUNS = {
    'TripoSR': (f'{BASE}/output/poses.csv',
                f'{BASE}/output/meshes/proxy_mesh_aligned.glb', '#1f77b4'),
    'Stream3D': (f'{BASE}/output/stream3d/poses.csv',
                 f'{BASE}/output/stream3d/proxy_mesh_stream3d_mm.glb', '#ff7f0e'),
}


def load_run(csv_path, mesh_path):
    df = pd.read_csv(csv_path)
    n = len(df)
    mesh = trimesh.load(mesh_path, force='mesh')
    mesh.vertices = mesh.vertices.astype(np.float32) / 1000.0
    mn, mx = mesh.vertices.min(0), mesh.vertices.max(0)
    corners = np.array([[mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
                        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
                        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
                        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]]],
                       dtype=np.float32)
    Rs, ts = [], []
    for _, r in df.iterrows():
        Rs.append(Rotation.from_quat([r.qx, r.qy, r.qz, r.qw]).as_matrix())
        ts.append([r.tx, r.ty, r.tz])
    return n, np.stack(Rs), np.stack(ts), corners


def project(pose, corners, K):
    ch = np.hstack([corners, np.ones((8, 1))])
    cam = (pose @ ch.T).T[:, :3]
    img = (K @ cam.T).T
    return img[:, :2] / img[:, 2:3]


def main():
    K = np.load(K_PATH).astype(np.float32)
    masks = np.memmap(MASKS, dtype=np.uint8, mode='r', shape=(737, H, W))
    cap = cv2.VideoCapture(VIDEO)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        h, w = f.shape[:2]
        s = 360 / min(h, w)
        frames.append(cv2.resize(f, (int(w * s), int(h * s))))
    cap.release()
    n = len(frames)

    data = {}
    for name, (csv, mesh_p, color) in RUNS.items():
        nn, Rs, ts, corners = load_run(csv, mesh_p)
        bbox_err, ious, rot_deltas = [], [], []
        for i in range(n):
            pose = np.eye(4)
            pose[:3, :3] = Rs[i]
            pose[:3, 3] = ts[i]
            img = project(pose, corners, K)
            mask = masks[i]
            ys, xs = np.where(mask > 0)
            if len(ys) < 20:
                bbox_err.append(np.nan); ious.append(np.nan); continue
            bb = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2])
            u, v = img[:, 0].mean(), img[:, 1].mean()
            bbox_err.append(np.hypot(u - bb[0], v - bb[1]))
            x1, y1, x2, y2 = img[:, 0].min(), img[:, 1].min(), img[:, 0].max(), img[:, 1].max()
            track = np.zeros((H, W), np.uint8)
            y1i, y2i = max(0, int(y1)), min(H - 1, int(y2))
            x1i, x2i = max(0, int(x1)), min(W - 1, int(x2))
            track[y1i:y2i + 1, x1i:x2i + 1] = 1
            inter = np.logical_and(track, mask).sum()
            union = np.logical_or(track, mask).sum()
            ious.append(inter / max(union, 1))
        for i in range(1, n):
            d = np.trace(Rs[i] @ Rs[i - 1].T) - 1
            rot_deltas.append(np.degrees(np.arccos(np.clip(d / 2, -1, 1))))
        data[name] = (np.array(bbox_err), np.array(ious), np.array(rot_deltas), color)

    # ── 1. 曲线对比图 ──
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    titles = ['Mesh bbox center vs XMem mask bbox center (px)',
              'Tracking AABB vs XMem mask IoU',
              'Frame-to-frame rotation delta (deg)']
    ylabels = ['px', 'IoU', 'deg/frame']
    for ax, idx, t, yl in zip(axes, range(3), titles, ylabels):
        for name, (be, iou, rd, color) in data.items():
            y = be if idx == 0 else (iou if idx == 1 else rd)
            x = np.arange(1, len(y) + 1) if idx == 2 else np.arange(len(y))
            ax.plot(x, y, color=color, label=name, linewidth=0.7, alpha=0.85)
        ax.set_title(t, fontsize=11)
        ax.set_ylabel(yl, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    axes[2].set_xlabel('frame')
    plt.tight_layout()
    plt.savefig(f'{BASE}/output/stream3d/tracking_eval_compare.png', dpi=110)
    plt.close()
    print('saved: output/stream3d/tracking_eval_compare.png')

    # ── 2. 关键帧叠加图 ──
    keyframes = [0, 120, 260, 400, 520, 650]
    cols = 3
    rows = (len(keyframes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.4 * rows))
    axes = np.array(axes).reshape(-1)
    for k, i in enumerate(keyframes):
        ax = axes[k]
        frame = frames[i].copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = masks[i]
        # XMem mask 轮廓
        cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, cont, -1, (0, 255, 0), 1)
        for name, (csv, mesh_p, color) in RUNS.items():
            _, Rs, ts, corners = data[name] if False else load_run(csv, mesh_p)
            pose = np.eye(4)
            pose[:3, :3] = Rs[i]
            pose[:3, 3] = ts[i]
            img = project(pose, corners, K)
            bb_color = (int(color[1:] and int(color[1:3], 16)),
                        int(color[3:5], 16), int(color[5:7], 16))
            x1, y1, x2, y2 = img[:, 0].min(), img[:, 1].min(), img[:, 0].max(), img[:, 1].max()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          bb_color, 2)
        ax.imshow(frame)
        ax.set_title(f'frame {i}')
        ax.axis('off')
    for k in range(len(keyframes), len(axes)):
        axes[k].axis('off')
    handles = [plt.Line2D([0], [0], color=data['TripoSR'][3], lw=3, label='TripoSR bbox'),
               plt.Line2D([0], [0], color=data['Stream3D'][3], lw=3, label='Stream3D bbox'),
               plt.Line2D([0], [0], color='green', lw=2, label='XMem mask')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f'{BASE}/output/stream3d/tracking_keyframes.png', dpi=110,
                bbox_inches='tight')
    plt.close()
    print('saved: output/stream3d/tracking_keyframes.png')


if __name__ == '__main__':
    main()
