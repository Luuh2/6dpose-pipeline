#!/usr/bin/env python3
"""
eval_stream3d_tracking.py — Stream3D vs TripoSR 追踪质量评估
=================================================================
在 Windows 端对同一视频 (demo/test_mustard.mp4) 的两套网格追踪结果做量化对比:

  A) TripoSR 网格追踪  -> output/poses.csv            (mesh: proxy_mesh_aligned.glb)
  B) Stream3D 网格追踪 -> output/stream3d/poses.csv   (mesh: proxy_mesh_stream3d_mm.glb)

指标 (与 METRICS.md 方法一致, 无真值 pose 的代理指标):
  1. 掩码贴合:   mesh bbox 投影中心 vs XMem mask bbox 中心 (mean/max px)
  2. 掩码质心:   mesh bbox 投影中心 vs XMem mask 质心 (mean/max px)
  3. 追踪 IoU:   追踪 AABB (投影) vs mask 的 IoU (mean/min/低帧数)
  4. 旋转平滑度: 帧间三轴方向夹角 (mean/max deg/frame)
  5. 平移连续性: 帧间平移增量 (mean/max mm)
  6. 旋转跟随范围: 绕相机 z 轴翻滚角范围 (deg)
  7. 置信度:     CSV 内 confidence 统计

用法:
  python scripts/eval_stream3d_tracking.py
"""
import os
import numpy as np
import cv2
import pandas as pd
import trimesh
from scipy.spatial.transform import Rotation

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = f'{BASE}/demo/test_mustard.mp4'
MASKS = f'{BASE}/output/intermediate/masks_xmem_full.dat'
K_PATH = f'{BASE}/output/K.npy'
H, W = 360, 480

RUNS = {
    'TripoSR@3k': {  # 原始 Windows 运行 (N_PTS=3000, rot_grid 8x120)
        'poses': f'{BASE}/output/poses.csv',
        'mesh': f'{BASE}/output/meshes/proxy_mesh_aligned.glb',
    },
    'TripoSR@8k': {  # 公平基线 (N_PTS=8000, rot_grid 16x60, 服务器重跑)
        'poses': f'{BASE}/output/stream3d/poses_triposr_fair.csv',
        'mesh': f'{BASE}/output/meshes/proxy_mesh_aligned.glb',
    },
    'Stream3D@8k': {  # Stream3D 网格 (服务器, N_PTS=8000)
        'poses': f'{BASE}/output/stream3d/poses.csv',
        'mesh': f'{BASE}/output/stream3d/proxy_mesh_stream3d_mm.glb',
    },
}


def project_corners(pose, corners, K):
    """将 mesh bbox 8 角点 (meters) 投影到图像平面, 返回 (u,v) (N,2)"""
    corners_h = np.hstack([corners, np.ones((8, 1))])
    cam = (pose @ corners_h.T).T[:, :3]
    img = (K @ cam.T).T
    valid = img[:, 2] > 0.01
    img = img[:, :2] / img[:, 2:3]
    return img, valid


def axis_angle_deg(Ra, Rb):
    """两旋转矩阵间轴角 (deg)"""
    d = np.trace(Ra @ Rb.T) - 1
    return np.degrees(np.arccos(np.clip(d / 2, -1, 1)))


def mask_bbox_center(mask):
    ys, xs = np.where(mask > 0)
    if len(ys) < 20:
        return None, None
    return np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2]), \
        np.array([xs.mean(), ys.mean()])


def compute_run(name, cfg):
    print(f'\n=== {name} ===')
    df = pd.read_csv(cfg['poses'])
    n = len(df)
    mesh = trimesh.load(cfg['mesh'], force='mesh')
    mesh.vertices = mesh.vertices.astype(np.float32) / 1000.0  # mm -> m
    mn, mx = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ], dtype=np.float32)
    mesh_diag = np.linalg.norm(mx - mn)
    print(f'  mesh: {len(mesh.vertices)}v, bbox={((mx-mn)*1000).round(1)}mm, '
          f'diag={mesh_diag*1000:.0f}mm')

    K = np.load(K_PATH).astype(np.float32)
    masks = np.memmap(MASKS, dtype=np.uint8, mode='r', shape=(n, H, W))

    # 姿态矩阵序列
    Rs, ts = [], []
    for _, row in df.iterrows():
        R = Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
        Rs.append(R)
        ts.append([row.tx, row.ty, row.tz])
    Rs = np.stack(Rs)
    ts = np.stack(ts)
    conf = df['confidence'].values

    # 1/2/3: 掩码贴合 (逐帧)
    bbox_center_err, centroid_err, ious = [], [], []
    bad = 0
    for i in range(n):
        mask = masks[i]
        pose = np.eye(4)
        pose[:3, :3] = Rs[i]
        pose[:3, 3] = ts[i]
        img, valid = project_corners(pose, corners, K)
        if not valid.all():
            bad += 1
            continue
        bb_c, ct_c = mask_bbox_center(mask)
        if bb_c is None:
            continue
        u = img[:, 0].mean()
        v = img[:, 1].mean()
        bbox_center_err.append(np.hypot(u - bb_c[0], v - bb_c[1]))
        centroid_err.append(np.hypot(u - ct_c[0], v - ct_c[1]))
        # 追踪 AABB vs mask IoU
        x1, y1 = img[:, 0].min(), img[:, 1].min()
        x2, y2 = img[:, 0].max(), img[:, 1].max()
        if x2 <= x1 or y2 <= y1:
            ious.append(0.0)
            continue
        track = np.zeros((H, W), np.uint8)
        y1i, y2i = max(0, int(y1)), min(H - 1, int(y2))
        x1i, x2i = max(0, int(x1)), min(W - 1, int(x2))
        track[y1i:y2i + 1, x1i:x2i + 1] = 1
        inter = np.logical_and(track, mask).sum()
        union = np.logical_or(track, mask).sum()
        ious.append(inter / max(union, 1))
    bbox_center_err = np.array(bbox_center_err)
    centroid_err = np.array(centroid_err)
    ious = np.array(ious)

    # 4: 旋转平滑度 (帧间轴角)
    rot_deltas = np.array([axis_angle_deg(Rs[i], Rs[i - 1])
                           for i in range(1, n)])

    # 5: 平移连续性 (帧间平移增量, mm)
    trans_deltas = np.linalg.norm(np.diff(ts, axis=0), axis=1) * 1000.0

    # 6: 绕相机 z 轴翻滚角范围
    # 物体 x 轴在图像平面的角度 alpha = atan2(Ry_x, Rx_x) (r 列投影)
    alphas = np.arctan2(Rs[:, 1, 0], Rs[:, 0, 0])
    alphas = np.degrees(alphas)
    # 展开避免环绕
    alphas_unwrapped = np.unwrap(alphas)
    yaw_range = alphas_unwrapped.max() - alphas_unwrapped.min()

    # 7: 置信度
    conf_low = (conf < 0.5).sum()

    # 掩码 IoU 低帧统计 (追踪贴合度差)
    iou_low = (ious < 0.30).sum() if len(ious) else 0

    stats = {
        'frames': n,
        'mesh_verts': len(mesh.vertices),
        'mesh_diag_mm': mesh_diag * 1000,
        'bbox_center_err_mean': bbox_center_err.mean(),
        'bbox_center_err_max': bbox_center_err.max(),
        'centroid_err_mean': centroid_err.mean(),
        'centroid_err_max': centroid_err.max(),
        'iou_mean': ious.mean(),
        'iou_min': ious.min(),
        'iou_low_pct': 100.0 * iou_low / max(len(ious), 1),
        'rot_smooth_mean': rot_deltas.mean(),
        'rot_smooth_max': rot_deltas.max(),
        'trans_mean_mm': trans_deltas.mean(),
        'trans_max_mm': trans_deltas.max(),
        'yaw_range_deg': yaw_range,
        'conf_mean': conf.mean(),
        'conf_low_count': int(conf_low),
        'bad_proj': bad,
    }
    print(f'  掩码贴合(bbox中心): mean={stats["bbox_center_err_mean"]:.1f}px, '
          f'max={stats["bbox_center_err_max"]:.1f}px')
    print(f'  掩码质心贴合:       mean={stats["centroid_err_mean"]:.1f}px, '
          f'max={stats["centroid_err_max"]:.1f}px')
    print(f'  追踪IoU:            mean={stats["iou_mean"]:.3f}, '
          f'min={stats["iou_min"]:.3f}, <0.3 占 {stats["iou_low_pct"]:.1f}%')
    print(f'  旋转平滑:           mean={stats["rot_smooth_mean"]:.2f}°/帧, '
          f'max={stats["rot_smooth_max"]:.1f}°/帧')
    print(f'  平移连续性:         mean={stats["trans_mean_mm"]:.1f}mm, '
          f'max={stats["trans_max_mm"]:.1f}mm')
    print(f'  翻滚范围:           {stats["yaw_range_deg"]:.0f}°')
    print(f'  置信度:             mean={stats["conf_mean"]:.3f}, '
          f'<0.5 共 {stats["conf_low_count"]} 帧, 投影失败 {bad}')
    return stats, (bbox_center_err, centroid_err, ious, rot_deltas, trans_deltas)


def main():
    results = {}
    series = {}
    for name, cfg in RUNS.items():
        stats, s = compute_run(name, cfg)
        results[name] = stats
        series[name] = s

    names = list(RUNS.keys())
    print('\n\n===== 对比汇总 =====')
    header = f'{"指标":<22}'
    for name in names:
        header += f'{name:>18}'
    print(header)
    rows = [
        ('网格顶点数', 'mesh_verts', '{:d}'),
        ('网格对角线(mm)', 'mesh_diag_mm', '{:.0f}'),
        ('掩码贴合 mean(px)', 'bbox_center_err_mean', '{:.2f}'),
        ('掩码贴合 max(px)', 'bbox_center_err_max', '{:.2f}'),
        ('掩码质心 mean(px)', 'centroid_err_mean', '{:.2f}'),
        ('掩码质心 max(px)', 'centroid_err_max', '{:.2f}'),
        ('追踪IoU mean', 'iou_mean', '{:.3f}'),
        ('IoU<0.3 占比(%)', 'iou_low_pct', '{:.1f}'),
        ('旋转平滑 mean(°/帧)', 'rot_smooth_mean', '{:.2f}'),
        ('旋转平滑 max(°/帧)', 'rot_smooth_max', '{:.1f}'),
        ('平移 mean(mm)', 'trans_mean_mm', '{:.1f}'),
        ('平移 max(mm)', 'trans_max_mm', '{:.1f}'),
        ('翻滚范围(°)', 'yaw_range_deg', '{:.0f}'),
        ('置信度 mean', 'conf_mean', '{:.3f}'),
        ('置信度<0.5 帧数', 'conf_low_count', '{:d}'),
    ]
    for label, key, fmt in rows:
        line = f'{label:<22}'
        for name in names:
            line += f'{fmt.format(results[name][key]):>18}'
        print(line)

    # 逐帧曲线图 (bbox 中心差 + IoU)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t = np.arange(results[names[0]]['frames'])
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        colors = ['#1f77b4', '#d62728', '#ff7f0e']
        for ax, idx, title, ylab in [
            (axes[0], 0, 'Mesh bbox center vs XMem mask bbox center (px)', 'px'),
            (axes[1], 2, 'Tracking AABB vs XMem mask IoU', 'IoU'),
            (axes[2], 3, 'Frame-to-frame rotation delta (deg)', 'deg/frame'),
        ]:
            for j, name in enumerate(names):
                ax.plot(t[1:] if idx == 3 else t, series[name][idx],
                        color=colors[j % len(colors)], label=name,
                        linewidth=0.7, alpha=0.85)
            ax.set_title(title)
            ax.set_ylabel(ylab)
            ax.legend()
            ax.grid(alpha=0.3)
        axes[2].set_xlabel('frame')
        plt.tight_layout()
        out = f'{BASE}/output/stream3d/tracking_eval_compare.png'
        plt.savefig(out, dpi=110)
        print(f'\n对比图: {out}')
    except ImportError:
        print('\n[skip] matplotlib 不可用, 跳过曲线图')


if __name__ == '__main__':
    main()
