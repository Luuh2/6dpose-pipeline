#!/usr/bin/env python3
"""
prepare_stream3d_data.py — 将追踪输出转成 Stream3D GSO 格式输入
=================================================================
把 demo 的追踪输出 (poses.csv + XMem mask + DA3 深度) 转换成
Stream3D 需要的目录结构:

<object_dir>/
  render_spiral_100/
    images/           # RGB 帧 (000000.png)
    masks/            # 前景掩码 (000000.png)
    da3/
      camera_poses.txt   # world-to-camera 位姿 (每帧 16 浮点)
      results_output/
        frame_000000.npz   # {depth, intrinsics}

用法:
  python scripts/prepare_stream3d_data.py \
    --frames demo/test_mustard.mp4 \
    --poses output/poses.csv \
    --masks output/intermediate/masks_xmem_full.dat \
    --depths output/intermediate/depths_metric.dat \
    --K output/K.npy \
    --out /tmp/STREAM3D/stream3d_data/bottle
"""
import argparse, os, sys, json
import numpy as np
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', required=True)
    parser.add_argument('--poses', required=True)
    parser.add_argument('--masks', required=True)
    parser.add_argument('--depths', required=True)
    parser.add_argument('--K', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--sample_every', type=int, default=1,
                        help='每 N 帧采样一帧 (控制数据量)')
    args = parser.parse_args()

    # 输出目录
    img_dir = os.path.join(args.out, 'render_spiral_100', 'images')
    mask_dir = os.path.join(args.out, 'render_spiral_100', 'masks')
    da3_dir = os.path.join(args.out, 'render_spiral_100', 'da3')
    da3_out = os.path.join(da3_dir, 'results_output')
    for d in [img_dir, mask_dir, da3_out]:
        os.makedirs(d, exist_ok=True)

    # 加载数据
    cap = cv2.VideoCapture(args.frames)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    n_video = len(frames)
    print(f'视频帧: {n_video}')

    df = pd.read_csv(args.poses)
    n_pose = len(df)
    print(f'位姿: {n_pose}')

    # memmap — 用实际帧分辨率 (native 帧尺寸, 与 masks/depth 一致)
    h, w = frames[0].shape[:2]
    masks = np.memmap(args.masks, dtype=np.uint8, mode='r',
                      shape=(n_video, h, w))
    depths = np.memmap(args.depths, dtype=np.float16, mode='r',
                       shape=(n_video, h, w))
    K = np.load(args.K).astype(np.float32)
    print(f'memmap: masks {masks.shape}, depths {depths.shape}')

    # 采样帧索引 (保证 masks/poses/depth 数量一致)
    n = min(n_video, n_pose)
    idxs = list(range(0, n, args.sample_every))
    print(f'采样 {len(idxs)} 帧')

    pose_lines = []
    last_mask = None  # 空掩码时沿用上一帧
    for out_i, vid_i in enumerate(idxs):
        # 帧号 (6 位, 以整数结尾)
        stem = f'{out_i:06d}'

        # ── 图像 (缩放到 360x480 匹配掩码/深度) ──
        img = frames[vid_i]
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        cv2.imwrite(os.path.join(img_dir, f'{stem}.png'), img)

        # ── 掩码 (单通道) ──
        m = masks[vid_i].astype(np.uint8)
        if m.sum() == 0 and last_mask is not None:
            m = last_mask.copy()  # 空掩码沿用上一帧, 避免 Stream3D 裁剪除零
            print(f'  [warn] frame {vid_i} 空掩码, 沿用上一帧')
        if m.sum() > 0:
            last_mask = m.copy()
        cv2.imwrite(os.path.join(mask_dir, f'{stem}.png'), m)

        # ── 深度 npz {depth, intrinsics} ──
        d = depths[vid_i].astype(np.float32)  # meters
        np.savez(os.path.join(da3_out, f'frame_{stem}.npz'),
                 depth=d, intrinsics=K)

        # ── 位姿 (world-to-camera) ──
        # 追踪 pose: 物体在相机系 (c2w 物体位姿) = T_cam_obj
        row = df.iloc[vid_i]
        R = Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
        T_cam_obj = np.eye(4)
        T_cam_obj[:3, :3] = R
        T_cam_obj[:3, 3] = [row.tx, row.ty, row.tz]
        # camera w2c = inv(T_cam_obj)  (物体系=世界系, 相机位姿 w2c)
        T_w2c = np.linalg.inv(T_cam_obj)
        pose_lines.append(' '.join(f'{v:.6f}' for v in T_w2c.flatten()))

    # 写入 camera_poses.txt
    with open(os.path.join(da3_dir, 'camera_poses.txt'), 'w') as f:
        f.write('\n'.join(pose_lines) + '\n')

    print(f'完成! 输出到 {args.out}')
    print(f'  图像: {len(idxs)} 张 -> {img_dir}')
    print(f'  掩码: {len(idxs)} 张 -> {mask_dir}')
    print(f'  深度: {len(idxs)} 个 npz -> {da3_out}')
    print(f'  位姿: {len(pose_lines)} 行 -> camera_poses.txt')


if __name__ == '__main__':
    main()
