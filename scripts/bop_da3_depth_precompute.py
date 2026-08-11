#!/usr/bin/env python3
"""
bop_da3_depth_precompute.py — Windows 端预计算 BOP 测试帧的 DA3 深度
====================================================================
BOP lmo 测试帧的 DA3 估算深度在 Windows 端计算 (DA3 环境已就绪),
保存为 npy 供 WSL 端 FoundationPose 评估使用.

用法 (Windows):
  python scripts/bop_da3_depth_precompute.py
"""
import sys, os, json
import numpy as np
import cv2

BASE = 'E:/zhijiyige'
SEQ_DIR = f'{BASE}/bop_data/lmo/test/000002'
OUT_DIR = f'{BASE}/bop_data/lmo/da3_depth'
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, f'{BASE}/Depth-Anything-3/Depth-Anything-3-main/src')
from depth_anything_3.api import DepthAnything3
import torch


def main():
    # 需要处理的帧
    with open(f'{SEQ_DIR}/scene_camera.json') as f:
        cam_info = json.load(f)
    frame_ids = sorted(cam_info.keys())
    print(f'处理 {len(frame_ids)} 帧')

    # 读取相机参数 (获取原始分辨率)
    first = list(cam_info.keys())[0]
    H, W = 480, 640

    # 加载 DA3
    print('加载 DA3...')
    da3 = DepthAnything3.from_pretrained(f'{BASE}/weights/da3_metric')
    da3 = da3.to('cuda:0').eval()

    for i, fid in enumerate(frame_ids):
        fid_str = f'{int(fid):06d}'
        rgb_path = f'{SEQ_DIR}/rgb/{fid_str}.png'
        img = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # DA3 推理
        out = da3.inference([rgb])
        depth = out.depth[0].astype(np.float32)  # meters

        # resize 到原始分辨率
        if depth.shape[:2] != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

        np.save(f'{OUT_DIR}/{fid_str}.npy', depth)

        if i % 20 == 0:
            print(f'  frame {i}/{len(frame_ids)}: depth {depth.shape} '
                  f'range {depth.min():.3f}-{depth.max():.3f}m')

    print(f'完成! 保存到 {OUT_DIR}')


if __name__ == '__main__':
    main()
