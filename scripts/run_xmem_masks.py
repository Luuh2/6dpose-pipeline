#!/usr/bin/env python3
"""
run_xmem_masks.py — M4: XMem 时序掩码传播
==========================================
将首帧物体掩码传播到视频全部帧, 生成逐帧时序掩码。

XMem 上游 bug 已修复 (见 PIPELINE_README 故障排查):
  1. group_modules.py — num_objects=1 时张量维度不匹配
  2. network.py segment() — 共享特征 num_objects 维 squeeze
  3. xmem_propagator.py — argmax 维度错误

用法:
  python scripts/run_xmem_masks.py --video demo/test_mustard.mp4 --output ./output
"""

import sys
import os
import gc
import time
import argparse
import numpy as np
import cv2
import torch

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)
sys.path.insert(0, os.path.join(_BASE, "src/efficientvit"))
sys.path.insert(0, os.path.join(_BASE, "src/XMem"))

from modules.video_decoder import VideoDecoder
from modules.yolo_world_detector import YOLOWorldDetector
from modules.sam_segmentor import EfficientViTSAMSegmentor
from modules.depth_estimator import DepthEstimator
from modules.xmem_propagator import XMemPropagator


def gcuda():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="XMem 时序掩码传播")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device
    os.makedirs(os.path.join(args.output, "intermediate"), exist_ok=True)

    # ── 1. 解码 ──────────────────────────────────────────────────
    print("[1] Decoding video...")
    decoder = VideoDecoder(target_short_edge=360)
    frames, fps, n_frames = decoder.decode_all(args.video)
    h, w = frames[0].shape[:2]
    print(f"  {n_frames} frames @ {fps:.1f}fps, {w}x{h}")

    first_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)

    # ── 2. 首帧深度 (供检测验证) ─────────────────────────────────
    print("[2] First-frame depth...")
    depth_est = DepthEstimator(device=device, model_size="da3")
    depth_0, K = depth_est.estimate_da3(first_rgb)
    depth_est.unload(); del depth_est; gcuda()

    # ── 3. 首帧检测 + 分割 → 首帧掩码 ────────────────────────────
    print("[3] First-frame detect + segment...")
    yolo = YOLOWorldDetector(
        model_path=os.path.join(_BASE, "weights/yolo_world/yolov8s-worldv2.pt"),
        device=device, conf_threshold=0.15, use_world=False)
    det = yolo.auto_detect(frames[0], depth_m=depth_0)
    yolo.unload(); del yolo; gcuda()

    if det is None:
        print("FATAL: No object detected in first frame!")
        return

    sam = EfficientViTSAMSegmentor(
        model_path=os.path.join(_BASE, "weights/efficientvit_sam/efficientvit_sam_l0.pt"),
        model_name="efficientvit-sam-l0", device=device)
    mask_0 = sam.segment_with_box(first_rgb, np.array(det["bbox"]))
    sam.unload(); del sam; gcuda()
    print(f"  First mask: {mask_0.sum()}px ({det['label']})")

    # ── 4. XMem 传播 ─────────────────────────────────────────────
    print(f"[4] XMem propagation ({n_frames} frames)...")
    out_path = os.path.join(args.output, "intermediate", "masks_xmem_full.dat")
    t0 = time.time()
    prop = XMemPropagator(
        model_path=os.path.join(_BASE, "weights/xmem/XMem-s012.pth"),
        device=device, resolution=360,
        segment_length=200, segment_overlap=5)
    masks = prop.propagate(frames, mask_0, output_memmap=out_path)
    prop.unload(); del prop; gcuda()
    dt = time.time() - t0
    print(f"  Propagated {n_frames}f in {dt:.0f}s ({n_frames/dt:.1f}fps)")

    # ── 5. 验证 ──────────────────────────────────────────────────
    print("[5] Verify trajectory:")
    for fi in [0, 100, 300, 600, n_frames - 1]:
        ys, xs = np.where(masks[fi] > 0)
        if len(ys):
            print(f"  frame {fi}: {masks[fi].sum()}px "
                  f"bbox=x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")
        else:
            print(f"  frame {fi}: EMPTY")

    print(f"\nDONE! Mask memmap: {out_path}")


if __name__ == "__main__":
    main()
