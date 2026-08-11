#!/usr/bin/env python3
"""
test_precompute.py — 测试预计算管线 (M1-M7)
============================================
在 Windows 上运行完整的预计算流程，输出:
  1. 检测+分割结果可视化 (detection_seg.png)
  2. 深度图可视化 (depth_vis.png)
  3. 3D 网格截图信息
  4. 中间结果 memmap 文件

用法:
  python test_precompute.py --video demo/test_mustard.mp4 --prompt "mustard bottle"
"""

import os, sys, time, gc
import numpy as np
import cv2
import argparse

# 添加源码路径
_BASE = os.path.dirname(os.path.abspath(__file__))
for _p in [
    "src/FoundationPose", "src/TripoSR", "src/XMem", "src/efficientvit",
    "Depth-Anything-3/Depth-Anything-3-main/src", "src/Depth-Anything-V2",
]:
    _full = os.path.join(_BASE, _p)
    if os.path.isdir(_full) and _full not in sys.path:
        sys.path.insert(0, _full)

from modules.video_decoder import VideoDecoder
from modules.yolo_world_detector import YOLOWorldDetector
from modules.sam_segmentor import EfficientViTSAMSegmentor
from modules.depth_estimator import DepthEstimator
from modules.triposr_mesh_generator import TripoSRMeshGenerator
from modules.depth_scale_recovery import MeshDepthAligner


def gcuda():
    gc.collect()
    import torch; torch.cuda.empty_cache()


def visualize_detection_mask(frame_rgb, bbox, mask, label, score, save_path):
    """绘制检测框 + 掩码叠加"""
    vis = frame_rgb.copy()
    h, w = vis.shape[:2]

    # 掩码半透明叠加 (绿色)
    overlay = np.zeros_like(vis)
    overlay[mask > 0] = [0, 255, 0]
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

    # 检测框
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 标签
    text = f"{label} ({score:.2f})"
    cv2.putText(vis, text, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"  [viz] Detection+mask saved: {save_path}")
    return vis


def visualize_depth(depth_m, mask, save_path):
    """深度图可视化 (彩色映射)"""
    # 全局深度归一化
    d_vis = depth_m.copy()
    d_valid = d_vis[d_vis > 0]
    if len(d_valid) > 0:
        vmin, vmax = np.percentile(d_valid, 2), np.percentile(d_valid, 98)
    else:
        vmin, vmax = 0, 5
    d_vis = np.clip((d_vis - vmin) / max(vmax - vmin, 0.01), 0, 1)
    d_vis = (d_vis * 255).astype(np.uint8)
    d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_TURBO)

    # 掩码轮廓
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(d_color, contours, -1, (0, 255, 0), 2)

    cv2.imwrite(save_path, d_color)
    print(f"  [viz] Depth visualization saved: {save_path} "
          f"(range: {vmin:.2f}m - {vmax:.2f}m)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None,
                        help="物体文本描述 (可选。为空时自动检测)")
    parser.add_argument("--output", type=str, default="./output")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "intermediate"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "meshes"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "debug"), exist_ok=True)

    t0 = time.time()
    device = "cuda:0"

    # ══════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("  PIPELINE PRECOMPUTATION TEST (M1-M7)")
    print(f"  Video: {args.video}")
    print(f"  Prompt: {args.prompt or '(auto-detect)'}")
    print("=" * 60)

    # ── M1: 视频解码 ──────────────────────────────────────────────
    print("\n[M1] Decoding video...")
    decoder = VideoDecoder(target_short_edge=720)
    meta = decoder.get_metadata(args.video)
    frames, fps, n_frames = decoder.decode_all(args.video)
    print(f"  {n_frames} frames @ {fps:.1f}fps, "
          f"native={meta['native_size']}, proc={meta['proc_size']}")

    first_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
    h, w = first_rgb.shape[:2]

    # ── M5 (先行): 首帧深度 ───────────────────────────────────────
    # 自动检测需要深度做平面验证, 所以深度必须在检测之前
    print("\n[M5] Estimating depth (first frame)...")
    depth_estimator = DepthEstimator(device=device, model_size="da3")
    depth_0, K_est = depth_estimator.estimate_da3(first_rgb)
    depth_estimator.unload(); del depth_estimator; gcuda()

    if K_est is not None:
        K = K_est
        print(f"  K (from DA3):\n{K}")
    else:
        # Heuristic K fallback
        fx = max(h, w) * 0.866
        K = np.array([[fx, 0, w/2], [0, fx, h/2], [0, 0, 1]], dtype=np.float64)
        print(f"  K (heuristic): fx={fx:.0f}")

    # ── M2: 检测 (带深度验证) ────────────────────────────────────
    print("\n[M2] Detecting object (depth-verified)...")
    yolo = YOLOWorldDetector(
        model_path="E:/zhijiyige/weights/yolo_world/yolov8s-worldv2.pt",
        device=device, conf_threshold=0.20, use_world=False)

    if args.prompt:
        detection = yolo.detect_top1(frames[0], args.prompt)
        detect_mode = f"prompt='{args.prompt}'"
    else:
        detection = yolo.auto_detect(frames[0], depth_m=depth_0)
        detect_mode = "auto-detect (COCO+depth)"

    yolo.unload(); del yolo; gcuda()

    if detection is None:
        print("  ERROR: No object detected! Retrying with YOLO-World...")
        yolo2 = YOLOWorldDetector(
            model_path="E:/zhijiyige/weights/yolo_world/yolov8s-worldv2.pt",
            device=device, conf_threshold=0.10, use_world=True)
        detection = yolo2.auto_detect(frames[0], depth_m=depth_0)
        yolo2.unload(); del yolo2; gcuda()
        if detection is None:
            print("  FATAL: No objects at all. Check video.")
            return

    print(f"  Detection ({detect_mode}): {detection['label']} "
          f"score={detection['score']:.3f} "
          f"bbox={[int(v) for v in detection['bbox']]}")

    bbox = np.array(detection["bbox"])
    print(f"  Detection: {detection['label']} score={detection['score']:.3f} "
          f"bbox={bbox.astype(int).tolist()}")

    # ── M3: 分割 ──────────────────────────────────────────────────
    print("\n[M3] Segmenting...")
    sam = EfficientViTSAMSegmentor(
        model_path="E:/zhijiyige/weights/efficientvit_sam/efficientvit_sam_l0.pt",
        model_name="efficientvit-sam-l0", device=device)
    mask_0 = sam.segment_with_box(first_rgb, bbox)
    sam.unload(); del sam; gcuda()
    print(f"  Mask: {mask_0.sum()}px ({100*mask_0.sum()/mask_0.size:.1f}% of frame)")

    # 保存检测+分割可视化
    vis_det = visualize_detection_mask(
        first_rgb, bbox, mask_0, detection["label"], detection["score"],
        os.path.join(args.output, "debug", "01_detection_seg.png"))

    obj_depths = depth_0[mask_0 > 0]
    if len(obj_depths) > 0:
        print(f"  Object depth: min={obj_depths.min():.3f}m, "
              f"median={np.median(obj_depths):.3f}m, max={obj_depths.max():.3f}m")

    # 保存深度可视化
    visualize_depth(depth_0, mask_0,
                    os.path.join(args.output, "debug", "02_depth.png"))

    # ── M6: 3D 网格 ────────────────────────────────────────────────
    print("\n[M6] Generating 3D mesh...")
    triposr = TripoSRMeshGenerator(
        device=device, mc_resolution=128,
        output_dir=os.path.join(args.output, "meshes"),
        enable_fallback=True,
        source_dir="E:/zhijiyige/src/TripoSR",
        model_dir="E:/zhijiyige/weights/triposr",
    )
    try:
        mesh_path, mesh_info = triposr.generate(
            first_rgb, mask_0, output_name="proxy_mesh")
        print(f"  Mesh: {mesh_info['vertices']}v/{mesh_info['faces']}f, "
              f"source={mesh_info.get('source','?')}, path={mesh_path}")
    except Exception as e:
        print(f"  TripoSR failed: {e}")
        mesh_path = os.path.join(args.output, "meshes", "proxy_mesh.glb")
        mesh_info = {"vertices": 8, "faces": 12, "source": "bbox_fallback"}
    triposr.unload(); del triposr; gcuda()

    # ── M7: 尺度对齐 ──────────────────────────────────────────────
    print("\n[M7] Aligning mesh scale to metric depth...")
    aligner = MeshDepthAligner(
        method="depth_guided", default_object_size_mm=100.0)
    aligned_path, scale_mm = aligner.align(
        mesh_path=mesh_path, depth_m=depth_0, mask=mask_0, K=K,
        output_dir=os.path.join(args.output, "meshes"))
    print(f"  Scale factor: {scale_mm:.1f} (mesh -> mm)")
    print(f"  Aligned mesh: {aligned_path}")

    # 参考尺寸 (尺度校准用): 必须来自独立于 DA3 深度的来源, 避免循环依赖.
    # MeshDepthAligner 用 DA3 反投影出的尺寸继承深度偏差, 不能用作参考.
    # 正确来源: 用户提供的 known_object_size_mm 或验证过的 CAD 模型.
    import yaml as _yaml
    _cfg_for_ref = {}
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'config.yaml')) as _f:
            _cfg_for_ref = _yaml.safe_load(_f)
    except Exception:
        pass
    _known_size_mm = (_cfg_for_ref.get('scale_calibration', {})
                      .get('known_object_size_mm', None))
    if _known_size_mm:
        reference_size_m = float(_known_size_mm) / 1000.0
        print(f"  [ScaleRef] 已知物体尺寸: {_known_size_mm}mm = {reference_size_m:.3f}m "
              f"(独立来源, 非DA3反投影)")
    else:
        # 未提供已知尺寸 → 禁用尺度校准 (DA3 深度可能已正确)
        reference_size_m = None
        print("  [ScaleRef] 未提供已知物体尺寸 → 跳过尺度校准 "
              "(避免用 DA3 反投影的循环依赖)")

    # ── M5b: 全帧深度 ─────────────────────────────────────────────
    print("\n[M5b] Estimating depth for all frames...")
    depth_memmap_path = os.path.join(args.output, "intermediate", "depths_metric.dat")
    depths_mmap = np.memmap(depth_memmap_path, dtype=np.float16,
                            mode='w+', shape=(n_frames, h, w))

    depth_estimator = DepthEstimator(device=device, model_size="da3")
    # 关键: 每帧估计深度 — track_one 的 RGB-D refiner 依赖实时深度跟随物体
    # 每10帧一次会导致物体移动后深度图陈旧, 追踪跟丢
    every_n = 1  # 每帧估计 (0.1s/帧, 737帧约75s, 可接受)
    last_depth = depth_0.astype(np.float16)
    for i in range(n_frames):
        if i % every_n == 0:
            rgb_i = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            d_i, _ = depth_estimator.estimate_da3(rgb_i)
            last_depth = d_i.astype(np.float16)
        depths_mmap[i] = last_depth
        if i % 200 == 0:
            print(f"  Depth frame {i}/{n_frames}")
    depth_estimator.unload(); del depth_estimator; gcuda()
    print(f"  Depth memmap: {depth_memmap_path} "
          f"({os.path.getsize(depth_memmap_path)/1e6:.1f}MB)")

    # ── M4: XMem 时序掩码跟踪 ────────────────────────────────────
    # 弃用固定周期重检测: 首帧 mask_0 初始化 XMem, 时序传播到所有帧,
    # 输出帧间平滑连贯的分割掩码, 消除掩码边缘跳动/轮廓突变.
    # 为跟踪阶段质心/主轴约束提供可靠输入.
    # (XMem 上游 3 个 bug 已修复: group_modules 维度, network squeeze, argmax)
    print("\n[M4] XMem temporal mask tracking...")
    mask_memmap_path = os.path.join(args.output, "intermediate", "masks.dat")
    mask_xmem_path = os.path.join(args.output, "intermediate", "masks_xmem_full.dat")

    try:
        sys.path.insert(0, "E:/zhijiyige/src/XMem")
        from modules.xmem_propagator import XMemPropagator
        propagator = XMemPropagator(
            model_path="E:/zhijiyige/weights/xmem/XMem-s012.pth",
            device=device, resolution=360,
            segment_length=200, segment_overlap=5)
        masks_mmap = propagator.propagate(
            frames, mask_0, output_memmap=mask_xmem_path)
        propagator.unload(); del propagator; gcuda()

        # 校验 XMem 输出 (掩码应平滑连贯, 帧间无跳变)
        n_zero = sum(1 for i in range(0, n_frames, 50) if masks_mmap[i].sum() == 0)
        print(f"  [M4] XMem done: {n_frames} 帧, "
              f"采样点零掩码帧: {n_zero}")
        # 拷贝 XMem 掩码到 masks.dat (追踪脚本兼容)
        masks_final = np.memmap(mask_memmap_path, dtype=np.uint8,
                                mode='w+', shape=(n_frames, h, w))
        for i in range(n_frames):
            masks_final[i] = masks_mmap[i]
        del masks_final
    except Exception as e:
        print(f"  [M4] XMem failed ({str(e)[:80]}), fallback to re-detection...")
        # 回退: 固定周期重检测 (每30帧)
        RE_DETECT_EVERY = 30
        yolo = YOLOWorldDetector(
            model_path="E:/zhijiyige/weights/yolo_world/yolov8s-worldv2.pt",
            device=device, conf_threshold=0.15, use_world=False)
        sam = EfficientViTSAMSegmentor(
            model_path="E:/zhijiyige/weights/efficientvit_sam/efficientvit_sam_l0.pt",
            model_name="efficientvit-sam-l0", device=device)
        masks_mmap = np.memmap(mask_memmap_path, dtype=np.uint8,
                               mode='w+', shape=(n_frames, h, w))
        last_mask = mask_0.copy()
        for i in range(n_frames):
            if i % RE_DETECT_EVERY == 0:
                rgb_i = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
                d_i = depths_mmap[i].astype(np.float32) if depths_mmap[i].max() > 0 else depth_0
                det = yolo.auto_detect(frames[i], depth_m=d_i)
                if det:
                    last_mask = sam.segment_with_box(rgb_i, np.array(det["bbox"]))
            masks_mmap[i] = last_mask
        yolo.unload(); del yolo
        sam.unload(); del sam
        gcuda()
        print("  [M4] Fallback: re-detection used (masked edges may jump)")

    print(f"  Mask memmap: {mask_memmap_path} "
          f"({os.path.getsize(mask_memmap_path)/1e6:.1f}MB)")

    # ── 保存 K ────────────────────────────────────────────────────
    K_path = os.path.join(args.output, "K.npy")
    np.save(K_path, K)
    print(f"\n  Camera intrinsics saved: {K_path}")

    # ── M7b: 深度尺度校准 + 稳定性监控 ────────────────────────────
    # 消除 DA3 系统性尺度偏差 (~47% 低估), 需独立参考尺寸 (known_object_size).
    # 稳定性监控: 过滤剧烈波动/异常跳变的深度帧.
    # 未提供已知尺寸时跳过 (避免用 DA3 反投影的循环依赖).
    if reference_size_m is None:
        print("\n[M7b] 跳过尺度校准 (未提供已知物体尺寸)")
        reference_size_m = 0.10  # 占位, 不实际校准
        _skip_calibration = True
    else:
        _skip_calibration = False

    from modules.depth_scale_calibration import DepthScaleCalibrator
    if _skip_calibration:
        print("  (保留原始 DA3 深度 — demo 场景 DA3 已基本正确, 校准可能引入误差)")
    else:
        print("\n[M7b] Depth scale calibration (参考尺寸 {:.0f}mm)...".format(
            reference_size_m * 1000))

    calibrator = DepthScaleCalibrator(
        reference_size_m=reference_size_m if not _skip_calibration else None,
        scale_min=0.5, scale_max=3.0,
        enable_stability=True,
    )
    # 读取已生成的 mask memmap
    mask_memmap_cal = np.memmap(mask_memmap_path, dtype=np.uint8,
                                mode='r', shape=(n_frames, h, w))
    # 校准深度 (写回 fp16)
    depths_cal = np.memmap(depth_memmap_path, dtype=np.float16,
                           mode='r+', shape=(n_frames, h, w))
    depths_f32 = np.asarray(depths_cal).astype(np.float32)
    cal_out = calibrator.calibrate_sequence(
        depths_f32, np.asarray(mask_memmap_cal).astype(np.uint8), K,
        output_memmap=None)
    # 写回 (fp16 校准深度)
    if not _skip_calibration:
        for i in range(n_frames):
            depths_cal[i] = cal_out[i].astype(np.float16)
        print(f"  [M7b] 校准完成, 全局尺度比={calibrator.global_scale:.3f}")
        print(f"  [M7b] 深度 memmap 已更新 (尺度校准后)")
    else:
        print("  [M7b] 跳过写回 — 保留原始 DA3 深度")
    del depths_cal, cal_out

    # ── 批量深度可视化 ────────────────────────────────────────────
    print("\n[Debug] Generating multi-frame depth visualization...")
    sample_frames = [0, n_frames//4, n_frames//2, 3*n_frames//4, n_frames-1]
    debug_vis = []
    for fi in sample_frames:
        frame_rgb = cv2.cvtColor(frames[fi], cv2.COLOR_BGR2RGB)
        d_vis = depths_mmap[fi].astype(np.float32).copy()
        vmin = np.percentile(d_vis[d_vis > 0], 2) if (d_vis > 0).any() else 0
        vmax = np.percentile(d_vis[d_vis > 0], 98) if (d_vis > 0).any() else 5
        d_norm = np.clip((d_vis - vmin) / max(vmax - vmin, 0.01), 0, 1)
        d_color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        # Concatenate RGB + depth side by side
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        combined = np.hstack([frame_bgr, d_color])
        cv2.putText(combined, f"Frame {fi}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        debug_vis.append(combined)

    # Stack vertically with labels
    debug_montage_path = os.path.join(args.output, "debug", "03_depth_montage.png")
    cv2.imwrite(debug_montage_path, np.vstack(debug_vis))
    print(f"  [viz] Depth montage saved: {debug_montage_path}")

    # ── 汇总 ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  PRECOMPUTATION COMPLETE")
    print(f"  Time: {elapsed:.1f}s ({elapsed/n_frames:.2f}s/frame)")
    print(f"  Frames: {n_frames}")
    print(f"  Resolution: {w}x{h}")
    print(f"  Output directory: {args.output}/")
    print(f"  Intermediate files:")
    print(f"    depths: {depth_memmap_path}")
    print(f"    masks:  {mask_memmap_path}")
    print(f"    K:      {K_path}")
    print(f"    mesh:   {aligned_path}")
    print(f"  Debug visualizations: {args.output}/debug/")
    print("=" * 60)
    print("\n  Next step (WSL2): Run FoundationPose tracking with:")
    print(f"    mesh:  {aligned_path}")
    print(f"    depths: {depth_memmap_path}")
    print(f"    masks:  {mask_memmap_path}")
    print(f"    K:      {K_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
