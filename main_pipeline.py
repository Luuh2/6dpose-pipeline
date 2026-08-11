#!/usr/bin/env python3
"""
main_pipeline.py — RGB → 6D Pose Tracking 顶层入口
====================================================
纯 RGB 视频输入 → 物体 3D 检测 + 6D 位姿追踪输出
无真值依赖 (无 GT 深度、无 GT 掩码、无 CAD 模型)

管线架构 (11 模块串联):
  M1  VideoDecoder          → 流式解码 + 360p 缩放
  M2  YOLOWorldDetector     → 首帧开集目标检测 → bbox
  M3  EfficientViTSAMSegmentor → bbox→mask 精细化分割
  M5  DepthEstimator        → DA3 度量深度 + 相机内参 K
  M6  TripoSRMeshGenerator  → 单视图 3D 网格生成
  M7  MeshDepthAligner      → 网格-深度尺度对齐 (核心改进)
  M4  XMemPropagator        → 全帧 mask 传播
  M8  FoundationPoseRunner  → 6D 位姿注册 + 追踪
  M10 FailureDetector       → 失败检测 + 自动恢复
  M9  SE3LieKalmanFilter    → SE(3) 轨迹平滑
  M11 PoseOutputWriter      → CSV + 可视化视频

核心改进 (相对 naive SAM3D→FP 方案):
  1. YOLO+SAM 两阶段分割 → 比 SAM3D 内置 masking 更鲁棒
  2. DA3 度量深度+K → 为 FP 提供真尺度深度, 而非相对深度
  3. Mesh→深度尺度对齐 → 从度量深度估算物理尺寸, 缩放 mesh
  4. XMem 时序传播 → 避免逐帧重检测, 节省计算
  5. LIEKF 平滑 + 失败恢复 → 处理遮挡和漂移

用法:
  python main_pipeline.py --config config.yaml --video demo/test.mp4 --prompt "blue mug"
"""

import argparse
import os
import sys
import time
import gc
import numpy as np
import torch
import cv2
import yaml
from typing import List, Tuple, Optional

# ── 添加 vendored 源码路径 (确保所有依赖可导入) ──────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_SRC_PATHS = [
    os.path.join(_BASE, "src/FoundationPose"),
    os.path.join(_BASE, "src/TripoSR"),
    os.path.join(_BASE, "src/XMem"),
    os.path.join(_BASE, "src/efficientvit"),
    os.path.join(_BASE, "src/nvdiffrast"),
    os.path.join(_BASE, "Depth-Anything-3/Depth-Anything-3-main/src"),
    os.path.join(_BASE, "src/Depth-Anything-V2"),
]
for _p in _SRC_PATHS:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# ── 内部模块 ──────────────────────────────────────────────────────────
from modules.video_decoder import VideoDecoder
from modules.yolo_world_detector import YOLOWorldDetector
from modules.sam_segmentor import EfficientViTSAMSegmentor
from modules.xmem_propagator import XMemPropagator
from modules.depth_estimator import DepthEstimator
from modules.triposr_mesh_generator import TripoSRMeshGenerator
from modules.depth_scale_recovery import DepthScaleRecovery, MeshDepthAligner
from modules.foundationpose_runner import FoundationPoseRunner
from modules.se3_kalman_filter import SE3LieKalmanFilter
from modules.failure_detector import FailureDetector, AutoRecoveryManager
from modules.output_writer import PoseOutputWriter, VisualizationRenderer


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        MAIN PIPELINE CLASS                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class PoseTrackingPipeline:
    """纯 RGB → 6D 位姿追踪 完整管线"""

    def __init__(self, config: dict):
        self.cfg = config
        self.device = config.get("pipeline", {}).get("device", "cuda:0")
        self.output_dir = config.get("pipeline", {}).get("output_dir", "./output")
        os.makedirs(self.output_dir, exist_ok=True)

        # 中间结果目录 (memmap)
        inter_dir = config.get("output", {}).get("intermediate_dir", "./output/intermediate")
        os.makedirs(inter_dir, exist_ok=True)
        self.depth_memmap_path = os.path.join(inter_dir, "depths_metric.dat")
        self.mask_memmap_path = os.path.join(inter_dir, "masks.dat")

        # 状态
        self.frames: List[np.ndarray] = []
        self.fps: float = 30.0
        self.frame_count: int = 0
        self.meta: dict = {}
        self.K: Optional[np.ndarray] = None  # 相机内参 (3,3)

        # GL context (给 FoundationPose)
        self._glctx = None
        self._glfw_window = None

    # ── nvdiffrast Context Management ──────────────────────────────────
    # FoundationPose 需要 nvdiffrast CUDA rasterizer (RasterizeCudaContext),
    # 不是普通的 OpenGL context. nvdiffrast 需要在 Linux (WSL2) 下编译 CUDA
    # 扩展, Windows 原生环境通常无法直接使用.

    def _init_nvdiffrast_context(self):
        """创建 nvdiffrast CUDA rasterizer context (FoundationPose 必需)

        Raises:
            RuntimeError: 如果 nvdiffrast 不可用 (需要 WSL2 环境)
        """
        try:
            import nvdiffrast.torch as dr
            self._glctx = dr.RasterizeCudaContext(device=self.device)
            print("[nvdiffrast] RasterizeCudaContext created successfully.")
        except ImportError:
            raise RuntimeError(
                "nvdiffrast not found. FoundationPose requires nvdiffrast with CUDA support.\n"
                "  Install: pip install nvdiffrast\n"
                "  On Windows, nvdiffrast CUDA rasterizer may not compile natively.\n"
                "  Recommendation: Run FoundationPose (phases 8-11) inside WSL2.\n"
                "  See scripts/run_da3_wsl.py for the WSL2 workflow pattern.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to create nvdiffrast context: {e}\n"
                f"FoundationPose requires nvdiffrast CUDA rasterizer.\n"
                f"On Windows, this typically requires running inside WSL2.")

    def _destroy_gl_context(self):
        """清理 nvdiffrast / GL 资源"""
        if self._glctx is not None:
            try:
                del self._glctx
                self._glctx = None
            except Exception:
                pass
        if self._glfw_window is not None:
            try:
                import glfw
                glfw.destroy_window(self._glfw_window)
                glfw.terminate()
            except Exception:
                pass
            self._glfw_window = None
        torch.cuda.empty_cache()

    # ── Phase 1: Video Decoding ────────────────────────────────────────

    def phase1_decode(self, video_path: str) -> List[np.ndarray]:
        """M1: 解码视频 → 全部帧 (XMem 需要全帧列表)"""
        print("\n" + "=" * 70)
        print("PHASE 1: Video Decoding (M1)")
        print("=" * 70)

        vcfg = self.cfg.get("video", {})
        decoder = VideoDecoder(
            target_short_edge=vcfg.get("target_short_edge", 360))

        self.meta = decoder.get_metadata(video_path)
        self.fps = self.meta["fps"]
        self.frame_count = self.meta["frame_count"]

        # 全帧加载 (XMem 需要随机访问)
        self.frames, _, _ = decoder.decode_all(video_path)

        print(f"[Phase 1] {len(self.frames)} frames @ {self.fps:.1f}fps, "
              f"native={self.meta['native_size']}, proc={self.meta['proc_size']}")
        return self.frames

    # ── Phase 2: First-Frame Detection + Segmentation ──────────────────

    def phase2_detect_and_segment(self, text_prompt: str = None) -> Tuple[np.ndarray, dict]:
        """M2+M3: 首帧检测 + 分割 → 获取高质量物体掩码

        Args:
            text_prompt: 物体描述 (可选。None 时自动检测最显著物体)
        """
        print("\n" + "=" * 70)
        print("PHASE 2: Object Detection & Segmentation (M2 + M3)")
        print("=" * 70)

        first_frame = self.frames[0]
        det_cfg = self.cfg.get("detection", {})
        seg_cfg = self.cfg.get("segmentation", {})

        # ── M2: YOLO-World 检测 ──
        use_world = (det_cfg.get("method", "yolo_world") == "yolo_world")
        yolo = YOLOWorldDetector(
            model_path=os.path.expanduser(det_cfg.get(
                "model_path", "E:/zhijiyige/weights/yolo_world/yolov8s-worldv2.pt")),
            device=self.device,
            conf_threshold=det_cfg.get("conf_threshold", 0.20),
            use_world=use_world,
        )

        if text_prompt:
            detection = yolo.detect_top1(first_frame, text_prompt)
            detect_mode = f"prompt='{text_prompt}'"
        else:
            detection = yolo.auto_detect(first_frame)
            detect_mode = "auto-detect"

        yolo.unload(); del yolo; self._gc()

        if detection is None:
            msg = (f"No object found in first frame (mode={detect_mode}). "
                   f"Try providing a --prompt or use a video with a clearly visible object.")
            raise RuntimeError(msg)

        bbox = np.array(detection["bbox"])
        print(f"[Phase 2] Detection ({detect_mode}): {detection['label']} "
              f"score={detection['score']:.3f} bbox={bbox.tolist()}")

        # ── M3: SAM 精细化分割 ──
        sam = EfficientViTSAMSegmentor(
            model_path=os.path.expanduser(seg_cfg.get(
                "model_path", "E:/zhijiyige/weights/efficientvit_sam/efficientvit_sam_l0.pt")),
            model_name=seg_cfg.get("model_name", "efficientvit-sam-l0"),
            device=self.device,
        )
        # SAM 需要 RGB 输入
        first_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        mask_0 = sam.segment_with_box(first_rgb, bbox)
        sam.unload(); del sam; self._gc()

        mask_area = mask_0.sum()
        print(f"[Phase 2] Mask: {mask_area}px ({100*mask_area/mask_0.size:.1f}% of frame)")

        return mask_0, detection

    # ── Phase 3: Depth Estimation + Camera Intrinsics (First Frame) ────

    def phase3_depth_first_frame(self, mask_0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """M5: 首帧度量深度估计 + 相机内参获取 (DA3)"""
        print("\n" + "=" * 70)
        print("PHASE 3: Depth Estimation — First Frame (M5)")
        print("=" * 70)

        depth_cfg = self.cfg.get("depth", {})
        first_rgb = cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2RGB)

        estimator = DepthEstimator(device=self.device,
                                   model_size=depth_cfg.get("method", "da3"))

        depth_0, K_est = estimator.estimate_da3(first_rgb)
        self.K = K_est if K_est is not None else self._estimate_k_from_fov()

        # 物体区域深度统计 (用于尺度对齐)
        obj_depths = depth_0[mask_0 > 0]
        if len(obj_depths) > 0:
            d_min, d_med, d_max = obj_depths.min(), np.median(obj_depths), obj_depths.max()
            print(f"[Phase 3] Object depth: min={d_min:.3f}m, median={d_med:.3f}m, "
                  f"max={d_max:.3f}m, extent={d_max-d_min:.3f}m")
        print(f"[Phase 3] Camera K:\n{self.K}")

        estimator.unload(); del estimator; self._gc()
        return depth_0, self.K

    # ── Phase 4: 3D Mesh Generation ────────────────────────────────────

    def phase4_generate_mesh(self, mask_0: np.ndarray) -> Tuple[str, dict]:
        """M6: 首帧 RGB+mask → TripoSR 生成 3D 代理网格"""
        print("\n" + "=" * 70)
        print("PHASE 4: 3D Mesh Generation (M6)")
        print("=" * 70)

        mesh_cfg = self.cfg.get("mesh", {})
        first_rgb = cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2RGB)

        triposr = TripoSRMeshGenerator(
            device=self.device,
            mc_resolution=mesh_cfg.get("mc_resolution", 128),
            output_dir=mesh_cfg.get("output_dir", "./output/meshes"),
            enable_fallback=mesh_cfg.get("enable_bbox_fallback", True),
            source_dir=mesh_cfg.get("source_dir", "E:/zhijiyige/src/TripoSR"),
            model_dir=mesh_cfg.get("model_dir", "E:/zhijiyige/weights/triposr"),
        )

        mesh_path, mesh_info = triposr.generate(first_rgb, mask_0,
                                                 output_name="proxy_mesh")
        triposr.unload(); del triposr; self._gc()

        mesh_source = mesh_info.get("source", "unknown")
        print(f"[Phase 4] Mesh: {mesh_info['vertices']}v/{mesh_info['faces']}f, "
              f"source={mesh_source}, path={mesh_path}")
        return mesh_path, mesh_info

    # ── Phase 5: Mesh-Depth Scale Alignment ────────────────────────────

    def phase5_align_scale(
        self, mesh_path: str, depth_0: np.ndarray, mask_0: np.ndarray
    ) -> Tuple[str, float]:
        """M7: 将 mesh 尺度对齐到 DA3 度量深度 (关键改进)

        原方案: 缩放深度到启发式物体尺寸 (depth_scale_recovery)
        改进方案: DA3 已提供度量深度 (m) → 从中估算物体物理尺寸 →
                 缩放 TripoSR mesh 使两者一致 → FP 直接使用度量深度

        Returns:
            aligned_mesh_path: 尺度对齐后的 mesh 路径
            mesh_scale_mm: mesh 顶点缩放因子 (到 mm 单位)
        """
        print("\n" + "=" * 70)
        print("PHASE 5: Mesh-Depth Scale Alignment (M7)")
        print("=" * 70)

        align_cfg = self.cfg.get("scale_alignment", {})
        method = align_cfg.get("method", "depth_guided")

        aligner = MeshDepthAligner(
            method=method,
            default_object_size_mm=align_cfg.get("default_object_size_mm", 100.0),
        )

        mesh_path_out, mesh_scale_mm = aligner.align(
            mesh_path=mesh_path,
            depth_m=depth_0,            # DA3 度量深度, meters
            mask=mask_0,
            K=self.K,
            output_dir=os.path.join(self.output_dir, "meshes"),
        )

        print(f"[Phase 5] Mesh scale factor (→mm): {mesh_scale_mm:.1f}")
        return mesh_path_out, mesh_scale_mm

    # ── Phase 6: Full-Video Depth Estimation ───────────────────────────

    def phase6_depth_all_frames(self) -> np.ndarray:
        """M5: 全帧深度估计 → memmap (DA3 度量深度, meters)

        策略: 每 every_n 帧估计一次, 其余帧最近邻填充 (节省算力)
        """
        print("\n" + "=" * 70)
        print("PHASE 6: Depth Estimation — All Frames (M5)")
        print("=" * 70)

        depth_cfg = self.cfg.get("depth", {})
        every_n = depth_cfg.get("every_n", 10)
        n = len(self.frames)
        h, w = self.frames[0].shape[:2]

        # 创建 memmap (FP16 节省空间)
        depths_mmap = np.memmap(self.depth_memmap_path, dtype=np.float16,
                                mode='w+', shape=(n, h, w))

        estimator = DepthEstimator(device=self.device,
                                   model_size=depth_cfg.get("method", "da3"))

        # 逐帧/间隔估计
        last_depth = None
        for i in range(n):
            if i % every_n == 0 or i == 0:
                rgb = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2RGB)
                d_m, _ = estimator.estimate_da3(rgb)
                last_depth = d_m.astype(np.float16)
                depths_mmap[i] = last_depth
            else:
                # 最近邻填充
                if last_depth is not None:
                    depths_mmap[i] = last_depth

            if i % 100 == 0:
                print(f"[Phase 6] Depth frame {i}/{n}")

        # 后向填充开头帧 (如果首帧未估计)
        for i in range(n):
            if depths_mmap[i].max() == 0:
                # 找到最近的有效帧
                for di in range(1, n):
                    ni = i + di
                    if ni < n and depths_mmap[ni].max() > 0:
                        depths_mmap[i] = depths_mmap[ni]
                        break
                    ni = i - di
                    if ni >= 0 and depths_mmap[ni].max() > 0:
                        depths_mmap[i] = depths_mmap[ni]
                        break

        estimator.unload(); del estimator; self._gc()
        print(f"[Phase 6] Depth memmap: {self.depth_memmap_path} "
              f"({n}x{h}x{w}, fp16, {os.path.getsize(self.depth_memmap_path)/1e6:.1f}MB)")
        return depths_mmap

    # ── Phase 7: Mask Propagation ──────────────────────────────────────

    def phase7_propagate_masks(self, mask_0: np.ndarray) -> np.ndarray:
        """M4: XMem 全帧 mask 传播 → memmap"""
        print("\n" + "=" * 70)
        print("PHASE 7: Mask Propagation (M4)")
        print("=" * 70)

        prop_cfg = self.cfg.get("mask_propagation", {})
        n = len(self.frames)
        h, w = self.frames[0].shape[:2]

        if not prop_cfg.get("enabled", True):
            # 不使用传播: 将首帧 mask 复制到所有帧
            print("[Phase 7] Mask propagation DISABLED — using first-frame mask for all frames.")
            masks_mmap = np.memmap(self.mask_memmap_path, dtype=np.uint8,
                                   mode='w+', shape=(n, h, w))
            for i in range(n):
                masks_mmap[i] = mask_0
            return masks_mmap

        # 设置 XMem 源码路径
        xmem_src = prop_cfg.get("xmem_source", "E:/zhijiyige/src/XMem")
        if xmem_src not in sys.path:
            sys.path.insert(0, xmem_src)

        propagator = XMemPropagator(
            model_path=os.path.expanduser(
                prop_cfg.get("model_path", "E:/zhijiyige/weights/xmem/XMem-s012.pth")),
            device=self.device,
            resolution=prop_cfg.get("resolution", 360),
            segment_length=prop_cfg.get("segment_length", 200),
            segment_overlap=prop_cfg.get("segment_overlap", 5),
        )

        all_masks = propagator.propagate(
            self.frames, mask_0, output_memmap=self.mask_memmap_path)
        propagator.unload(); del propagator; self._gc()

        print(f"[Phase 7] Mask memmap: {self.mask_memmap_path} ({n}x{h}x{w}, uint8)")
        return all_masks

    # ── Phase 8: FoundationPose 6D Pose Tracking ───────────────────────

    def phase8_foundationpose(
        self,
        mesh_path: str,
        mesh_scale_mm: float,
        depths: np.ndarray,
        masks: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """M8: FoundationPose 核心 — 首帧注册 + 逐帧追踪

        Args:
            mesh_path: 尺度对齐后的 mesh 路径 (或原始路径+scale)
            mesh_scale_mm: mesh→mm 缩放因子
            depths: memmap (N,H,W) float16, 单位 meters
            masks: memmap (N,H,W) uint8

        Returns:
            poses: list of (4,4) ndarray in camera frame
            confidences: list of float
        """
        print("\n" + "=" * 70)
        print("PHASE 8: FoundationPose 6D Tracking (M8)")
        print("=" * 70)

        fp_cfg = self.cfg.get("foundationpose", {})
        n = len(self.frames)

        # ── 创建 nvdiffrast context ──
        self._init_nvdiffrast_context()

        # ── 初始化 FoundationPose ──
        fp = FoundationPoseRunner(
            foundationpose_dir=os.path.expanduser(
                fp_cfg.get("source_dir", "E:/zhijiyige/src/FoundationPose")),
            scorer_path=os.path.expanduser(
                fp_cfg.get("scorer_path", "E:/zhijiyige/weights/foundationpose/2023-10-28-18-33-37.pth")),
            refiner_path=os.path.expanduser(
                fp_cfg.get("refiner_path", "E:/zhijiyige/weights/foundationpose/2023-11-07-02-29-13.pth")),
            sampler_encoder_path=os.path.expanduser(
                fp_cfg.get("encoder_path", "E:/zhijiyige/weights/efficientvit_sam/l0.pt")),
            device=self.device,
        )

        # ── 加载物体 mesh ──
        # 使用对齐后的 mesh (如果存在)
        aligned_path = mesh_path.replace(".glb", "_aligned.glb").replace(".obj", "_aligned.obj")
        if os.path.exists(aligned_path):
            load_path = aligned_path
            print(f"[Phase 8] Using aligned mesh: {aligned_path}")
        else:
            load_path = mesh_path
            print(f"[Phase 8] Using original mesh: {mesh_path}")

        fp.set_object(load_path, self._glctx, model_scale=1.0)
        # model_scale=1.0 因为对齐后的 mesh 已经是 mm 单位
        # (原代码中 model_scale=100.0 是因为 mesh 是归一化单位)

        # ── 转换深度 m→mm ──
        print(f"[Phase 8] Starting tracking: {n} frames...")

        poses, confidences = [], []

        # 首帧注册
        depth_0_mm = (depths[0].astype(np.float32) * 1000.0)
        mask_0 = masks[0].astype(np.uint8)

        pose_0 = fp.register(self.frames[0], depth_0_mm, mask_0, self.K, self._glctx)
        conf_0 = fp._compute_confidence(self.frames[0], depth_0_mm, mask_0,
                                        self.K, pose_0, self._glctx)
        poses.append(pose_0)
        confidences.append(conf_0)
        print(f"[Phase 8] Frame 0/{n}: registered, conf={conf_0:.3f}")

        # 逐帧追踪
        for i in range(1, n):
            depth_i_mm = (depths[i].astype(np.float32) * 1000.0)
            mask_i = masks[i].astype(np.uint8)

            try:
                pose_i, conf_i = fp.track(
                    self.frames[i], depth_i_mm, mask_i, self.K, poses[-1], self._glctx)
            except Exception as e:
                print(f"[Phase 8] Frame {i}: tracking error ({e}), using prev pose.")
                pose_i = poses[-1].copy()
                conf_i = 0.1

            poses.append(pose_i)
            confidences.append(float(conf_i))

            if i % 100 == 0:
                print(f"[Phase 8] Frame {i}/{n}, conf={confidences[-1]:.3f}")

        fp.unload(); del fp; self._gc()
        mean_conf = np.mean(confidences)
        print(f"[Phase 8] Complete. {n} poses, mean confidence={mean_conf:.3f}")

        return poses, confidences

    # ── Phase 9: Failure Detection & Recovery ──────────────────────────

    def phase9_failure_recovery(
        self,
        text_prompt: str,
        poses: List[np.ndarray],
        confidences: List[float],
        depths: np.ndarray,
        masks: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[float], np.ndarray]:
        """M10: 检测追踪失败 + 按需恢复"""
        print("\n" + "=" * 70)
        print("PHASE 9: Failure Detection & Recovery (M10)")
        print("=" * 70)

        fail_cfg = self.cfg.get("failure_detection", {})
        if not fail_cfg.get("enabled", True):
            print("[Phase 9] Failure detection DISABLED.")
            return poses, confidences, masks

        detector = FailureDetector(
            mask_area_threshold=fail_cfg.get("mask_area_threshold", 0.50),
            confidence_threshold=fail_cfg.get("confidence_threshold", 0.35),
            recovery_lookback=fail_cfg.get("recovery_lookback", 5),
        )
        detector.set_baseline(masks[0])

        failure_count = 0
        recovery_mgr = None

        for i in range(1, len(self.frames)):
            is_fail, reason = detector.check(masks[i], confidences[i])
            if is_fail:
                failure_count += 1
                print(f"[Phase 9] FAILURE frame {i}: {reason}")

                if fail_cfg.get("enable_recovery", True):
                    recovery_idx = detector.get_recovery_frame_idx(i)
                    print(f"[Phase 9]   Attempting recovery from frame {recovery_idx}...")

                    if recovery_mgr is None:
                        recovery_mgr = AutoRecoveryManager(
                            text_prompt=text_prompt,
                            yolo_model_path=self.cfg.get("detection", {}).get(
                                "model_path", "E:/zhijiyige/weights/yolo_world/yolov8s-worldv2.pt"),
                            sam_model_path=self.cfg.get("segmentation", {}).get(
                                "model_path", "E:/zhijiyige/weights/efficientvit_sam/efficientvit_sam_l0.pt"),
                            sam_model_name=self.cfg.get("segmentation", {}).get(
                                "model_name", "efficientvit-sam-l0"),
                            device=self.device,
                        )

                    try:
                        masks_r, poses_r, confs_r = recovery_mgr.recover(
                            failure_idx=recovery_idx,
                            frames=self.frames,
                            masks=masks,
                            poses=poses,
                            confidences=confidences,
                            K=self.K,
                            glctx=self._glctx,
                            fp_runner=None,
                            depths=depths,
                        )
                        # Copy recovered values forward
                        for j in range(recovery_idx, min(i + 5, len(self.frames))):
                            if j < len(poses_r):
                                poses[j] = poses_r[j]
                                confidences[j] = confs_r[j]
                                masks[j] = masks_r[j]
                        print(f"[Phase 9]   Recovery applied (frames {recovery_idx}→{i}).")
                    except Exception as e:
                        print(f"[Phase 9]   Recovery FAILED: {e}")

        if failure_count == 0:
            print("[Phase 9] No failures detected.")
        else:
            print(f"[Phase 9] {failure_count} failures detected & handled.")

        return poses, confidences, masks

    # ── Phase 10: SE(3) Kalman Smoothing ───────────────────────────────

    def phase10_smooth(self, poses: List[np.ndarray],
                       confidences: List[float]) -> List[np.ndarray]:
        """M9: SE(3) LIEKF 前向滤波 + RTS 后向平滑"""
        print("\n" + "=" * 70)
        print("PHASE 10: SE(3) Kalman Smoothing (M9)")
        print("=" * 70)

        kalman_cfg = self.cfg.get("kalman", {})
        if not kalman_cfg.get("enabled", True):
            print("[Phase 10] Kalman filter DISABLED.")
            return poses

        kf = SE3LieKalmanFilter(
            dt=kalman_cfg.get("dt", 1.0 / self.fps) if self.fps > 0 else 0.033,
            process_noise_pos=kalman_cfg.get("process_noise_pos", 0.01),
            process_noise_rot=kalman_cfg.get("process_noise_rot", 0.001),
            measurement_noise_pos=kalman_cfg.get("measurement_noise_pos", 0.005),
            measurement_noise_rot=kalman_cfg.get("measurement_noise_rot", 0.002),
        )

        smoothed = kf.smooth(poses, confidences)
        print(f"[Phase 10] Smoothed {len(smoothed)} poses.")
        return smoothed

    # ── Phase 11: Output ───────────────────────────────────────────────

    def phase11_output(self, poses: List[np.ndarray], confidences: List[float],
                       mesh_path: str, mesh_scale_mm: float):
        """M11: CSV 输出 + 可视化视频"""
        print("\n" + "=" * 70)
        print("PHASE 11: Output (M11)")
        print("=" * 70)

        out_cfg = self.cfg.get("output", {})
        n = len(self.frames)

        # ── CSV 输出 ──
        csv_path = out_cfg.get("csv_path", "./output/poses.csv")
        timestamps = [i / self.fps for i in range(n)] if self.fps > 0 else list(range(n))
        PoseOutputWriter.write_csv(poses, confidences, timestamps, csv_path)

        # ── 可视化视频 ──
        vis_path = out_cfg.get("vis_video_path", "./output/tracking_vis.avi")
        fps_render = out_cfg.get("render_fps", self.fps)
        if fps_render <= 0:
            fps_render = 30.0

        # 使用对齐后的 mesh (已为 mm 单位, 故 model_scale=1.0)
        viz_mesh_path = mesh_path
        aligned_path = mesh_path.replace(".glb", "_aligned.glb").replace(".obj", "_aligned.obj")
        if os.path.exists(aligned_path):
            viz_mesh_path = aligned_path
            viz_scale = 1.0  # 对齐后的 mesh 已经是 mm 单位
        else:
            viz_scale = mesh_scale_mm  # 原始 mesh 需要缩放

        renderer = VisualizationRenderer(
            mesh_path=viz_mesh_path, model_scale=viz_scale)
        renderer.render_video(self.frames, poses, self.K, vis_path, fps=fps_render)

        print(f"[Phase 11] Done. CSV: {csv_path}, Video: {vis_path}")

    # ── Utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _estimate_k_from_fov(hfov_deg: float = 60.0) -> np.ndarray:
        """从假设的 HFOV 估算相机内参 (DA3 失败时的 fallback)"""
        h, w = 360, 640  # 默认 360p
        fx = w / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
        fy = fx
        cx, cy = w / 2.0, h / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    @staticmethod
    def _gc():
        """强制 GPU 内存回收"""
        gc.collect()
        torch.cuda.empty_cache()

    def cleanup(self):
        """清理所有资源"""
        self._destroy_gl_context()
        self._gc()
        # 保留中间文件 (memmap), 用户可手动删除
        print("[Pipeline] Cleanup complete.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                            CLI ENTRY POINT                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_args():
    parser = argparse.ArgumentParser(
        description="RGB 6D Pose Tracking Pipeline — 纯 RGB 视频 → 物体 6D 位姿追踪")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="YAML 配置文件路径 (default: config.yaml)")
    parser.add_argument("--video", type=str, required=True,
                        help="输入 RGB 视频路径")
    parser.add_argument("--prompt", type=str, default=None,
                        help="物体文本描述 (可选。为空时自动检测画面中最显著的物体)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录 (覆盖 config 中的设置)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="计算设备 (default: cuda:0)")
    parser.add_argument("--resolution", type=int, default=None,
                        help="短边分辨率 (覆盖 config, default: 360)")
    parser.add_argument("--no-kalman", action="store_true",
                        help="禁用 Kalman 滤波器")
    parser.add_argument("--no-recovery", action="store_true",
                        help="禁用失败自动恢复")
    parser.add_argument("--no-xmem", action="store_true",
                        help="禁用 XMem mask 传播 (逐帧重检测)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 加载配置 ──
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # ── CLI 覆盖配置 ──
    config["pipeline"]["video_path"] = args.video
    config["pipeline"]["text_prompt"] = args.prompt  # None = auto-detect
    if args.output:
        config["pipeline"]["output_dir"] = args.output
    if args.device:
        config["pipeline"]["device"] = args.device
    if args.resolution:
        config["video"]["target_short_edge"] = args.resolution
    if args.no_kalman:
        config["kalman"]["enabled"] = False
    if args.no_recovery:
        config["failure_detection"]["enable_recovery"] = False
    if args.no_xmem:
        config["mask_propagation"]["enabled"] = False

    # ── 验证 ──
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    print("=" * 70)
    print("  RGB 6D POSE TRACKING PIPELINE")
    print("  Input : Pure RGB Video (no GT depth/mask/CAD)")
    print("  Output: 3D Detection + 6D Pose Trajectory")
    print("  Core  : FoundationPose + Monocular Depth + Single-View Mesh")
    print("=" * 70)
    print(f"  Video  : {args.video}")
    print(f"  Prompt : {args.prompt or '(auto-detect)'}")
    print(f"  Device : {config['pipeline']['device']}")
    print(f"  Output : {config['pipeline']['output_dir']}")
    print("=" * 70)

    # ── 运行管线 ──
    pipeline = PoseTrackingPipeline(config)
    t_start = time.time()

    try:
        # Phase 1: 视频解码
        pipeline.phase1_decode(args.video)

        # Phase 2: 首帧检测 + 分割 → mask_0
        mask_0, detection = pipeline.phase2_detect_and_segment(args.prompt)

        # Phase 3: 首帧深度 → metric depth + K
        depth_0, K = pipeline.phase3_depth_first_frame(mask_0)

        # Phase 4: 3D 网格生成
        mesh_path, mesh_info = pipeline.phase4_generate_mesh(mask_0)

        # Phase 5: 尺度对齐 (mesh ↔ metric depth)
        mesh_path, mesh_scale_mm = pipeline.phase5_align_scale(
            mesh_path, depth_0, mask_0)

        # Phase 6: 全帧深度 → memmap
        depths = pipeline.phase6_depth_all_frames()

        # Phase 7: Mask 传播 → memmap
        masks = pipeline.phase7_propagate_masks(mask_0)

        # Phase 8: FoundationPose 追踪
        poses, confidences = pipeline.phase8_foundationpose(
            mesh_path, mesh_scale_mm, depths, masks)

        # Phase 9: 失败检测 + 恢复
        poses, confidences, masks = pipeline.phase9_failure_recovery(
            args.prompt, poses, confidences, depths, masks)

        # Phase 10: SE(3) Kalman 平滑
        poses = pipeline.phase10_smooth(poses, confidences)

        # Phase 11: 输出
        pipeline.phase11_output(poses, confidences, mesh_path, mesh_scale_mm)

    except Exception as e:
        print(f"\n[FATAL] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        pipeline.cleanup()

    elapsed = time.time() - t_start
    n_frames = pipeline.frame_count
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE")
    print(f"  Frames  : {n_frames}")
    print(f"  Time    : {elapsed:.1f}s ({elapsed/n_frames:.2f}s/frame)")
    print(f"  Output  : {config['pipeline']['output_dir']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
