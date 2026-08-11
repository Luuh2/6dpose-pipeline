"""
failure_detector.py — Module 10
功能: 追踪失败检测 + 自动重初始化
低配适配: 阈值放宽 (0.50/0.35), 按需加载 YOLO+SAM+FoundationPose
"""

import numpy as np
import cv2
import torch
from typing import List, Tuple, Optional


class FailureDetector:
    """追踪失败检测器 — 基于掩码面积变化 + 置信度"""

    def __init__(
        self,
        mask_area_threshold: float = 0.50,
        confidence_threshold: float = 0.35,
        recovery_lookback: int = 5,
    ):
        self.mask_area_threshold = mask_area_threshold
        self.confidence_threshold = confidence_threshold
        self.recovery_lookback = recovery_lookback
        self.baseline_mask_area: Optional[float] = None

    def set_baseline(self, mask: np.ndarray):
        self.baseline_mask_area = float(mask.sum())

    def check(self, mask: np.ndarray, confidence: float) -> Tuple[bool, str]:
        """检测当前帧是否失败

        Returns:
            (is_failure, reason)
        """
        if self.baseline_mask_area is None:
            self.set_baseline(mask)
            return False, "ok"

        current_area = float(mask.sum())
        area_ratio = current_area / max(self.baseline_mask_area, 1)
        reasons = []

        if area_ratio < self.mask_area_threshold:
            reasons.append(f"mask_drop({area_ratio:.2f})")
        elif area_ratio > (2.0 - self.mask_area_threshold):
            reasons.append(f"mask_spike({area_ratio:.2f})")

        if confidence < self.confidence_threshold:
            reasons.append(f"low_conf({confidence:.3f})")

        if reasons:
            return True, "|".join(reasons)
        else:
            # 更新基线 (EMA)
            self.baseline_mask_area = 0.9 * self.baseline_mask_area + 0.1 * current_area
            return False, "ok"

    def get_recovery_frame_idx(self, current_idx: int) -> int:
        return max(0, current_idx - self.recovery_lookback)


class AutoRecoveryManager:
    """自动重初始化管理器 — 按需加载 YOLO+SAM+FoundationPose 恢复

    注意: 使用延迟导入避免循环依赖, 模块在 recover() 内按需加载。
    """

    def __init__(
        self,
        text_prompt: str,
        yolo_model_path: str = "E:/zhijiyige/weights/yolo_world/yolov8s-worldv2.pt",
        sam_model_path: str = "E:/zhijiyige/weights/efficientvit_sam/efficientvit_sam_l0.pt",
        sam_model_name: str = "efficientvit-sam-l0",
        device: str = "cuda:0",
    ):
        self.text_prompt = text_prompt
        self.yolo_model_path = yolo_model_path
        self.sam_model_path = sam_model_path
        self.sam_model_name = sam_model_name
        self.device = device

    def recover(
        self,
        failure_idx: int,
        frames: List[np.ndarray],
        masks: np.ndarray,
        poses: List[np.ndarray],
        confidences: List[float],
        K: np.ndarray,
        glctx,
        fp_runner,
        depths: np.ndarray = None,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
        """在失败帧重新初始化. 按需加载 YOLO+SAM, 用完即卸.

        Args:
            failure_idx: 失败帧索引
            frames: 全部帧列表
            masks: memmap (N,H,W) uint8
            poses: 位姿列表
            confidences: 置信度列表
            K: 相机内参
            glctx: OpenGL context
            fp_runner: FoundationPoseRunner 实例 (如 None 则创建新的)
            depths: memmap (N,H,W) float16, 单位 meters (可选)

        Returns:
            (masks, poses, confidences) — 原地修改后的引用
        """
        # 延迟导入 (避免循环依赖)
        from modules.yolo_world_detector import YOLOWorldDetector
        from modules.sam_segmentor import EfficientViTSAMSegmentor

        recovery_frame = frames[failure_idx]

        # Step 1: YOLO-World 重新检测
        yolo = YOLOWorldDetector(
            model_path=self.yolo_model_path,
            device=self.device,
            conf_threshold=0.20,
        )
        detection = yolo.detect_top1(recovery_frame, self.text_prompt)
        yolo.unload()
        del yolo
        torch.cuda.empty_cache()

        if detection is None:
            print(f"[Recovery] No object found at frame {failure_idx}")
            return masks, poses, confidences

        print(f"[Recovery] Re-detected: {detection['label']} "
              f"score={detection['score']:.3f}")

        # Step 2: SAM 重新分割
        sam = EfficientViTSAMSegmentor(
            model_path=self.sam_model_path,
            model_name=self.sam_model_name,
            device=self.device,
        )
        recovery_rgb = cv2.cvtColor(recovery_frame, cv2.COLOR_BGR2RGB)
        new_mask = sam.segment_with_box(recovery_rgb, np.array(detection["bbox"]))
        sam.unload()
        del sam
        torch.cuda.empty_cache()

        masks[failure_idx] = new_mask

        # Step 3: FoundationPose 重新注册
        depth_for_register = np.zeros_like(new_mask, dtype=np.float32)
        if depths is not None:
            depth_for_register = (depths[failure_idx].astype(np.float32) * 1000.0)

        if fp_runner is not None:
            try:
                new_pose = fp_runner.register(
                    frames[failure_idx], depth_for_register,
                    new_mask, K, glctx)
                poses[failure_idx] = new_pose
                confidences[failure_idx] = 1.0
                print(f"[Recovery] Re-registered pose at frame {failure_idx}")
            except Exception as e:
                print(f"[Recovery] Pose registration failed: {e}")
        else:
            print("[Recovery] No fp_runner provided, skipping pose re-registration.")

        return masks, poses, confidences
