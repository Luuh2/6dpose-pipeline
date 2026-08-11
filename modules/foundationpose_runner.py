"""
foundationpose_runner.py — Module 8
功能: FoundationPose 核心 6D 姿态估计与追踪
低配适配: 轻量 encoder (l0) + n_pts=3000 + 低渲染分辨率
"""

import torch
import numpy as np
import os
import sys
from typing import List, Tuple, Optional


class FoundationPoseRunner:
    """FoundationPose 姿态估计与追踪 — 低配版"""

    def __init__(
        self,
        foundationpose_dir: str = "E:/zhijiyige/src/FoundationPose",
        scorer_path: str = "E:/zhijiyige/weights/foundationpose/FoundationPosescorer.pth",
        refiner_path: str = "E:/zhijiyige/weights/foundationpose/FoundationPoserefiner.pth",
        sampler_encoder_path: str = "E:/zhijiyige/weights/efficientvit_sam/efficientvit_sam_l0.pt",
        device: str = "cuda:0",
    ):
        sys.path.insert(0, os.path.expanduser(foundationpose_dir))

        from estimater import FoundationPose

        self.estimator = FoundationPose(
            model_pts_path=None,
            model_normals_path=None,
            mesh_file_path=None,
            scorer_path=os.path.expanduser(scorer_path),
            refiner_path=os.path.expanduser(refiner_path),
            sampler_encoder_path=os.path.expanduser(sampler_encoder_path),
            device=device,
        )
        self.device = device
        self._mesh = None

    def set_object(self, mesh_path: str, glctx, model_scale: float = 100.0):
        """设置被追踪物体的 3D 模型

        Args:
            mesh_path: GLB/OBJ 文件路径
            glctx: OpenGL context
            model_scale: 归一化网格 → mm 尺度映射因子 (默认100mm物体)
        """
        import trimesh
        from estimater import sample_points_and_normals

        self._mesh = trimesh.load(mesh_path, force="mesh")
        self._mesh.vertices *= model_scale

        pts, normals = sample_points_and_normals(self._mesh, n_pts=3000)  # 低配
        self.estimator.set_model_pts(pts, normals)
        self.estimator.mesh = self._mesh

        print(f"[FoundationPose] Object set: {mesh_path}, scale={model_scale}, "
              f"pts={pts.shape[0]}")

    def register(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        glctx,
    ) -> np.ndarray:
        """首帧姿态注册 (全局搜索)

        Returns:
            pose: (4, 4) ndarray [R|t] 相机坐标系
        """
        pose = self.estimator.register(
            rgb=rgb, depth=depth.astype(np.float32),
            ob_mask=mask, ob_id=0, K=K, glctx=glctx,
        )
        print(f"[FoundationPose] Registration done.")
        return pose

    def track(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        prev_pose: np.ndarray,
        glctx,
    ) -> Tuple[np.ndarray, float]:
        """逐帧追踪 (局部优化)

        Returns:
            pose: (4, 4) ndarray
            confidence: float
        """
        pose = self.estimator.track(
            rgb=rgb, depth=depth.astype(np.float32),
            ob_mask=mask, ob_id=0, K=K, glctx=glctx,
            init_pose=prev_pose,
        )
        confidence = self._compute_confidence(rgb, depth.astype(np.float32), mask, K, pose, glctx)
        return pose, confidence

    def _compute_confidence(self, rgb, depth, mask, K, pose, glctx) -> float:
        """Scorer 评分"""
        try:
            score = self.estimator.scorer.evaluate(
                rgb=rgb, depth=depth, ob_mask=mask, K=K, pose=pose, glctx=glctx,
            )
            return float(score)
        except Exception:
            return 0.5

    def run_full_pipeline(
        self,
        frames: List[np.ndarray],
        depths: np.ndarray,   # memmap (N, H, W) float16
        masks: np.ndarray,     # memmap (N, H, W) uint8
        K: np.ndarray,
        glctx,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """完整追踪 pipeline

        Returns:
            poses: list of (4,4) ndarray
            confidences: list of float
        """
        poses = []
        confidences = []

        # 首帧注册
        pose0 = self.register(
            frames[0], depths[0].astype(np.float32),
            masks[0].astype(np.uint8), K, glctx)
        poses.append(pose0)
        confidences.append(self._compute_confidence(
            frames[0], depths[0].astype(np.float32),
            masks[0].astype(np.uint8), K, pose0, glctx))

        # 逐帧追踪
        for i in range(1, len(frames)):
            pose, conf = self.track(
                frames[i], depths[i].astype(np.float32),
                masks[i].astype(np.uint8), K, poses[-1], glctx)
            poses.append(pose)
            confidences.append(conf)

            if i % 100 == 0:
                print(f"[FoundationPose] Frame {i}/{len(frames)}, conf={confidences[-1]:.3f}")

        return poses, confidences

    def unload(self):
        """释放 FoundationPose 模型"""
        if hasattr(self, 'estimator') and self.estimator is not None:
            del self.estimator
            self.estimator = None
        self._mesh = None
        torch.cuda.empty_cache()
        print("[FoundationPose] Unloaded.")
