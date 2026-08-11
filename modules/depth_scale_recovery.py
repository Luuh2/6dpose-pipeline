"""
depth_scale_recovery.py — Module 7
功能: 深度尺度恢复 + Mesh-深度尺度对齐
方法:
  DepthScaleRecovery — 相对深度→伪度量 (DA2 模式, 已弃用)
  MeshDepthAligner  — DA3 度量深度→缩放 mesh (当前推荐)
"""

import numpy as np
import os
from typing import Tuple, Optional


class DepthScaleRecovery:
    """深度尺度恢复器 — 相对→度量 (DA2 兼容, 已弃用)

    当使用 DA3 度量深度时, 不需要此类。
    保留仅为向后兼容 DA2 相对深度管线。
    """

    def __init__(
        self,
        method: str = "triposr_bbox",
        known_object_size_mm: float = None,
        heuristic_depth_range_mm: float = 500.0,
    ):
        self.method = method
        self.known_object_size_mm = known_object_size_mm
        self.heuristic_depth_range_mm = heuristic_depth_range_mm

    def recover(
        self,
        relative_depth: np.ndarray,  # (H, W) float16, [0, 1]
        mask: np.ndarray,             # (H, W) uint8
        triposr_bbox_size: float = None,
    ) -> np.ndarray:
        """
        Returns:
            pseudo_metric_depth: ndarray (H, W) float16, 单位 mm
        """
        masked_depths = relative_depth[mask > 0].astype(np.float32)
        if len(masked_depths) == 0:
            return np.zeros_like(relative_depth, dtype=np.float16)

        d_rel_min = masked_depths.min()
        d_rel_max = masked_depths.max()
        d_rel_range = max(d_rel_max - d_rel_min, 0.05)

        if self.method == "known_size" and self.known_object_size_mm is not None:
            scale = self.known_object_size_mm / d_rel_range
            d_metric_min = 500.0
        elif self.method == "triposr_bbox" and triposr_bbox_size is not None:
            real_size_mm = 100.0
            scale = real_size_mm / d_rel_range
            d_metric_min = 400.0
        else:
            scale = self.heuristic_depth_range_mm / d_rel_range
            d_metric_min = 400.0

        metric_depth = d_metric_min + (relative_depth.astype(np.float32) - d_rel_min) * scale
        return metric_depth.astype(np.float16)

    def recover_batch(
        self,
        relative_depths: np.ndarray,
        masks: np.ndarray,
        output_memmap: str = None,
    ) -> np.ndarray:
        """批量尺度恢复 + memmap输出"""
        n_frames = relative_depths.shape[0]
        h, w = relative_depths.shape[1:3]

        mmap_path = output_memmap or "./output/depths_metric.dat"
        os.makedirs(os.path.dirname(mmap_path) or ".", exist_ok=True)
        metric_mmap = np.memmap(mmap_path, dtype=np.float16, mode='w+',
                                shape=(n_frames, h, w))

        base_result = self.recover(relative_depths[0], masks[0])

        for i in range(n_frames):
            if i == 0:
                metric_mmap[i] = base_result
            else:
                metric_mmap[i] = self.recover(relative_depths[i], masks[i])
            if i % 100 == 0:
                print(f"[ScaleRecovery] Frame {i}/{n_frames}")

        return metric_mmap


class MeshDepthAligner:
    """Mesh-深度尺度对齐器 — DA3 度量深度 → 缩放 mesh (推荐)

    核心思路 (改进):
      原方案: 缩放深度到启发式物体尺寸 → FP 使用伪度量深度
      改进方案: DA3 已提供度量深度 (m) → 从物体区域深度 + 相机内参 K
                反投影到 3D, 估算物体物理尺寸 → 缩放 TripoSR mesh 对齐
                → FP 直接使用 DA3 度量深度 (转 mm)

    方法:
      - "depth_guided": 从深度+mask+K 反投影, 计算物体 3D bbox, 缩放 mesh
      - "heuristic":    使用默认物体尺寸缩放 mesh
    """

    def __init__(
        self,
        method: str = "depth_guided",
        default_object_size_mm: float = 100.0,
    ):
        self.method = method
        self.default_object_size_mm = default_object_size_mm

    def align(
        self,
        mesh_path: str,
        depth_m: np.ndarray,       # (H, W) float32, DA3 度量深度 (meters)
        mask: np.ndarray,           # (H, W) uint8
        K: np.ndarray,              # (3, 3) 相机内参
        output_dir: str = "./output/meshes",
    ) -> Tuple[str, float]:
        """将 mesh 尺度对齐到度量深度

        Args:
            mesh_path: TripoSR 生成的 mesh 路径 (.glb/.obj)
            depth_m: DA3 度量深度 (meters)
            mask: 物体掩码
            K: 相机内参矩阵
            output_dir: 输出目录

        Returns:
            aligned_mesh_path: 缩放后的 mesh 路径
            scale_mm: mesh→mm 的缩放因子
        """
        if self.method == "depth_guided":
            aligned_path, scale_mm = self._align_depth_guided(
                mesh_path, depth_m, mask, K, output_dir)
        else:
            aligned_path, scale_mm = self._align_heuristic(mesh_path, output_dir)

        return aligned_path, float(scale_mm)

    def _align_depth_guided(
        self,
        mesh_path: str,
        depth_m: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        output_dir: str,
    ) -> Tuple[str, float]:
        """从度量深度估算物体物理尺寸, 缩放 mesh

        步骤:
          1. 掩码区域深度点 → 相机内参反投影 → 3D 点云 (meters)
          2. 统计滤波去除离群值
          3. 计算物体 3D bbox 对角线 (meters)
          4. 加载 mesh, 计算其 bbox 对角线 (任意单位)
          5. 缩放因子 = obj_diag_mm / mesh_diag
          6. 导出缩放后的 mesh
        """
        import trimesh

        ys, xs = np.where(mask > 0)
        if len(ys) < 100:
            print("[MeshDepthAligner] Mask too small (<100px), falling back to heuristic.")
            return self._align_heuristic(mesh_path, output_dir)

        # 采样控制计算量
        n_sample = min(3000, len(ys))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(ys), n_sample, replace=False)
        ys_s, xs_s = ys[idx], xs[idx]

        z_cam = depth_m[ys_s, xs_s]  # meters
        valid = (z_cam > 0.1) & (z_cam < 10.0) & np.isfinite(z_cam)
        if valid.sum() < 50:
            print("[MeshDepthAligner] Too few valid depth points, falling back to heuristic.")
            return self._align_heuristic(mesh_path, output_dir)

        ys_s, xs_s, z_cam = ys_s[valid], xs_s[valid], z_cam[valid]

        # 反投影: pixel → camera 3D
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        x_cam = (xs_s - cx) * z_cam / fx
        y_cam = (ys_s - cy) * z_cam / fy
        pts_3d = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3) meters

        # IQR 离群值过滤
        q1 = np.percentile(pts_3d, 25, axis=0)
        q3 = np.percentile(pts_3d, 75, axis=0)
        iqr = q3 - q1
        center = np.median(pts_3d, axis=0)
        inlier = np.all(np.abs(pts_3d - center) < 2.5 * iqr, axis=1)
        if inlier.sum() >= 30:
            pts_3d = pts_3d[inlier]

        # 物体 3D 包围盒对角线
        obj_min = pts_3d.min(axis=0)
        obj_max = pts_3d.max(axis=0)
        obj_diag_m = float(np.linalg.norm(obj_max - obj_min))
        obj_diag_mm = obj_diag_m * 1000.0

        # 钳制在合理范围 (1cm ~ 2m)
        obj_diag_mm = max(10.0, min(obj_diag_mm, 2000.0))

        # Mesh 包围盒对角线
        mesh = trimesh.load(mesh_path, force="mesh")
        mesh_min = mesh.vertices.min(axis=0)
        mesh_max = mesh.vertices.max(axis=0)
        mesh_diag = float(np.linalg.norm(mesh_max - mesh_min))

        if mesh_diag < 1e-8:
            mesh_diag = 1.0

        scale_mm = obj_diag_mm / mesh_diag

        # 导出缩放后的 mesh
        mesh.vertices = mesh.vertices.astype(np.float64) * scale_mm
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(mesh_path))[0]
        aligned_path = os.path.join(output_dir, f"{base}_aligned.glb")
        mesh.export(aligned_path)

        print(f"[MeshDepthAligner] Object 3D extent: {obj_diag_mm:.0f}mm "
              f"(from {len(pts_3d)} back-projected points)")
        print(f"[MeshDepthAligner] Mesh diag: {mesh_diag:.3f}u → scale factor: {scale_mm:.1f}")
        print(f"[MeshDepthAligner] Aligned mesh: {aligned_path}")

        # 更新 mesh_path 引用 (副作用, 供调用者使用)
        return aligned_path, scale_mm

    def _align_heuristic(self, mesh_path: str, output_dir: str) -> Tuple[str, float]:
        """启发式缩放: 假设物体为默认尺寸, 缩放 mesh 到 mm"""
        import trimesh

        mesh = trimesh.load(mesh_path, force="mesh")
        mesh_min = mesh.vertices.min(axis=0)
        mesh_max = mesh.vertices.max(axis=0)
        mesh_diag = float(np.linalg.norm(mesh_max - mesh_min))

        if mesh_diag < 1e-8:
            mesh_diag = 1.0

        scale_mm = self.default_object_size_mm / mesh_diag

        mesh.vertices = mesh.vertices.astype(np.float64) * scale_mm
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(mesh_path))[0]
        aligned_path = os.path.join(output_dir, f"{base}_aligned.glb")
        mesh.export(aligned_path)

        print(f"[MeshDepthAligner] Heuristic scale: {self.default_object_size_mm:.0f}mm "
              f"/ {mesh_diag:.3f}u → factor: {scale_mm:.1f}")
        print(f"[MeshDepthAligner] Aligned mesh: {aligned_path}")

        return aligned_path, scale_mm
