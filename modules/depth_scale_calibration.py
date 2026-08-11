"""
depth_scale_calibration.py — DA3 深度尺度校准 + 稳定性监控
============================================================
消除 DA3 度量深度的系统性尺度偏差 (BOP 实测 ~47% 低估).

核心发现:
  DA3-Metric 深度有固定的缩放偏差 (~×1.88 低估), 导致:
  - FoundationPose 的 guess_translation 物体中心偏离 500mm+
  - 所有基于深度的引导/修正失效

方案:
  1. 参考尺寸 (已知): CAD 模型直径, 或 TripoSR mesh 物理尺寸
  2. 逐帧: mask 区域反投影 → 观测尺寸 → 尺度比 = 参考尺寸/观测尺寸
  3. 校准: 深度 × 尺度比 (消除固定偏差)

稳定性监控:
  尺度校准前提是 DA3 深度相对一致. 对剧烈波动/异常跳变帧:
  - 计算 mask 区域深度中位数的帧间跳变
  - 超过阈值标记为不稳定 → 用前帧深度回退/跳过校准
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class DepthStabilityMonitor:
    """深度稳定性监控 — 检测剧烈波动/异常跳变的深度帧"""

    def __init__(
        self,
        jump_threshold: float = 0.15,     # 帧间深度中位数跳变阈值 (m)
        max_deviation_ratio: float = 0.25,  # 相对中位数的最大偏离比
        history_len: int = 5,             # 稳定性历史窗口
    ):
        self.jump_threshold = jump_threshold
        self.max_deviation_ratio = max_deviation_ratio
        self.history_len = history_len
        self._depth_median_hist = []
        self._stability_stats = {'total': 0, 'unstable': 0, 'recovered': 0}

    def _mask_depth_median(self, depth: np.ndarray, mask: np.ndarray) -> Optional[float]:
        """mask 区域深度中位数"""
        ys, xs = np.where(mask > 0)
        if len(ys) < 20:
            return None
        vals = depth[ys, xs]
        valid = (vals > 0.05) & (vals < 10) & np.isfinite(vals)
        if valid.sum() < 10:
            return None
        return float(np.median(vals[valid]))

    def check(self, depth: np.ndarray, mask: np.ndarray) -> Tuple[bool, str]:
        """检测当前帧深度是否稳定

        Returns:
            (is_stable, reason)
        """
        self._stability_stats['total'] += 1
        d_med = self._mask_depth_median(depth, mask)
        if d_med is None:
            self._stability_stats['unstable'] += 1
            return False, "no_valid_depth"

        # ① 帧间跳变检测
        if self._depth_median_hist:
            prev_med = np.median(self._depth_median_hist)
            if abs(d_med - prev_med) > self.jump_threshold:
                self._stability_stats['unstable'] += 1
                self._depth_median_hist.append(d_med)
                if len(self._depth_median_hist) > self.history_len:
                    self._depth_median_hist.pop(0)
                return False, f"jump({abs(d_med-prev_med):.2f}m)"

        # ② 深度离散度检测 (mask 区域 std 过大 → 深度不可靠)
        ys, xs = np.where(mask > 0)
        vals = depth[ys, xs]
        valid = (vals > 0.05) & (vals < 10) & np.isfinite(vals)
        if valid.sum() > 10:
            d_std = float(np.std(vals[valid]))
            if d_std > self.max_deviation_ratio * d_med:
                self._stability_stats['unstable'] += 1
                self._depth_median_hist.append(d_med)
                if len(self._depth_median_hist) > self.history_len:
                    self._depth_median_hist.pop(0)
                return False, f"high_std({d_std:.2f}m)"

        # 稳定
        self._depth_median_hist.append(d_med)
        if len(self._depth_median_hist) > self.history_len:
            self._depth_median_hist.pop(0)
        return True, "ok"

    def summary(self) -> str:
        s = self._stability_stats
        if s['total'] == 0:
            return ""
        pct = 100 * s['unstable'] / s['total']
        return f"{s['total']} 帧, {s['unstable']} 不稳定 ({pct:.0f}%)"


class DepthScaleCalibrator:
    """基于参考尺寸的逐帧深度尺度校准"""

    def __init__(
        self,
        reference_size_m: Optional[float] = None,
        scale_min: float = 0.5,
        scale_max: float = 3.0,
        enable_stability: bool = True,
    ):
        """
        Args:
            reference_size_m: 参考物体尺寸 (m), 如 CAD 直径或 mesh 对角线
            scale_min/max: 尺度比钳制范围 (防校准发散)
            enable_stability: 启用稳定性监控 (过滤异常帧)
        """
        self.reference_size_m = reference_size_m
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.monitor = DepthStabilityMonitor() if enable_stability else None
        self.scale_hist = []

    def set_reference_size(self, size_m: float):
        """设置参考尺寸 (已知物体尺寸)"""
        self.reference_size_m = float(size_m)

    def estimate_scale(self, depth: np.ndarray, mask: np.ndarray,
                       K: np.ndarray) -> Optional[float]:
        """估算尺度比 = 参考尺寸 / 观测尺寸

        观测尺寸: mask 区域像素反投影 → 3D bbox 对角线
        """
        if self.reference_size_m is None:
            return None

        ys, xs = np.where(mask > 0)
        if len(ys) < 50:
            return None
        z = depth[ys, xs]
        valid = (z > 0.05) & (z < 10) & np.isfinite(z)
        if valid.sum() < 30:
            return None
        ys_v, xs_v, z_v = ys[valid], xs[valid], z[valid]

        # 反投影
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        x_cam = (xs_v - cx) * z_v / fx
        y_cam = (ys_v - cy) * z_v / fy
        pts = np.stack([x_cam, y_cam, z_v], axis=1)

        # IQR 去离群
        q1, q3 = np.percentile(pts, 25, axis=0), np.percentile(pts, 75, axis=0)
        iqr = q3 - q1
        center = np.median(pts, axis=0)
        inlier = np.all(np.abs(pts - center) < 2.5 * iqr, axis=1)
        if inlier.sum() >= 30:
            pts = pts[inlier]

        obs_size = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        if obs_size < 1e-4:
            return None

        scale = self.reference_size_m / obs_size
        scale = np.clip(scale, self.scale_min, self.scale_max)
        return float(scale)

    def calibrate(self, depth: np.ndarray, mask: np.ndarray,
                  K: np.ndarray, use_stability: bool = True
                  ) -> Tuple[np.ndarray, Optional[float]]:
        """校准单帧深度

        Returns:
            (calibrated_depth, scale_ratio or None)
        """
        # 稳定性检查 (异常帧不校准, 返回原深度)
        if use_stability and self.monitor is not None:
            is_stable, reason = self.monitor.check(depth, mask)
            if not is_stable:
                return depth.astype(np.float32), None

        scale = self.estimate_scale(depth, mask, K)
        if scale is None:
            return depth.astype(np.float32), None

        self.scale_hist.append(scale)
        calibrated = depth.astype(np.float32) * scale
        return calibrated, scale

    def calibrate_sequence(self, depths: np.ndarray, masks: np.ndarray,
                           K: np.ndarray, output_memmap: str = None
                           ) -> np.ndarray:
        """批量校准深度序列

        Args:
            depths: (N,H,W) float32 meters
            masks: (N,H,W) uint8
            K: (3,3)

        Returns:
            calibrated_depths: (N,H,W) float32
        """
        n = depths.shape[0]
        H, W = depths.shape[1:]
        out = np.memmap(output_memmap, dtype=np.float32, mode='w+',
                        shape=(n, H, W)) if output_memmap else \
            np.zeros((n, H, W), dtype=np.float32)

        # 全局尺度: 用所有稳定帧的尺度中位数 (更鲁棒)
        self.scale_hist = []
        valid_scales = []

        for i in range(n):
            cal, scale = self.calibrate(depths[i], masks[i], K)
            out[i] = cal
            if scale is not None:
                valid_scales.append(scale)

        if valid_scales:
            global_scale = float(np.median(valid_scales))
            self.global_scale = global_scale
            # 用全局尺度重新校准所有帧 (消除帧间波动)
            for i in range(n):
                out[i] = depths[i].astype(np.float32) * global_scale
            print(f'[ScaleCalibrate] 全局尺度比={global_scale:.3f} '
                  f'(参考尺寸={self.reference_size_m*1000:.0f}mm, '
                  f'{len(valid_scales)}/{n} 稳定帧)')
        else:
            self.global_scale = 1.0
            print('[ScaleCalibrate] 无有效尺度估计, 跳过校准')

        if self.monitor is not None:
            print(f'[Stability] {self.monitor.summary()}')

        return out

    def apply_global_scale(self, depth: np.ndarray) -> np.ndarray:
        """用已估计的全局尺度校准单帧 (追踪阶段使用)"""
        if hasattr(self, 'global_scale') and self.global_scale != 1.0:
            return depth.astype(np.float32) * self.global_scale
        return depth.astype(np.float32)
