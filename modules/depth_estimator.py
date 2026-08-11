"""
depth_estimator.py — Module 5
功能: 单目深度估计
支持: Depth Anything V3 metric (首选) / Depth Anything V2 (备用)
"""

import cv2, torch, numpy as np, os, sys
from typing import Optional, List, Tuple


class DepthEstimator:
    """统一深度估计器 — DA3 metric 优先, DA2 备用"""

    def __init__(self, device="cuda:0", model_size="da3"):
        self.device = device
        self.model_size = model_size
        self._model = None
        self._da3 = None

    def _load_da3(self):
        if self._da3 is None:
            from depth_anything_3.api import DepthAnything3
            import tempfile
            self._da3 = DepthAnything3.from_pretrained(
                'E:/zhijiyige/weights/da3_metric')
            self._da3 = self._da3.to(self.device).eval()
        return self._da3

    def estimate_da3(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """DA3 度量深度估计 + 内参估计

        Args:
            image: RGB ndarray (H, W, 3) uint8

        Returns:
            depth: (H, W) float32 — metric depth in meters
            K: (3, 3) float64 — estimated camera intrinsics
        """
        if image.shape[-1] == 3 and image.dtype == np.uint8:
            rgb = image
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        model = self._load_da3()
        import tempfile
        tmpdir = tempfile.mkdtemp()
        pred = model.inference([rgb], export_dir=tmpdir)

        depth = pred.depth[0]  # (H_da3, W_da3) in meters
        conf = pred.conf[0] if pred.conf is not None else np.ones_like(depth)
        K_da3 = pred.intrinsics[0] if pred.intrinsics is not None else None

        # Resize depth to match input image
        h_img, w_img = image.shape[:2]
        if depth.shape[:2] != (h_img, w_img):
            depth = cv2.resize(depth, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

        # Scale K if DA3 used different resolution
        if K_da3 is not None:
            h_da3, w_da3 = pred.depth[0].shape[:2]
            sx, sy = w_img / w_da3, h_img / h_da3
            K_da3[0] *= sx
            K_da3[1] *= sy

        return depth.astype(np.float32), K_da3

    def estimate_da3_batch(self, images: List[np.ndarray], every_n: int = 10
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """批量 DA3 深度估计

        Returns:
            depths: (N, H, W) float32 meters
            K: (3, 3) estimated camera intrinsics (from first frame)
        """
        n = len(images)
        h_img, w_img = images[0].shape[:2]
        depths = np.zeros((n, h_img, w_img), dtype=np.float32)

        K = None
        for i in range(0, n, every_n):
            d, Ki = self.estimate_da3(images[i])
            depths[i] = d
            if K is None and Ki is not None:
                K = Ki
            if i % 100 == 0:
                print(f"[DA3] Frame {i}/{n}")

        # Fill gaps with nearest
        for i in range(n):
            if i % every_n != 0:
                base = (i // every_n) * every_n
                depths[i] = depths[min(base, n - 1)]

        return depths, K

    def unload(self):
        if self._da3 is not None:
            del self._da3; self._da3 = None
        torch.cuda.empty_cache()
        print("[DA3] Unloaded.")
