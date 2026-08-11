"""
sam_segmentor.py — Module 3
功能: EfficientViT-SAM l0 实例分割 + triton shim for Windows
"""

import numpy as np
import torch
import sys, types


def _install_triton_shim():
    """Install triton shim for Windows (no CUDA triton support)."""
    if 'triton' in sys.modules:
        return
    import importlib
    spec = importlib.machinery.ModuleSpec('triton', None)
    tl = types.ModuleType('triton.language'); tl.__spec__ = spec
    tl.constexpr = type('constexpr', (), {})
    tl.float32 = None; tl.float16 = None
    tl.pid = lambda x: 0; tl.load = lambda *a, **kw: None
    tl.store = lambda *a, **kw: None; tl.arange = lambda x, y: range(x, y)
    tl.zeros = lambda *a: []; tl.program_id = lambda x: 0
    tl.num_programs = lambda x: 1
    tr = types.ModuleType('triton'); tr.__spec__ = spec; tr.language = tl
    class _JIT:
        def __init__(self, fn): self.fn = fn
        def __getitem__(self, grid): return self
        def __call__(self, *a, **kw): return None
    tr.jit = _JIT
    tr.autotune = lambda *a, **kw: (lambda x: x)
    tr.Config = type('Config', (), {'__init__': lambda s, **kw: None})
    sys.modules['triton'] = tr
    sys.modules['triton.language'] = tl


class EfficientViTSAMSegmentor:
    """EfficientViT-SAM l0 分割器 (512x512, 35M params, ~0.4GB VRAM)"""

    def __init__(
        self,
        model_path: str = "E:/zhijiyige/weights/efficientvit_sam/efficientvit_sam_l0.pt",
        model_name: str = "efficientvit-sam-l0",
        device: str = "cuda:0",
    ):
        _install_triton_shim()
        from efficientvit.sam_model_zoo import create_efficientvit_sam_model
        from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

        self.model = create_efficientvit_sam_model(
            name=model_name, weight_url=model_path,
        ).to(device).eval()
        self.predictor = EfficientViTSamPredictor(self.model)
        self.device = device

    def segment_with_box(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Args:
            image: RGB ndarray (H, W, 3) uint8
            bbox: [x1, y1, x2, y2]

        Returns:
            mask: ndarray (H, W) uint8 {0, 1}
        """
        if image.ndim == 3 and image.shape[2] == 3:
            image_rgb = image
        else:
            import cv2
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(image_rgb)
        masks, scores, logits = self.predictor.predict(
            point_coords=None, point_labels=None,
            box=bbox[None, :], multimask_output=False,
        )
        best_mask = masks[0].astype(np.uint8)
        best_score = float(scores[0])
        print(f"[SAM] EfficientViT-l0: mask={best_mask.sum()}px conf={best_score:.3f}")
        return best_mask

    def unload(self):
        if hasattr(self, 'model'):
            del self.model
            del self.predictor
        torch.cuda.empty_cache()
        print("[SAM] Unloaded.")
