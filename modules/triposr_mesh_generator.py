"""
triposr_mesh_generator.py — Module 6
功能: 从首帧 RGB+mask 生成 3D 代理网格 (proxy CAD)
低配适配: mc_resolution=128 + FP16 + unload() + bbox fallback
"""

import torch
import numpy as np
import os
import sys
from typing import Tuple


class TripoSRMeshGenerator:
    """单图->3D网格 — 使用本地 TripoSR 权重"""

    def __init__(
        self,
        device: str = "cuda:0",
        mc_resolution: int = 128,
        output_dir: str = "./output/meshes",
        enable_fallback: bool = True,
        source_dir: str = "E:/zhijiyige/src/TripoSR",
        model_dir: str = "E:/zhijiyige/weights/triposr",
    ):
        self.device = device
        self.mc_resolution = mc_resolution
        self.output_dir = output_dir
        self.enable_fallback = enable_fallback
        self.source_dir = source_dir
        self.model_dir = model_dir
        self._model = None
        self._tsr_available = False
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_model(self):
        """加载 TripoSR 模型 (延迟加载, 用完即卸)"""
        if self._model is None and not self._tsr_available:
            try:
                tsr_src = self.source_dir
                if tsr_src not in sys.path:
                    sys.path.insert(0, tsr_src)
                from tsr.system import TSR
                self._model = TSR.from_pretrained(
                    self.model_dir, config_name='config.yaml', weight_name='model.ckpt')
                self._model.to(self.device).eval()
                self._tsr_available = True
                print('[TripoSR] Model loaded from local weights')
            except Exception as e:
                print(f'[TripoSR] Load failed: {e}')
                self._tsr_available = False
                self._model = None
        return self._model

    def generate(self, image: np.ndarray, mask: np.ndarray, output_name: str = "proxy_mesh") -> Tuple[str, dict]:
        from PIL import Image

        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            raise RuntimeError("Empty mask!")
        cx, cy = int(xs.mean()), int(ys.mean())
        size = max(image.shape[0], image.shape[1])
        x1, y1 = max(0, cx - size // 2), max(0, cy - size // 2)
        x2, y2 = min(image.shape[1], x1 + size), min(image.shape[0], y1 + size)

        try:
            model = self._load_model()
            if model is None:
                raise RuntimeError("TSR not available")

            masked_image = image.copy()
            masked_image[mask == 0] = 255
            cropped = masked_image[y1:y2, x1:x2]
            pil_img = Image.fromarray(cropped)
            pil_img = pil_img.resize((512, 512), Image.LANCZOS)

            with torch.no_grad():
                scene_codes = model([pil_img], device=self.device)
            mesh = model.extract_mesh(scene_codes, has_vertex_color=False,
                                       resolution=self.mc_resolution)[0]
            del scene_codes
            mesh_source = "triposr"
        except torch.cuda.OutOfMemoryError:
            print("[TripoSR] OOM! Falling back to bbox proxy mesh...")
            torch.cuda.empty_cache()
            mesh = self._generate_bbox_mesh()
            mesh_source = "bbox_fallback_oom"
        except Exception as e:
            print(f"[TripoSR] {type(e).__name__}: {e}, falling back to bbox mesh...")
            torch.cuda.empty_cache()
            mesh = self._generate_bbox_mesh()
            mesh_source = "bbox_fallback_error"

        mesh_path = os.path.join(self.output_dir, f"{output_name}.glb")
        mesh.export(mesh_path)

        mesh_info = {
            "vertices": len(mesh.vertices), "faces": len(mesh.faces),
            "path": mesh_path, "source": mesh_source,
            "crop_bbox": [x1, y1, x2, y2],
        }
        print(f"[TripoSR] Mesh ({mesh_source}): {mesh_info['vertices']}v/{mesh_info['faces']}f -> {mesh_path}")
        return mesh_path, mesh_info

    def _generate_bbox_mesh(self):
        import trimesh
        return trimesh.creation.box(extents=[1.0, 1.0, 1.0])

    def unload(self):
        if self._model is not None:
            del self._model; self._model = None
        torch.cuda.empty_cache()
        print("[TripoSR] Unloaded.")
