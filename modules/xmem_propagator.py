"""
xmem_propagator.py — Module 4
功能: XMem 视频mask传播 (使用官方 MaskMapper API)
"""

import torch, numpy as np, os, sys, cv2
from typing import List


class XMemPropagator:
    """XMem mask 传播器 — MaskMapper API (兼容 single_object=False)"""

    def __init__(self, model_path="E:/zhijiyige/weights/xmem/XMem-s012.pth",
                 device="cuda:0", resolution=360, segment_length=200, segment_overlap=5):
        sys.path.insert(0, "E:/zhijiyige/src/XMem")
        from model.network import XMem
        from inference.inference_core import InferenceCore
        from inference.data.mask_mapper import MaskMapper

        config = {'mem_every': 5, 'deep_update_every': -1,
            'enable_long_term': True, 'enable_long_term_count_usage': True,
            'max_mid_term_frames': 10, 'min_mid_term_frames': 5,
            'num_prototypes': 128, 'max_long_term_elements': 10000,
            'top_k': 30, 'num_objects': 1,
            'key_dim': 64, 'value_dim': 512, 'hidden_dim': 64}
        network = XMem(config=config, model_path=model_path, map_location=device)
        network.to(device).eval()
        self.processor = InferenceCore(network, config=config)
        self.device = device; self.resolution = resolution
        self.segment_length = segment_length; self.segment_overlap = segment_overlap

    def propagate(self, frames: List[np.ndarray], first_mask: np.ndarray,
                  output_memmap: str = None) -> np.ndarray:
        """XMem 传播, 使用 MaskMapper 处理 multi-object 格式"""
        from inference.data.mask_mapper import MaskMapper

        h_orig, w_orig = frames[0].shape[:2]
        scale = self.resolution / min(h_orig, w_orig)
        h_proc, w_proc = int(h_orig * scale), int(w_orig * scale)
        n_frames = len(frames)

        if output_memmap:
            os.makedirs(os.path.dirname(output_memmap) or ".", exist_ok=True)
            all_masks = np.memmap(output_memmap, dtype=np.uint8, mode='w+', shape=(n_frames, h_orig, w_orig))
        else:
            all_masks = np.zeros((n_frames, h_orig, w_orig), dtype=np.uint8)

        # Prepare first mask
        mask_0 = cv2.resize(first_mask.astype(np.uint8), (w_proc, h_proc), interpolation=cv2.INTER_NEAREST)
        mapper = MaskMapper()
        msk, labels = mapper.convert_mask(mask_0)
        self.processor.set_all_labels(list(mapper.remappings.values()))

        # Preprocess frames
        proc_frames = []
        for frame in frames:
            f = cv2.resize(frame, (w_proc, h_proc))
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            proc_frames.append(torch.from_numpy(f).permute(2, 0, 1).float() / 255.0)

        with torch.no_grad():
            for i, f_tensor in enumerate(proc_frames):
                f_tensor = f_tensor.unsqueeze(0).to(self.device)
                if i == 0:
                    if isinstance(msk, np.ndarray):
                        m = torch.from_numpy(msk).to(self.device)
                    else:
                        m = msk.to(self.device)
                    prob = self.processor.step(f_tensor, m, labels, end=(i == n_frames - 1))
                else:
                    prob = self.processor.step(f_tensor)
                # prob: (num_objects+1, H, W) — argmax 沿物体维
                out = torch.argmax(prob, dim=0).cpu().numpy().astype(np.uint8)
                out = mapper.remap_index_mask(out)
                if scale != 1.0:
                    out = cv2.resize(out, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                all_masks[i] = out
                if i % 100 == 0:
                    print(f"[XMem] Frame {i}/{n_frames}")

        self.processor.clear_memory()
        torch.cuda.empty_cache()
        return all_masks

    def unload(self):
        if hasattr(self, 'processor'): del self.processor
        torch.cuda.empty_cache()
        print("[XMem] Unloaded.")
