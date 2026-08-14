"""
xmem_propagator.py — Module 4
功能: XMem 视频mask传播 (使用官方 MaskMapper API) + 自动恢复
"""

import torch, numpy as np, os, sys, cv2
from typing import List, Callable, Optional


class XMemPropagator:
    """XMem mask 传播器 — MaskMapper API (兼容 single_object=False), 支持丢失自动恢复"""

    def __init__(self,
                 model_path="/mnt/20T/xieyongling/zhijiyige/weights/xmem/XMem-s012.pth",
                 device="cuda:0", resolution=720, segment_length=200, segment_overlap=5):
        sys.path.insert(0, "/mnt/20T/xieyongling/zhijiyige/src/XMem")
        from model.network import XMem
        from inference.inference_core import InferenceCore
        from inference.data.mask_mapper import MaskMapper

        self.config = {'mem_every': 5, 'deep_update_every': -1,
            'enable_long_term': True, 'enable_long_term_count_usage': True,
            'max_mid_term_frames': 10, 'min_mid_term_frames': 5,
            'num_prototypes': 128, 'max_long_term_elements': 10000,
            'top_k': 30, 'num_objects': 1,
            'key_dim': 64, 'value_dim': 512, 'hidden_dim': 64}
        network = XMem(config=self.config, model_path=model_path, map_location=device)
        network.to(device).eval()
        self.network = network
        self.processor = InferenceCore(network, config=self.config)
        self.device = device; self.resolution = resolution
        self.segment_length = segment_length; self.segment_overlap = segment_overlap

    def _fresh_processor(self):
        """创建全新的 InferenceCore (用于丢失后重新初始化)"""
        from inference.inference_core import InferenceCore
        self.processor = InferenceCore(self.network, config=self.config)

    def _step(self, f_tensor, mask, labels, end):
        """处理一帧; mask 非 None 时为首次/恢复步"""
        if mask is not None:
            if isinstance(mask, np.ndarray):
                m = torch.from_numpy(mask).to(self.device)
            else:
                m = mask.to(self.device)
            prob = self.processor.step(f_tensor, m, labels, end=end)
        else:
            prob = self.processor.step(f_tensor)
        out = torch.argmax(prob, dim=0).cpu().numpy().astype(np.uint8)
        return out

    def propagate(self, frames: List[np.ndarray], first_mask: np.ndarray,
                  output_memmap: str = None,
                  recovery_fn: Optional[Callable] = None,
                  loss_ratio: float = 0.15, lost_frames: int = 5) -> np.ndarray:
        """XMem 传播 + 自动恢复

        Args:
            frames: BGR 帧列表
            first_mask: 首帧掩码
            output_memmap: 输出 memmap 路径
            recovery_fn: 丢失恢复回调 fn(frame_bgr, frame_idx) -> mask or None.
                返回的新掩码需为 (H_orig, W_orig) uint8, 面积足够才接受.
            loss_ratio: 掩码面积低于 首帧面积×loss_ratio 判定为丢失
            lost_frames: 连续丢失 N 帧触发恢复
        """
        from inference.data.mask_mapper import MaskMapper

        h_orig, w_orig = frames[0].shape[:2]
        # 目标短边 = min(resolution, 源短边) — 不超过原生分辨率, 避免无意义放大
        scale = min(1.0, self.resolution / min(h_orig, w_orig))
        h_proc, w_proc = int(h_orig * scale), int(w_orig * scale)
        n_frames = len(frames)

        if output_memmap:
            os.makedirs(os.path.dirname(output_memmap) or ".", exist_ok=True)
            all_masks = np.memmap(output_memmap, dtype=np.uint8, mode='w+', shape=(n_frames, h_orig, w_orig))
        else:
            all_masks = np.zeros((n_frames, h_orig, w_orig), dtype=np.uint8)

        ref_area = max(int(first_mask.sum()), 1)

        def _to_proc(mask_orig):
            return cv2.resize(mask_orig.astype(np.uint8), (w_proc, h_proc),
                              interpolation=cv2.INTER_NEAREST)

        def _init_xmem(mask_proc):
            """用给定掩码初始化 MaskMapper + labels"""
            mapper = MaskMapper()
            msk, labels = mapper.convert_mask(mask_proc)
            self.processor.set_all_labels(list(mapper.remappings.values()))
            return mapper, msk, labels

        mask_proc0 = _to_proc(first_mask)
        mapper, msk, labels = _init_xmem(mask_proc0)

        # Preprocess frames
        proc_frames = []
        for frame in frames:
            f = cv2.resize(frame, (w_proc, h_proc))
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            proc_frames.append(torch.from_numpy(f).permute(2, 0, 1).float() / 255.0)

        n_recoveries = 0
        lost_count = 0
        with torch.no_grad():
            for i, f_tensor in enumerate(proc_frames):
                f_tensor = f_tensor.unsqueeze(0).to(self.device)
                if i == 0:
                    out = self._step(f_tensor, msk, labels, end=(i == n_frames - 1))
                else:
                    out = self._step(f_tensor, None, None, end=(i == n_frames - 1))
                out = mapper.remap_index_mask(out)
                if scale != 1.0:
                    out = cv2.resize(out, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                all_masks[i] = out
                if i % 100 == 0:
                    print(f"[XMem] Frame {i}/{n_frames}")

                # ── 丢失检测 + 自动恢复 ─────────────────────────────
                if recovery_fn is not None and i > 0:
                    area = int(out.sum())
                    if area < loss_ratio * ref_area:
                        lost_count += 1
                    else:
                        lost_count = 0
                    if lost_count >= lost_frames:
                        print(f"[XMem] 丢失 {lost_count} 帧 (area={area} < "
                              f"{loss_ratio}×{ref_area}), 尝试恢复 @frame {i}...")
                        new_mask = recovery_fn(frames[i], i)
                        if new_mask is not None and new_mask.sum() > 100:
                            # 重新初始化 XMem 从当前帧继续
                            self._fresh_processor()
                            mask_proc_new = _to_proc(new_mask)
                            mapper, msk, labels = _init_xmem(mask_proc_new)
                            out = self._step(f_tensor, msk, labels,
                                             end=(i == n_frames - 1))
                            out = mapper.remap_index_mask(out)
                            if scale != 1.0:
                                out = cv2.resize(out, (w_orig, h_orig),
                                                 interpolation=cv2.INTER_NEAREST)
                            all_masks[i] = out
                            ref_area = max(ref_area, int(new_mask.sum()))
                            lost_count = 0
                            n_recoveries += 1
                            print(f"[XMem] 恢复成功 @frame {i}: "
                                  f"{new_mask.sum()}px (累计恢复 {n_recoveries})")
                        else:
                            print(f"[XMem] 恢复失败 @frame {i}, 继续跟踪")

        self.processor.clear_memory()
        torch.cuda.empty_cache()
        print(f"[XMem] 传播完成: {n_frames} 帧, 自动恢复 {n_recoveries} 次")
        return all_masks

    def unload(self):
        if hasattr(self, 'processor'): del self.processor
        if hasattr(self, 'network'): del self.network
        torch.cuda.empty_cache()
        print("[XMem] Unloaded.")
