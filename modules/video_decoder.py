"""
video_decoder.py — Module 1
功能: 将 mp4/avi 视频流式解码为缩放帧序列
默认分辨率: 720p (源视频高于 720p 时降采样, 低于则保持原生)
"""

import cv2
import numpy as np
from typing import Tuple, Generator


class VideoDecoder:
    """通用视频解码器 — 支持流式逐帧解码"""

    def __init__(self, target_short_edge: int = 720):
        """
        Args:
            target_short_edge: 短边缩放目标(px)。
                720p 为服务器默认值; 源分辨率低于目标时保持原生。
        """
        self.target_short_edge = target_short_edge

    def get_metadata(self, video_path: str) -> dict:
        """获取视频元信息 (不解码所有帧)"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        scale = self.target_short_edge / min(h, w) if min(h, w) > self.target_short_edge else 1.0
        return {
            "fps": fps,
            "frame_count": frame_count,
            "native_size": (w, h),
            "proc_size": (int(w * scale), int(h * scale)),
            "scale": scale,
        }

    def decode_stream(self, video_path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
        """流式解码生成器 — 逐帧 yield，不一次性加载全帧到内存

        Yields:
            (frame_idx, frame): 帧索引 + ndarray (H, W, 3) BGR uint8
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        idx = 0
        proc_w, proc_h = None, None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 短边缩放（保持宽高比）
            h, w = frame.shape[:2]
            scale = self.target_short_edge / min(h, w)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
                proc_w, proc_h = new_w, new_h
            else:
                proc_w, proc_h = w, h
            yield idx, frame
            idx += 1

        cap.release()
        print(f"[VideoDecoder] Streamed {idx} frames @ {self.fps:.2f} FPS, "
              f"resolution: {proc_w}x{proc_h}")

    def decode_all(self, video_path: str) -> Tuple[list, float, int]:
        """一次性解码所有帧 (用于需要全帧列表的场景, 如 XMem)

        Returns:
            frames: list of ndarray (H, W, 3) BGR uint8
            fps: 原始帧率
            frame_count: 总帧数
        """
        frames = []
        for idx, frame in self.decode_stream(video_path):
            frames.append(frame)
        return frames, self.fps, self.frame_count
