"""
output_writer.py — Module 11
功能: CSV 姿态输出 + 3D bbox 可视化视频渲染
低配适配: 360p 渲染, 不保留原生分辨率版本
"""

import numpy as np
import cv2
import pandas as pd
import os
from typing import List


class PoseOutputWriter:
    """姿态结果 CSV 输出"""

    @staticmethod
    def write_csv(
        poses: List[np.ndarray],
        confidences: List[float],
        timestamps: List[float],
        output_path: str,
    ):
        """CSV: timestamp, qw, qx, qy, qz, tx, ty, tz, confidence"""
        from scipy.spatial.transform import Rotation

        rows = []
        for i, (T, conf, ts) in enumerate(zip(poses, confidences, timestamps)):
            R_mat = T[:3, :3]
            t = T[:3, 3]
            quat = Rotation.from_matrix(R_mat).as_quat()  # [x,y,z,w]
            rows.append({
                "frame": i,
                "timestamp": round(ts, 6),
                "qw": round(float(quat[3]), 8),
                "qx": round(float(quat[0]), 8),
                "qy": round(float(quat[1]), 8),
                "qz": round(float(quat[2]), 8),
                "tx": round(float(t[0]), 6),
                "ty": round(float(t[1]), 6),
                "tz": round(float(t[2]), 6),
                "confidence": round(float(conf), 6),
            })

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[Output] CSV written: {output_path} ({len(df)} rows)")


class VisualizationRenderer:
    """可视化渲染器 — 3D bbox + 坐标轴叠加 (360p)"""

    def __init__(self, mesh_path: str, model_scale: float = 0.12):
        """Args: model_scale must match FoundationPose's mesh.vertices *= scale"""
        self.model_scale = model_scale
        try:
            import trimesh
            mesh = trimesh.load(mesh_path, force="mesh")
            mesh.vertices *= model_scale  # Match FoundationPose scale
            self.bbox_3d = self._compute_bbox_corners(mesh.vertices)
            axis_len = np.linalg.norm(mesh.vertices.max(0) - mesh.vertices.min(0)) * 0.5
        except Exception:
            self.bbox_3d = self._compute_bbox_corners(np.array([
                [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])) * model_scale
            axis_len = 0.5 * model_scale

        self.axis_3d = np.array([
            [0, 0, 0],
            [axis_len, 0, 0],
            [0, axis_len, 0],
            [0, 0, axis_len],
        ])
        self.color_bbox = (0, 255, 0)       # Green
        self.color_axes = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: Red=X, Green=Y, Blue=Z
        self.thickness = 2

    @staticmethod
    def _compute_bbox_corners(vertices: np.ndarray) -> np.ndarray:
        v_min = vertices.min(axis=0)
        v_max = vertices.max(axis=0)
        return np.array([
            [v_min[0], v_min[1], v_min[2]],
            [v_max[0], v_min[1], v_min[2]],
            [v_max[0], v_max[1], v_min[2]],
            [v_min[0], v_max[1], v_min[2]],
            [v_min[0], v_min[1], v_max[2]],
            [v_max[0], v_min[1], v_max[2]],
            [v_max[0], v_max[1], v_max[2]],
            [v_min[0], v_max[1], v_max[2]],
        ])

    def project_points(self, points_3d: np.ndarray, pose: np.ndarray, K: np.ndarray) -> np.ndarray:
        pts_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        pts_cam = (pose @ pts_h.T).T[:, :3]
        pts_img = (K @ pts_cam.T).T
        pts_img = pts_img[:, :2] / pts_img[:, 2:3]
        return pts_img

    def render_frame(self, frame: np.ndarray, pose: np.ndarray, K: np.ndarray) -> np.ndarray:
        vis = frame.copy()
        bbox_2d = self.project_points(self.bbox_3d, pose, K).astype(np.int32)

        edges = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7),
        ]
        for e1, e2 in edges:
            cv2.line(vis, tuple(bbox_2d[e1]), tuple(bbox_2d[e2]),
                     self.color_bbox, self.thickness)

        axis_2d = self.project_points(self.axis_3d, pose, K).astype(np.int32)
        origin = tuple(axis_2d[0])
        for i in range(3):
            cv2.line(vis, origin, tuple(axis_2d[i+1]),
                     self.color_axes[i], self.thickness + 1)

        return vis

    def render_video(
        self,
        frames: List[np.ndarray],
        poses: List[np.ndarray],
        K: np.ndarray,
        output_path: str,
        fps: float = 30.0,
    ):
        h, w = frames[0].shape[:2]
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # 按容器选择 codec:
        #   .mp4 → mp4v (内置, 兼容主流播放器; OpenH264 未装, H264 不可用)
        #   .avi → XVID (Windows 兼容)
        ext = os.path.splitext(output_path)[1].lower()
        if ext == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")

        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            # 回退: mp4v 写失败则尝试 XVID (可能容器/编解码器兼容问题)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for i, (frame, pose) in enumerate(zip(frames, poses)):
            vis = self.render_frame(frame, pose, K)
            writer.write(vis)
            if i % 100 == 0:
                print(f"[Viz] Frame {i}/{len(frames)}")

        writer.release()
        print(f"[Viz] Video saved: {output_path}")
