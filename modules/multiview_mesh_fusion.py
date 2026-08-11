"""
multiview_mesh_fusion.py — 多视图时序网格融合
==============================================
用追踪得到的每帧 pose, 将多帧 mask 区域深度反投影点云累积到物体坐标系,
覆盖物体正面/侧面/背面几何, 重建完整 3D 网格.

替代 TripoSR 首帧静态重建:
  TripoSR 只有正面几何 (背面是 hallucination), 物体旋转/遮挡时失准.
  本模块利用连续帧看到的多个侧面, 补全背面几何.

流程:
  1. 每帧: mask 区域像素 + 深度反投影 → 相机系 3D 点云 (m)
  2. 用追踪 pose 的逆变换 → 物体坐标系
  3. 累积所有帧 → 完整物体点云
  4. 体素降采样 → Poisson 表面重建 → 完整网格
"""

import numpy as np
import cv2
import os
from typing import List, Optional, Tuple


def unproject_depth(depth: np.ndarray, mask: np.ndarray, K: np.ndarray,
                    max_points: int = 8000) -> np.ndarray:
    """mask 区域深度反投影 → 相机系 3D 点 (m)

    Args:
        depth: (H,W) float32 meters
        mask: (H,W) uint8 物体掩码
        K: (3,3) 相机内参
        max_points: 采样上限

    Returns:
        pts: (N,3) 相机系点云 (m)
    """
    ys, xs = np.where(mask > 0)
    if len(ys) < 30:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth[ys, xs]
    valid = (z > 0.05) & (z < 10) & np.isfinite(z)
    if valid.sum() < 30:
        return np.zeros((0, 3), dtype=np.float32)
    ys_v, xs_v, z_v = ys[valid], xs[valid], z[valid]

    # 采样 (控制点数)
    if len(z_v) > max_points:
        idx = np.random.RandomState(0).choice(len(z_v), max_points, replace=False)
        ys_v, xs_v, z_v = ys_v[idx], xs_v[idx], z_v[idx]

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x_cam = (xs_v - cx) * z_v / fx
    y_cam = (ys_v - cy) * z_v / fy
    return np.stack([x_cam, y_cam, z_v], axis=1).astype(np.float32)


def pose_to_camera_inv(pose: np.ndarray) -> np.ndarray:
    """SE(3) 逆变换: 相机系 → 物体系"""
    R, t = pose[:3, :3], pose[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T = np.eye(4)
    T[:3, :3] = R_inv
    T[:3, 3] = t_inv
    return T


def accumulate_object_cloud(
    frames: List[np.ndarray],
    masks: np.ndarray,
    depths: np.ndarray,
    poses: List[np.ndarray],
    K: np.ndarray,
    frame_indices: Optional[List[int]] = None,
    max_points_per_frame: int = 6000,
    max_depth_std: float = 0.15,
    stat_outlier_k: float = 1.5,
    stat_outlier_nb: int = 20,
) -> np.ndarray:
    """多帧累积物体坐标系点云

    Args:
        frames: 帧列表 (BGR)
        masks: (N,H,W) uint8 逐帧掩码
        depths: (N,H,W) float32 深度 (m)
        poses: 每帧 4x4 追踪 pose (相机系, mesh 在原点)
        K: (3,3)
        frame_indices: 使用的帧 (None = 全部)
        max_points_per_frame: 每帧采样点数
        max_depth_std: 跳过深度离散度大的帧 (mask 区域 std > 此值)
        stat_outlier_k / nb: 统计滤波参数

    Returns:
        cloud: (M,3) 物体坐标系点云 (m)
    """
    if frame_indices is None:
        frame_indices = list(range(len(frames)))

    clouds = []
    n_skipped_std = 0
    for i in frame_indices:
        # 深度稳定性: 跳过 mask 区域深度离散度过大的帧
        ys, xs = np.where(masks[i] > 0)
        if len(ys) > 20:
            zvals = depths[i][ys, xs]
            valid = (zvals > 0.05) & (zvals < 10) & np.isfinite(zvals)
            if valid.sum() > 10:
                d_std = float(np.std(zvals[valid]))
                if d_std > max_depth_std:
                    n_skipped_std += 1
                    continue

        # 相机系点云
        pts_cam = unproject_depth(depths[i], masks[i], K, max_points_per_frame)
        if len(pts_cam) == 0:
            continue
        # 相机系 → 物体系 (用追踪 pose 逆变换)
        T_inv = pose_to_camera_inv(poses[i])
        pts_obj = (T_inv[:3, :3] @ pts_cam.T).T + T_inv[:3, 3]
        clouds.append(pts_obj)

    if not clouds:
        return np.zeros((0, 3), dtype=np.float32)
    cloud = np.concatenate(clouds, axis=0)
    if n_skipped_std > 0:
        print(f'[MeshFusion] 跳过 {n_skipped_std} 深度不稳定帧')

    # 统计滤波 (去除深度噪声产生的离群点)
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.astype(np.float64))
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=stat_outlier_nb, std_ratio=stat_outlier_k)
    cloud_filtered = np.asarray(pcd.points).astype(np.float32)
    if len(cloud_filtered) > 0:
        return cloud_filtered
    return cloud


def reconstruct_mesh(cloud: np.ndarray, output_path: str,
                     voxel_size: float = 0.003,
                     poisson_depth: int = 9,
                     min_density_ratio: float = 0.05) -> Optional[str]:
    """点云 → 完整网格 (Poisson 表面重建)

    Args:
        cloud: (M,3) 物体系点云 (m)
        output_path: 输出 .glb 路径
        voxel_size: 体素降采样尺寸 (m)
        poisson_depth: Poisson 重建深度 (越大细节越多)
        min_density_ratio: 低密度区域裁剪阈值

    Returns:
        output_path or None (失败)
    """
    import open3d as o3d

    if len(cloud) < 100:
        print(f'[MeshFusion] 点云太少 ({len(cloud)}), 无法重建')
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.astype(np.float64))

    # 体素降采样 (均匀密度)
    pcd = pcd.voxel_down_sample(voxel_size)
    n_down = len(pcd.points)
    print(f'[MeshFusion] 点云 {len(cloud)} → 降采样 {n_down}')

    # 法线估计
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 10, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(15)

    # Poisson 重建
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth)

    # 裁剪低密度区域 (减少噪声/伪影)
    densities = np.asarray(densities)
    d_max = densities.max()
    if d_max > 0:
        mesh.remove_vertices_by_mask(
            densities < min_density_ratio * d_max)

    # 清理
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()

    # 转为 trimesh 保存 (open3d glb 导出可能有问题)
    import trimesh
    verts = np.asarray(mesh.vertices).astype(np.float32)
    faces = np.asarray(mesh.triangles).astype(np.int64)
    if len(verts) == 0 or len(faces) == 0:
        return None
    tri_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tri_mesh.export(output_path)

    print(f'[MeshFusion] 网格: {len(verts)}v/{len(faces)}f -> {output_path}')
    print(f'[MeshFusion] 尺寸: {tri_mesh.bounds[1]-tri_mesh.bounds[0]}')
    return output_path


def fuse_multiview_mesh(
    frames: List[np.ndarray],
    masks: np.ndarray,
    depths: np.ndarray,
    poses: List[np.ndarray],
    K: np.ndarray,
    output_path: str,
    frame_indices: Optional[List[int]] = None,
    voxel_size: float = 0.003,
) -> Optional[str]:
    """多视图网格融合完整流程"""
    print('[MeshFusion] 累积多视图点云...')
    cloud = accumulate_object_cloud(
        frames, masks, depths, poses, K, frame_indices=frame_indices)
    print(f'[MeshFusion] 累积点云: {len(cloud)} 点 (物体系)')

    if len(cloud) == 0:
        return None
    return reconstruct_mesh(cloud, output_path, voxel_size=voxel_size)
