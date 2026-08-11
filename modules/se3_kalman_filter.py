"""
se3_kalman_filter.py — Module 9
功能: SE(3) 李代数卡尔曼滤波 — 姿态轨迹平滑
低配适配: 无额外模型加载, CPU 计算, 低开销
"""

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from typing import List, Tuple


class SE3LieKalmanFilter:
    """SE(3) 左不变扩展卡尔曼滤波器 + RTS 平滑器"""

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_noise_pos: float = 0.01,
        process_noise_rot: float = 0.001,
        measurement_noise_pos: float = 0.005,
        measurement_noise_rot: float = 0.002,
    ):
        self.dt = dt
        # 基础噪声 (用于自适应时作为基准)
        self.base_process_pos = process_noise_pos
        self.base_process_rot = process_noise_rot
        self.base_meas_pos = measurement_noise_pos
        self.base_meas_rot = measurement_noise_rot
        self.Q = np.diag([process_noise_pos] * 3 + [process_noise_rot] * 3) * dt
        self.R = np.diag([measurement_noise_pos] * 3 + [measurement_noise_rot] * 3)
        self.X = None
        self.P = None
        self.v = None
        # 自适应运动级别 [0,1]: 0=静止, 1=剧烈运动
        self.motion_level = 0.0
        # 静止时噪声参数 (强平滑, 抑制抖动漂移)
        self.static_process_pos = process_noise_pos * 0.5
        self.static_process_rot = process_noise_rot * 0.3
        self.static_meas_pos = measurement_noise_pos * 2.0
        self.static_meas_rot = measurement_noise_rot * 4.0
        # 运动时噪声参数 (放大旋转过程噪声, 降低观测平滑 → 减少滞后)
        self.motion_process_pos = process_noise_pos * 2.0
        self.motion_process_rot = process_noise_rot * 8.0
        self.motion_meas_pos = measurement_noise_pos * 0.5
        self.motion_meas_rot = measurement_noise_rot * 0.5

    def set_motion(self, motion_level: float):
        """自适应噪声: 根据运动强度动态调整过程/观测噪声

        不动就磨平噪声 (静止: 高观测噪声 → 强力平滑, 低过程噪声 → 抑制漂移)
        一动就放开枷锁 (运动: 高过程噪声 → 允许跟随, 低观测噪声 → 减少滞后)
        """
        self.motion_level = float(np.clip(motion_level, 0.0, 1.0))
        # 线性插值: 静止参数 → 运动参数
        p_pos = self.static_process_pos + (self.motion_process_pos - self.static_process_pos) * self.motion_level
        p_rot = self.static_process_rot + (self.motion_process_rot - self.static_process_rot) * self.motion_level
        m_pos = self.static_meas_pos + (self.motion_meas_pos - self.static_meas_pos) * self.motion_level
        m_rot = self.static_meas_rot + (self.motion_meas_rot - self.static_meas_rot) * self.motion_level
        self.Q = np.diag([p_pos] * 3 + [p_rot] * 3) * self.dt
        self.R = np.diag([m_pos] * 3 + [m_rot] * 3)
        return self.motion_level

    # ---- Lie Algebra Utilities ----
    @staticmethod
    def _hat(omega: np.ndarray) -> np.ndarray:
        return np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0],
        ])

    @staticmethod
    def _vee(Omega: np.ndarray) -> np.ndarray:
        return np.array([Omega[2, 1], Omega[0, 2], Omega[1, 0]])

    @staticmethod
    def so3_exp(omega: np.ndarray) -> np.ndarray:
        theta = np.linalg.norm(omega)
        if theta < 1e-12:
            return np.eye(3)
        w_hat = SE3LieKalmanFilter._hat(omega)
        R = np.eye(3) + np.sin(theta)/theta * w_hat + \
            (1 - np.cos(theta))/(theta**2) * w_hat @ w_hat
        return R

    @staticmethod
    def so3_log(R: np.ndarray) -> np.ndarray:
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if abs(theta) < 1e-12:
            return np.zeros(3)
        omega_hat = theta / (2 * np.sin(theta)) * (R - R.T)
        return SE3LieKalmanFilter._vee(omega_hat)

    @staticmethod
    def se3_exp(xi: np.ndarray) -> np.ndarray:
        rho, omega = xi[:3], xi[3:6]
        theta = np.linalg.norm(omega)
        R_mat = SE3LieKalmanFilter.so3_exp(omega)
        if theta < 1e-12:
            V = np.eye(3)
        else:
            w_hat = SE3LieKalmanFilter._hat(omega)
            V = np.eye(3) + (1-np.cos(theta))/(theta**2)*w_hat + \
                (theta-np.sin(theta))/(theta**3) * w_hat @ w_hat
        t = V @ rho
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t
        return T

    @staticmethod
    def se3_log(T: np.ndarray) -> np.ndarray:
        R_mat, t = T[:3, :3], T[:3, 3]
        omega = SE3LieKalmanFilter.so3_log(R_mat)
        theta = np.linalg.norm(omega)
        if theta < 1e-12:
            V_inv = np.eye(3)
        else:
            w_hat = SE3LieKalmanFilter._hat(omega)
            V_inv = np.eye(3) - 0.5*w_hat + \
                (1/(theta**2) - (1+np.cos(theta))/(2*theta*np.sin(theta))) * w_hat @ w_hat
        rho = V_inv @ t
        return np.concatenate([rho, omega])

    @staticmethod
    def _adjoint_se3(xi: np.ndarray) -> np.ndarray:
        rho_hat = SE3LieKalmanFilter._hat(xi[:3])
        omega_hat = SE3LieKalmanFilter._hat(xi[3:6])
        ad = np.zeros((6, 6))
        ad[:3, :3] = omega_hat
        ad[:3, 3:6] = rho_hat
        ad[3:6, 3:6] = omega_hat
        return ad

    # ---- LIEKF Core ----
    def initialize(self, T0: np.ndarray, v0: np.ndarray = None):
        self.X = T0.copy()
        self.P = np.eye(6) * 0.1
        self.v = v0 if v0 is not None else np.zeros(6)

    def predict(self):
        self.X = self.X @ self.se3_exp(self.v * self.dt)
        ad_v = self._adjoint_se3(self.v)
        F_mat = np.eye(6) + ad_v * self.dt
        self.P = F_mat @ self.P @ F_mat.T + self.Q

    def update(self, T_obs: np.ndarray):
        X_inv_obs = np.linalg.inv(self.X) @ T_obs
        z = self.se3_log(X_inv_obs)
        H_mat = np.eye(6)
        S = H_mat @ self.P @ H_mat.T + self.R
        K = self.P @ H_mat.T @ np.linalg.inv(S)
        correction = K @ z
        self.X = self.X @ self.se3_exp(correction)
        self.P = (np.eye(6) - K @ H_mat) @ self.P
        self.v = correction / self.dt

    # ---- Batch Smoothing ----
    def smooth(
        self, poses: List[np.ndarray], confidences: List[float]
    ) -> List[np.ndarray]:
        """前向滤波 + 后向 RTS 平滑"""
        self.initialize(poses[0])
        forward_poses = [poses[0].copy()]

        for i in range(1, len(poses)):
            conf = max(confidences[i], 0.01)
            scale = 1.0 / conf
            R_adaptive = self.R * scale
            R_orig = self.R.copy()
            self.R = R_adaptive

            self.predict()
            self.update(poses[i])

            self.R = R_orig
            forward_poses.append(self.X.copy())

        # 后向 RTS 平滑
        smoothed = forward_poses.copy()
        for i in range(len(poses) - 2, -1, -1):
            X_i = forward_poses[i]
            X_ip1 = smoothed[i + 1]
            C = 0.5
            diff = self.se3_log(np.linalg.inv(X_i) @ X_ip1)
            smoothed[i] = X_i @ self.se3_exp(C * diff)

        return smoothed


# ---- Utility: Pose Conversions ----

def pose_matrix_to_quat_translation(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """4x4 SE(3) → (quat_wxyz, trans_xyz)"""
    R_mat = T[:3, :3]
    t_vec = T[:3, 3]
    quat_xyzw = R_scipy.from_matrix(R_mat).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return quat_wxyz, t_vec


def quat_translation_to_pose_matrix(quat_wxyz: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
    """(quat_wxyz, trans_xyz) → 4x4 SE(3)"""
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    R_mat = R_scipy.from_quat(quat_xyzw).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t_vec
    return T
