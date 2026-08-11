"""
FoundationPose launcher — patches pytorch3d dependency with local implementations.
Run from within WSL2:
    /root/miniconda3/envs/rgb6d/bin/python /mnt/e/zhijiyige/wsl_fp_launcher.py
"""
import sys, os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# ===== pytorch3d shim =====
class _FakePytorch3DTransforms:
    """Drop-in replacement for pytorch3d.transforms SE(3) functions"""

    @staticmethod
    def so3_log_map(R_mat, eps=1e-4):
        """SO(3) log map -> axis-angle vectors"""
        if isinstance(R_mat, torch.Tensor):
            R_np = R_mat.detach().cpu().numpy()
            is_torch = True
        else:
            R_np = R_mat
            is_torch = False
        rotvecs = R.from_matrix(R_np.reshape(-1,3,3)).as_rotvec()
        out = rotvecs.reshape(R_mat.shape[:-2] + (3,))
        return torch.from_numpy(out).float().to(R_mat.device) if is_torch else out

    @staticmethod
    def so3_exp_map(log_rot):
        """SO(3) exp map -> rotation matrices"""
        if isinstance(log_rot, torch.Tensor):
            log_np = log_rot.detach().cpu().numpy()
            is_torch = True
        else:
            log_np = log_rot
            is_torch = False
        mats = R.from_rotvec(log_np.reshape(-1,3)).as_matrix()
        out = mats.reshape(log_rot.shape[:-1] + (3,3))
        return torch.from_numpy(out).float().to(log_rot.device) if is_torch else out

    @staticmethod
    def se3_log_map(transform, eps=1e-4):
        """SE(3) log -> [rho, omega]"""
        R_mat = transform[..., :3, :3]
        t = transform[..., :3, 3]
        omega = _FakePytorch3DTransforms.so3_log_map(R_mat, eps)
        theta = torch.norm(omega, dim=-1, keepdim=True)
        mask = theta < eps
        theta_safe = torch.where(mask, torch.ones_like(theta), theta)
        omega_hat = _FakePytorch3DTransforms._hat(omega)
        V_inv = (torch.eye(3, device=transform.device) -
                 0.5 * omega_hat +
                 (1/(theta_safe**2) - (1+torch.cos(theta_safe))/(2*theta_safe*torch.sin(theta_safe))).unsqueeze(-1).unsqueeze(-1) * omega_hat @ omega_hat)
        V_inv = torch.where(mask.unsqueeze(-1), torch.eye(3, device=transform.device).expand_as(V_inv), V_inv)
        rho = (V_inv @ t.unsqueeze(-1)).squeeze(-1)
        return torch.cat([rho, omega], dim=-1)

    @staticmethod
    def se3_exp_map(log_transform, eps=1e-4):
        """se3 -> SE(3)"""
        rho, omega = log_transform[..., :3], log_transform[..., 3:]
        R_mat = _FakePytorch3DTransforms.so3_exp_map(omega)
        theta = torch.norm(omega, dim=-1, keepdim=True)
        mask = theta < eps
        theta_safe = torch.where(mask, torch.ones_like(theta), theta)
        omega_hat = _FakePytorch3DTransforms._hat(omega)
        V = (torch.eye(3, device=log_transform.device) +
             (1-torch.cos(theta_safe))/(theta_safe**2)).unsqueeze(-1).unsqueeze(-1) * omega_hat +
             ((theta_safe-torch.sin(theta_safe))/(theta_safe**3)).unsqueeze(-1).unsqueeze(-1) * omega_hat @ omega_hat)
        V = torch.where(mask.unsqueeze(-1), torch.eye(3, device=log_transform.device).expand_as(V), V)
        t = (V @ rho.unsqueeze(-1)).squeeze(-1)
        T = torch.zeros(log_transform.shape[:-1] + (4,4), device=log_transform.device)
        T[..., :3, :3] = R_mat
        T[..., :3, 3] = t
        T[..., 3, 3] = 1.0
        return T

    @staticmethod
    def matrix_to_axis_angle(matrix):
        return _FakePytorch3DTransforms.so3_log_map(matrix)

    @staticmethod
    def matrix_to_euler_angles(matrix, convention='XYZ'):
        if isinstance(matrix, torch.Tensor):
            m_np = matrix.detach().cpu().numpy()
            is_torch = True
        else:
            m_np = matrix
            is_torch = False
        euler = R.from_matrix(m_np.reshape(-1,3,3)).as_euler(convention.lower(), degrees=False)
        out = euler.reshape(matrix.shape[:-2] + (3,))
        return torch.from_numpy(out).float().to(matrix.device) if is_torch else out

    @staticmethod
    def euler_angles_to_matrix(euler_angles, convention='XYZ'):
        if isinstance(euler_angles, torch.Tensor):
            ea_np = euler_angles.detach().cpu().numpy()
            is_torch = True
        else:
            ea_np = euler_angles
            is_torch = False
        mats = R.from_euler(convention.lower(), ea_np.reshape(-1,3), degrees=False).as_matrix()
        out = mats.reshape(euler_angles.shape[:-1] + (3,3))
        return torch.from_numpy(out).float().to(euler_angles.device) if is_torch else out

    @staticmethod
    def rotation_6d_to_matrix(d6):
        """6D rotation representation -> 3x3 matrix"""
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = torch.nn.functional.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = torch.nn.functional.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)

    @staticmethod
    def _hat(v):
        """batch hat operator"""
        v1, v2, v3 = v[..., 0], v[..., 1], v[..., 2]
        z = torch.zeros_like(v1)
        return torch.stack([
            z, -v3, v2,
            v3, z, -v1,
            -v2, v1, z,
        ], dim=-1).reshape(v.shape[:-1] + (3,3))

# Inject the shim
import pytorch3d_shim_loader
sys.modules['pytorch3d'] = type(sys)('pytorch3d')
sys.modules['pytorch3d.transforms'] = _FakePytorch3DTransforms()

# ===== Import patched Utils =====
sys.path.insert(0, '/mnt/e/zhijiyige/src/FoundationPose')

# Patch Utils BEFORE import
import importlib
# Use a custom finder that intercepts pytorch3d imports
class Pytorch3DFinder:
    def find_module(self, fullname, path=None):
        if fullname.startswith('pytorch3d'):
            return self
        return None
    def load_module(self, fullname):
        if fullname == 'pytorch3d':
            m = type(sys)('pytorch3d')
            m.__path__ = []
            return m
        if fullname == 'pytorch3d.transforms':
            return sys.modules['pytorch3d.transforms']
        return None

sys.meta_path.insert(0, Pytorch3DFinder())

# Now import
from estimater import FoundationPose
print("FoundationPose imported OK (patched)")
