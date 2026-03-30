"""
Utility functions for camera pose estimation training and evaluation.
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Rotation / transform helpers
# ---------------------------------------------------------------------------

def compute_pose_2_to_1(pose1, pose2):
    """Relative transform that maps coordinates from pose2 into pose1."""
    return np.linalg.inv(pose1) @ pose2


def orthogonalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Orthogonalise the rotation part of batched 4×4 transforms (flat shape ``(B, 16)``)."""
    B = matrix.size(0)
    mat = matrix.view(B, 4, 4)
    q, _ = torch.linalg.qr(mat[:, :3, :3])
    mat = mat.clone()
    mat[:, :3, :3] = q
    return mat.view(B, 16)


def orthogonalize_rotation(rot: torch.Tensor) -> torch.Tensor:
    """SVD-based orthogonalisation of batched 3×3 rotation matrices (flat ``(B, 9)``)."""
    B = rot.size(0)
    R_mat = rot.view(B, 3, 3)
    U, _, Vh = torch.linalg.svd(R_mat)
    R_ortho = U @ Vh
    det = torch.det(R_ortho)
    mask = det < 0
    if mask.any():
        U_fix = U.clone()
        U_fix[mask, :, -1] = -U_fix[mask, :, -1]
        R_ortho = U_fix @ Vh
    return R_ortho.view(B, 9)


def rotation_angle_from_matrix(R_mat: torch.Tensor) -> torch.Tensor:
    """Rotation angle (radians) from batched 3×3 rotation matrices."""
    v = torch.stack([
        R_mat[:, 2, 1] - R_mat[:, 1, 2],
        R_mat[:, 0, 2] - R_mat[:, 2, 0],
        R_mat[:, 1, 0] - R_mat[:, 0, 1],
    ], dim=1)
    trace = R_mat[:, 0, 0] + R_mat[:, 1, 1] + R_mat[:, 2, 2]
    return torch.atan2(torch.linalg.norm(v, dim=1), trace - 1)


# ---------------------------------------------------------------------------
# Batch conversions: representation → rotation matrix
# ---------------------------------------------------------------------------

def batch_rvecs_to_matrix(rvecs: torch.Tensor) -> torch.Tensor:
    """Rodrigues: ``(B, 3)`` rotation vectors → ``(B, 3, 3)`` matrices."""
    theta = torch.norm(rvecs, dim=1, keepdim=True)
    k = rvecs / (theta + 1e-8)
    K = torch.zeros(rvecs.shape[0], 3, 3, dtype=rvecs.dtype, device=rvecs.device)
    K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2];  K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] = k[:, 0]
    I = torch.eye(3, device=rvecs.device).unsqueeze(0)
    s = torch.sin(theta).unsqueeze(-1)
    c = torch.cos(theta).unsqueeze(-1)
    return I + s * K + (1 - c) * torch.bmm(K, K)


def batch_euler_to_matrix(eulers: torch.Tensor, intrinsic=True, degree=False) -> torch.Tensor:
    """``(B, 3)`` Euler angles → ``(B, 3, 3)`` rotation matrices."""
    if degree:
        eulers = torch.deg2rad(eulers)

    def _axis_rot(angle, axis):
        c, s = torch.cos(angle), torch.sin(angle)
        z, o = torch.zeros_like(angle), torch.ones_like(angle)
        if axis == "x":
            return torch.stack([torch.stack([o, z, z], -1),
                                torch.stack([z, c, -s], -1),
                                torch.stack([z, s, c], -1)], -2)
        if axis == "y":
            return torch.stack([torch.stack([c, z, s], -1),
                                torch.stack([z, o, z], -1),
                                torch.stack([-s, z, c], -1)], -2)
        return torch.stack([torch.stack([c, -s, z], -1),
                            torch.stack([s, c, z], -1),
                            torch.stack([z, z, o], -1)], -2)

    Rs = [_axis_rot(eulers[:, i], a) for i, a in enumerate("xyz")]
    if intrinsic:
        return torch.bmm(torch.bmm(Rs[2], Rs[1]), Rs[0])
    return torch.bmm(torch.bmm(Rs[0], Rs[1]), Rs[2])


def batch_quaternion_to_matrix(quats: torch.Tensor) -> torch.Tensor:
    """``(B, 4)`` quaternions (xyzw) → ``(B, 3, 3)`` rotation matrices."""
    quats = F.normalize(quats, dim=1)
    x, y, z, w = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack([
        torch.stack([1 - 2*(yy+zz), 2*(xy-wz),     2*(xz+wy)],     -1),
        torch.stack([2*(xy+wz),     1 - 2*(xx+zz), 2*(yz-wx)],      -1),
        torch.stack([2*(xz-wy),     2*(yz+wx),     1 - 2*(xx+yy)],  -1),
    ], -2)


def batch_construct_transform(rot: torch.Tensor, tvec: torch.Tensor) -> torch.Tensor:
    """``(B, 3, 3)`` + ``(B, 3)`` → ``(B, 4, 4)``."""
    B = rot.shape[0]
    T = torch.eye(4, device=rot.device).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = rot
    T[:, :3, 3] = tvec
    return T


def get_batched_transformation_matrix(q_est, q_true, euler_style="", orthogonalize=True):
    """Build 4×4 transforms from flat predicted / ground-truth vectors."""
    B = q_true.shape[0]
    t_est = torch.zeros(B, 3, device=q_est.device)
    t_true = torch.zeros(B, 3, device=q_true.device)

    if q_true.shape[1] in (3, 6):
        if euler_style == "":
            r_true = batch_rvecs_to_matrix(q_true[:, :3])
            r_est = batch_rvecs_to_matrix(q_est[:, :3])
        elif euler_style == "intrinsic":
            r_true = batch_euler_to_matrix(q_true[:, :3], intrinsic=True)
            r_est = batch_euler_to_matrix(q_est[:, :3], intrinsic=True)
        else:
            r_true = batch_euler_to_matrix(q_true[:, :3], intrinsic=False)
            r_est = batch_euler_to_matrix(q_est[:, :3], intrinsic=False)
        if q_true.shape[1] == 6:
            t_est = q_est[:, 3:]
            t_true = q_true[:, 3:]
    elif q_true.shape[1] in (4, 7):
        r_true = batch_quaternion_to_matrix(q_true[:, :4])
        r_est = batch_quaternion_to_matrix(q_est[:, :4])
        if q_true.shape[1] == 7:
            t_est = q_est[:, 4:]
            t_true = q_true[:, 4:]
    elif q_true.shape[1] in (9, 12):
        r_true = q_true[:, :9].reshape(B, 3, 3)
        r_est = q_est[:, :9].reshape(B, 3, 3)
        if q_true.shape[1] == 12:
            t_est = q_est[:, 9:]
            t_true = q_true[:, 9:]
    elif q_true.shape[1] == 16:
        return q_est.view(B, 4, 4), q_true.view(B, 4, 4)
    else:
        raise ValueError(f"Unexpected vector size {q_true.shape[1]}")

    return batch_construct_transform(r_est, t_est), batch_construct_transform(r_true, t_true)


# ---------------------------------------------------------------------------
# Loss / error metrics
# ---------------------------------------------------------------------------

def transformation_loss(t_est, t_true, alpha=0.01, orthogonolize=False):
    """Geodesic rotation error + weighted L2 translation error."""
    eps = 1e-5
    R_est = t_est[:, :3, :3]
    if orthogonolize:
        R_est = orthogonalize_rotation(R_est.reshape(-1, 9)).reshape(-1, 3, 3)
    R_true = t_true[:, :3, :3]
    R_diff = torch.bmm(R_est.transpose(1, 2), R_true)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    rot_err = torch.acos(torch.clamp((trace - 1) / 2, -1.0 + eps, 1.0 - eps))
    trans_err = torch.norm(t_est[:, :3, 3] - t_true[:, :3, 3], dim=1)
    return rot_err.mean() + alpha * trans_err.mean()


def translation_loss(t_pred, t_gt, reduction="sum"):
    errors = torch.norm(t_pred - t_gt, dim=1)
    if reduction == "mean":
        return errors.mean()
    elif reduction == "sum":
        return errors.sum()
    return errors


def get_rot_trans_error(t_est, t_true):
    """Return per-sample ``(rot_error_deg, trans_error)``."""
    eps = 1e-5
    R_diff = torch.bmm(t_est[:, :3, :3].transpose(1, 2), t_true[:, :3, :3])
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    rot_err = torch.rad2deg(torch.acos(torch.clamp((trace - 1) / 2, -1 + eps, 1 - eps)))
    trans_err = torch.norm(t_est[:, :3, 3] - t_true[:, :3, 3], dim=1)
    return rot_err, trans_err


def print_rot_error(t_est, t_true, orthogonolize=True):
    eps = 1e-5
    R_est = t_est[:, :3, :3]
    if orthogonolize:
        U, _, Vt = torch.linalg.svd(R_est)
        R_est = U @ Vt
    R_diff = torch.bmm(R_est.transpose(1, 2), t_true[:, :3, :3])
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    rot_err = torch.rad2deg(torch.acos(torch.clamp((trace - 1) / 2, -1 + eps, 1 - eps)))
    print(f"rot error (deg): {rot_err}, orthogonalised={orthogonolize}")
    return rot_err


# ---------------------------------------------------------------------------
# CSV metric writer
# ---------------------------------------------------------------------------

def write_metrics_to_csv(val_metrics, log_dir, current_epoch):
    csv_path = os.path.join(log_dir, "aggregated_val_metrics.csv")
    header_needed = not os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["epoch", "loader_name", "metric_name",
                             "mean", "median", "min", "max", "count"])
        for loader_name, metrics in val_metrics.items():
            for metric_name, values in metrics.items():
                if values:
                    t = torch.abs(torch.tensor(values))
                    writer.writerow([
                        current_epoch, loader_name, metric_name,
                        t.mean().item(), t.median().item(),
                        t.min().item(), t.max().item(), len(values),
                    ])


# ---------------------------------------------------------------------------
# RoPE2D (Rotary Position Embedding for 2-D spatial tokens)
# ---------------------------------------------------------------------------

class RoPE2D(nn.Module):
    """2-D Rotary Position Embedding (used in the cross-decoder)."""

    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        key = (D, seq_len, device, dtype)
        if key not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2, device=device).float() / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat([freqs, freqs], dim=-1)
            self.cache[key] = (freqs.cos(), freqs.sin())
        return self.cache[key]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        cos = F.embedding(pos1d, cos)[:, None, :, :]
        sin = F.embedding(pos1d, sin)[:, None, :, :]
        return tokens * cos + self.rotate_half(tokens) * sin

    def forward(self, tokens, positions):
        D = tokens.size(3) // 2
        cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.device, tokens.dtype)
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
        return torch.cat([y, x], dim=-1)
