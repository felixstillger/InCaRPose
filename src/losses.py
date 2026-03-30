"""
Loss functions for relative camera pose estimation.
"""

import torch
import torch.nn.functional as F


def rot_ang_loss(R: torch.Tensor, Rgt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Geodesic rotation angular error (radians).

    Args:
        R:   predicted rotation matrices ``(B, 3, 3)``
        Rgt: ground-truth rotation matrices ``(B, 3, 3)``
    """
    residual = torch.matmul(R.transpose(1, 2), Rgt)
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
    cosine = (trace - 1) / 2
    return torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps)).mean()


def transl_ang_loss(t: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Translation direction angular error (radians).

    Args:
        t:   predicted translation ``(B, 3)``
        tgt: ground-truth translation ``(B, 3)``
    """
    t_n = t / (torch.norm(t, dim=1, keepdim=True) + eps)
    tgt_n = tgt / (torch.norm(tgt, dim=1, keepdim=True) + eps)
    cosine = torch.sum(t_n * tgt_n, dim=1)
    return torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps)).mean()
