"""
Camera pose estimation model – DINOv2 backbone + cross-attention decoder.

Only the production architecture (``Camera_ref_Calibration_Model_v3_dinov3_v4_v3``)
is kept.  The model is imported via its short name ``PoseEstimationModel`` for
convenience.
"""

import os
import copy
import math
from functools import partial
import collections.abc
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy.spatial.transform import Rotation as R

from .losses import rot_ang_loss, transl_ang_loss
from .pose_utils import (
    RoPE2D,
    rotation_angle_from_matrix,
    batch_rvecs_to_matrix,
    batch_euler_to_matrix,
    batch_quaternion_to_matrix,
    batch_construct_transform,
    get_batched_transformation_matrix,
    transformation_loss,
    translation_loss,
    get_rot_trans_error,
    write_metrics_to_csv,
    print_rot_error,
    orthogonalize_matrix,
    orthogonalize_rotation,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class PositionGetter:
    """Return (y, x) position indices for every patch in a feature map."""

    def __init__(self):
        self.cache = {}

    def __call__(self, b, h, w, device="cuda"):
        if (h, w) not in self.cache:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache[h, w] = torch.cartesian_prod(y, x)
        return self.cache[h, w].view(1, h * w, 2).expand(b, -1, 2).clone()


# ---------------------------------------------------------------------------
# Decoder building blocks
# ---------------------------------------------------------------------------

class DecoderAttention(nn.Module):
    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x, xpos):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class DecoderCrossAttention(nn.Module):
    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, query, key, value, qpos, kpos):
        B, Nq, C = query.shape
        Nk, Nv = key.shape[1], value.shape[1]
        h = self.num_heads
        q = self.projq(query).reshape(B, Nq, h, C // h).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B, Nk, h, C // h).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B, Nv, h, C // h).permute(0, 2, 1, 3)
        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        x = x.transpose(1, 2).reshape(B, Nq, C)
        return self.proj_drop(self.proj(x))


class DecoderMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, bias=True, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop = to_2tuple(drop)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop[1])

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


class DecoderBlock(nn.Module):
    """Self-attention + cross-attention + FFN with RoPE."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True):
        super().__init__()
        self.rope = RoPE2D(freq=100)
        self.norm1 = norm_layer(dim)
        self.attn = DecoderAttention(dim, rope=self.rope, num_heads=num_heads,
                                     qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = DecoderCrossAttention(dim, rope=self.rope, num_heads=num_heads,
                                                qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.mlp = DecoderMlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                              act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y, xpos, ypos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        y_ = self.norm_y(y)
        x = x + self.drop_path(self.cross_attn(self.norm2(x), y_, y_, xpos, ypos))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, y


class DecoderResConvBlock(nn.Module):
    """1×1-conv residual block used after the decoder."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        self.c1 = nn.Conv2d(in_ch, out_ch, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 1)
        self.c3 = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, x):
        return self.skip(x) + F.relu(self.c3(F.relu(self.c2(F.relu(self.c1(x))))))


# ---------------------------------------------------------------------------
# Base Lightning class with training / validation logic
# ---------------------------------------------------------------------------

class _BasePoseModule(pl.LightningModule):
    """All loss / logging / optimiser logic lives here."""

    _SEVEN_SCENE_NAMES = ["chess", "fire", "heads", "office",
                          "pumpkin", "redkitchen", "stairs"]

    def __init__(self, output_size=7, loss_style="quatloss", euler_style="",
                 inverse_prediction=True, translation_weight=0.01,
                 l_r=3e-6, angular_error=False):
        super().__init__()
        self.output_size = output_size
        self.loss_style = loss_style
        self.euler_style = euler_style
        self.inverse_prediction = inverse_prediction
        self.translation_weight = translation_weight
        self.l_r = l_r
        self.translation_angular_error = angular_error
        self.orthogonalize_in_loss = False
        self.log_dir = None
        self.nan_counter = 0

        # These are overwritten per-dataset in train.py
        self.dataloader_names = ["standard"]

    # ---- losses ----------------------------------------------------------

    def _reloc_loss(self, q_est, q_true):
        def _c(te, tt):
            return rot_ang_loss(te[:, :3, :3], tt[:, :3, :3]) + \
                   transl_ang_loss(te[:, :3, 3], tt[:, :3, 3])
        if self.inverse_prediction:
            te1, tt1 = get_batched_transformation_matrix(
                q_est[:, :self.output_size], q_true[:, :self.output_size], self.euler_style)
            te2, tt2 = get_batched_transformation_matrix(
                q_est[:, self.output_size:], q_true[:, self.output_size:], self.euler_style)
            return _c(te1, tt1) + _c(te2, tt2)
        te, tt = get_batched_transformation_matrix(q_est, q_true, self.euler_style)
        return _c(te, tt)

    def _custom_loss(self, q_est, q_true):
        if self.inverse_prediction:
            te1, tt1 = get_batched_transformation_matrix(
                q_est[:, :self.output_size], q_true[:, :self.output_size], self.euler_style)
            te2, tt2 = get_batched_transformation_matrix(
                q_est[:, self.output_size:], q_true[:, self.output_size:], self.euler_style)
            return transformation_loss(te1, tt1, alpha=self.translation_weight) + \
                   transformation_loss(te2, tt2, alpha=self.translation_weight)
        te, tt = get_batched_transformation_matrix(q_est, q_true, self.euler_style)
        return transformation_loss(te, tt, alpha=self.translation_weight)

    def _quat_loss(self, q_est, q_true):
        eps = 1e-6

        def _rot_err(q1, q2):
            q2 = F.normalize(q2, dim=-1)
            dot = torch.sum(q1 * q2, dim=-1).clamp(-1 + eps, 1 - eps)
            return 2 * torch.acos(torch.abs(dot))

        def _trans_err(t1, t2):
            if self.translation_angular_error:
                return transl_ang_loss(t1, t2)
            return torch.norm(t1 - t2, p=2, dim=-1)

        def _pair(est, true):
            return _rot_err(est[:, :4], true[:, :4]) + _trans_err(est[:, 4:], true[:, 4:])

        if self.inverse_prediction:
            err = _pair(q_est[:, :self.output_size], q_true[:, :self.output_size])
            err = err + _pair(q_est[:, self.output_size:], q_true[:, self.output_size:])
        else:
            err = _pair(q_est, q_true)
        return err.mean()

    def _compute_loss(self, y_hat, y):
        if self.loss_style == "quatloss":
            return self._quat_loss(y_hat, y)
        if self.loss_style == "reloc":
            return self._reloc_loss(y_hat, y)
        if self.loss_style == "custom":
            return self._custom_loss(y_hat, y)
        return F.mse_loss(y_hat, y)

    # ---- normalisation ---------------------------------------------------

    def ortho_or_norm(self, pred):
        if self.output_size == 7:
            fwd = F.normalize(pred[:, :4], dim=1)
            if self.inverse_prediction:
                inv = F.normalize(pred[:, self.output_size:self.output_size + 4], dim=1)
                return torch.cat([fwd, pred[:, 4:7], inv, pred[:, 11:14]], dim=1)
            return torch.cat([fwd, pred[:, 4:7]], dim=1)
        if self.output_size == 16:
            fwd = orthogonalize_matrix(pred[:, :16])
            if self.inverse_prediction:
                return torch.cat([fwd, orthogonalize_matrix(pred[:, 16:])], dim=1)
            return fwd
        if self.output_size == 12:
            fwd_rot = orthogonalize_rotation(pred[:, :9])
            fwd = torch.cat([fwd_rot, pred[:, 9:12]], dim=1)
            if self.inverse_prediction:
                inv_rot = orthogonalize_rotation(pred[:, 12:21])
                return torch.cat([fwd, inv_rot, pred[:, 21:24]], dim=1)
            return fwd
        return pred

    # ---- logging helpers -------------------------------------------------

    def _log_rot_components(self, rot_est, rot_true, prefix):
        try:
            R_rel = torch.bmm(rot_est, rot_true.transpose(1, 2)).detach().cpu().numpy()
            euler = R.from_matrix(R_rel).as_euler("xyz", degrees=True)
            eps = 1e-6
            trace = torch.bmm(rot_est, rot_true.transpose(1, 2))
            trace = trace[:, 0, 0] + trace[:, 1, 1] + trace[:, 2, 2]
            geo = torch.rad2deg(torch.acos(torch.clamp((trace - 1) / 2, -1 + eps, 1 - eps)))
            self.log(f"{prefix}_geodesic_rot", geo.abs().mean(), prog_bar=True)
            self.log(f"{prefix}_er_x", np.abs(euler[:, 0]).mean())
            self.log(f"{prefix}_er_y", np.abs(euler[:, 1]).mean())
            self.log(f"{prefix}_er_z", np.abs(euler[:, 2]).mean())
        except Exception:
            pass

    def _log_transform_errors(self, y_hat, y, prefix="train"):
        if self.inverse_prediction:
            te, tt = get_batched_transformation_matrix(
                y_hat[:, :self.output_size], y[:, :self.output_size], self.euler_style)
        else:
            te, tt = get_batched_transformation_matrix(y_hat, y, self.euler_style)
        self._log_rot_components(te[:, :3, :3], tt[:, :3, :3], prefix)
        self.log(f"{prefix}_trans_err", translation_loss(te[:, :3, 3], tt[:, :3, 3]).mean())

    def _get_val_errors(self, y_hat, y):
        if self.inverse_prediction:
            te, tt = get_batched_transformation_matrix(
                y_hat[:, :self.output_size], y[:, :self.output_size], self.euler_style)
        else:
            te, tt = get_batched_transformation_matrix(y_hat, y, self.euler_style)
        R_rel = torch.bmm(te[:, :3, :3], tt[:, :3, :3].transpose(1, 2)).detach().cpu().numpy()
        # SVD fix
        fixed = []
        for M in R_rel:
            U, _, Vt = np.linalg.svd(M)
            P = U @ Vt
            if np.linalg.det(P) < 0:
                U[:, -1] *= -1
                P = U @ Vt
            fixed.append(P)
        euler = R.from_matrix(np.stack(fixed)).as_euler("xyz", degrees=True)
        return (euler[:, 0], euler[:, 1], euler[:, 2]), get_rot_trans_error(te, tt)

    # ---- training / validation -------------------------------------------

    def training_step(self, batch, batch_idx):
        x_ref, x_data, y = batch
        y_hat = self(x_ref, x_data)
        if torch.isnan(y).any() or torch.isnan(y_hat).any():
            self.nan_counter += 1
            if self.nan_counter > 50:
                raise ValueError("Too many NaN batches")
            return None
        loss = self._compute_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        with torch.no_grad():
            self._log_transform_errors(y_hat, y, "train")
        return loss

    def on_validation_epoch_start(self):
        self.val_metrics = {
            name: {k: [] for k in ("loss", "x_error", "y_error", "z_error",
                                   "rotation_error", "translation_error")}
            for name in self.dataloader_names
        }

    def validation_step(self, batch, batch_idx, *args):
        dl_idx = args[0] if args else "standard"
        x_ref, x_data, y = batch
        y_hat = self(x_ref, x_data)
        name = self.dataloader_names[dl_idx] if isinstance(dl_idx, int) else dl_idx
        loss = self._compute_loss(y_hat, y)
        self.log(f"val_{name}", loss, on_step=True, on_epoch=False)
        self._log_transform_errors(y_hat, y, f"val_{name}")
        (xe, ye, ze), (rot_e, tr_e) = self._get_val_errors(y_hat, y)
        self.val_metrics[name]["x_error"].extend(xe.tolist())
        self.val_metrics[name]["y_error"].extend(ye.tolist())
        self.val_metrics[name]["z_error"].extend(ze.tolist())
        self.val_metrics[name]["rotation_error"].extend(rot_e.cpu().tolist())
        self.val_metrics[name]["translation_error"].extend(tr_e.cpu().tolist())

    def validation_epoch_end(self, outputs):
        for name, metrics in self.val_metrics.items():
            for mname, vals in metrics.items():
                if vals:
                    t = np.abs(np.array(vals))
                    self.log(f"{name}/{mname}_mean", t.mean())
                    self.log(f"{name}/{mname}_median", np.median(t))
        if self.log_dir:
            write_metrics_to_csv(self.val_metrics, self.log_dir, self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.l_r, weight_decay=1e-5)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class PoseEstimationModel(_BasePoseModule):
    """DINOv2 backbone + cross-attention decoder for relative pose estimation.

    This is the cleaned-up version of ``Camera_ref_Calibration_Model_v3_dinov3_v4_v3``.

    Args:
        backbone_size: ``'small'``, ``'base'``, or ``'large'``.
        output_size: Dimension of the output pose vector (default 7 = quat + t).
        out_indices: List of intermediate ViT layer indices to extract.
        loss_style: ``'quatloss'``, ``'custom'``, ``'reloc'``, or ``'mse'``.
        inverse_prediction: Also predict the inverse relative pose.
        img_size: Input image size (square, after padding).
        decoder_depth: Number of cross-decoder layers.
        dinov3_repo: Path to a local clone of the DINOv3 repo.
        dinov3_weights: Path to the pre-trained ``.pth`` checkpoint.
    """

    BACKBONE_DIMS = {"small": 384, "base": 768, "large": 1024}
    BACKBONE_ARCHS = {"small": "dinov3_vits16", "base": "dinov3_vitb16", "large": "dinov3_vitl16"}

    def __init__(
        self,
        backbone_size="large",
        output_size=7,
        mode="7scenes",
        out_indices=(23,),
        loss_style="quatloss",
        euler_style="",
        inverse_prediction=True,
        img_size=224,
        decoder_depth=6,
        dinov3_repo=None,
        dinov3_weights=None,
        # passed through to base
        translation_weight=0.01,
        l_r=3e-6,
        angular_error=False,
        # ignored legacy kwargs
        **kwargs,
    ):
        kwargs.pop("pose_encoding_style", None)
        super().__init__(
            output_size=output_size,
            loss_style=loss_style,
            euler_style=euler_style,
            inverse_prediction=inverse_prediction,
            translation_weight=translation_weight,
            l_r=l_r,
            angular_error=angular_error,
        )

        self.patch_size = 16
        self.img_size = img_size
        self.dec_depth = decoder_depth
        self.out_indices = list(out_indices)
        self.embed_dim = self.BACKBONE_DIMS[backbone_size]
        self.mode = mode
        self.reduced_dim = 1024

        # ---- Load DINOv2 backbone ----------------------------------------
        if dinov3_repo is None or dinov3_weights is None:
            raise ValueError(
                "You must supply `dinov3_repo` (path to local DINOv3 repo) "
                "and `dinov3_weights` (path to .pth checkpoint). "
                "See README for details."
            )
        arch = self.BACKBONE_ARCHS[backbone_size]
        self.backbone_model = torch.hub.load(
            dinov3_repo, arch, source="local", weights=dinov3_weights
        )
        for p in self.backbone_model.parameters():
            p.requires_grad = False

        # ---- Cross-decoder -----------------------------------------------
        self.pos_getter = PositionGetter()
        self.dec_embed_dim = 768
        self.dec_num_heads = 12
        norm = partial(nn.LayerNorm, eps=1e-6)

        self.decoder_lin = nn.Linear(self.embed_dim, self.dec_embed_dim, bias=True)
        self.enc_norm = norm(self.dec_embed_dim)
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(self.dec_embed_dim, self.dec_num_heads,
                         mlp_ratio=4.0, qkv_bias=True, norm_layer=norm, norm_mem=True)
            for _ in range(self.dec_depth)
        ])

        # ---- Head --------------------------------------------------------
        self.head_proj = nn.Linear(self.dec_embed_dim, self.reduced_dim)
        self.reduce_convs = nn.ModuleList([
            copy.deepcopy(DecoderResConvBlock(self.reduced_dim, self.reduced_dim))
            for _ in self.out_indices
        ])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pred_heads = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.output_size),
        )

    # ------------------------------------------------------------------
    def extract_features(self, x):
        ph = x.size(2) // self.patch_size
        pw = x.size(3) // self.patch_size
        feats = self.backbone_model.get_intermediate_layers(x, n=self.out_indices)
        reshaped = [
            f.squeeze().reshape(x.size(0), ph, pw, self.embed_dim).permute(0, 3, 1, 2)
            for f in feats
        ]
        return reshaped, ph, pw

    def _decoder(self, x_ref, x_data, ph, pw):
        pos1 = self.pos_getter(x_ref.shape[0], ph, pw)
        pos2 = self.pos_getter(x_data.shape[0], ph, pw)
        x_ref = x_ref.permute(0, 2, 3, 1).reshape(x_ref.shape[0], -1, x_ref.shape[1])
        x_data = x_data.permute(0, 2, 3, 1).reshape(x_data.shape[0], -1, x_data.shape[1])
        f1, f2 = self.decoder_lin(x_ref), self.decoder_lin(x_data)
        stack = [(f1, f2)]
        for blk in self.dec_blocks:
            f1, _ = blk(*stack[-1][::1], pos1, pos2)
            f2, _ = blk(*stack[-1][::-1], pos2, pos1)
            stack.append((f1, f2))
        stack[-1] = tuple(map(self.enc_norm, stack[-1]))
        return zip(*stack)

    def _head_prep(self, dec_out, B, ph, pw):
        feat = self.head_proj(dec_out[-1])
        feat = feat.transpose(-1, -2).view(B, -1, ph, pw)
        for blk in self.reduce_convs:
            feat = blk(feat)
        feat = self.avgpool(feat).view(B, -1)
        return feat

    def forward(self, x_ref, x_data):
        self.backbone_model.eval()
        feats_ref, ph, pw = self.extract_features(x_ref)
        feats_data, _, _ = self.extract_features(x_data)
        B = x_ref.shape[0]
        dec1, dec2 = self._decoder(feats_ref[0], feats_data[0], ph, pw)
        pred = self.pred_heads(self._head_prep(dec1, B, ph, pw))
        if self.inverse_prediction:
            pred_inv = self.pred_heads(self._head_prep(dec2, B, ph, pw))
            pred = torch.cat([pred, pred_inv], dim=1)
        return self.ortho_or_norm(pred)


# Alias used by the old config files
Camera_ref_Calibration_Model_v3_dinov3_v4_v3 = PoseEstimationModel
