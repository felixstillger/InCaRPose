"""
Inference utilities – load a trained checkpoint and run prediction on image pairs.
"""

import os
import re
import ast
import glob

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Mappings (must match train.py)
# ---------------------------------------------------------------------------

VERSION_MAP = {
    "v0": 3, "v1": 3, "v2": 3, "v3": 4, "v4": 9,
    "v5": 6, "v6": 6, "v7": 6, "v8": 7, "v9": 12, "v10": 16,
}


# ---------------------------------------------------------------------------
# Image preprocessing (must match training exactly)
# ---------------------------------------------------------------------------

def resize_and_pad(image: Image.Image, target_size: int) -> Image.Image:
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = TF.resize(image, [new_h, new_w])
    pl = (target_size - new_w) // 2
    pt = (target_size - new_h) // 2
    pr = target_size - new_w - pl
    pb = target_size - new_h - pt
    return TF.pad(image, [pl, pt, pr, pb], fill=0)


def make_inference_transform(pad_size: int = 224) -> transforms.Compose:
    """Transform used at inference (no color jitter)."""
    return transforms.Compose([
        transforms.Lambda(lambda img: resize_and_pad(img, pad_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0),
            std=(58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0),
        ),
    ])


def preprocess_pair(img_path1: str, img_path2: str, pad_size: int = 224):
    """Load and preprocess an image pair for inference.

    Returns:
        (img1_tensor, img2_tensor)  – each of shape ``(1, 3, H, W)``.
    """
    transform = make_inference_transform(pad_size)
    img1 = Image.open(img_path1).convert("RGB")
    img2 = Image.open(img_path2).convert("RGB")
    return transform(img1).unsqueeze(0), transform(img2).unsqueeze(0)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_from_directory(directory: str, dinov3_repo: str, dinov3_weights: str):
    """Load a trained model from a log directory containing ``args_vXX.txt`` and a ``.ckpt``.

    Args:
        directory: Path to the log folder produced by ``train.py``.
        dinov3_repo: Path to the local DINOv3 repository.
        dinov3_weights: Path to the DINOv3 ``.pth`` backbone weights.

    Returns:
        (model, output_size)
    """
    from .pose_model import PoseEstimationModel

    # Find args file
    args_file = None
    for f in os.listdir(directory):
        if re.match(r"args_v\d+\.txt", f):
            args_file = os.path.join(directory, f)
            break
    if args_file is None:
        raise FileNotFoundError("No args_vXX.txt found in " + directory)

    settings = {}
    with open(args_file) as f:
        for line in f:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                settings[k.strip()] = v.strip()

    version = settings["version"]
    output_size = VERSION_MAP[version]
    backbone_size = settings.get("backbone", settings.get("backbone_size", "large"))
    out_indices = ast.literal_eval(settings.get("out_indices",
                                                settings.get("intermediate_layers", "[23]")))
    decoder_depth = int(settings.get("decoder_depth", "6"))
    datatype = settings.get("datatype", "7scenes")
    img_size = int(settings.get("img_size", settings.get("crop_size", "224")))

    # Find checkpoint
    ckpts = sorted(glob.glob(os.path.join(directory, "*.ckpt")))
    if not ckpts:
        raise FileNotFoundError("No .ckpt file in " + directory)
    ckpt_path = ckpts[-1]
    print(f"Loading checkpoint: {ckpt_path}")

    model = PoseEstimationModel.load_from_checkpoint(
        ckpt_path,
        backbone_size=backbone_size,
        output_size=output_size,
        mode=datatype,
        out_indices=out_indices,
        loss_style="quatloss",
        inverse_prediction=False,
        img_size=img_size,
        decoder_depth=decoder_depth,
        dinov3_repo=dinov3_repo,
        dinov3_weights=dinov3_weights,
        strict=True,
    )
    model.eval()
    return model, output_size


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def postprocess_prediction(pred: torch.Tensor, out_size: int, euler_style: str = "") -> np.ndarray:
    """Convert a raw network output to a 4×4 transformation matrix.

    Args:
        pred: Tensor of shape ``(1, out_size)`` or ``(out_size,)``.
        out_size: Output vector length (3, 6, 7, 9, 12, or 16).
        euler_style: ``''`` (rotvec), ``'intrinsic'``, or ``'extrinsic'``.

    Returns:
        T: (4, 4) numpy array.
    """
    p = pred.detach().cpu().numpy().flatten()
    T = np.eye(4)

    if out_size == 3:
        T[:3, :3] = R.from_rotvec(p[:3]).as_matrix()
    elif out_size == 6:
        if euler_style == "intrinsic":
            T[:3, :3] = R.from_euler("XYZ", p[:3]).as_matrix()
        elif euler_style == "extrinsic":
            T[:3, :3] = R.from_euler("xyz", p[:3]).as_matrix()
        else:
            T[:3, :3] = R.from_rotvec(p[:3]).as_matrix()
        T[:3, 3] = p[3:6]
    elif out_size == 7:
        T[:3, :3] = R.from_quat(p[:4]).as_matrix()
        T[:3, 3] = p[4:7]
    elif out_size == 9:
        T[:3, :3] = p[:9].reshape(3, 3)
    elif out_size == 12:
        T[:3, :3] = p[:9].reshape(3, 3)
        T[:3, 3] = p[9:12]
    elif out_size == 16:
        T = p[:16].reshape(4, 4)
    else:
        raise ValueError(f"Unsupported out_size={out_size}")
    return T
