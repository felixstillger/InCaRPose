"""
Data loading utilities for camera pose estimation.
Supports 7Scenes and Cambridge Landmarks datasets.
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

def make_transform_center(resize: int) -> transforms.Compose:
    """Center-crop transform with ImageNet normalization and color jitter."""
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(resize),
        transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4),
                               saturation=(0.6, 1.4), hue=0.0),
        transforms.ToTensor(),
        lambda x: x[:3],  # discard alpha channel if present
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


def resize_and_pad(image: Image.Image, target_size: int) -> Image.Image:
    """Resize *image* so the longer edge equals *target_size*, then zero-pad."""
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = F.resize(image, [new_h, new_w])

    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    pad_right = target_size - new_w - pad_left
    pad_bottom = target_size - new_h - pad_top
    image = F.pad(image, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
    return image


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------

def relpose_to_matrix(vals, quat_order="wxyz"):
    """Convert [tx, ty, tz, qw, qx, qy, qz] to a 4×4 homogeneous matrix.

    Args:
        vals: 7 floats – translation followed by quaternion.
        quat_order: ``'wxyz'`` (scalar-first) or ``'xyzw'`` (scalar-last).

    Returns:
        T: (4, 4) numpy array.
    """
    vals = list(vals)
    if len(vals) != 7:
        raise ValueError(f"Expected 7 floats, got {len(vals)}")
    tx, ty, tz = vals[:3]
    q = vals[3:]
    if quat_order == "wxyz":
        q = [q[1], q[2], q[3], q[0]]  # SciPy expects scalar-last
    rot = R.from_quat(q)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = [tx, ty, tz]
    return T


def relpose_from_absolute(p1, q1, p2, q2):
    """Compute relative pose T_{1→2} from two absolute poses.

    Uses the same convention as the 7Scenes / Cambridge data loaders.
    """
    R1 = R.from_quat(q1, scalar_first=True).as_matrix()
    R2 = R.from_quat(q2, scalar_first=True).as_matrix()
    t1 = np.array(p1).reshape(3, 1)
    t2 = np.array(p2).reshape(3, 1)

    def _pose(rot, trans):
        T_cw = np.eye(4)
        T_cw[:3, :3] = rot
        T_cw[:3, 3] = (-rot @ trans).flatten()
        return np.linalg.inv(T_cw)

    pose1 = _pose(R1, t1)
    pose2 = _pose(R2, t2)
    return np.linalg.inv(pose1) @ pose2


def compute_pose_2_to_1(pose1, pose2):
    """Relative transform that maps coordinates from pose2 into pose1."""
    return np.linalg.inv(pose1) @ pose2


# ---------------------------------------------------------------------------
# 7Scenes data loader
# ---------------------------------------------------------------------------

def _parse_7scenes_txt(txt_path, img_dir, per_scene=False):
    """Parse a 7Scenes train / validation split file.

    Returns either a flat list or a dict keyed by scene name.
    """
    scene_names = ["chess", "fire", "heads", "office",
                   "pumpkin", "redkitchen", "stairs"]
    if per_scene:
        data = {s: [] for s in scene_names}
    else:
        data = []

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 10:
                continue
            img1_rel, img2_rel = parts[0], parts[1]
            scene = scene_names[int(parts[2])]
            rel_pose = relpose_to_matrix(list(map(float, parts[3:])))
            img1 = os.path.join(img_dir, scene, img1_rel.lstrip("/"))
            img2 = os.path.join(img_dir, scene, img2_rel.lstrip("/"))
            entry = (img1, img2, rel_pose, None)
            if per_scene:
                data[scene].append(entry)
            else:
                data.append(entry)
    return data


def get_7scene_data(img_dir):
    """Return ``(train_data, val_data)`` for 7Scenes."""
    train_txt = os.path.join(img_dir, "db_all_med_hard_train.txt")
    val_txt = os.path.join(img_dir, "db_all_med_hard_valid.txt")
    train_data = _parse_7scenes_txt(train_txt, img_dir)
    val_data = _parse_7scenes_txt(val_txt, img_dir)
    return train_data, val_data


def get_7scene_val_lists(img_dir):
    """Return a dict ``{scene_name: [(img1, img2, rel_pose, None), ...]}``."""
    val_txt = os.path.join(img_dir, "db_all_med_hard_valid.txt")
    return _parse_7scenes_txt(val_txt, img_dir, per_scene=True)


# ---------------------------------------------------------------------------
# Cambridge Landmarks data loader
# ---------------------------------------------------------------------------

def _parse_cambridge_file(txt_path, img_root):
    """Parse a Cambridge Landmarks RPNet-style split file."""
    pairs = []
    with open(txt_path, "r") as f:
        lines = f.read().strip().splitlines()
    lines = lines[2:]  # skip header
    for line in lines:
        parts = line.split()
        if len(parts) != 16:
            continue
        img1_rel = "/".join(parts[0].split("/")[-2:])
        img2_rel = "/".join(parts[8].split("/")[-2:])
        img1 = os.path.join(img_root, img1_rel)
        img2 = os.path.join(img_root, img2_rel)
        p1 = list(map(float, parts[1:4]))
        q1 = list(map(float, parts[4:8]))
        p2 = list(map(float, parts[9:12]))
        q2 = list(map(float, parts[12:16]))
        rel_pose = relpose_from_absolute(p1, q1, p2, q2)
        pairs.append((img1, img2, rel_pose, None))
    return pairs


def get_cambridge_data(
    base_dir,
    scene_list=("KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch"),
):
    """Return ``(train_data, val_data)`` for Cambridge Landmarks."""
    all_train, all_val = [], []
    for scene in scene_list:
        scene_img_root = os.path.join(base_dir, scene)
        train_txt = os.path.join(base_dir, "rpnet", scene, "train_set.txt")
        val_txt = os.path.join(base_dir, "rpnet", scene, "validation_set.txt")
        all_train.extend(_parse_cambridge_file(train_txt, scene_img_root))
        all_val.extend(_parse_cambridge_file(val_txt, scene_img_root))
    return all_train, all_val


# ---------------------------------------------------------------------------
# Torch Datasets
# ---------------------------------------------------------------------------

class PaddedPoseDataset(Dataset):
    """Image-pair dataset with resize-and-pad preprocessing.

    Each sample returns ``(img_ref, img_query, label_tensor)``.
    """

    def __init__(self, data, pad_size=224, out_size=7,
                 euler_style="", inverse_out=False, grayscale=False):
        self.data = data
        self.pad_size = pad_size
        self.out_size = out_size
        self.euler_style = euler_style
        self.inverse_out = inverse_out
        self.grayscale = grayscale

        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4),
                                   saturation=(0.6, 1.4), hue=0.0),
            transforms.Lambda(lambda img: resize_and_pad(img, self.pad_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0),
                std=(58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0),
            ),
        ])

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _to_rgb(img):
        if img.mode in ("I;16", "I"):
            img = img.point(lambda p: p * (255.0 / 65535.0))
        return img.convert("RGB")

    # ------------------------------------------------------------------
    def _process_label(self, label):
        """Convert a 4×4 transform into the requested output representation."""
        # Orthogonalise the rotation via SVD
        U, _, Vt = np.linalg.svd(label[:3, :3])
        label = label.copy()
        label[:3, :3] = U @ Vt

        out = self._label_to_vector(label)
        if self.inverse_out:
            inv_label = np.linalg.inv(label)
            out = np.concatenate([out, self._label_to_vector(inv_label)])
        return out.flatten().astype(np.float32)

    def _label_to_vector(self, label):
        rot = label[:3, :3]
        if self.out_size in (3, 6):
            if self.euler_style == "intrinsic":
                vec = R.from_matrix(rot).as_euler("XYZ")
            elif self.euler_style == "extrinsic":
                vec = R.from_matrix(rot).as_euler("xyz")
            else:
                vec = R.from_matrix(rot).as_rotvec()
            if self.out_size == 6:
                vec = np.concatenate([vec, label[:3, 3]])
        elif self.out_size in (4, 7):
            vec = R.from_matrix(rot).as_quat()
            if vec[3] < 0:
                vec = -vec
            if self.out_size == 7:
                vec = np.concatenate([vec, label[:3, 3]])
        elif self.out_size == 9:
            vec = rot.flatten()
        elif self.out_size == 12:
            vec = np.concatenate([rot.flatten(), label[:3, 3]])
        elif self.out_size == 16:
            vec = label.flatten()
        else:
            raise ValueError(f"Unsupported out_size={self.out_size}")
        return vec.flatten()

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        img_path_ref, img_path_data, label, optional = self.data[idx]

        image_ref = self._to_rgb(Image.open(img_path_ref))
        image_data = self._to_rgb(Image.open(img_path_data))

        if self.grayscale:
            image_ref = image_ref.convert("L").convert("RGB")
            image_data = image_data.convert("L").convert("RGB")

        # Handle optional rotation (ARKitScenes orientation)
        if optional and isinstance(optional, dict) and "rotate" in optional:
            deg = optional["rotate"]
            rot_map = {"90": Image.Transpose.ROTATE_90,
                       "180": Image.Transpose.ROTATE_180,
                       "270": Image.Transpose.ROTATE_270}
            if str(deg) != "0" and str(deg) in rot_map:
                image_ref = image_ref.transpose(rot_map[str(deg)])
                image_data = image_data.transpose(rot_map[str(deg)])

        t_ref = self.transform(image_ref)
        t_data = self.transform(image_data)
        out = self._process_label(label)
        return t_ref, t_data, torch.tensor(out, dtype=torch.float32)
