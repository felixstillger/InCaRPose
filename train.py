"""
Training script for relative camera pose estimation.

Supports:
  - 7Scenes  (--datatype 7scenes)
  - Cambridge Landmarks  (--datatype cambridge)

Example
-------
python train.py \
    --datatype 7scenes \
    --data_dir /path/to/7scenes \
    --dinov3_repo /path/to/dinov3 \
    --dinov3_weights /path/to/ckpts/dinov3_vitl16_pretrain.pth \
    --backbone large \
    --out_indices "[23]" \
    --decoder_depth 6 \
    --learning_rate 3e-6 \
    --batch_size 8 \
    --img_size 224 \
    --predict_inverse \
    --loss_style quatloss \
    --translation_weight 10 \
    --max_epochs 1000 \
    --log_dir ./logs
"""

import os
import ast
import argparse
import random
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader

from src.dataloading import (
    get_7scene_data,
    get_7scene_val_lists,
    get_cambridge_data,
    PaddedPoseDataset,
)
from src.pose_model import PoseEstimationModel

torch.set_float32_matmul_precision("medium")

# ---------------------------------------------------------------------------
# Version mapping  (output vector size)
# ---------------------------------------------------------------------------
VERSION_MAP = {
    "v0": 3,   # rotation vector
    "v1": 3,   # Euler intrinsic
    "v2": 3,   # Euler extrinsic
    "v3": 4,   # quaternion
    "v4": 9,   # rotation matrix
    "v5": 6,   # rotvec + translation
    "v6": 6,   # Euler intrinsic + t
    "v7": 6,   # Euler extrinsic + t
    "v8": 7,   # quaternion + translation  ← default
    "v9": 12,  # rotation matrix + t
    "v10": 16, # full 4×4
}

NUM_WORKERS = 4


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.strip().lower() in ("yes", "true", "1"):
        return True
    if v.strip().lower() in ("no", "false", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean expected")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    output_size = VERSION_MAP[args.version]
    euler_style = ""
    if args.version in ("v1", "v6"):
        euler_style = "intrinsic"
    elif args.version in ("v2", "v7"):
        euler_style = "extrinsic"

    out_indices = ast.literal_eval(args.out_indices)

    # ---- Data -----------------------------------------------------------
    if args.datatype == "7scenes":
        train_data, _ = get_7scene_data(args.data_dir)
        val_scene_dict = get_7scene_val_lists(args.data_dir)
        scene_names = list(val_scene_dict.keys())

        train_ds = PaddedPoseDataset(
            train_data, pad_size=args.img_size, out_size=output_size,
            euler_style=euler_style, inverse_out=args.predict_inverse)

        val_datasets = {
            s: PaddedPoseDataset(
                val_scene_dict[s], pad_size=args.img_size, out_size=output_size,
                euler_style=euler_style, inverse_out=args.predict_inverse)
            for s in scene_names
        }

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
        val_loaders = [
            DataLoader(val_datasets[s], batch_size=args.batch_size, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
            for s in scene_names
        ]
        dataloader_names = scene_names

    elif args.datatype == "cambridge":
        train_data, val_data = get_cambridge_data(args.data_dir)
        train_ds = PaddedPoseDataset(
            train_data, pad_size=args.img_size, out_size=output_size,
            euler_style=euler_style, inverse_out=args.predict_inverse)
        val_ds = PaddedPoseDataset(
            val_data, pad_size=args.img_size, out_size=output_size,
            euler_style=euler_style, inverse_out=args.predict_inverse)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
        val_loaders = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
        dataloader_names = ["cambridge_val"]
    else:
        raise ValueError(f"Unknown datatype: {args.datatype}")

    # ---- Model ----------------------------------------------------------
    model = PoseEstimationModel(
        backbone_size=args.backbone,
        output_size=output_size,
        mode=args.datatype,
        out_indices=out_indices,
        loss_style=args.loss_style,
        euler_style=euler_style,
        inverse_prediction=args.predict_inverse,
        img_size=args.img_size,
        decoder_depth=args.decoder_depth,
        dinov3_repo=args.dinov3_repo,
        dinov3_weights=args.dinov3_weights,
        translation_weight=args.translation_weight,
        l_r=args.learning_rate,
        angular_error=args.angular_error,
    )
    model.dataloader_names = dataloader_names

    # ---- Logger / callbacks ---------------------------------------------
    log_dir = os.path.join(
        args.log_dir,
        f"{args.datatype}_{args.backbone}_{datetime.now():%Y%m%d_%H%M%S}",
    )
    os.makedirs(log_dir, exist_ok=True)
    logger = TensorBoardLogger(log_dir, name="train")
    model.log_dir = logger.log_dir
    os.makedirs(model.log_dir, exist_ok=True)

    # Save args
    with open(os.path.join(model.log_dir, f"args_{args.version}.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    ckpt_cb = ModelCheckpoint(
        dirpath=model.log_dir,
        filename="{epoch}-{step}",
        save_top_k=1,
        every_n_train_steps=max(1, len(train_loader)),
    )
    bar = TQDMProgressBar(refresh_rate=1)

    # ---- Train ----------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        logger=logger,
        log_every_n_steps=200,
        callbacks=[ckpt_cb, bar],
        val_check_interval=len(train_loader),
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )
    trainer.fit(model, train_loader, val_loaders)
    print("Training finished.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train relative camera pose estimator")

    # Data
    p.add_argument("--datatype", type=str, required=True,
                   choices=["7scenes", "cambridge"],
                   help="Dataset to train on.")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Root directory of the chosen dataset.")

    # Backbone
    p.add_argument("--dinov3_repo", type=str, required=True,
                   help="Path to local DINOv3 repository.")
    p.add_argument("--dinov3_weights", type=str, required=True,
                   help="Path to DINOv3 pre-trained .pth checkpoint.")
    p.add_argument("--backbone", type=str, default="large",
                   choices=["small", "base", "large"],
                   help="Backbone size (small=ViT-S, base=ViT-B, large=ViT-L).")

    # Model
    p.add_argument("--version", type=str, default="v8",
                   help="Output representation version (default: v8 = quat+t).")
    p.add_argument("--out_indices", type=str, default="[23]",
                   help="ViT intermediate layer indices to use, e.g. '[11]' for base.")
    p.add_argument("--decoder_depth", type=int, default=6,
                   help="Number of cross-decoder transformer layers.")
    p.add_argument("--img_size", type=int, default=224,
                   help="Input image size (images are resized & padded to this).")
    p.add_argument("--predict_inverse", type=str2bool, default=True,
                   help="Also predict inverse relative pose.")

    # Loss
    p.add_argument("--loss_style", type=str, default="quatloss",
                   choices=["quatloss", "custom", "reloc", "mse"],
                   help="Loss function.")
    p.add_argument("--translation_weight", type=float, default=10.0,
                   help="Weight for translation component in the loss.")
    p.add_argument("--angular_error", type=str2bool, default=False,
                   help="Use angular error for translation (instead of L2).")

    # Training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=3e-6)
    p.add_argument("--max_epochs", type=int, default=1000)
    p.add_argument("--log_dir", type=str, default="./logs",
                   help="Root directory for TensorBoard logs and checkpoints.")

    main(p.parse_args())
