# Relative Camera Pose Estimation

DINOv3-based model for pairwise relative camera pose estimation.  
Given two RGB images the network predicts the relative 6-DoF pose (rotation as a unit quaternion + translation vector).

---

## Table of Contents

1. [Installation](#installation)  
2. [DINOv3 Backbone Setup](#dinov3-backbone-setup)  
3. [Dataset Preparation](#dataset-preparation)  
   - [7Scenes](#7scenes)  
   - [Cambridge Landmarks](#cambridge-landmarks)  
4. [Training](#training)  
   - [Training Parameters](#training-parameters)  
   - [7Scenes Example](#train-on-7scenes)  
   - [Cambridge Landmarks Example](#train-on-cambridge-landmarks)  
5. [Inference](#inference)  
6. [Project Structure](#project-structure)  

---

## Installation

### Option A – Conda (recommended)

```bash
conda env create -f environment.yml
conda activate camera_pose
```

### Option B – pip

```bash
pip install -r requirements.txt
```

---

## DINOv3 Backbone Setup

The model uses a **DINOv3** Vision Transformer as a frozen feature extractor.  
You need two things:

1. **A local clone of the DINOv3 repository**  
   ```bash
   git clone https://github.com/facebookresearch/dinov3.git
   cd dinov3 && pip install -e .
   ```

2. **Pre-trained weights** (`.pth` file)  
   Download the checkpoint that matches your chosen backbone size:

   | Backbone | Arch flag | Weights |
   |----------|-----------|---------|
   | ViT-S/16 | `small`   | `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` |
   | ViT-B/16 | `base`    | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` |
   | ViT-L/16 | `large`   | `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` |

   Place the downloaded `.pth` file in a convenient location and pass its path via `--dinov3_weights`.

---

## Dataset Preparation

### 7Scenes

1. **Download the dataset** from the official source:  
   <https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/>

   The dataset consists of 7 indoor scenes:  
   `chess`, `fire`, `heads`, `office`, `pumpkin`, `redkitchen`, `stairs`

2. **Download the train/test split files** from the **RelPoseNet** repository:  
   <https://github.com/AaltoVision/RelPoseNet>

   You need two text files placed in the **root** of the 7Scenes directory:
   - `db_all_med_hard_train.txt`
   - `db_all_med_hard_valid.txt`

3. **Expected directory layout:**

   ```
   /path/to/7scenes/
   ├── chess/
   │   ├── seq-01/
   │   │   ├── frame-000000.color.png
   │   │   └── ...
   │   └── ...
   ├── fire/
   ├── heads/
   ├── office/
   ├── pumpkin/
   ├── redkitchen/
   ├── stairs/
   ├── db_all_med_hard_train.txt
   └── db_all_med_hard_valid.txt
   ```

### Cambridge Landmarks

1. **Download the dataset** from the official source:  
   <https://www.repository.cam.ac.uk/handle/1810/251342>

   The dataset contains 4 outdoor scenes:  
   `KingsCollege`, `OldHospital`, `ShopFacade`, `StMarysChurch`

2. **Download the train/test split files** from the **RPNet** repository:  
   <https://github.com/ensv/RPNet>

   Place the split files under an `rpnet/` subdirectory:
   - `rpnet/<Scene>/train_set.txt`
   - `rpnet/<Scene>/validation_set.txt`

3. **Expected directory layout:**

   ```
   /path/to/cambridge_landmarks/
   ├── KingsCollege/
   │   ├── seq1/
   │   │   ├── frame00001.png
   │   │   └── ...
   │   └── ...
   ├── OldHospital/
   ├── ShopFacade/
   ├── StMarysChurch/
   └── rpnet/
       ├── KingsCollege/
       │   ├── train_set.txt
       │   └── validation_set.txt
       ├── OldHospital/
       │   ├── train_set.txt
       │   └── validation_set.txt
       ├── ShopFacade/
       │   ├── train_set.txt
       │   └── validation_set.txt
       └── StMarysChurch/
           ├── train_set.txt
           └── validation_set.txt
   ```

---

## Training

### Training Parameters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Dataset | `--datatype` | *(required)* | `7scenes` or `cambridge` |
| Data directory | `--data_dir` | *(required)* | Root path to the dataset |
| DINOv3 repo | `--dinov3_repo` | *(required)* | Path to cloned DINOv3 repo |
| DINOv3 weights | `--dinov3_weights` | *(required)* | Path to `.pth` backbone weights |
| Backbone size | `--backbone` | `large` | `small` (ViT-S), `base` (ViT-B), `large` (ViT-L) |
| Output version | `--version` | `v8` | Pose representation: `v8` = quaternion + translation |
| ViT layer indices | `--out_indices` | `"[23]"` | Intermediate ViT layers to extract (use `"[11]"` for `small`/`base`) |
| Decoder depth | `--decoder_depth` | `6` | Number of cross-attention decoder layers |
| Image size | `--img_size` | `224` | Input resolution (images are resized & padded) |
| Predict inverse | `--predict_inverse` | `True` | Also predict the inverse relative pose |
| Loss function | `--loss_style` | `quatloss` | `quatloss`, `custom`, `reloc`, or `mse` |
| Translation weight | `--translation_weight` | `10.0` | Weight of the translation term in the loss |
| Angular trans. error | `--angular_error` | `False` | Use angular error for translation instead of L2 |
| Batch size | `--batch_size` | `8` | Mini-batch size |
| Learning rate | `--learning_rate` | `3e-6` | AdamW learning rate |
| Max epochs | `--max_epochs` | `1000` | Maximum training epochs |
| Log directory | `--log_dir` | `./logs` | Where to save TensorBoard logs & checkpoints |

> **Note on `--out_indices`:** Use `"[11]"` for `small` and `base` backbones (12 ViT layers), and `"[23]"` for the `large` backbone (24 ViT layers).

### Train on 7Scenes

```bash
python train.py \
    --datatype 7scenes \
    --data_dir /path/to/7scenes \
    --dinov3_repo /path/to/dinov3 \
    --dinov3_weights /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --backbone large \
    --out_indices "[23]" \
    --decoder_depth 6 \
    --img_size 224 \
    --batch_size 8 \
    --learning_rate 3e-6 \
    --loss_style quatloss \
    --translation_weight 10 \
    --predict_inverse True \
    --max_epochs 1000 \
    --log_dir ./logs
```

### Train on Cambridge Landmarks

```bash
python train.py \
    --datatype cambridge \
    --data_dir /path/to/cambridge_landmarks \
    --dinov3_repo /path/to/dinov3 \
    --dinov3_weights /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --backbone large \
    --out_indices "[23]" \
    --decoder_depth 6 \
    --img_size 224 \
    --batch_size 8 \
    --learning_rate 3e-6 \
    --loss_style quatloss \
    --translation_weight 10 \
    --predict_inverse True \
    --max_epochs 1000 \
    --log_dir ./logs
```

#### Multiple backbone sizes in parallel

```bash
# ViT-S
python train.py --datatype cambridge --data_dir /data/cambridge \
    --dinov3_repo /path/to/dinov3 --dinov3_weights /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --backbone small --out_indices "[11]" --batch_size 8 &

# ViT-B
python train.py --datatype cambridge --data_dir /data/cambridge \
    --dinov3_repo /path/to/dinov3 --dinov3_weights /path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
    --backbone base --out_indices "[11]" --batch_size 8 &

# ViT-L
python train.py --datatype cambridge --data_dir /data/cambridge \
    --dinov3_repo /path/to/dinov3 --dinov3_weights /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --backbone large --out_indices "[23]" --batch_size 4 &

wait
```

### Monitor training

```bash
tensorboard --logdir ./logs
```

---

## Inference

See **`notebooks/inference.ipynb`** for a complete example with visualization.

Quick usage in Python:

```python
from src.inference_utils import load_model_from_directory, preprocess_pair, postprocess_prediction

model, out_size = load_model_from_directory(
    "logs/7scenes_large_20250101/train/version_0",
    dinov3_repo="/path/to/dinov3",
    dinov3_weights="/path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
)
model = model.cuda()

img1, img2 = preprocess_pair("scene/img1.png", "scene/img2.png", pad_size=224)
with torch.no_grad():
    pred = model(img1.cuda(), img2.cuda())

T_rel = postprocess_prediction(pred, out_size)
print(T_rel)  # 4x4 relative pose matrix
```

---

## Project Structure

```
camera_ready/
├── README.md               # This file
├── requirements.txt        # pip dependencies
├── environment.yml         # Conda environment
├── train.py                # Training entry point (argparse CLI)
│
├── src/                    # Python package – all library code
│   ├── __init__.py
│   ├── pose_model.py       # PoseEstimationModel (DINOv3 + cross-attention decoder)
│   ├── dataloading.py      # Dataset loaders & augmentation (7Scenes, Cambridge)
│   ├── losses.py           # Rotation & translation angular losses
│   ├── pose_utils.py       # Utility functions (RoPE2D, batch transforms, metrics)
│   └── inference_utils.py  # Model loading, preprocessing & post-processing
│
├── notebooks/
│   └── inference.ipynb     # Interactive inference & visualisation notebook
│
└── logs/                   # Created automatically during training
    └── <experiment>/
        └── train/
            └── version_0/
                ├── checkpoints/
                │   └── *.ckpt
                ├── args_v8.txt
                └── hparams.yaml
```

---

## Acknowledgements

- **DINOv3**: Based on Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023.  
- **7Scenes**: Shotton et al., *Scene Coordinate Regression Forests for Camera Relocalization in RGB-D Images*, CVPR 2013.  
- **Cambridge Landmarks**: Kendall et al., *PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization*, ICCV 2015.  
- **RelPoseNet** (7Scenes splits): <https://github.com/AaltoVision/RelPoseNet>  
- **RPNet** (Cambridge splits): <https://github.com/ensv/RPNet>
