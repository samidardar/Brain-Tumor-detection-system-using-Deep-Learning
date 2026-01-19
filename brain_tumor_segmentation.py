"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    BRAIN TUMOR SEGMENTATION PIPELINE                          ║
║                         U-Net Architecture with PyTorch                       ║
║═══════════════════════════════════════════════════════════════════════════════║
║  Author: Medical AI Pipeline                                                  ║
║  Dataset: LGG MRI Segmentation (Kaggle)                                       ║
║  Purpose: Hospital-ready brain tumor segmentation from MRI scans              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Features:
- U-Net architecture with skip connections
- Dice + BCE combined loss for class imbalance handling
- Albumentations data augmentation
- IoU and Dice Score metrics
- Patient-level train/val split to prevent data leakage
- Early stopping and model checkpointing
- GPU acceleration with mixed precision training support

Usage:
    # Google Colab / Local GPU
    python brain_tumor_segmentation.py --data_path /path/to/kaggle_3m
    
    # Or import as module
    from brain_tumor_segmentation import UNet, train_model, get_dataloaders
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import os
import glob
import random
import argparse
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Configuration for training pipeline - optimized for Google Colab Tesla T4."""
    
    # Data paths
    data_path: str = r"C:\Users\PC\Downloads\archive (7)\kaggle_3m"
    output_dir: str = "./outputs"
    
    # Image settings
    image_size: int = 256
    in_channels: int = 3
    out_channels: int = 1
    
    # Training hyperparameters
    batch_size: int = 4  # Reduced for GTX 1650 VRAM conservation
    num_epochs: int = 50
    learning_rate: float = 3e-5  # Lower LR for stability
    weight_decay: float = 1e-5
    
    # Train/Val split
    val_split: float = 0.2
    random_seed: int = 42
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Hardware
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 4
    pin_memory: bool = True
    use_amp: bool = True  # Automatic Mixed Precision
    
    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        # Adjust workers for Windows
        if os.name == 'nt':
            self.num_workers = 0


# ═══════════════════════════════════════════════════════════════════════════════
# DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_training_augmentation(image_size: int = 256) -> A.Compose:
    """
    Get training augmentation pipeline optimized for medical imaging.
    
    Includes:
    - Spatial transforms (rotations, flips, elastic deformation)
    - Intensity transforms (brightness, contrast)
    - All transforms preserve mask-image correspondence
    """
    return A.Compose([
        # Spatial transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=30,
            border_mode=0,
            p=0.5
        ),
        A.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            p=0.3
        ),
        
        # Intensity transforms (applied to image only)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.5),
        
        # Noise for robustness
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
        
        # Ensure correct size
        A.Resize(image_size, image_size),
        
        # Normalize and convert to tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_validation_augmentation(image_size: int = 256) -> A.Compose:
    """Get validation/inference augmentation (no random transforms)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for LGG MRI Segmentation.
    
    Features:
    - Loads .tif images and corresponding _mask.tif files
    - Patient-level splitting to prevent data leakage
    - Albumentations augmentation support
    - Automatic mask binarization
    
    Args:
        data_path: Path to kaggle_3m directory
        patient_ids: List of patient folder names to include
        transform: Albumentations compose object
    """
    
    def __init__(
        self,
        data_path: str,
        patient_ids: List[str],
        transform: Optional[A.Compose] = None
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.image_mask_pairs = []
        
        # Collect all image-mask pairs from specified patients
        for patient_id in patient_ids:
            patient_dir = self.data_path / patient_id
            if not patient_dir.exists():
                continue
                
            # Find all non-mask images
            for img_path in patient_dir.glob("*.tif"):
                if "_mask" in img_path.stem:
                    continue
                    
                # Find corresponding mask
                mask_path = patient_dir / f"{img_path.stem}_mask.tif"
                if mask_path.exists():
                    self.image_mask_pairs.append((str(img_path), str(mask_path)))
        
        print(f"Loaded {len(self.image_mask_pairs)} image-mask pairs from {len(patient_ids)} patients")
    
    def __len__(self) -> int:
        return len(self.image_mask_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.image_mask_pairs[idx]
        
        # Load image (RGB)
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Load mask (grayscale, binarize)
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)  # Binarize
        
        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        # Ensure mask has channel dimension
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)
        
        return image, mask.float()


def get_dataloaders(
    data_path: str,
    batch_size: int = 16,
    val_split: float = 0.2,
    image_size: int = 256,
    num_workers: int = 4,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders with patient-level split.
    
    Args:
        data_path: Path to kaggle_3m directory
        batch_size: Batch size for dataloaders
        val_split: Fraction of patients for validation
        image_size: Target image size
        num_workers: Number of dataloader workers
        random_seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader
    """
    data_path = Path(data_path)
    
    # Get all patient directories
    patient_ids = [
        d.name for d in data_path.iterdir()
        if d.is_dir() and d.name.startswith("TCGA")
    ]
    
    print(f"Found {len(patient_ids)} patients")
    
    # Patient-level split (prevents data leakage between train/val)
    train_patients, val_patients = train_test_split(
        patient_ids,
        test_size=val_split,
        random_state=random_seed
    )
    
    print(f"Train: {len(train_patients)} patients, Val: {len(val_patients)} patients")
    
    # Create datasets
    train_dataset = BrainTumorDataset(
        data_path=str(data_path),
        patient_ids=train_patients,
        transform=get_training_augmentation(image_size)
    )
    
    val_dataset = BrainTumorDataset(
        data_path=str(data_path),
        patient_ids=val_patients,
        transform=get_validation_augmentation(image_size)
    )
    
    # Adjust workers for Windows
    if os.name == 'nt':
        num_workers = 0
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════════════
# U-NET ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    """
    Double convolution block: (Conv2d -> BN -> ReLU) × 2
    
    Standard building block for U-Net encoder and decoder.
    """
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block: MaxPool -> DoubleConv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block: Upsample -> Concat skip connection -> DoubleConv
    
    Includes skip connection from encoder for feature reuse.
    """
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle size mismatch (padding if needed)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution: 1x1 conv to produce final segmentation map."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    
    Architecture:
    - Encoder: 4 downsampling blocks (doubles channels, halves resolution)
    - Bridge: 1024 channels at bottleneck
    - Decoder: 4 upsampling blocks with skip connections
    - Output: 1-channel sigmoid for binary segmentation
    
    Total parameters: ~7.7M
    
    Reference: Ronneberger et al., 2015
    https://arxiv.org/abs/1505.04597
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        out_channels: Number of output channels (1 for binary mask)
        features: Base number of features (doubled at each level)
        bilinear: Use bilinear upsampling (True) or transposed conv (False)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = None,
        bilinear: bool = True
    ):
        super().__init__()
        
        if features is None:
            features = [64, 128, 256, 512]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor)
        
        # Decoder
        self.up1 = Up(features[3] * 2, features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)
        
        # Output
        self.outc = OutConv(features[0], out_channels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path with skip connections
        x1 = self.inc(x)      # 64 channels
        x2 = self.down1(x1)   # 128 channels
        x3 = self.down2(x2)   # 256 channels
        x4 = self.down3(x3)   # 512 channels
        x5 = self.down4(x4)   # 512 channels (bottleneck)
        
        # Decoder path
        x = self.up1(x5, x4)  # Skip from x4
        x = self.up2(x, x3)   # Skip from x3
        x = self.up3(x, x2)   # Skip from x2
        x = self.up4(x, x1)   # Skip from x1
        
        # Output with sigmoid activation
        logits = self.outc(x)
        if self.training:
            return logits  # Return logits during training for BCEWithLogitsLoss
        return torch.sigmoid(logits)  # Return probabilities during inference
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice
    
    Excellent for class imbalance (small tumors vs large background).
    
    Args:
        smooth: Smoothing factor to prevent division by zero
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class DiceBCELoss(nn.Module):
    """
    Combined Dice + Binary Cross-Entropy Loss.
    
    Combines:
    - Dice Loss: Handles class imbalance well
    - BCE Loss: Provides stable gradients
    
    This combination is the gold standard for medical image segmentation.
    
    Args:
        smooth: Smoothing factor for Dice
        dice_weight: Weight for Dice loss (default 0.5)
        bce_weight: Weight for BCE loss (default 0.5)
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5
    ):
        super().__init__()
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Clamp predictions for numerical stability
        pred_clamped = torch.clamp(pred, min=-20, max=20)
        
        # Apply sigmoid to logits for Dice calculation
        pred_sig = torch.sigmoid(pred_clamped)
        
        # Dice Loss with numerical stability
        pred_flat = pred_sig.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        
        # BCE Loss - use BCEWithLogitsLoss which is AMP-safe
        bce_loss = F.binary_cross_entropy_with_logits(pred_clamped, target, reduction='mean')
        
        # Combined loss - clamp to prevent NaN
        loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return torch.clamp(loss, min=0, max=10)


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - alternative loss for highly imbalanced data.
    
    Tversky Index generalizes Dice with different weights for FP and FN.
    Focal variant adds gamma for focusing on hard examples.
    
    Args:
        alpha: Weight for false positives (default 0.7)
        beta: Weight for false negatives (default 0.3)
        gamma: Focal parameter (default 0.75)
        smooth: Smoothing factor
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        smooth: float = 1.0
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        # True positives, false positives, false negatives
        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0) -> float:
    """
    Calculate Dice Score (F1 Score for segmentation).
    
    Dice = 2 * TP / (2 * TP + FP + FN)
    
    Args:
        pred: Predicted probabilities (B, 1, H, W)
        target: Ground truth mask (B, 1, H, W)
        threshold: Threshold to binarize predictions
        smooth: Smoothing factor
    
    Returns:
        Dice score (0-1, higher is better)
    """
    pred = (pred > threshold).float()
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0) -> float:
    """
    Calculate IoU (Intersection over Union / Jaccard Index).
    
    IoU = TP / (TP + FP + FN)
    
    Args:
        pred: Predicted probabilities (B, 1, H, W)
        target: Ground truth mask (B, 1, H, W)
        threshold: Threshold to binarize predictions
        smooth: Smoothing factor
    
    Returns:
        IoU score (0-1, higher is better)
    """
    pred = (pred > threshold).float()
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def calculate_precision_recall(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> Tuple[float, float]:
    """
    Calculate Precision (PPV) and Recall (Sensitivity).
    
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    """
    pred = (pred > threshold).float()
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    TP = (pred * target).sum()
    FP = ((1 - target) * pred).sum()
    FN = (target * (1 - pred)).sum()
    
    precision = (TP + smooth) / (TP + FP + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)
    
    return precision.item(), recall.item()


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Calculate all metrics at once."""
    precision, recall = calculate_precision_recall(pred, target, threshold)
    return {
        'dice': calculate_dice_score(pred, target, threshold),
        'iou': calculate_iou(pred, target, threshold),
        'precision': precision,
        'recall': recall
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EARLY STOPPING
# ═══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors validation loss/metric and stops training if no improvement.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for metrics like Dice
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = True
) -> Tuple[float, float, float, float, float]:
    """
    Train for one epoch.
    
    Returns:
        avg_loss, avg_dice, avg_iou, avg_precision, avg_recall
    """
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Calculate metrics (apply sigmoid since model returns logits during training)
        with torch.no_grad():
            outputs_prob = torch.sigmoid(outputs)
            metrics = calculate_metrics(outputs_prob, masks)
        
        total_loss += loss.item()
        total_dice += metrics['dice']
        total_iou += metrics['iou']
        total_precision += metrics['precision']
        total_recall += metrics['recall']
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{metrics['dice']:.4f}",
            'prec': f"{metrics['precision']:.4f}"
        })
    
    n_batches = len(train_loader)
    return total_loss / n_batches, total_dice / n_batches, total_iou / n_batches, total_precision / n_batches, total_recall / n_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float, float]:
    """
    Validate the model.
    
    Returns:
        avg_loss, avg_dice, avg_iou, avg_precision, avg_recall
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    
    pbar = tqdm(val_loader, desc="Validation", leave=False)
    
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        metrics = calculate_metrics(outputs, masks)
        
        total_loss += loss.item()
        total_dice += metrics['dice']
        total_iou += metrics['iou']
        total_precision += metrics['precision']
        total_recall += metrics['recall']
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{metrics['dice']:.4f}",
            'prec': f"{metrics['precision']:.4f}"
        })
    
    n_batches = len(val_loader)
    return total_loss / n_batches, total_dice / n_batches, total_iou / n_batches, total_precision / n_batches, total_recall / n_batches


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config = None
) -> Dict[str, List[float]]:
    """
    Full training loop with validation, early stopping, and checkpointing.
    
    Args:
        model: U-Net model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Training configuration
    
    Returns:
        Dictionary with training history
    """
    if config is None:
        config = Config()
    
    device = torch.device(config.device)
    model = model.to(device)
    
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Mixed precision: {config.use_amp}")
    print(f"{'='*60}\n")
    
    # Loss function, optimizer, scheduler
    criterion = DiceBCELoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Mixed precision scaler
    scaler = GradScaler() if config.use_amp and device.type == 'cuda' else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience, mode='max')
    
    # Training history
    history = {
        'train_loss': [], 'train_dice': [], 'train_iou': [], 'train_prec': [], 'train_recall': [],
        'val_loss': [], 'val_dice': [], 'val_iou': [], 'val_prec': [], 'val_recall': [],
        'lr': []
    }
    
    best_dice = 0.0
    
    for epoch in range(config.num_epochs):
        torch.cuda.empty_cache()  # Clear VRAM at start of epoch
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 40)
        
        # Training
        train_loss, train_dice, train_iou, train_prec, train_recall = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, config.use_amp
        )
        
        # Validation
        val_loss, val_dice, val_iou, val_prec, val_recall = validate(
            model, val_loader, criterion, device
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['train_iou'].append(train_iou)
        history['train_prec'].append(train_prec)
        history['train_recall'].append(train_recall)
        
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        history['val_prec'].append(val_prec)
        history['val_recall'].append(val_recall)
        
        history['lr'].append(current_lr)
        
        # Print metrics
        print(f"Train - Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IoU: {train_iou:.4f} | Prec: {train_prec:.4f} | Recall: {train_recall:.4f}")
        print(f"Val   - Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | Prec: {val_prec:.4f} | Recall: {val_recall:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Learning rate scheduling (based on validation Dice)
        scheduler.step(val_dice)
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
                'config': config
            }, os.path.join(config.output_dir, 'best_model.pth'))
            print(f"[*] New best model saved! Dice: {val_dice:.4f}")
        
        # Early stopping
        if early_stopping(val_dice):
            print(f"\n[!] Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_dice': val_dice,
        'val_iou': val_iou,
        'history': history,
        'config': config
    }, os.path.join(config.output_dir, 'final_model.pth'))
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"Models saved to: {config.output_dir}")
    
    return history


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """Plot training history curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train', color='blue')
    axes[0].plot(history['val_loss'], label='Validation', color='orange')
    axes[0].set_title('Loss (Dice + BCE)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice Score
    axes[1].plot(history['train_dice'], label='Train', color='blue')
    axes[1].plot(history['val_dice'], label='Validation', color='orange')
    axes[1].set_title('Dice Score')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # IoU
    axes[2].plot(history['train_iou'], label='Train', color='blue')
    axes[2].plot(history['val_iou'], label='Validation', color='orange')
    axes[2].set_title('IoU (Intersection over Union)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()


def visualize_predictions(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_samples: int = 5,
    save_path: Optional[str] = None
):
    """Visualize model predictions vs ground truth."""
    model.eval()
    
    # Get a batch
    images, masks = next(iter(val_loader))
    images = images.to(device)
    
    with torch.no_grad():
        predictions = model(images)
    
    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    images_denorm = images * std + mean
    images_denorm = images_denorm.clamp(0, 1)
    
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        img = images_denorm[i].cpu().permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('MRI Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        gt = masks[i, 0].cpu().numpy()
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        pred = predictions[i, 0].cpu().numpy()
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction (Prob)')
        axes[i, 2].axis('off')
        
        # Overlay
        pred_binary = (pred > 0.5).astype(np.float32)
        overlay = img.copy()
        overlay[pred_binary > 0.5] = [1, 0, 0]  # Red for tumor
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Predictions saved to: {save_path}")
    
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_for_inference(checkpoint_path: str, device: str = 'cuda') -> nn.Module:
    """
    Load trained model for inference.
    
    Args:
        checkpoint_path: Path to saved model (.pth file)
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = UNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Validation Dice: {checkpoint.get('val_dice', 'N/A'):.4f}")
    print(f"Validation IoU: {checkpoint.get('val_iou', 'N/A'):.4f}")
    
    return model


def predict_single_image(
    model: nn.Module,
    image_path: str,
    device: str = 'cuda',
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict on a single image.
    
    Args:
        model: Trained U-Net model
        image_path: Path to .tif image
        device: Device to run inference on
        threshold: Threshold for binary mask
    
    Returns:
        probability_map, binary_mask
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = np.array(Image.open(image_path).convert("RGB"))
    transform = get_validation_augmentation()
    augmented = transform(image=image)
    image_tensor = augmented["image"].unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        prob_map = model(image_tensor)
    
    prob_map = prob_map.squeeze().cpu().numpy()
    binary_mask = (prob_map > threshold).astype(np.uint8)
    
    return prob_map, binary_mask


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Brain Tumor Segmentation with U-Net')
    parser.add_argument('--data_path', type=str, default=r"C:\Users\PC\Downloads\archive (7)\kaggle_3m",
                        help='Path to kaggle_3m dataset folder')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save models and outputs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training (default 4 for GTX 1650)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision training')
    args = parser.parse_args()
    
    # Create configuration
    config = Config(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        use_amp=not args.no_amp
    )
    
    # Print banner
    print("\n" + "=" * 70)
    print("  BRAIN TUMOR SEGMENTATION PIPELINE")
    print("  U-Net Architecture | PyTorch")
    print("  Hospital-Ready Medical AI")
    print("=" * 70 + "\n")
    
    # Set random seeds for reproducibility
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
    
    # Create dataloaders
    print("[1] Loading dataset...")
    train_loader, val_loader = get_dataloaders(
        data_path=config.data_path,
        batch_size=config.batch_size,
        val_split=config.val_split,
        image_size=config.image_size,
        num_workers=config.num_workers,
        random_seed=config.random_seed
    )
    
    # Create model
    print("\n[2] Building U-Net model...")
    model = UNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels
    )
    print(f"   Parameters: {model.count_parameters():,}")
    
    # Train
    print("\n[3] Starting training...")
    history = train_model(model, train_loader, val_loader, config)
    
    # Plot training curves
    print("\n[4] Generating training curves...")
    plot_training_history(history, os.path.join(config.output_dir, 'training_curves.png'))
    
    # Visualize predictions
    print("\n[5] Visualizing predictions...")
    device = torch.device(config.device)
    visualize_predictions(model, val_loader, device, num_samples=5,
                         save_path=os.path.join(config.output_dir, 'predictions.png'))
    
    print("\n[OK] Training complete! Check the outputs folder for results.")


# ═══════════════════════════════════════════════════════════════════════════════
# GOOGLE COLAB QUICK START
# ═══════════════════════════════════════════════════════════════════════════════

"""
# 🚀 GOOGLE COLAB QUICK START

# 1. Install dependencies
!pip install albumentations tqdm

# 2. Mount Google Drive (if dataset is there)
from google.colab import drive
drive.mount('/content/drive')

# 3. Upload this script or clone from GitHub
# Upload brain_tumor_segmentation.py

# 4. Run training
!python brain_tumor_segmentation.py --data_path /content/drive/MyDrive/kaggle_3m --epochs 50

# OR use as module:

from brain_tumor_segmentation import (
    Config, UNet, BrainTumorDataset, 
    get_dataloaders, train_model,
    plot_training_history, visualize_predictions,
    load_model_for_inference, predict_single_image
)

# Create config
config = Config(
    data_path='/content/drive/MyDrive/kaggle_3m',
    batch_size=16,
    num_epochs=50
)

# Get dataloaders
train_loader, val_loader = get_dataloaders(
    config.data_path, 
    batch_size=config.batch_size
)

# Create and train model
model = UNet()
history = train_model(model, train_loader, val_loader, config)

# Visualize results
plot_training_history(history)
visualize_predictions(model, val_loader, torch.device('cuda'))

# Load best model for inference
model = load_model_for_inference('./outputs/best_model.pth')
prob_map, mask = predict_single_image(model, '/path/to/image.tif')
"""


if __name__ == "__main__":
    main()
