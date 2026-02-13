"""
PyTorch DataLoader for Diabetic Retinopathy Detection
=====================================================
Custom Dataset class with Albumentations augmentation pipeline,
stratified train/val/test splitting, and weighted sampling for
class imbalance handling.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RetinopathyDataset(Dataset):
    """
    PyTorch Dataset for fundus images with diabetic retinopathy labels.

    Supports both multi-class (5 grades: 0-4) and binary (No DR / DR Present)
    classification modes.

    Args:
        dataframe: DataFrame with 'id_code' and 'diagnosis' columns.
        image_dir: Directory containing the images.
        transform: Albumentations transform pipeline.
        image_format: Image file extension (e.g., 'png').
        binary_mode: If True, convert labels to binary (0 vs 1-4).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str,
        transform: Optional[A.Compose] = None,
        image_format: str = "png",
        binary_mode: bool = False,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_format = image_format
        self.binary_mode = binary_mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_id = row["id_code"]
        label = int(row["diagnosis"])

        # Convert to binary if needed
        if self.binary_mode:
            label = 0 if label == 0 else 1

        # Load image (BGR → RGB)
        img_path = self.image_dir / f"{image_id}.{self.image_format}"
        image = cv2.imread(str(img_path))

        if image is None:
            logger.error(f"Could not load image: {img_path}")
            # Return a blank image as fallback
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

    def get_labels(self) -> List[int]:
        """Return all labels as a list (for weighted sampler)."""
        labels = self.df["diagnosis"].tolist()
        if self.binary_mode:
            labels = [0 if l == 0 else 1 for l in labels]
        return labels


def get_train_transforms(config: Dict[str, Any]) -> A.Compose:
    """
    Build the Albumentations augmentation pipeline for training.

    Applies medical-image-appropriate augmentations including geometric
    transforms, color jitter, elastic deformation, and coarse dropout.

    Args:
        config: Full configuration dictionary.

    Returns:
        Albumentations Compose pipeline.
    """
    aug_cfg = config.get("augmentation", {})
    img_size = config.get("data", {}).get("image_size", 512)
    p = aug_cfg.get("transform_probability", 0.5)

    transforms_list = [
        # Resize to target
        A.Resize(img_size, img_size),
        # Geometric
        A.HorizontalFlip(p=0.5 if aug_cfg.get("horizontal_flip", True) else 0),
        A.VerticalFlip(p=0.5 if aug_cfg.get("vertical_flip", True) else 0),
        A.ShiftScaleRotate(
            shift_limit=aug_cfg.get("shift_limit", 0.1),
            scale_limit=aug_cfg.get("scale_limit", 0.15),
            rotate_limit=aug_cfg.get("rotation_limit", 30),
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=p,
        ),
        # Color / intensity
        A.RandomBrightnessContrast(
            brightness_limit=aug_cfg.get("brightness_limit", 0.2),
            contrast_limit=aug_cfg.get("contrast_limit", 0.2),
            p=p,
        ),
        A.HueSaturationValue(
            hue_shift_limit=aug_cfg.get("hue_shift_limit", 10),
            sat_shift_limit=aug_cfg.get("saturation_limit", 20),
            val_shift_limit=20,
            p=p * 0.5,
        ),
    ]

    # Elastic transform (medical imaging standard)
    if aug_cfg.get("elastic_transform", True):
        transforms_list.append(
            A.ElasticTransform(
                alpha=aug_cfg.get("elastic_alpha", 120),
                sigma=aug_cfg.get("elastic_sigma", 6.0),
                p=p * 0.3,
            )
        )

    # Grid distortion
    if aug_cfg.get("grid_distortion", True):
        transforms_list.append(A.GridDistortion(p=p * 0.3))

    # Optical distortion
    if aug_cfg.get("optical_distortion", True):
        transforms_list.append(A.OpticalDistortion(p=p * 0.3))

    # Coarse dropout (cutout)
    if aug_cfg.get("coarse_dropout", True):
        transforms_list.append(
            A.CoarseDropout(
                max_holes=aug_cfg.get("dropout_max_holes", 8),
                max_height=aug_cfg.get("dropout_max_height", 32),
                max_width=aug_cfg.get("dropout_max_width", 32),
                fill_value=0,
                p=p * 0.5,
            )
        )

    # Normalize + to tensor (always applied)
    transforms_list.extend([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    return A.Compose(transforms_list)


def get_val_transforms(config: Dict[str, Any]) -> A.Compose:
    """
    Build the transform pipeline for validation/test sets.
    Only resize and normalize — no augmentation.

    Args:
        config: Full configuration dictionary.

    Returns:
        Albumentations Compose pipeline.
    """
    img_size = config.get("data", {}).get("image_size", 512)

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for loss function balancing.

    Args:
        labels: List of integer labels.
        num_classes: Total number of classes.

    Returns:
        Tensor of class weights normalized so that the minimum weight is 1.0.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)  # Avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.min()  # Normalize: least frequent = highest weight
    logger.info(f"Class weights: {dict(enumerate(weights.tolist()))}")
    return torch.FloatTensor(weights)


def create_weighted_sampler(labels: List[int], num_classes: int) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler for handling class imbalance.

    Each sample gets a weight inversely proportional to its class frequency,
    so minority classes are sampled more often.

    Args:
        labels: List of integer labels for the training set.
        num_classes: Total number of classes.

    Returns:
        WeightedRandomSampler instance.
    """
    class_weights = compute_class_weights(labels, num_classes)
    sample_weights = [class_weights[label].item() for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def create_dataloaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create train, validation, and test DataLoaders with stratified splitting.

    Handles:
        - Stratified train/val/test split
        - Data augmentation (train only)
        - Class imbalance via weighted sampling (train only)
        - Binary mode conversion

    Args:
        config: Full configuration dictionary.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, metadata_dict).
        metadata_dict contains class_weights, class_counts, num_classes, etc.
    """
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    balance_cfg = config.get("class_balance", {})

    # Load labels CSV
    labels_path = data_cfg.get("labels_csv", "data/raw/train.csv")
    df = pd.read_csv(labels_path)
    logger.info(f"Loaded {len(df)} samples from {labels_path}")

    # Determine image directory (processed if available, else raw)
    processed_dir = data_cfg.get("processed_dir", "data/processed")
    raw_dir = data_cfg.get("raw_dir", "data/raw")
    if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
        image_dir = processed_dir
        logger.info(f"Using preprocessed images from {processed_dir}")
    else:
        image_dir = os.path.join(raw_dir, "train_images")
        logger.info(f"Using raw images from {image_dir}")

    binary_mode = data_cfg.get("binary_mode", False)
    num_classes = 2 if binary_mode else len(data_cfg.get("class_names", range(5)))

    # Stratified split: train / (val + test)
    train_ratio = data_cfg.get("train_ratio", 0.70)
    val_ratio = data_cfg.get("val_ratio", 0.15)
    test_ratio = data_cfg.get("test_ratio", 0.15)

    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio),
        stratify=df["diagnosis"], random_state=config.get("project", {}).get("seed", 42),
    )
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_ratio,
        stratify=temp_df["diagnosis"],
        random_state=config.get("project", {}).get("seed", 42),
    )

    logger.info(
        f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    # Log class distribution
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split_df["diagnosis"].value_counts().sort_index().to_dict()
        logger.info(f"  {split_name} distribution: {dist}")

    # Build transforms
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)

    # Create datasets
    image_format = data_cfg.get("image_format", "png")
    train_dataset = RetinopathyDataset(
        train_df, image_dir, train_transform, image_format, binary_mode
    )
    val_dataset = RetinopathyDataset(
        val_df, image_dir, val_transform, image_format, binary_mode
    )
    test_dataset = RetinopathyDataset(
        test_df, image_dir, val_transform, image_format, binary_mode
    )

    # Handle class imbalance for training
    batch_size = train_cfg.get("batch_size", 16)
    num_workers = data_cfg.get("num_workers", 4)
    pin_memory = data_cfg.get("pin_memory", True)

    sampler = None
    shuffle_train = True
    strategy = balance_cfg.get("strategy", "weighted_sampling")

    if strategy == "weighted_sampling":
        train_labels = train_dataset.get_labels()
        sampler = create_weighted_sampler(train_labels, num_classes)
        shuffle_train = False  # Sampler and shuffle are mutually exclusive
        logger.info("Using weighted random sampling for class imbalance")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Compute class weights for loss function
    all_train_labels = train_dataset.get_labels()
    class_weights = compute_class_weights(all_train_labels, num_classes)
    class_counts = np.bincount(all_train_labels, minlength=num_classes)

    metadata = {
        "num_classes": num_classes,
        "class_weights": class_weights,
        "class_counts": class_counts.tolist(),
        "binary_mode": binary_mode,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }

    return train_loader, val_loader, test_loader, metadata
