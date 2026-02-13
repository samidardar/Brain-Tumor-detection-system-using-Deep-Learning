"""
Diabetic Retinopathy Detection - GPU Training Script
=====================================================
Optimized for GTX 1650 (4GB VRAM).
Trains EfficientNet-B0 with progressive unfreezing on APTOS 2019 dataset.
"""

import os
import sys
import time
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================================================
# CONFIGURATION - Optimized for GTX 1650 (4GB VRAM)
# ============================================================================
DATASET_DIR = r"C:\Users\PC\Downloads\archive (8)"
TRAIN_CSV = os.path.join(DATASET_DIR, "train_1.csv")
VALID_CSV = os.path.join(DATASET_DIR, "valid.csv")
TRAIN_IMAGES = os.path.join(DATASET_DIR, "train_images", "train_images")
VAL_IMAGES = os.path.join(DATASET_DIR, "val_images", "val_images")

PROJECT_DIR = r"c:\Users\PC\Downloads\New folder\diabetic-retinopathy-detection"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

# Hyperparameters (GTX 1650 optimized)
IMAGE_SIZE = 380               # Balance quality vs VRAM
BATCH_SIZE = 8                 # Small for 4GB VRAM
ACCUMULATION_STEPS = 4         # Effective batch = 32
NUM_WORKERS = 2
ARCHITECTURE = "efficientnet_b0"  # Lighter model for 4GB
NUM_CLASSES = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 30
PATIENCE = 7
UNFREEZE_AFTER = 3
DROPOUT = 0.4
HIDDEN_DIM = 256
SEED = 42

# ImageNet stats
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# ============================================================================
# Setup
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(PROJECT_DIR, "training.log")),
    ],
)
logger = logging.getLogger(__name__)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Dataset
# ============================================================================
class RetinopathyDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id_code"]
        label = int(row["diagnosis"])

        img_path = os.path.join(self.image_dir, f"{img_id}.png")
        image = cv2.imread(img_path)
        if image is None:
            # Try jpg
            img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
            image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label

    def get_labels(self):
        return self.df["diagnosis"].tolist()


# ============================================================================
# Augmentations
# ============================================================================
def get_train_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=6.0, p=1.0),
            A.GridDistortion(p=1.0),
            A.OpticalDistortion(p=1.0),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=IMAGE_SIZE//16, max_width=IMAGE_SIZE//16, fill_value=0, p=0.3),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


# ============================================================================
# Model
# ============================================================================
class RetinopathyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            ARCHITECTURE, pretrained=True, num_classes=0, global_pool="avg"
        )
        with torch.no_grad():
            feat_dim = self.backbone(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)).shape[-1]

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )
        self._frozen = False

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._frozen = True
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Backbone frozen. Trainable params: {t:,}")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._frozen = False
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Backbone unfrozen. Trainable params: {t:,}")


# ============================================================================
# Focal Loss
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        p = torch.softmax(logits, dim=1)
        targets_oh = torch.nn.functional.one_hot(targets, NUM_CLASSES).float()
        p_t = (p * targets_oh).sum(dim=1).clamp(min=1e-8)
        focal_weight = (1 - p_t) ** self.gamma
        ce = -torch.log(p_t)
        if self.alpha is not None:
            alpha_t = self.alpha.to(targets.device)[targets]
            loss = alpha_t * focal_weight * ce
        else:
            loss = focal_weight * ce
        return loss.mean()


# ============================================================================
# Training
# ============================================================================
def train():
    logger.info("=" * 60)
    logger.info("DIABETIC RETINOPATHY DETECTION - TRAINING")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"VRAM: {vram:.1f} GB")
    logger.info(f"Architecture: {ARCHITECTURE}")
    logger.info(f"Image size: {IMAGE_SIZE}")
    logger.info(f"Batch size: {BATCH_SIZE} x {ACCUMULATION_STEPS} = {BATCH_SIZE * ACCUMULATION_STEPS}")
    logger.info("=" * 60)

    # Load data
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VALID_CSV)
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")
    logger.info(f"Train distribution:\n{train_df['diagnosis'].value_counts().sort_index()}")
    logger.info(f"Val distribution:\n{val_df['diagnosis'].value_counts().sort_index()}")

    # Datasets
    train_dataset = RetinopathyDataset(train_df, TRAIN_IMAGES, get_train_transforms())
    val_dataset = RetinopathyDataset(val_df, VAL_IMAGES, get_val_transforms())

    # Weighted sampler for class imbalance
    labels = train_dataset.get_labels()
    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.min()
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    logger.info(f"Class weights: {dict(enumerate(class_weights.tolist()))}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # Model
    model = RetinopathyModel().to(device)
    model.freeze_backbone()

    # Loss with class weights
    alpha = torch.FloatTensor(class_weights).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    # Optimizer (only trainable params)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )
    scaler = GradScaler()

    # Training loop
    best_recall = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_recall": [],
               "val_precision": [], "val_f1": [], "val_auc": [], "lr": []}

    for epoch in range(MAX_EPOCHS):
        # Unfreeze backbone after N epochs
        if epoch == UNFREEZE_AFTER and model._frozen:
            logger.info(f"\n*** Epoch {epoch+1}: Unfreezing backbone ***")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LEARNING_RATE * 0.1, weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6,
            )

        # --- Train ---
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUMULATION_STEPS
            pbar.set_postfix({"loss": f"{running_loss / (batch_idx + 1):.4f}"})

        train_loss = running_loss / len(train_loader)
        scheduler.step()

        # --- Validate ---
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        val_loss = 0.0

        with torch.no_grad():
            for images, labels_batch in tqdm(val_loader, desc="Validating", leave=False):
                images = images.to(device, non_blocking=True)
                labels_batch = labels_batch.to(device, non_blocking=True)

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels_batch)

                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_loss /= len(val_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        except:
            auc = 0.0

        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(acc)
        history["val_recall"].append(rec)
        history["val_precision"].append(prec)
        history["val_f1"].append(f1)
        history["val_auc"].append(auc)
        history["lr"].append(lr)

        logger.info(
            f"Epoch {epoch+1}/{MAX_EPOCHS} — "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Acc: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | "
            f"AUC: {auc:.4f} | LR: {lr:.6f}"
        )

        # Per-class recall
        per_class_rec = recall_score(all_labels, all_preds, average=None, zero_division=0)
        for i, r in enumerate(per_class_rec):
            logger.info(f"  {CLASS_NAMES[i]:20s} recall: {r:.4f}")

        # Save best model
        if rec > best_recall:
            best_recall = rec
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": {"accuracy": acc, "recall": rec, "precision": prec, "f1": f1, "auc_roc": auc},
                "config": {
                    "architecture": ARCHITECTURE, "image_size": IMAGE_SIZE,
                    "num_classes": NUM_CLASSES, "dropout": DROPOUT, "hidden_dim": HIDDEN_DIM,
                },
                "architecture": ARCHITECTURE,
                "num_classes": NUM_CLASSES,
            }
            torch.save(checkpoint, os.path.join(MODEL_DIR, "best_model.pth"))
            logger.info(f"  ★ New best recall: {best_recall:.4f} — model saved!")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE and epoch >= 10:
                logger.info(f"Early stopping at epoch {epoch+1} (patience={PATIENCE})")
                break

        # Free VRAM
        torch.cuda.empty_cache()

    # ========================================================================
    # EVALUATION
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)

    # Load best model
    best_ckpt = torch.load(os.path.join(MODEL_DIR, "best_model.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Evaluate on val set with best model
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels_batch in val_loader:
            images = images.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            with autocast():
                outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Classification report
    logger.info("\n" + classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Final metrics
    final_acc = accuracy_score(all_labels, all_preds)
    final_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    final_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    final_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    try:
        final_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except:
        final_auc = 0.0

    logger.info(f"Accuracy:  {final_acc:.4f}")
    logger.info(f"Recall:    {final_rec:.4f}  {'✅' if final_rec >= 0.95 else '⚠️'} (target ≥ 0.95)")
    logger.info(f"Precision: {final_prec:.4f}")
    logger.info(f"F1-Score:  {final_f1:.4f}")
    logger.info(f"AUC-ROC:   {final_auc:.4f}  {'✅' if final_auc >= 0.90 else '⚠️'} (target ≥ 0.90)")

    # ========================================================================
    # PLOTS
    # ========================================================================
    eval_dir = os.path.join(OUTPUT_DIR, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # 1. Training history
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs_range = range(1, len(history["train_loss"]) + 1)

    axes[0, 0].plot(epochs_range, history["train_loss"], "b-", label="Train")
    axes[0, 0].plot(epochs_range, history["val_loss"], "r-", label="Val")
    axes[0, 0].set_title("Loss", fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs_range, history["val_acc"], "g-", label="Accuracy")
    axes[0, 1].set_title("Accuracy", fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs_range, history["val_recall"], "r-", label="Recall")
    axes[1, 0].plot(epochs_range, history["val_precision"], "b-", label="Precision")
    axes[1, 0].plot(epochs_range, history["val_f1"], "g--", label="F1")
    axes[1, 0].axhline(y=0.95, color="r", ls=":", alpha=0.5, label="Target (95%)")
    axes[1, 0].set_title("Recall / Precision / F1", fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs_range, history["val_auc"], "purple", label="AUC-ROC")
    axes[1, 1].axhline(y=0.90, color="r", ls=":", alpha=0.5, label="Target (0.90)")
    axes[1, 1].set_title("AUC-ROC", fontweight="bold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Training History", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "training_history.png"), dpi=150)
    plt.close()
    logger.info(f"Training history saved to {eval_dir}/training_history.png")

    # 2. Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {eval_dir}/confusion_matrix.png")

    # 3. ROC curves
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, NUM_CLASSES))
    for i in range(NUM_CLASSES):
        binary_true = (all_labels == i).astype(int)
        if binary_true.sum() == 0:
            continue
        try:
            fpr, tpr, _ = roc_curve(binary_true, all_probs[:, i])
            class_auc = roc_auc_score(binary_true, all_probs[:, i])
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f"{CLASS_NAMES[i]} (AUC={class_auc:.3f})")
        except:
            pass
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "roc_curves.png"), dpi=150)
    plt.close()
    logger.info(f"ROC curves saved to {eval_dir}/roc_curves.png")

    # ========================================================================
    # Grad-CAM on sample images
    # ========================================================================
    logger.info("\nGenerating Grad-CAM visualizations...")
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image

        target_layer = model.backbone.blocks[-1]
        cam = GradCAM(model=model, target_layers=[target_layer])

        # Get a few val images
        gradcam_dir = os.path.join(eval_dir, "gradcam")
        os.makedirs(gradcam_dir, exist_ok=True)

        val_transform = get_val_transforms()
        sample_df = val_df.sample(min(12, len(val_df)), random_state=SEED)

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()

        for i, (_, row) in enumerate(sample_df.iterrows()):
            if i >= 12:
                break
            img_path = os.path.join(VAL_IMAGES, f"{row['id_code']}.png")
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            image_float = image_resized.astype(np.float32) / 255.0

            transformed = val_transform(image=image_resized)
            input_tensor = transformed["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(input_tensor).argmax(1).item()

            targets = [ClassifierOutputTarget(pred)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
            overlay = show_cam_on_image(image_float, grayscale_cam, use_rgb=True)

            true_label = CLASS_NAMES[int(row["diagnosis"])]
            pred_label = CLASS_NAMES[pred]
            color = "green" if pred == int(row["diagnosis"]) else "red"

            axes[i].imshow(overlay)
            axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=9, color=color)
            axes[i].axis("off")

        plt.suptitle("Grad-CAM Visualizations (Green=Correct, Red=Wrong)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(gradcam_dir, "gradcam_samples.png"), dpi=150)
        plt.close()
        logger.info(f"Grad-CAM visualizations saved to {gradcam_dir}/gradcam_samples.png")
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Best model: {os.path.join(MODEL_DIR, 'best_model.pth')}")
    logger.info(f"Plots: {eval_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
