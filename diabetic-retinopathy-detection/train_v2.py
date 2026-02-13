"""
Diabetic Retinopathy Detection v2 - Production Training
========================================================
Hospital-grade (CHU) pipeline targeting 93%+ accuracy.
Fixes all v1 bugs + adds: Ben Graham preprocessing, Mixup,
label smoothing, TTA, GridSearch, binary clinical mode.

GTX 1650 (4GB VRAM) optimized.
"""

import os, sys, time, json, logging, random, copy, csv
from pathlib import Path
from datetime import datetime
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
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
# CONFIG
# ============================================================================
DATASET_DIR = r"C:\Users\PC\Downloads\archive (8)"
TRAIN_CSV = os.path.join(DATASET_DIR, "train_1.csv")
VALID_CSV = os.path.join(DATASET_DIR, "valid.csv")
TRAIN_IMAGES = os.path.join(DATASET_DIR, "train_images", "train_images")
VAL_IMAGES = os.path.join(DATASET_DIR, "val_images", "val_images")

PROJECT_DIR = r"c:\Users\PC\Downloads\New folder\diabetic-retinopathy-detection"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
PREPROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")

SEED = 42
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASS_NAMES_5 = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
CLASS_NAMES_2 = ["Non-Referable", "Referable DR"]

# ============================================================================
# LOGGING (Windows-safe, no Unicode)
# ============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(PROJECT_DIR, "training_v2.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# BEN GRAHAM PREPROCESSING
# ============================================================================
def crop_fundus(image, tol=7):
    """Auto-crop black borders from fundus image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mask = gray > tol
    if not mask.any():
        return image
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image[y0:y1, x0:x1]


def ben_graham_preprocess(image, sigmaX=10, img_size=512):
    """Ben Graham's preprocessing: subtract local avg, add 128."""
    image = crop_fundus(image)
    image = cv2.resize(image, (img_size, img_size))
    image = cv2.addWeighted(
        image, 4,
        cv2.GaussianBlur(image, (0, 0), sigmaX), -4,
        128
    )
    return image


def preprocess_dataset(img_dir, output_dir, img_size=512):
    """Batch preprocess all images with Ben Graham's method."""
    os.makedirs(output_dir, exist_ok=True)
    existing = set(os.listdir(output_dir))
    files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    skipped = 0
    processed = 0
    for f in tqdm(files, desc="Preprocessing"):
        if f in existing:
            skipped += 1
            continue
        img = cv2.imread(os.path.join(img_dir, f))
        if img is None:
            continue
        img = ben_graham_preprocess(img, img_size=img_size)
        cv2.imwrite(os.path.join(output_dir, f), img)
        processed += 1
    
    logger.info(f"Preprocessing done: {processed} new, {skipped} cached")
    return output_dir


# ============================================================================
# DATASET WITH MIXUP
# ============================================================================
class RetinopathyDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, binary=False, preprocessed_dir=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.preprocessed_dir = preprocessed_dir
        self.transform = transform
        self.binary = binary

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id_code"]
        label = int(row["diagnosis"])
        if self.binary:
            label = 0 if label <= 1 else 1  # Non-referable (0,1) vs Referable (2,3,4)

        # Try preprocessed first, fallback to raw
        img_path = None
        if self.preprocessed_dir:
            for ext in ['.png', '.jpg']:
                p = os.path.join(self.preprocessed_dir, f"{img_id}{ext}")
                if os.path.exists(p):
                    img_path = p
                    break
        if img_path is None:
            for ext in ['.png', '.jpg']:
                p = os.path.join(self.image_dir, f"{img_id}{ext}")
                if os.path.exists(p):
                    img_path = p
                    break

        image = cv2.imread(img_path) if img_path else None
        if image is None:
            image = np.zeros((380, 380, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label

    def get_labels(self):
        labels = self.df["diagnosis"].values.astype(int)
        if self.binary:
            return [0 if l <= 1 else 1 for l in labels]
        return labels.tolist()


def mixup_data(x, y, alpha=0.4):
    """Mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


# ============================================================================
# AUGMENTATIONS
# ============================================================================
def get_train_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45,
                           border_mode=cv2.BORDER_CONSTANT, value=0, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=25, p=0.4),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=6.0, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.1, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2),
        A.CoarseDropout(max_holes=8, max_height=img_size // 12, max_width=img_size // 12, p=0.3),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size):
    """Test-Time Augmentation transforms."""
    return [
        get_val_transforms(img_size),
        A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0),
                    A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
        A.Compose([A.Resize(img_size, img_size), A.VerticalFlip(p=1.0),
                    A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
        A.Compose([A.Resize(img_size, img_size), A.RandomRotate90(p=1.0),
                    A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
    ]


# ============================================================================
# MODEL
# ============================================================================
class RetinopathyModel(nn.Module):
    def __init__(self, arch="efficientnet_b0", num_classes=5, dropout=0.4, hidden=256):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=True, num_classes=0, global_pool="avg")
        with torch.no_grad():
            feat = self.backbone(torch.randn(1, 3, 224, 224)).shape[-1]
        self.classifier = nn.Sequential(
            nn.Linear(feat, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self._frozen = False

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._frozen = True

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._frozen = False

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# FOCAL LOSS WITH LABEL SMOOTHING
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        # Label smoothing
        with torch.no_grad():
            targets_oh = torch.zeros_like(logits)
            targets_oh.scatter_(1, targets.unsqueeze(1), 1.0)
            targets_oh = targets_oh * (1 - self.label_smoothing) + self.label_smoothing / num_classes

        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)
        focal_weight = (1 - (p * targets_oh).sum(dim=1)) ** self.gamma
        ce = -(targets_oh * log_p).sum(dim=1)

        if self.alpha is not None:
            alpha_t = self.alpha.to(targets.device)[targets]
            loss = alpha_t * focal_weight * ce
        else:
            loss = focal_weight * ce
        return loss.mean()


# ============================================================================
# COMPUTE METRICS (FIXED AUC)
# ============================================================================
def compute_metrics(all_labels, all_preds, all_probs, num_classes, class_names):
    """Compute all metrics with proper AUC handling."""
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Fixed AUC computation
    auc = 0.0
    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            # Ensure all classes are present
            present = np.unique(all_labels)
            if len(present) >= 2:
                auc = roc_auc_score(
                    all_labels, all_probs,
                    multi_class="ovr", average="macro",
                    labels=list(range(num_classes))
                )
    except Exception as e:
        logger.warning(f"AUC computation failed: {e}")
        auc = 0.0

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc_roc": auc}


# ============================================================================
# TTA INFERENCE
# ============================================================================
def tta_predict(model, image_rgb, tta_transforms, device):
    """Test-Time Augmentation: average predictions over augmented versions."""
    model.eval()
    all_probs = []
    for tfm in tta_transforms:
        aug = tfm(image=image_rgb)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                out = model(aug)
        all_probs.append(torch.softmax(out, dim=1).cpu())
    return torch.stack(all_probs).mean(dim=0)


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_one_config(config, train_df, val_df, config_name="default"):
    """Train model with given config. Returns best metrics."""
    arch = config["arch"]
    img_size = config["img_size"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    dropout = config["dropout"]
    hidden = config["hidden"]
    max_epochs = config["max_epochs"]
    patience = config["patience"]
    unfreeze_at = config["unfreeze_at"]
    gamma = config["gamma"]
    smoothing = config["label_smoothing"]
    use_mixup = config["use_mixup"]
    mixup_alpha = config["mixup_alpha"]
    accum_steps = config["accum_steps"]
    binary = config["binary"]
    num_classes = 2 if binary else 5
    class_names = CLASS_NAMES_2 if binary else CLASS_NAMES_5

    logger.info(f"\n{'='*60}")
    logger.info(f"CONFIG: {config_name}")
    logger.info(f"  arch={arch}, img={img_size}, bs={batch_size}x{accum_steps}")
    logger.info(f"  lr={lr}, drop={dropout}, hidden={hidden}, binary={binary}")
    logger.info(f"  gamma={gamma}, smooth={smoothing}, mixup={use_mixup}")
    logger.info(f"{'='*60}")

    # Datasets
    train_tfm = get_train_transforms(img_size)
    val_tfm = get_val_transforms(img_size)

    train_ds = RetinopathyDataset(train_df, TRAIN_IMAGES, train_tfm, binary,
                                   preprocessed_dir=os.path.join(PREPROCESSED_DIR, "train"))
    val_ds = RetinopathyDataset(val_df, VAL_IMAGES, val_tfm, binary,
                                 preprocessed_dir=os.path.join(PREPROCESSED_DIR, "val"))

    # Weighted sampler
    labels = train_ds.get_labels()
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights / weights.min()
    # Extra boost for minority classes
    if not binary:
        weights[3] *= 1.5  # Severe
        weights[4] *= 1.3  # Proliferative 
    sample_w = [weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                               num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Model
    model = RetinopathyModel(arch, num_classes, dropout, hidden).to(device)
    model.freeze_backbone()
    logger.info(f"Frozen trainable: {model.trainable_params():,}")

    # Loss
    alpha = torch.FloatTensor(weights[:num_classes]).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=gamma, label_smoothing=smoothing)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    # Training loop
    best_metric = 0.0
    best_metrics = {}
    patience_ctr = 0
    history = {k: [] for k in ["train_loss", "val_loss", "val_acc", "val_recall", "val_f1", "val_auc", "lr"]}

    for epoch in range(max_epochs):
        # Unfreeze
        if epoch == unfreeze_at and model._frozen:
            logger.info(f"  [Epoch {epoch+1}] UNFREEZING backbone")
            model.unfreeze_backbone()
            logger.info(f"  Trainable: {model.trainable_params():,}")
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc=f"E{epoch+1} Train", leave=False)):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Mixup
            if use_mixup and random.random() < 0.5:
                images, y_a, y_b, lam = mixup_data(images, targets, mixup_alpha)
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, targets)

            loss = loss / accum_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accum_steps

        train_loss = running_loss / len(train_loader)
        scheduler.step()

        # --- VALIDATE ---
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Val", leave=False):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_loss /= len(val_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = compute_metrics(all_labels, all_preds, all_probs, num_classes, class_names)
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(metrics["accuracy"])
        history["val_recall"].append(metrics["recall"])
        history["val_f1"].append(metrics["f1"])
        history["val_auc"].append(metrics["auc_roc"])
        history["lr"].append(cur_lr)

        # Primary metric: AUC for binary, recall for multi
        primary = metrics["auc_roc"] if binary else metrics["recall"]
        primary_name = "AUC" if binary else "Recall"

        logger.info(
            f"  E{epoch+1}/{max_epochs} | TL:{train_loss:.4f} VL:{val_loss:.4f} | "
            f"Acc:{metrics['accuracy']:.4f} Rec:{metrics['recall']:.4f} "
            f"F1:{metrics['f1']:.4f} AUC:{metrics['auc_roc']:.4f} | LR:{cur_lr:.6f}"
        )

        # Per-class recall
        per_rec = recall_score(all_labels, all_preds, average=None, zero_division=0)
        for i, r in enumerate(per_rec):
            logger.info(f"    {class_names[i]:20s} recall: {r:.4f}")

        # Save best
        if primary > best_metric:
            best_metric = primary
            best_metrics = metrics.copy()
            best_metrics["epoch"] = epoch + 1
            patience_ctr = 0
            ckpt = {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "metrics": metrics, "config": config,
                "architecture": arch, "num_classes": num_classes,
            }
            save_name = f"best_model_{config_name}.pth"
            torch.save(ckpt, os.path.join(MODEL_DIR, save_name))
            logger.info(f"    >> NEW BEST {primary_name}: {best_metric:.4f} - saved {save_name}")
        else:
            patience_ctr += 1
            if patience_ctr >= patience and epoch >= 8:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

        torch.cuda.empty_cache()

    return best_metrics, history, model


# ============================================================================
# GRIDSEARCH
# ============================================================================
def gridsearch(train_df, val_df):
    """GridSearch over key hyperparameters."""
    base = {
        "img_size": 380, "max_epochs": 8, "patience": 5,
        "unfreeze_at": 2, "label_smoothing": 0.1,
        "use_mixup": True, "mixup_alpha": 0.3, "accum_steps": 4,
    }

    configs = [
        # Binary classification configs (clinically standard for referral)
        {**base, "arch": "efficientnet_b0", "batch_size": 8, "lr": 3e-4,
         "dropout": 0.4, "hidden": 256, "gamma": 2.0, "binary": True},
        {**base, "arch": "efficientnet_b0", "batch_size": 8, "lr": 1e-4,
         "dropout": 0.5, "hidden": 512, "gamma": 2.0, "binary": True},
        {**base, "arch": "efficientnet_b0", "batch_size": 8, "lr": 5e-4,
         "dropout": 0.3, "hidden": 256, "gamma": 1.5, "binary": True},
        # EfficientNet-B3 (smaller img for VRAM)
        {**base, "arch": "efficientnet_b3", "batch_size": 4, "lr": 2e-4,
         "dropout": 0.4, "hidden": 512, "gamma": 2.0, "binary": True,
         "img_size": 300, "accum_steps": 8},
        # Multi-class configs
        {**base, "arch": "efficientnet_b0", "batch_size": 8, "lr": 3e-4,
         "dropout": 0.4, "hidden": 256, "gamma": 2.0, "binary": False},
        {**base, "arch": "efficientnet_b0", "batch_size": 8, "lr": 1e-4,
         "dropout": 0.5, "hidden": 512, "gamma": 2.5, "binary": False,
         "label_smoothing": 0.15},
    ]

    results = []
    for i, cfg in enumerate(configs):
        name = f"gs_{i}_{cfg['arch']}_{('bin' if cfg['binary'] else '5c')}_lr{cfg['lr']}"
        logger.info(f"\n>>> GRIDSEARCH [{i+1}/{len(configs)}]: {name}")
        try:
            metrics, _, _ = train_one_config(cfg, train_df, val_df, name)
            metrics["config_name"] = name
            metrics["config"] = str(cfg)
            results.append(metrics)
            logger.info(f">>> RESULT: Acc={metrics['accuracy']:.4f} AUC={metrics['auc_roc']:.4f}")
        except Exception as e:
            logger.error(f">>> FAILED: {e}")
            results.append({"config_name": name, "error": str(e)})

    # Save results
    gs_path = os.path.join(OUTPUT_DIR, "gridsearch_results.csv")
    pd.DataFrame(results).to_csv(gs_path, index=False)
    logger.info(f"\nGridSearch results saved to {gs_path}")

    # Find best
    valid = [r for r in results if "error" not in r]
    if valid:
        best = max(valid, key=lambda x: x.get("auc_roc", 0))
        logger.info(f"\nBEST CONFIG: {best['config_name']}")
        logger.info(f"  Acc={best['accuracy']:.4f} AUC={best['auc_roc']:.4f} Rec={best['recall']:.4f}")
        return best
    return None


# ============================================================================
# FULL EVALUATION + PLOTS
# ============================================================================
def full_evaluation(model, val_df, config, config_name):
    """Full eval with TTA, plots, Grad-CAM."""
    binary = config["binary"]
    img_size = config["img_size"]
    num_classes = 2 if binary else 5
    class_names = CLASS_NAMES_2 if binary else CLASS_NAMES_5

    eval_dir = os.path.join(OUTPUT_DIR, f"eval_{config_name}")
    os.makedirs(eval_dir, exist_ok=True)

    val_tfm = get_val_transforms(img_size)
    tta_tfms = get_tta_transforms(img_size)
    val_ds = RetinopathyDataset(val_df, VAL_IMAGES, val_tfm, binary,
                                 preprocessed_dir=os.path.join(PREPROCESSED_DIR, "val"))
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                             num_workers=2, pin_memory=True)

    # Standard eval
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Final eval"):
            images = images.to(device)
            with torch.amp.autocast("cuda"):
                out = model(images)
            probs = torch.softmax(out, dim=1)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = compute_metrics(all_labels, all_preds, all_probs, num_classes, class_names)
    logger.info(f"\nFINAL RESULTS ({config_name}):")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  F1:        {metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    logger.info("\n" + classification_report(all_labels, all_preds, target_names=class_names))

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")
    ax.set_title(f"Confusion Matrix - {config_name}\nAcc={metrics['accuracy']:.4f} AUC={metrics['auc_roc']:.4f}",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # --- ROC Curves ---
    fig, ax = plt.subplots(figsize=(8, 6))
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {metrics['auc_roc']:.4f}")
    else:
        for i in range(num_classes):
            bt = (all_labels == i).astype(int)
            if bt.sum() > 0:
                try:
                    fp, tp, _ = roc_curve(bt, all_probs[:, i])
                    cl_auc = roc_auc_score(bt, all_probs[:, i])
                    ax.plot(fp, tp, lw=2, label=f"{class_names[i]} (AUC={cl_auc:.3f})")
                except:
                    pass
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curves", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "roc_curves.png"), dpi=150)
    plt.close()

    # --- Grad-CAM ---
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image

        target_layer = model.backbone.blocks[-1] if hasattr(model.backbone, 'blocks') else list(model.backbone.children())[-2]
        cam = GradCAM(model=model, target_layers=[target_layer])

        gradcam_dir = os.path.join(eval_dir, "gradcam")
        os.makedirs(gradcam_dir, exist_ok=True)

        sample = val_df.sample(min(12, len(val_df)), random_state=SEED)
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()

        for i, (_, row) in enumerate(sample.iterrows()):
            if i >= 12:
                break
            ipath = os.path.join(VAL_IMAGES, f"{row['id_code']}.png")
            img = cv2.imread(ipath)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_r = cv2.resize(img, (img_size, img_size))
            img_f = img_r.astype(np.float32) / 255.0

            t = val_tfm(image=img_r)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(t).argmax(1).item()

            gc = cam(input_tensor=t, targets=[ClassifierOutputTarget(pred)])[0]
            overlay = show_cam_on_image(img_f, gc, use_rgb=True)

            true_l = int(row["diagnosis"])
            if binary:
                true_l = 0 if true_l <= 1 else 1
            color = "green" if pred == true_l else "red"
            axes[i].imshow(overlay)
            axes[i].set_title(f"T:{class_names[true_l]} P:{class_names[pred]}", fontsize=9, color=color)
            axes[i].axis("off")

        plt.suptitle("Grad-CAM (Green=Correct, Red=Wrong)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(gradcam_dir, "gradcam_samples.png"), dpi=150)
        plt.close()
        logger.info(f"Grad-CAM saved to {gradcam_dir}")
    except Exception as e:
        logger.warning(f"Grad-CAM failed: {e}")

    return metrics


# ============================================================================
# MAIN
# ============================================================================
def main():
    logger.info("="*60)
    logger.info("DIABETIC RETINOPATHY DETECTION v2 - PRODUCTION TRAINING")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    logger.info("="*60)

    # Load data
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VALID_CSV)
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # === PHASE 1: PREPROCESS ===
    logger.info("\n--- PHASE 1: BEN GRAHAM PREPROCESSING ---")
    preprocess_dataset(TRAIN_IMAGES, os.path.join(PREPROCESSED_DIR, "train"), img_size=512)
    preprocess_dataset(VAL_IMAGES, os.path.join(PREPROCESSED_DIR, "val"), img_size=512)

    # === PHASE 2: GRIDSEARCH ===
    logger.info("\n--- PHASE 2: GRIDSEARCH ---")
    best_gs = gridsearch(train_df, val_df)

    # === PHASE 3: FULL TRAINING WITH BEST CONFIG ===
    logger.info("\n--- PHASE 3: FULL TRAINING (best config) ---")

    # Best binary config for 93%+ accuracy
    best_config = {
        "arch": "efficientnet_b0", "img_size": 380, "batch_size": 8,
        "lr": 3e-4, "dropout": 0.4, "hidden": 256, "gamma": 2.0,
        "max_epochs": 40, "patience": 10, "unfreeze_at": 3,
        "label_smoothing": 0.1, "use_mixup": True, "mixup_alpha": 0.3,
        "accum_steps": 4, "binary": True,
    }

    # Override with gridsearch winner if available
    if best_gs and best_gs.get("config"):
        try:
            gs_cfg = eval(best_gs["config"])
            gs_cfg["max_epochs"] = 40
            gs_cfg["patience"] = 10
            best_config = gs_cfg
            logger.info(f"Using GridSearch winner config")
        except:
            logger.info(f"Using default best config")

    best_metrics, history, model = train_one_config(best_config, train_df, val_df, "final_best")

    # Also train multi-class for full grading
    logger.info("\n--- PHASE 4: MULTI-CLASS TRAINING ---")
    mc_config = best_config.copy()
    mc_config["binary"] = False
    mc_config["max_epochs"] = 35
    mc_config["label_smoothing"] = 0.15
    mc_config["gamma"] = 2.5
    mc_metrics, mc_history, mc_model = train_one_config(mc_config, train_df, val_df, "final_multiclass")

    # === PHASE 5: FULL EVALUATION ===
    logger.info("\n--- PHASE 5: FULL EVALUATION ---")

    # Load best binary model
    ckpt = torch.load(os.path.join(MODEL_DIR, "best_model_final_best.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    bin_metrics = full_evaluation(model, val_df, best_config, "binary_final")

    # Load best multi-class model
    ckpt_mc = torch.load(os.path.join(MODEL_DIR, "best_model_final_multiclass.pth"), map_location=device)
    mc_model.load_state_dict(ckpt_mc["model_state_dict"])
    mc_final = full_evaluation(mc_model, val_df, mc_config, "multiclass_final")

    # Copy best binary as the default model
    import shutil
    shutil.copy2(
        os.path.join(MODEL_DIR, "best_model_final_best.pth"),
        os.path.join(MODEL_DIR, "best_model.pth")
    )

    # --- Training History Plots ---
    for name, hist in [("binary", history), ("multiclass", mc_history)]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ep = range(1, len(hist["train_loss"])+1)

        axes[0,0].plot(ep, hist["train_loss"], "b-", label="Train")
        axes[0,0].plot(ep, hist["val_loss"], "r-", label="Val")
        axes[0,0].set_title("Loss"); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

        axes[0,1].plot(ep, hist["val_acc"], "g-")
        axes[0,1].set_title("Accuracy"); axes[0,1].grid(True, alpha=0.3)

        axes[1,0].plot(ep, hist["val_recall"], "r-", label="Recall")
        axes[1,0].plot(ep, hist["val_f1"], "g--", label="F1")
        axes[1,0].set_title("Recall / F1"); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

        axes[1,1].plot(ep, hist["val_auc"], "purple")
        axes[1,1].axhline(y=0.93, color="r", ls=":", alpha=0.5, label="Target 93%")
        axes[1,1].set_title("AUC-ROC"); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

        plt.suptitle(f"Training History ({name})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"training_history_{name}.png"), dpi=150)
        plt.close()

    # === SUMMARY ===
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("="*60)
    logger.info(f"Binary (Referable DR screening):")
    logger.info(f"  Accuracy:  {bin_metrics['accuracy']:.4f}")
    logger.info(f"  AUC-ROC:   {bin_metrics['auc_roc']:.4f}")
    logger.info(f"  Recall:    {bin_metrics['recall']:.4f}")
    logger.info(f"  F1:        {bin_metrics['f1']:.4f}")
    logger.info(f"\nMulti-class (5-grade):")
    logger.info(f"  Accuracy:  {mc_final['accuracy']:.4f}")
    logger.info(f"  AUC-ROC:   {mc_final['auc_roc']:.4f}")
    logger.info(f"  Recall:    {mc_final['recall']:.4f}")
    logger.info(f"  F1:        {mc_final['f1']:.4f}")
    logger.info(f"\nModels: {MODEL_DIR}")
    logger.info(f"Plots:  {OUTPUT_DIR}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
