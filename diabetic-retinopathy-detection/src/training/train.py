"""
Training Pipeline for Diabetic Retinopathy Detection
=====================================================
Complete training loop with:
- Mixed-precision training (FP16)
- Gradient accumulation
- Learning rate scheduling (Cosine Annealing / ReduceLROnPlateau)
- Progressive backbone unfreezing
- Early stopping on validation recall
- Model checkpointing
- MLflow experiment tracking
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    StepLR,
)
from tqdm import tqdm

from src.data.dataloader import create_dataloaders
from src.models.cnn_model import get_model
from src.models.loss_functions import get_loss_function
from src.evaluation.metrics import compute_metrics, compute_auc_roc

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to halt training when a monitored metric stops improving.

    Args:
        patience: Number of epochs to wait for improvement.
        mode: 'min' for loss, 'max' for recall/AUC.
        min_delta: Minimum change to qualify as improvement.
    """

    def __init__(self, patience: int = 7, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered after {self.counter} epochs "
                    f"without improvement (best={self.best_score:.4f})"
                )

        return self.should_stop


class Trainer:
    """
    Full training pipeline orchestrator.

    Handles the complete training lifecycle including:
    - Model training with mixed precision
    - Validation with comprehensive metrics
    - LR scheduling and early stopping
    - Progressive backbone unfreezing
    - Model checkpointing
    - MLflow logging

    Args:
        config: Full configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.train_cfg = config.get("training", {})
        self.model_cfg = config.get("model", {})
        self.log_cfg = config.get("logging", {})
        self.device = self._get_device()
        self.seed = config.get("project", {}).get("seed", 42)

        # Set seeds for reproducibility
        self._set_seed()

        # Initialize components
        logger.info("=" * 60)
        logger.info("DIABETIC RETINOPATHY DETECTION - TRAINING")
        logger.info("=" * 60)

        # DataLoaders
        logger.info("Creating data loaders...")
        self.train_loader, self.val_loader, self.test_loader, self.data_meta = (
            create_dataloaders(config)
        )

        # Model
        logger.info("Building model...")
        self.model = get_model(config).to(self.device)

        # Loss function
        self.criterion = get_loss_function(
            config, class_counts=self.data_meta["class_counts"]
        ).to(self.device)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # LR Scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.use_amp = self.train_cfg.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Early stopping
        if self.train_cfg.get("early_stopping", True):
            self.early_stopping = EarlyStopping(
                patience=self.train_cfg.get("patience", 7),
                mode=self.train_cfg.get("mode", "max"),
            )
        else:
            self.early_stopping = None

        # Tracking
        self.best_metric = 0.0 if self.train_cfg.get("mode", "max") == "max" else float("inf")
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_recall": [],
            "val_precision": [],
            "val_f1": [],
            "val_auc": [],
            "lr": [],
        }

        # MLflow
        self.mlflow_run = None
        if self.log_cfg.get("use_mlflow", False):
            self._init_mlflow()

        # Checkpoint directory
        self.checkpoint_dir = Path(self.log_cfg.get("checkpoint_dir", "models"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info(f"Gradient accumulation steps: {self.train_cfg.get('gradient_accumulation_steps', 1)}")

    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        else:
            device = torch.device("cpu")
            logger.warning("No GPU detected. Training on CPU (will be slow).")
        return device

    def _set_seed(self):
        """Set all random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        opt_name = self.train_cfg.get("optimizer", "adamw").lower()
        lr = self.train_cfg.get("learning_rate", 3e-4)
        wd = self.train_cfg.get("weight_decay", 0.01)

        # Only optimize parameters that require gradients
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        if opt_name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        elif opt_name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            momentum = self.train_cfg.get("momentum", 0.9)
            return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _create_scheduler(self):
        """Create LR scheduler from config."""
        sched_name = self.train_cfg.get("scheduler", "cosine").lower()

        if sched_name == "cosine":
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.train_cfg.get("cosine_T_0", 10),
                T_mult=self.train_cfg.get("cosine_T_mult", 2),
                eta_min=self.train_cfg.get("cosine_eta_min", 1e-6),
            )
        elif sched_name == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode=self.train_cfg.get("mode", "max"),
                factor=self.train_cfg.get("plateau_factor", 0.5),
                patience=self.train_cfg.get("plateau_patience", 3),
                min_lr=self.train_cfg.get("plateau_min_lr", 1e-6),
                verbose=True,
            )
        elif sched_name == "step":
            return StepLR(self.optimizer, step_size=10, gamma=0.1)
        else:
            return None

    def _init_mlflow(self):
        """Initialize MLflow experiment tracking."""
        try:
            import mlflow

            tracking_uri = self.log_cfg.get("mlflow_tracking_uri", "mlruns")
            mlflow.set_tracking_uri(tracking_uri)

            experiment_name = self.log_cfg.get(
                "mlflow_experiment_name", "diabetic_retinopathy"
            )
            mlflow.set_experiment(experiment_name)

            self.mlflow_run = mlflow.start_run()
            mlflow.log_params({
                "architecture": self.model_cfg.get("architecture"),
                "num_classes": self.model_cfg.get("num_classes"),
                "learning_rate": self.train_cfg.get("learning_rate"),
                "batch_size": self.train_cfg.get("batch_size"),
                "optimizer": self.train_cfg.get("optimizer"),
                "dropout": self.model_cfg.get("dropout"),
                "image_size": self.config.get("data", {}).get("image_size"),
            })
            logger.info(f"MLflow tracking initialized: {experiment_name}")
        except ImportError:
            logger.warning("MLflow not installed. Skipping experiment tracking.")
        except Exception as e:
            logger.warning(f"MLflow init failed: {e}. Continuing without tracking.")

    def train_one_epoch(self, epoch: int) -> float:
        """
        Train for one epoch with gradient accumulation and mixed precision.

        Args:
            epoch: Current epoch number.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        accum_steps = self.train_cfg.get("gradient_accumulation_steps", 1)
        clip_norm = self.train_cfg.get("gradient_clip_max_norm", 1.0)
        log_interval = self.log_cfg.get("log_interval", 10)

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1} [Train]",
            leave=False,
        )

        self.optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(progress):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Forward pass (mixed precision)
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / accum_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / accum_steps
                loss.backward()

                if (batch_idx + 1) % accum_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            running_loss += loss.item() * accum_steps
            num_batches += 1

            # Update progress bar
            if batch_idx % log_interval == 0:
                progress.set_postfix({"loss": f"{running_loss / num_batches:.4f}"})

        avg_loss = running_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation and compute all metrics.

        Returns:
            Dictionary with all validation metrics.
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        running_loss = 0.0
        num_batches = 0

        num_classes = self.model_cfg.get("num_classes", 5)

        for images, labels in tqdm(self.val_loader, desc="Validating", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            num_batches += 1

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        avg_loss = running_loss / max(num_batches, 1)

        # Compute metrics
        metrics = compute_metrics(all_labels, all_preds, num_classes=num_classes)
        try:
            auc = compute_auc_roc(all_labels, all_probs, num_classes=num_classes)
            metrics["auc_roc"] = auc["auc_roc"]
        except Exception as e:
            logger.warning(f"AUC computation failed: {e}")
            metrics["auc_roc"] = 0.0

        metrics["loss"] = avg_loss

        return metrics

    def fit(self) -> Dict[str, Any]:
        """
        Full training loop.

        Orchestrates:
        1. Epoch-level training and validation
        2. LR scheduling
        3. Progressive unfreezing
        4. Early stopping
        5. Best model checkpointing
        6. Metrics logging

        Returns:
            Dictionary with complete training history and best metrics.
        """
        max_epochs = self.train_cfg.get("max_epochs", 50)
        min_epochs = self.train_cfg.get("min_epochs", 10)
        unfreeze_after = self.model_cfg.get("unfreeze_after_epochs", 3)
        monitor_metric = self.train_cfg.get("monitor", "val_recall")
        monitor_mode = self.train_cfg.get("mode", "max")

        logger.info(f"\nStarting training for up to {max_epochs} epochs")
        logger.info(f"Monitoring: {monitor_metric} (mode={monitor_mode})")
        logger.info("-" * 60)

        start_time = time.time()

        for epoch in range(max_epochs):
            epoch_start = time.time()

            # Progressive unfreezing
            if epoch == unfreeze_after and self.model._backbone_frozen:
                logger.info(f"\n*** Unfreezing backbone at epoch {epoch + 1} ***")
                self.model.unfreeze_backbone()
                # Re-create optimizer with all params and lower LR
                self.optimizer = self._create_optimizer()
                # Reduce LR for fine-tuning
                for pg in self.optimizer.param_groups:
                    pg["lr"] = pg["lr"] * 0.1
                self.scheduler = self._create_scheduler()
                logger.info(f"LR reduced to {self.optimizer.param_groups[0]['lr']:.6f} for fine-tuning")

            # Train one epoch
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Get current LR
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics.get("accuracy", 0))
            self.history["val_recall"].append(val_metrics.get("recall", 0))
            self.history["val_precision"].append(val_metrics.get("precision", 0))
            self.history["val_f1"].append(val_metrics.get("f1", 0))
            self.history["val_auc"].append(val_metrics.get("auc_roc", 0))
            self.history["lr"].append(current_lr)

            # Log epoch results
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch + 1}/{max_epochs} ({epoch_time:.1f}s) — "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Acc: {val_metrics.get('accuracy', 0):.4f} | "
                f"Recall: {val_metrics.get('recall', 0):.4f} | "
                f"F1: {val_metrics.get('f1', 0):.4f} | "
                f"AUC: {val_metrics.get('auc_roc', 0):.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # MLflow logging
            self._log_mlflow_metrics(epoch, train_loss, val_metrics, current_lr)

            # LR Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    metric_key = monitor_metric.replace("val_", "")
                    self.scheduler.step(val_metrics.get(metric_key, val_metrics["loss"]))
                else:
                    self.scheduler.step()

            # Checkpointing (save best model)
            metric_key = monitor_metric.replace("val_", "")
            current_metric = val_metrics.get(metric_key, val_metrics.get("recall", 0))

            is_best = (
                (monitor_mode == "max" and current_metric > self.best_metric)
                or (monitor_mode == "min" and current_metric < self.best_metric)
            )
            if is_best:
                self.best_metric = current_metric
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info(f"  ★ New best {monitor_metric}: {self.best_metric:.4f}")

            # Early stopping
            if self.early_stopping and epoch >= min_epochs:
                if self.early_stopping(current_metric):
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Training complete in {total_time / 60:.1f} minutes")
        logger.info(f"Best {monitor_metric}: {self.best_metric:.4f}")
        logger.info("=" * 60)

        # End MLflow run
        if self.mlflow_run:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass

        return {
            "history": self.history,
            "best_metric": self.best_metric,
            "total_time_minutes": total_time / 60,
        }

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ):
        """
        Save model checkpoint with metadata.

        Args:
            epoch: Current epoch number.
            metrics: Validation metrics dictionary.
            is_best: If True, also save as 'best_model.pth'.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "architecture": self.model_cfg.get("architecture"),
            "num_classes": self.model_cfg.get("num_classes"),
        }

        # Save latest
        path = self.checkpoint_dir / "latest_model.pth"
        torch.save(checkpoint, path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")

    def load_checkpoint(self, path: str):
        """Load a model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(
            f"Loaded checkpoint from {path} "
            f"(epoch {checkpoint.get('epoch', '?')})"
        )
        return checkpoint

    def _log_mlflow_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: Dict[str, float],
        lr: float,
    ):
        """Log metrics to MLflow."""
        if not self.mlflow_run:
            return
        try:
            import mlflow

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics.get("accuracy", 0),
                    "val_recall": val_metrics.get("recall", 0),
                    "val_precision": val_metrics.get("precision", 0),
                    "val_f1": val_metrics.get("f1", 0),
                    "val_auc_roc": val_metrics.get("auc_roc", 0),
                    "learning_rate": lr,
                },
                step=epoch,
            )
        except Exception as e:
            logger.debug(f"MLflow logging failed: {e}")


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train DR Detection Model")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log"),
        ],
    )

    # Load config
    config = load_config(args.config)
    logger.info(f"Configuration loaded from {args.config}")

    # Train
    trainer = Trainer(config)
    results = trainer.fit()

    logger.info(f"\nFinal Results:")
    logger.info(f"  Best Validation Recall: {results['best_metric']:.4f}")
    logger.info(f"  Training Time: {results['total_time_minutes']:.1f} minutes")


if __name__ == "__main__":
    main()
