"""
Custom Loss Functions for Diabetic Retinopathy Detection
========================================================
Focal Loss and Weighted Cross-Entropy for handling severe
class imbalance in medical image classification.

Key principle: Minimize false negatives (missed diagnoses).
"""

import logging
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification with class imbalance.

    Focal Loss reduces the contribution of easy-to-classify examples and
    focuses training on hard, misclassified examples. This is critical for
    diabetic retinopathy where severe grades are rare but must not be missed.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class-specific weighting factor. Can be:
            - None: No class weighting
            - Tensor of shape (num_classes,): Per-class weights
        gamma: Focusing parameter (default: 2.0).
            - gamma=0 → standard Cross-Entropy
            - gamma>0 → down-weights easy examples
        reduction: 'mean', 'sum', or 'none'.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.FloatTensor(alpha)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model output of shape (B, C) — raw logits (not softmax).
            targets: Ground truth labels of shape (B,) — class indices.

        Returns:
            Scalar loss value.
        """
        num_classes = logits.shape[1]

        # Compute softmax probabilities
        p = F.softmax(logits, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # Get probability of true class
        p_t = (p * targets_one_hot).sum(dim=1)
        p_t = torch.clamp(p_t, min=1e-8, max=1.0)  # Numerical stability

        # Focal modulating factor
        focal_weight = (1 - p_t) ** self.gamma

        # Cross-entropy term
        ce = -torch.log(p_t)

        # Apply class weights (alpha)
        if self.alpha is not None:
            alpha_t = self.alpha.to(targets.device)[targets]
            loss = alpha_t * focal_weight * ce
        else:
            loss = focal_weight * ce

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss with per-class weighting.

    Applies higher weights to minority classes (severe DR grades)
    to prevent the model from ignoring rare but critical cases.

    Args:
        class_weights: Tensor of shape (num_classes,) with per-class weights.
        label_smoothing: Label smoothing factor (0.0 = no smoothing).
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            if isinstance(class_weights, (list, np.ndarray)):
                class_weights = torch.FloatTensor(class_weights)
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.

        Args:
            logits: Model output of shape (B, C).
            targets: Ground truth labels of shape (B,).

        Returns:
            Scalar loss value.
        """
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        return F.cross_entropy(
            logits, targets,
            weight=weight,
            label_smoothing=self.label_smoothing,
        )


def get_loss_function(
    config: Dict[str, Any],
    class_counts: Optional[List[int]] = None,
) -> nn.Module:
    """
    Factory function to create the appropriate loss function.

    Automatically computes class weights from training set distribution
    if auto_compute_weights is enabled.

    Args:
        config: Full configuration dictionary.
        class_counts: List of per-class sample counts from the training set.

    Returns:
        Loss function (nn.Module).
    """
    balance_cfg = config.get("class_balance", {})
    model_cfg = config.get("model", {})
    num_classes = model_cfg.get("num_classes", 5)

    # Compute class weights
    class_weights = None
    if balance_cfg.get("auto_compute_weights", True) and class_counts is not None:
        counts = np.array(class_counts, dtype=float)
        counts = np.maximum(counts, 1.0)
        # Inverse frequency weighting
        class_weights = 1.0 / counts
        class_weights = class_weights / class_weights.min()
        class_weights = torch.FloatTensor(class_weights)
        logger.info(f"Auto-computed class weights: {class_weights.tolist()}")
    elif balance_cfg.get("manual_weights") is not None:
        class_weights = torch.FloatTensor(balance_cfg["manual_weights"])
        logger.info(f"Using manual class weights: {class_weights.tolist()}")

    # Select loss function (default: Focal Loss for medical imaging)
    loss_type = config.get("training", {}).get("loss_function", "focal")

    if loss_type == "focal":
        gamma = config.get("training", {}).get("focal_gamma", 2.0)
        loss_fn = FocalLoss(alpha=class_weights, gamma=gamma)
        logger.info(f"Using Focal Loss (gamma={gamma})")
    elif loss_type == "weighted_ce":
        smoothing = config.get("training", {}).get("label_smoothing", 0.0)
        loss_fn = WeightedCrossEntropyLoss(
            class_weights=class_weights,
            label_smoothing=smoothing,
        )
        logger.info(f"Using Weighted Cross-Entropy (smoothing={smoothing})")
    else:
        # Default to Focal Loss
        loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
        logger.info("Using default Focal Loss (gamma=2.0)")

    return loss_fn
