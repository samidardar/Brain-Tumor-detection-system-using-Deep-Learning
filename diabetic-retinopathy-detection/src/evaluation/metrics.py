"""
Evaluation Metrics for Diabetic Retinopathy Detection
=====================================================
Comprehensive metrics suite for clinical-grade evaluation including
AUC-ROC, sensitivity/recall optimization, confusion matrices,
and detailed false negative/positive analysis.

Priority: Maximize recall (sensitivity) to minimize missed diagnoses.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    For multi-class, uses macro-averaging to give equal weight to all classes
    (important when minority classes are clinically significant).

    Args:
        y_true: Ground truth labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        num_classes: Number of classes.

    Returns:
        Dictionary with accuracy, precision, recall, f1, specificity.
    """
    average = "binary" if num_classes == 2 else "macro"

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    # Per-class recall (sensitivity)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    for i, r in enumerate(per_class_recall):
        metrics[f"recall_class_{i}"] = float(r)

    # Specificity (for binary or per-class)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["false_negative_rate"] = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["false_positive_rate"] = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        # Per-class specificity
        for i in range(num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics[f"specificity_class_{i}"] = spec

    return metrics


def compute_auc_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 5,
) -> Dict[str, Any]:
    """
    Compute AUC-ROC score and curve data.

    For multi-class, computes One-vs-Rest AUC per class and macro average.

    Args:
        y_true: Ground truth labels, shape (N,).
        y_prob: Predicted probabilities, shape (N, C).
        num_classes: Number of classes.

    Returns:
        Dictionary with 'auc_roc' (float), 'per_class_auc' (dict),
        and 'curve_data' for plotting.
    """
    result = {}

    if num_classes == 2:
        # Binary: use probability of positive class
        probs = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
        auc = float(roc_auc_score(y_true, probs))
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        result["auc_roc"] = auc
        result["curve_data"] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
    else:
        # Multi-class: One-vs-Rest
        try:
            auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except ValueError:
            auc = 0.0
        result["auc_roc"] = auc

        # Per-class AUC
        per_class_auc = {}
        curve_data = {}
        for i in range(num_classes):
            binary_true = (y_true == i).astype(int)
            if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
                per_class_auc[i] = 0.0
                continue
            try:
                class_auc = float(roc_auc_score(binary_true, y_prob[:, i]))
                fpr, tpr, thresholds = roc_curve(binary_true, y_prob[:, i])
                per_class_auc[i] = class_auc
                curve_data[i] = {"fpr": fpr, "tpr": tpr}
            except ValueError:
                per_class_auc[i] = 0.0

        result["per_class_auc"] = per_class_auc
        result["curve_data"] = curve_data

    return result


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.95,
) -> Tuple[float, Dict[str, float]]:
    """
    Find the classification threshold that achieves the target recall.

    For medical applications, we prioritize high recall (sensitivity)
    to minimize missed diagnoses, even at the cost of lower precision.

    Args:
        y_true: Binary ground truth labels.
        y_prob: Predicted probability of positive class.
        target_recall: Minimum target recall (default: 0.95).

    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold).
    """
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        recall = recall_score(y_true, preds, zero_division=0)
        if recall >= target_recall:
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

    # Compute metrics at threshold
    final_preds = (y_prob >= best_threshold).astype(int)
    metrics = {
        "threshold": float(best_threshold),
        "recall": float(recall_score(y_true, final_preds, zero_division=0)),
        "precision": float(precision_score(y_true, final_preds, zero_division=0)),
        "f1": float(f1_score(y_true, final_preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, final_preds)),
    }

    logger.info(
        f"Optimal threshold for recall≥{target_recall}: {best_threshold:.2f} "
        f"(Recall={metrics['recall']:.4f}, Precision={metrics['precision']:.4f})"
    )

    return best_threshold, metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot a styled confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class names for axis labels.
        save_path: If provided, save figure to this path.
        title: Plot title.
        normalize: If True, show percentages instead of counts.

    Returns:
        Matplotlib Figure.
    """
    if class_names is None:
        num_classes = max(len(set(y_true)), len(set(y_pred)))
        class_names = [f"Class {i}" for i in range(num_classes)]

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 5,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curve(s) with AUC annotation.

    For multi-class, plots One-vs-Rest curves for each class.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities, shape (N, C).
        num_classes: Number of classes.
        class_names: List of class names.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))

    if num_classes == 2:
        probs = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        ax.plot(fpr, tpr, color=colors[0], lw=2, label=f"ROC (AUC = {auc:.3f})")
    else:
        for i in range(num_classes):
            binary_true = (y_true == i).astype(int)
            if binary_true.sum() == 0:
                continue
            try:
                fpr, tpr, _ = roc_curve(binary_true, y_prob[:, i])
                auc = roc_auc_score(binary_true, y_prob[:, i])
                ax.plot(
                    fpr, tpr, color=colors[i], lw=2,
                    label=f"{class_names[i]} (AUC = {auc:.3f})",
                )
            except ValueError:
                continue

    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC curve saved to {save_path}")

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training curves (loss, accuracy, recall, etc.).

    Args:
        history: Dictionary of metric lists from training.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history.get("train_loss", []), "b-", label="Train Loss")
    ax.plot(epochs, history.get("val_loss", []), "r-", label="Val Loss")
    ax.set_title("Loss", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history.get("val_accuracy", []), "g-", label="Val Accuracy")
    ax.set_title("Accuracy", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Recall & Precision
    ax = axes[1, 0]
    ax.plot(epochs, history.get("val_recall", []), "r-", label="Recall (Sensitivity)")
    ax.plot(epochs, history.get("val_precision", []), "b-", label="Precision")
    ax.plot(epochs, history.get("val_f1", []), "g--", label="F1-Score")
    ax.axhline(y=0.95, color="r", linestyle=":", alpha=0.5, label="Target Recall (95%)")
    ax.set_title("Recall / Precision / F1", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AUC
    ax = axes[1, 1]
    ax.plot(epochs, history.get("val_auc", []), "purple", label="AUC-ROC")
    ax.axhline(y=0.90, color="r", linestyle=":", alpha=0.5, label="Target AUC (0.90)")
    ax.set_title("AUC-ROC", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Training History", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training history saved to {save_path}")

    return fig


def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    image_paths: Optional[List[str]] = None,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    num_examples: int = 10,
) -> Dict[str, Any]:
    """
    Analyze false negatives and false positives.

    For medical AI, FN analysis is critical — these are missed diagnoses.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        image_paths: Optional list of image paths for each sample.
        y_prob: Optional predicted probabilities.
        class_names: Optional class names.
        num_examples: Number of FN/FP examples to collect.

    Returns:
        Dictionary with 'false_negatives', 'false_positives', and summary stats.
    """
    analysis = {
        "total_samples": len(y_true),
        "correct": int((y_true == y_pred).sum()),
        "incorrect": int((y_true != y_pred).sum()),
    }

    # Find false negatives (true positive cases predicted as negative)
    fn_mask = (y_true > 0) & (y_pred == 0) if len(set(y_true)) <= 5 else (y_true != y_pred)
    fn_indices = np.where(fn_mask)[0]
    analysis["false_negative_count"] = len(fn_indices)

    fn_examples = []
    for idx in fn_indices[:num_examples]:
        example = {
            "index": int(idx),
            "true_label": int(y_true[idx]),
            "predicted_label": int(y_pred[idx]),
        }
        if image_paths:
            example["image_path"] = image_paths[idx]
        if y_prob is not None:
            example["confidence"] = float(y_prob[idx].max())
            example["probabilities"] = y_prob[idx].tolist()
        fn_examples.append(example)
    analysis["false_negative_examples"] = fn_examples

    # Find false positives
    fp_mask = (y_true == 0) & (y_pred > 0) if len(set(y_true)) <= 5 else (y_true != y_pred) & ~fn_mask
    fp_indices = np.where(fp_mask)[0]
    analysis["false_positive_count"] = len(fp_indices)

    fp_examples = []
    for idx in fp_indices[:num_examples]:
        example = {
            "index": int(idx),
            "true_label": int(y_true[idx]),
            "predicted_label": int(y_pred[idx]),
        }
        if image_paths:
            example["image_path"] = image_paths[idx]
        if y_prob is not None:
            example["confidence"] = float(y_prob[idx].max())
        fp_examples.append(example)
    analysis["false_positive_examples"] = fp_examples

    # Clinical impact summary
    total_positive = int((y_true > 0).sum())
    if total_positive > 0:
        analysis["missed_diagnosis_rate"] = analysis["false_negative_count"] / total_positive
    else:
        analysis["missed_diagnosis_rate"] = 0.0

    logger.info(f"Error Analysis:")
    logger.info(f"  Total: {analysis['total_samples']}, Wrong: {analysis['incorrect']}")
    logger.info(f"  False Negatives (missed diagnoses): {analysis['false_negative_count']}")
    logger.info(f"  False Positives (unnecessary referrals): {analysis['false_positive_count']}")
    logger.info(f"  Missed Diagnosis Rate: {analysis['missed_diagnosis_rate']:.2%}")

    return analysis


def generate_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 5,
    class_names: Optional[List[str]] = None,
    history: Optional[Dict[str, List[float]]] = None,
    save_dir: str = "outputs/evaluation",
    image_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate a complete evaluation report with all plots and metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities.
        num_classes: Number of classes.
        class_names: Optional list of class names.
        history: Optional training history for training curves.
        save_dir: Directory to save all plots and reports.
        image_paths: Optional image paths for error analysis.

    Returns:
        Complete results dictionary.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    report = {}

    # 1. Classification metrics
    logger.info("\n📊 Computing classification metrics...")
    metrics = compute_metrics(y_true, y_pred, num_classes)
    report["metrics"] = metrics

    # 2. AUC-ROC
    logger.info("📈 Computing AUC-ROC...")
    auc_result = compute_auc_roc(y_true, y_prob, num_classes)
    report["auc_roc"] = auc_result["auc_roc"]
    report["per_class_auc"] = auc_result.get("per_class_auc", {})

    # 3. Classification report (sklearn)
    cls_report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    report["classification_report"] = cls_report

    # Print text report
    logger.info("\n" + classification_report(y_true, y_pred, target_names=class_names))

    # 4. Confusion matrix
    logger.info("📉 Generating confusion matrix...")
    plot_confusion_matrix(
        y_true, y_pred, class_names=class_names,
        save_path=str(save_path / "confusion_matrix.png"),
    )
    plot_confusion_matrix(
        y_true, y_pred, class_names=class_names, normalize=False,
        save_path=str(save_path / "confusion_matrix_counts.png"),
        title="Confusion Matrix (Counts)",
    )

    # 5. ROC curve
    logger.info("📈 Generating ROC curve...")
    plot_roc_curve(
        y_true, y_prob, num_classes, class_names,
        save_path=str(save_path / "roc_curve.png"),
    )

    # 6. Training history (if available)
    if history:
        logger.info("📊 Plotting training history...")
        plot_training_history(history, save_path=str(save_path / "training_history.png"))

    # 7. Error analysis
    logger.info("🔍 Analyzing errors...")
    error_analysis = analyze_errors(
        y_true, y_pred, image_paths=image_paths, y_prob=y_prob,
        class_names=class_names
    )
    report["error_analysis"] = error_analysis

    # 8. Summary
    report["summary"] = {
        "accuracy": metrics["accuracy"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "auc_roc": report["auc_roc"],
        "total_samples": len(y_true),
        "false_negatives": error_analysis["false_negative_count"],
        "false_positives": error_analysis["false_positive_count"],
    }

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    for k, v in report["summary"].items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    # Check targets
    recall_target = metrics["recall"] >= 0.95
    auc_target = report["auc_roc"] >= 0.90
    logger.info(f"\n  Recall ≥ 95%: {'✅ PASS' if recall_target else '❌ FAIL'}")
    logger.info(f"  AUC ≥ 0.90:   {'✅ PASS' if auc_target else '❌ FAIL'}")

    return report
