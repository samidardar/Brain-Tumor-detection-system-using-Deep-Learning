"""
Grad-CAM Visualization for Diabetic Retinopathy
================================================
Generates visual explanations (heatmaps) showing which regions
of the fundus image the model focuses on for its predictions.

Essential for clinical trust and validation — clinicians need to
verify that the model looks at pathologically relevant areas
(microaneurysms, hemorrhages, exudates) rather than artifacts.

Uses pytorch-grad-cam library for reliable Grad-CAM computation.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.data.dataloader import get_val_transforms, IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


class GradCAMVisualizer:
    """
    Grad-CAM heatmap generator for fundus image interpretation.

    Produces overlay visualizations showing which image regions
    contributed most to the model's classification decision.

    Args:
        model: Trained RetinopathyModel.
        target_layer: Target convolutional layer for Grad-CAM.
            If None, auto-detects the last conv layer.
        device: Torch device.
        method: Grad-CAM variant ('gradcam', 'gradcam++', 'scorecam').
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer=None,
        device: Optional[torch.device] = None,
        method: str = "gradcam",
    ):
        self.model = model
        self.model.eval()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Auto-detect target layer if not specified
        if target_layer is None:
            target_layer = self._auto_detect_layer()

        # Initialize Grad-CAM
        if method == "gradcam++":
            self.cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
        elif method == "scorecam":
            self.cam = ScoreCAM(model=model, target_layers=[target_layer])
        else:
            self.cam = GradCAM(model=model, target_layers=[target_layer])

        logger.info(f"GradCAM initialized (method={method})")

    def _auto_detect_layer(self):
        """Auto-detect the last convolutional layer of the backbone."""
        if hasattr(self.model, "get_gradcam_target_layer"):
            return self.model.get_gradcam_target_layer()

        # Fallback: find last conv layer
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Conv2d,)):
                last_conv = module
        if last_conv is None:
            raise ValueError("Could not auto-detect target layer for Grad-CAM")
        return last_conv

    def generate_heatmap(
        self,
        image: np.ndarray,
        transform=None,
        target_class: Optional[int] = None,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap for a single image.

        Args:
            image: RGB image as numpy array (H, W, 3), values 0-255.
            transform: Albumentations transform for preprocessing.
            target_class: Class to explain. If None, uses predicted class.
            alpha: Heatmap overlay transparency.
            colormap: OpenCV colormap for heatmap.

        Returns:
            Tuple of (overlay_image, raw_heatmap, predicted_class, confidence).
        """
        # Normalize image to [0, 1] for overlay
        image_float = image.astype(np.float32) / 255.0

        # Preprocess for model
        if transform:
            transformed = transform(image=image)
            input_tensor = transformed["image"].unsqueeze(0).to(self.device)
        else:
            # Manual preprocessing
            resized = cv2.resize(image, (512, 512))
            tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
            # Normalize
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            tensor = (tensor - mean) / std
            input_tensor = tensor.unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probs[0, predicted_class].item()

        # Set target (explain predicted class or specified class)
        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]
        else:
            targets = [ClassifierOutputTarget(predicted_class)]

        # Generate Grad-CAM
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # First (and only) image

        # Resize heatmap to original image size
        heatmap = cv2.resize(grayscale_cam, (image.shape[1], image.shape[0]))

        # Create overlay
        overlay = show_cam_on_image(image_float, heatmap, use_rgb=True)

        return overlay, heatmap, predicted_class, confidence

    def visualize_single(
        self,
        image_path: str,
        transform=None,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualize Grad-CAM for a single image with original + heatmap side by side.

        Args:
            image_path: Path to the fundus image.
            transform: Preprocessing transform.
            class_names: Optional class name mapping.
            save_path: If provided, save the figure.

        Returns:
            Matplotlib Figure.
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        overlay, heatmap, pred_class, confidence = self.generate_heatmap(
            image, transform
        )

        if class_names:
            pred_name = class_names[pred_class]
        else:
            pred_name = f"Class {pred_class}"

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(image)
        axes[0].set_title("Original Fundus Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(
            f"Prediction: {pred_name} ({confidence:.1%})",
            fontsize=12, fontweight="bold",
        )
        axes[2].axis("off")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Grad-CAM visualization saved to {save_path}")

        return fig

    def visualize_batch(
        self,
        image_paths: List[str],
        transform=None,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        max_images: int = 8,
    ) -> plt.Figure:
        """
        Visualize Grad-CAM for multiple images in a grid.

        Args:
            image_paths: List of image paths.
            transform: Preprocessing transform.
            class_names: Optional class name mapping.
            save_path: If provided, save the figure.
            max_images: Maximum number of images to display.

        Returns:
            Matplotlib Figure.
        """
        paths = image_paths[:max_images]
        n = len(paths)
        fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))

        if n == 1:
            axes = axes[np.newaxis, :]

        for i, path in enumerate(paths):
            try:
                image = cv2.imread(str(path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                overlay, heatmap, pred_class, confidence = self.generate_heatmap(
                    image, transform
                )
                pred_name = class_names[pred_class] if class_names else f"Class {pred_class}"

                axes[i, 0].imshow(image)
                axes[i, 0].set_title(f"Image {i + 1}", fontsize=10)
                axes[i, 0].axis("off")

                axes[i, 1].imshow(heatmap, cmap="jet")
                axes[i, 1].set_title("Grad-CAM", fontsize=10)
                axes[i, 1].axis("off")

                axes[i, 2].imshow(overlay)
                axes[i, 2].set_title(f"{pred_name} ({confidence:.1%})", fontsize=10)
                axes[i, 2].axis("off")
            except Exception as e:
                logger.warning(f"Failed to process {path}: {e}")
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, "Error", ha="center")
                    axes[i, j].axis("off")

        plt.suptitle(
            "Grad-CAM Interpretability Report",
            fontsize=16, fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Batch Grad-CAM visualization saved to {save_path}")

        return fig

    def generate_interpretability_report(
        self,
        image_paths: List[str],
        true_labels: Optional[List[int]] = None,
        transform=None,
        class_names: Optional[List[str]] = None,
        save_dir: str = "outputs/interpretability",
        num_correct: int = 5,
        num_incorrect: int = 5,
    ):
        """
        Generate a full interpretability report showing:
        - Correct predictions with Grad-CAM
        - Incorrect predictions (errors) with Grad-CAM
        - Saved as individual and grid figures

        Args:
            image_paths: List of all image paths.
            true_labels: Corresponding true labels.
            transform: Preprocessing transform.
            class_names: Class name mapping.
            save_dir: Directory to save all outputs.
            num_correct: Number of correct examples to show.
            num_incorrect: Number of incorrect examples to show.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info("Generating interpretability report...")

        if true_labels is not None:
            correct_paths = []
            incorrect_paths = []

            for path, true_label in zip(image_paths, true_labels):
                try:
                    image = cv2.imread(str(path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    _, _, pred, _ = self.generate_heatmap(image, transform)

                    if pred == true_label:
                        correct_paths.append(path)
                    else:
                        incorrect_paths.append(path)
                except Exception:
                    continue

            # Visualize correct predictions
            if correct_paths:
                self.visualize_batch(
                    correct_paths[:num_correct],
                    transform, class_names,
                    save_path=str(save_path / "correct_predictions.png"),
                )

            # Visualize errors (critical for medical review)
            if incorrect_paths:
                self.visualize_batch(
                    incorrect_paths[:num_incorrect],
                    transform, class_names,
                    save_path=str(save_path / "incorrect_predictions.png"),
                )
                logger.warning(
                    f"Found {len(incorrect_paths)} misclassifications. "
                    f"Review saved to {save_path / 'incorrect_predictions.png'}"
                )
        else:
            # Just visualize a sample
            self.visualize_batch(
                image_paths[:num_correct + num_incorrect],
                transform, class_names,
                save_path=str(save_path / "sample_predictions.png"),
            )

        logger.info(f"Interpretability report saved to {save_dir}")
