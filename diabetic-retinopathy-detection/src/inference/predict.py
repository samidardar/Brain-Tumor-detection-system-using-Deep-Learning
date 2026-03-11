"""
Inference Pipeline for Diabetic Retinopathy Detection
=====================================================
Standalone prediction module with CLI interface.
Supports single image, batch, and directory-level inference
with optional Grad-CAM visualization.

Usage:
    python -m src.inference.predict --image path/to/fundus.jpg
    python -m src.inference.predict --dir path/to/images/ --output results.csv
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

import cv2
import numpy as np
import torch
import yaml

from src.data.dataloader import get_val_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.models.cnn_model import RetinopathyModel

logger = logging.getLogger(__name__)

# Default class names
CLASS_NAMES_MULTI = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
CLASS_NAMES_BINARY = ["No DR", "DR Present"]


def load_model(
    model_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[RetinopathyModel, Dict[str, Any]]:
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to the .pth checkpoint file.
        device: Target device. Auto-detects if None.

    Returns:
        Tuple of (model, checkpoint_metadata).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract model config from checkpoint
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})

    architecture = checkpoint.get("architecture", model_cfg.get("architecture", "efficientnet_b3"))
    num_classes = checkpoint.get("num_classes", model_cfg.get("num_classes", 5))

    # Reconstruct model
    model = RetinopathyModel(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=False,  # We're loading weights
        dropout=model_cfg.get("dropout", 0.4),
        hidden_dim=model_cfg.get("hidden_dim", 512),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    metadata = {
        "architecture": architecture,
        "num_classes": num_classes,
        "epoch": checkpoint.get("epoch", "unknown"),
        "metrics": checkpoint.get("metrics", {}),
    }

    logger.info(
        f"Model loaded: {architecture} (classes={num_classes}, "
        f"epoch={metadata['epoch']})"
    )

    return model, metadata


def preprocess_image(
    image_path: str,
    image_size: int = 512,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load and preprocess a single image for inference.

    Args:
        image_path: Path to the image file.
        image_size: Target image size.

    Returns:
        Tuple of (preprocessed_tensor, original_rgb_image).
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply validation transforms (resize + normalize)
    config = {"data": {"image_size": image_size}}
    transform = get_val_transforms(config)
    transformed = transform(image=image_rgb)
    tensor = transformed["image"]

    return tensor, image_rgb


def predict_single(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    image_size: int = 512,
    class_names: Optional[List[str]] = None,
    binary_mode: bool = False,
) -> Dict[str, Any]:
    """
    Predict diabetic retinopathy grade for a single fundus image.

    Args:
        image_path: Path to the fundus image.
        model: Trained model in eval mode.
        device: Torch device.
        image_size: Target image size.
        class_names: Optional class name mapping.
        binary_mode: Whether model is binary mode.

    Returns:
        Prediction dictionary with:
        - predicted_class: int
        - predicted_label: str
        - confidence: float (0-1)
        - probabilities: dict mapping class names to probabilities
        - inference_time_ms: float
    """
    if class_names is None:
        class_names = CLASS_NAMES_BINARY if binary_mode else CLASS_NAMES_MULTI

    start_time = time.time()

    # Preprocess
    tensor, _ = preprocess_image(image_path, image_size)
    input_batch = tensor.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class = outputs.argmax(dim=1).item()
        confidence = probabilities[predicted_class].item()

    inference_time = (time.time() - start_time) * 1000  # ms

    # Build result
    result = {
        "image_path": str(image_path),
        "predicted_class": int(predicted_class),
        "predicted_label": class_names[predicted_class],
        "confidence": float(confidence),
        "probabilities": {
            name: float(prob)
            for name, prob in zip(class_names, probabilities.cpu().numpy())
        },
        "inference_time_ms": float(inference_time),
        "timestamp": datetime.now().isoformat(),
    }

    return result


def predict_batch(
    image_paths: List[str],
    model: torch.nn.Module,
    device: torch.device,
    image_size: int = 512,
    class_names: Optional[List[str]] = None,
    binary_mode: bool = False,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    Predict for multiple images with batch processing.

    Args:
        image_paths: List of image paths.
        model: Trained model.
        device: Torch device.
        image_size: Target image size.
        class_names: Optional class names.
        binary_mode: Whether binary classification.
        batch_size: Batch size for inference.

    Returns:
        List of prediction dictionaries.
    """
    if class_names is None:
        class_names = CLASS_NAMES_BINARY if binary_mode else CLASS_NAMES_MULTI

    results = []
    total = len(image_paths)

    for i in range(0, total, batch_size):
        batch_paths = image_paths[i : i + batch_size]
        tensors = []
        valid_indices = []

        for j, path in enumerate(batch_paths):
            try:
                tensor, _ = preprocess_image(path, image_size)
                tensors.append(tensor)
                valid_indices.append(j)
            except Exception as e:
                logger.warning(f"Skipping {path}: {e}")
                results.append({
                    "image_path": str(path),
                    "error": str(e),
                })

        if tensors:
            batch_tensor = torch.stack(tensors).to(device)

            start_time = time.time()
            with torch.no_grad():
                outputs = model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = outputs.argmax(dim=1)
            batch_time = (time.time() - start_time) * 1000

            for k, idx in enumerate(valid_indices):
                pred_class = predicted_classes[k].item()
                conf = probabilities[k, pred_class].item()

                result = {
                    "image_path": str(batch_paths[idx]),
                    "predicted_class": int(pred_class),
                    "predicted_label": class_names[pred_class],
                    "confidence": float(conf),
                    "probabilities": {
                        name: float(p)
                        for name, p in zip(class_names, probabilities[k].cpu().numpy())
                    },
                    "inference_time_ms": float(batch_time / len(valid_indices)),
                }
                results.append(result)

        logger.info(f"Processed {min(i + batch_size, total)}/{total} images")

    return results


def predict_directory(
    input_dir: str,
    model: torch.nn.Module,
    device: torch.device,
    output_path: Optional[str] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Predict for all images in a directory.

    Args:
        input_dir: Directory containing images.
        model: Trained model.
        device: Torch device.
        output_path: Optional CSV/JSON path to save results.
        **kwargs: Additional arguments for predict_batch.

    Returns:
        List of prediction dictionaries.
    """
    extensions = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
    input_path = Path(input_dir)

    image_paths = [
        str(f)
        for f in sorted(input_path.iterdir())
        if f.suffix.lower() in extensions
    ]

    logger.info(f"Found {len(image_paths)} images in {input_dir}")

    if not image_paths:
        logger.warning("No images found!")
        return []

    results = predict_batch(image_paths, model, device, **kwargs)

    # Save results
    if output_path:
        output_ext = Path(output_path).suffix.lower()
        if output_ext == ".json":
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        elif output_ext == ".csv":
            import pandas as pd
            df = pd.DataFrame([
                {
                    "image": r.get("image_path", ""),
                    "predicted_class": r.get("predicted_class", ""),
                    "predicted_label": r.get("predicted_label", ""),
                    "confidence": r.get("confidence", ""),
                }
                for r in results
            ])
            df.to_csv(output_path, index=False)

        logger.info(f"Results saved to {output_path}")

    return results


def main():
    """CLI entry point for inference."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Diabetic Retinopathy Detection - Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.inference.predict --image fundus.jpg --model models/best_model.pth
  python -m src.inference.predict --dir images/ --output results.csv
  python -m src.inference.predict --image fundus.jpg --gradcam --output gradcam.png

DISCLAIMER: This tool is for research and screening assistance only.
Not a certified medical device. All results must be reviewed by a
qualified ophthalmologist.
        """,
    )
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--dir", type=str, help="Path to a directory of images")
    parser.add_argument(
        "--model", type=str, default="models/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument("--output", type=str, help="Output path (JSON/CSV/PNG)")
    parser.add_argument(
        "--gradcam", action="store_true",
        help="Generate Grad-CAM visualization",
    )
    parser.add_argument(
        "--image-size", type=int, default=512,
        help="Input image size (default: 512)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cuda', or 'cpu'",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.image and not args.dir:
        parser.error("Either --image or --dir must be specified")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load model
    model, metadata = load_model(args.model, device)
    num_classes = metadata["num_classes"]
    binary_mode = num_classes == 2
    class_names = CLASS_NAMES_BINARY if binary_mode else CLASS_NAMES_MULTI

    if args.image:
        # Single image prediction
        result = predict_single(
            args.image, model, device,
            image_size=args.image_size,
            class_names=class_names,
            binary_mode=binary_mode,
        )

        # Print result
        print("\n" + "=" * 50)
        print("DIABETIC RETINOPATHY DETECTION RESULT")
        print("=" * 50)
        print(f"Image:       {result['image_path']}")
        print(f"Prediction:  {result['predicted_label']}")
        print(f"Confidence:  {result['confidence']:.1%}")
        print(f"Inference:   {result['inference_time_ms']:.1f} ms")
        print("\nProbabilities:")
        for name, prob in result["probabilities"].items():
            bar = "█" * int(prob * 30)
            print(f"  {name:20s} {prob:6.1%} {bar}")
        print("\n⚠️  DISCLAIMER: This is a screening aid, not a diagnosis.")
        print("    All results must be reviewed by a qualified specialist.")

        # Grad-CAM
        if args.gradcam:
            from src.inference.gradcam import GradCAMVisualizer

            visualizer = GradCAMVisualizer(model, device=device)
            save_path = args.output or "gradcam_output.png"
            visualizer.visualize_single(
                args.image,
                transform=get_val_transforms({"data": {"image_size": args.image_size}}),
                class_names=class_names,
                save_path=save_path,
            )
            print(f"\nGrad-CAM saved to {save_path}")

    elif args.dir:
        # Directory batch prediction
        results = predict_directory(
            args.dir, model, device,
            output_path=args.output,
            image_size=args.image_size,
            class_names=class_names,
            binary_mode=binary_mode,
        )

        # Print summary
        if results:
            from collections import Counter
            label_counts = Counter(r.get("predicted_label", "Error") for r in results)
            print(f"\nProcessed {len(results)} images")
            print("Distribution:")
            for label, count in label_counts.most_common():
                print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
