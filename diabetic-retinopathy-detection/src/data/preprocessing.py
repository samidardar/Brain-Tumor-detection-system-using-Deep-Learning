"""
Fundus Image Preprocessing Pipeline
====================================
Preprocessing utilities for diabetic retinopathy fundus images.
Includes Ben Graham's method, auto-cropping, quality checks, and
batch processing.

Reference:
    Ben Graham's preprocessing (1st place KAGGLE DR competition):
    https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15801
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def crop_fundus(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Auto-crop black borders around the circular fundus region.

    The fundus image typically sits within a dark (black) background.
    This function detects the circular region and crops tightly around it.

    Args:
        image: Input BGR image (H, W, 3).
        threshold: Pixel intensity threshold to distinguish fundus from background.

    Returns:
        Cropped BGR image containing only the fundus region.
    """
    if image is None or image.size == 0:
        return image

    # Convert to grayscale for thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to create binary mask
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find contours of the fundus region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.warning("No contours found during fundus cropping, returning original.")
        return image

    # Get the largest contour (the fundus)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Add small padding (2% of dimension)
    pad_x = int(w * 0.02)
    pad_y = int(h * 0.02)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(image.shape[1] - x, w + 2 * pad_x)
    h = min(image.shape[0] - y, h + 2 * pad_y)

    cropped = image[y : y + h, x : x + w]
    return cropped


def ben_graham_preprocessing(
    image: np.ndarray,
    sigma: int = 10,
    image_size: int = 512,
) -> np.ndarray:
    """
    Apply Ben Graham's preprocessing method for fundus images.

    This technique was used by the 1st place winner of the Kaggle
    Diabetic Retinopathy Detection competition. It subtracts the local
    average color and adds 128, enhancing local contrast and reducing
    illumination variation.

    Steps:
        1. Resize to target size
        2. Subtract Gaussian-blurred version of the image
        3. Add 128 to re-center pixel values

    Args:
        image: Input BGR image.
        sigma: Gaussian blur sigma (controls locality of averaging).
        image_size: Target image size (square).

    Returns:
        Preprocessed BGR image with enhanced local contrast.
    """
    # Resize image
    image = cv2.resize(image, (image_size, image_size))

    # Compute local average color via Gaussian blur
    # Kernel size must be odd and large enough
    kernel_size = 2 * sigma + 1
    local_avg = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Subtract local average and add 128 (re-center)
    result = cv2.addWeighted(image, 4, local_avg, -4, 128)

    return result


def resize_and_normalize(
    image: np.ndarray,
    size: int = 512,
) -> np.ndarray:
    """
    Resize image to target size (square).

    Note: ImageNet normalization is applied in the DataLoader transforms,
    not here, so that augmentation operates on natural pixel values.

    Args:
        image: Input BGR image.
        size: Target size (image will be resized to size x size).

    Returns:
        Resized BGR image.
    """
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def quality_check(
    image: np.ndarray,
    blur_threshold: float = 100.0,
    dark_threshold: float = 30.0,
) -> Dict[str, Any]:
    """
    Assess the quality of a fundus image.

    Checks for:
        - Blurriness using the Laplacian variance method
        - Darkness using mean pixel intensity
        - Overall contrast via standard deviation

    Args:
        image: Input BGR image.
        blur_threshold: Laplacian variance below this = blurry.
        dark_threshold: Mean intensity below this = too dark.

    Returns:
        Dictionary with quality metrics and pass/fail flags:
        {
            "blur_score": float,
            "mean_intensity": float,
            "contrast": float,
            "is_blurry": bool,
            "is_dark": bool,
            "quality_ok": bool
        }
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur detection via Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Darkness detection
    mean_intensity = float(np.mean(gray))

    # Contrast via standard deviation
    contrast = float(np.std(gray))

    is_blurry = laplacian_var < blur_threshold
    is_dark = mean_intensity < dark_threshold
    quality_ok = not is_blurry and not is_dark

    return {
        "blur_score": float(laplacian_var),
        "mean_intensity": mean_intensity,
        "contrast": contrast,
        "is_blurry": is_blurry,
        "is_dark": is_dark,
        "quality_ok": quality_ok,
    }


def preprocess_single_image(
    image_path: str,
    output_path: str,
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Preprocess a single fundus image through the full pipeline.

    Pipeline:
        1. Load image
        2. Auto-crop black borders (if enabled)
        3. Apply Ben Graham preprocessing (if enabled)
        4. Resize to target size
        5. Quality check (if enabled)
        6. Save processed image

    Args:
        image_path: Path to the input image.
        output_path: Path to save the processed image.
        config: Configuration dictionary with preprocessing settings.

    Returns:
        Quality check results dict, or None if image could not be loaded.
    """
    prep_cfg = config.get("preprocessing", {})
    data_cfg = config.get("data", {})
    image_size = data_cfg.get("image_size", 512)

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return None

    # Step 1: Auto-crop
    if prep_cfg.get("auto_crop", True):
        threshold = prep_cfg.get("crop_threshold", 10)
        image = crop_fundus(image, threshold=threshold)

    # Step 2: Ben Graham preprocessing
    if prep_cfg.get("ben_graham", True):
        sigma = prep_cfg.get("ben_graham_sigma", 10)
        image = ben_graham_preprocessing(image, sigma=sigma, image_size=image_size)
    else:
        # Just resize
        image = resize_and_normalize(image, size=image_size)

    # Step 3: Quality check
    quality_result = None
    if prep_cfg.get("quality_check", True):
        quality_result = quality_check(
            image,
            blur_threshold=prep_cfg.get("blur_threshold", 100.0),
            dark_threshold=prep_cfg.get("dark_threshold", 30.0),
        )

    # Save processed image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(str(output_path), image)

    return quality_result


def process_dataset(
    input_dir: str,
    output_dir: str,
    config: Dict[str, Any],
    labels_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Batch-preprocess an entire dataset of fundus images.

    Args:
        input_dir: Directory containing raw images.
        output_dir: Directory to save processed images.
        config: Configuration dictionary.
        labels_df: Optional DataFrame with 'id_code' and 'diagnosis' columns.
                   If None, processes all images in input_dir.

    Returns:
        DataFrame with columns: id_code, diagnosis (if provided),
        quality metrics, and processing status.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_format = config.get("data", {}).get("image_format", "png")

    # Collect image files
    if labels_df is not None:
        image_files = [
            (row["id_code"], input_path / f"{row['id_code']}.{image_format}")
            for _, row in labels_df.iterrows()
        ]
    else:
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.tiff"]
        all_files = []
        for ext in extensions:
            all_files.extend(input_path.glob(ext))
        image_files = [(f.stem, f) for f in all_files]

    results = []

    logger.info(f"Processing {len(image_files)} images from {input_dir}")
    for id_code, img_path in tqdm(image_files, desc="Preprocessing"):
        out_file = output_path / f"{id_code}.{image_format}"

        quality = preprocess_single_image(str(img_path), str(out_file), config)

        record = {"id_code": id_code, "processed": quality is not None}
        if quality:
            record.update(quality)

        # Add label if available
        if labels_df is not None:
            match = labels_df[labels_df["id_code"] == id_code]
            if len(match) > 0:
                record["diagnosis"] = match.iloc[0]["diagnosis"]

        results.append(record)

    results_df = pd.DataFrame(results)

    # Log quality summary
    if "quality_ok" in results_df.columns:
        total = len(results_df)
        passed = results_df["quality_ok"].sum()
        logger.info(
            f"Quality check: {passed}/{total} images passed "
            f"({passed / total * 100:.1f}%)"
        )
        blurry = results_df["is_blurry"].sum()
        dark = results_df["is_dark"].sum()
        if blurry > 0:
            logger.warning(f"  → {blurry} blurry images detected")
        if dark > 0:
            logger.warning(f"  → {dark} dark images detected")

    return results_df
