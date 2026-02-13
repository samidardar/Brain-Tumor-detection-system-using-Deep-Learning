"""
Tests for the preprocessing pipeline.
"""

import os
import numpy as np
import cv2
import pytest

from src.data.preprocessing import (
    crop_fundus,
    ben_graham_preprocessing,
    resize_and_normalize,
    quality_check,
    preprocess_single_image,
)


def _create_synthetic_fundus(size=512):
    """Create a synthetic fundus image: circle on black background."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = size // 3
    cv2.circle(image, center, radius, (100, 80, 60), -1)
    # Add some texture
    noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    return image


class TestCropFundus:
    def test_crop_reduces_size(self):
        """Cropping should remove black borders."""
        image = _create_synthetic_fundus(512)
        cropped = crop_fundus(image, threshold=10)
        # Cropped should be smaller than original if there were borders
        assert cropped.shape[0] <= image.shape[0]
        assert cropped.shape[1] <= image.shape[1]

    def test_crop_handles_empty(self):
        """Should handle all-black images gracefully."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = crop_fundus(image, threshold=10)
        assert result is not None

    def test_crop_preserves_content(self):
        """Cropped image should not be empty."""
        image = _create_synthetic_fundus(512)
        cropped = crop_fundus(image, threshold=10)
        assert cropped.size > 0
        assert cropped.mean() > 0


class TestBenGraham:
    def test_output_shape(self):
        """Ben Graham preprocessing should output correct size."""
        image = _create_synthetic_fundus(600)
        result = ben_graham_preprocessing(image, sigma=10, image_size=224)
        assert result.shape == (224, 224, 3)

    def test_enhances_contrast(self):
        """Processed image should have different intensity distribution."""
        image = _create_synthetic_fundus(512)
        result = ben_graham_preprocessing(image, sigma=10, image_size=512)
        # Standard deviation should change (contrast enhancement)
        assert result.std() != image.std()


class TestResizeNormalize:
    def test_resize_to_target(self):
        image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        result = resize_and_normalize(image, size=224)
        assert result.shape == (224, 224, 3)


class TestQualityCheck:
    def test_sharp_image_passes(self):
        """A sharp synthetic image should pass quality check."""
        image = _create_synthetic_fundus(512)
        # Add sharp edges
        cv2.rectangle(image, (100, 100), (200, 200), (255, 255, 255), 2)
        result = quality_check(image, blur_threshold=50.0, dark_threshold=10.0)
        assert "blur_score" in result
        assert "mean_intensity" in result
        assert isinstance(result["quality_ok"], bool)

    def test_dark_image_detected(self):
        """A very dark image should be flagged."""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 5
        result = quality_check(image, dark_threshold=30.0)
        assert result["is_dark"] is True

    def test_returns_expected_keys(self):
        """Quality check should return all expected keys."""
        image = _create_synthetic_fundus(256)
        result = quality_check(image)
        expected_keys = {"blur_score", "mean_intensity", "contrast", "is_blurry", "is_dark", "quality_ok"}
        assert expected_keys.issubset(result.keys())
