"""
Tests for the FastAPI application.
"""

import sys
from pathlib import Path

import pytest

# Note: These tests require a trained model to be present.
# For CI/CD, mock the model loading or use a lightweight test model.

# The following tests validate API structure and error handling
# without requiring a full model.


class TestAPIStructure:
    """Test API configuration and structure."""

    def test_app_imports(self):
        """Verify API module can be imported."""
        # This tests that all dependencies are installed
        from api import app as api_module
        assert hasattr(api_module, "app")

    def test_response_models(self):
        """Verify response models are properly defined."""
        from api.app import PredictionResponse, HealthResponse, ModelInfoResponse
        assert PredictionResponse is not None
        assert HealthResponse is not None
        assert ModelInfoResponse is not None

    def test_health_response_schema(self):
        """Health response should have required fields."""
        from api.app import HealthResponse
        health = HealthResponse(
            status="healthy",
            model_loaded=True,
            device="cpu",
            timestamp="2026-01-01T00:00:00",
        )
        assert health.status == "healthy"
        assert health.model_loaded is True

    def test_prediction_response_schema(self):
        """Prediction response should include disclaimer."""
        from api.app import PredictionResponse
        pred = PredictionResponse(
            predicted_class=0,
            predicted_label="No DR",
            confidence=0.95,
            probabilities={"No DR": 0.95},
            inference_time_ms=50.0,
        )
        assert "screening aid" in pred.disclaimer.lower()
        assert pred.predicted_label == "No DR"
