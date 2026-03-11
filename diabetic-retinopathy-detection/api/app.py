"""
FastAPI Application for Diabetic Retinopathy Detection
======================================================
Production-ready REST API for fundus image screening.

Endpoints:
    - POST /predict       → Single image prediction + Grad-CAM
    - POST /predict/batch → Batch prediction
    - GET  /health        → Service health check
    - GET  /model/info    → Model metadata

Swagger docs available at /docs

DISCLAIMER: This API is a screening aid, NOT a medical device.
"""

import os
import io
import sys
import time
import base64
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import load_model, predict_single, CLASS_NAMES_MULTI, CLASS_NAMES_BINARY
from src.data.dataloader import get_val_transforms

logger = logging.getLogger(__name__)

# ============================================================================
# Global state
# ============================================================================
model = None
model_metadata = None
device = None
transform = None
class_names = None
gradcam_visualizer = None

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pth")
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "512"))
ENABLE_GRADCAM = os.environ.get("ENABLE_GRADCAM", "true").lower() == "true"
MAX_IMAGE_SIZE_MB = int(os.environ.get("MAX_IMAGE_SIZE_MB", "10"))


# ============================================================================
# Startup / Shutdown
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, model_metadata, device, transform, class_names, gradcam_visualizer

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Diabetic Retinopathy Detection API...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    try:
        model, model_metadata = load_model(MODEL_PATH, device)
        num_classes = model_metadata["num_classes"]
        class_names = CLASS_NAMES_BINARY if num_classes == 2 else CLASS_NAMES_MULTI
        transform = get_val_transforms({"data": {"image_size": IMAGE_SIZE}})
        logger.info(f"Model loaded: {model_metadata['architecture']}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        # Do not raise, allow app to start in degraded mode
        model = None
        model_metadata = {}

    # Initialize Grad-CAM
    if ENABLE_GRADCAM:
        try:
            from src.inference.gradcam import GradCAMVisualizer
            gradcam_visualizer = GradCAMVisualizer(model, device=device)
            logger.info("Grad-CAM visualizer initialized")
        except Exception as e:
            logger.warning(f"Grad-CAM initialization failed: {e}")

    logger.info("API ready for requests")
    yield
    logger.info("Shutting down API...")


# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description=(
        "AI-assisted screening for diabetic retinopathy from fundus images.\n\n"
        "**⚠️ DISCLAIMER**: This is a screening aid, NOT a certified medical device. "
        "All predictions must be reviewed by qualified ophthalmologists."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Response Models
# ============================================================================
class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    confidence: float
    probabilities: dict
    inference_time_ms: float
    referable: bool
    gradcam_base64: Optional[str] = None
    disclaimer: str = (
        "This is a screening aid. Not a medical diagnosis. "
        "Consult a qualified ophthalmologist."
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    timestamp: str


class ModelInfoResponse(BaseModel):
    architecture: str
    num_classes: int
    class_names: list
    training_epoch: str
    training_metrics: dict
    image_size: int
    gradcam_enabled: bool


# ============================================================================
# Helper functions
# ============================================================================
async def read_image(file: UploadFile) -> np.ndarray:
    """Read and validate an uploaded image."""
    # Check file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large: {size_mb:.1f}MB (max {MAX_IMAGE_SIZE_MB}MB)",
        )

    # Check extension
    allowed = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
    ext = Path(file.filename or "image.jpg").suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {ext}. Allowed: {allowed}",
        )

    # Decode image
    try:
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def generate_gradcam_base64(image_rgb: np.ndarray) -> Optional[str]:
    """Generate Grad-CAM overlay and return as base64 PNG."""
    if gradcam_visualizer is None:
        return None
    try:
        overlay, _, _, _ = gradcam_visualizer.generate_heatmap(image_rgb, transform)
        
        # Ensure uint8
        if overlay.dtype != np.uint8:
            overlay = (overlay * 255).clip(0, 255).astype(np.uint8)
            
        # Convert to base64
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".png", overlay_bgr)
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        logger.error(f"Grad-CAM generation failed: {e}", exc_info=True)
        return None


# ============================================================================
# Endpoints
# ============================================================================
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    file: UploadFile = File(..., description="Fundus image (JPG/PNG)"),
    include_gradcam: bool = True,
):
    """
    Predict diabetic retinopathy grade from a fundus image.

    Returns the severity grade, confidence score, class probabilities,
    and optionally a Grad-CAM heatmap as base64-encoded PNG.
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Read image
        image_bgr = await read_image(file)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess
        start_time = time.time()
        transformed = transform(image=image_rgb)
        input_tensor = transformed["image"].unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_class = outputs.argmax(dim=1).item()
        
        # Custom Thresholding for Screening Sensitivity
        if len(class_names) == 2:
            threshold = 0.40
            dr_prob = probs[1].item()
            if dr_prob >= threshold:
                pred_class = 1
                confidence = dr_prob
            else:
                pred_class = 0
                confidence = probs[0].item()
        else:
            confidence = probs[pred_class].item()
        
        # Determine referability
        is_referable = False
        if len(class_names) == 2:
            is_referable = (pred_class == 1)
        else:
            is_referable = (pred_class >= 2)

        inference_ms = (time.time() - start_time) * 1000

        # Log probabilities for debugging
        logger.info(f"Prediction: {class_names[pred_class]} (Class {pred_class})")
        logger.info(f"Probabilities: {probs.tolist()}")

        # Grad-CAM
        gradcam_b64 = None
        if include_gradcam and ENABLE_GRADCAM:
            gradcam_b64 = generate_gradcam_base64(image_rgb)

        return PredictionResponse(
            predicted_class=pred_class,
            predicted_label=class_names[pred_class],
            confidence=round(confidence, 4),
            probabilities={
                name: round(float(p), 4)
                for name, p in zip(class_names, probs.cpu().numpy())
            },
            inference_time_ms=round(inference_ms, 1),
            referable=is_referable,
            gradcam_base64=gradcam_b64,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", tags=["System"])
async def health_check():
    """Simple health check."""
    return {"status": "ok", "timestamp": str(datetime.now())}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Get model metadata and configuration."""
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        architecture=model_metadata["architecture"],
        num_classes=model_metadata["num_classes"],
        class_names=class_names,
        training_epoch=str(model_metadata.get("epoch", "unknown")),
        training_metrics=model_metadata.get("metrics", {}),
        image_size=IMAGE_SIZE,
        gradcam_enabled=ENABLE_GRADCAM and gradcam_visualizer is not None,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )
