# ===========================================================
# Diabetic Retinopathy Detection API - Docker Image
# ===========================================================
# Build:  docker build -t dr-detection .
# Run:    docker run -p 8000:8000 -v ./models:/app/models dr-detection
# GPU:    docker run --gpus all -p 8000:8000 dr-detection
# ===========================================================

FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and API
COPY src/ ./src/
COPY api/app.py ./api/app.py
COPY config/ ./config/

# Copy model (if available at build time)
COPY models/ ./models/ 2>/dev/null || true

# Environment variables
ENV MODEL_PATH=/app/models/best_model.pth
ENV IMAGE_SIZE=512
ENV ENABLE_GRADCAM=true
ENV MAX_IMAGE_SIZE_MB=10
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
