# Deployment Guide

## Prerequisites
- Docker installed
- Trained model checkpoint at `models/best_model.pth`
- NVIDIA GPU (optional, for faster inference)

## Docker Deployment

### Build
```bash
docker build -t dr-detection -f api/Dockerfile .
```

### Run (CPU)
```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e MODEL_PATH=/app/models/best_model.pth \
  dr-detection
```

### Run (GPU)
```bash
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  dr-detection
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/best_model.pth` | Path to model checkpoint |
| `IMAGE_SIZE` | `512` | Input image size |
| `ENABLE_GRADCAM` | `true` | Enable Grad-CAM visualization |
| `MAX_IMAGE_SIZE_MB` | `10` | Max upload size |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Upload fundus image → prediction + Grad-CAM |
| `/health` | GET | Health check |
| `/model/info` | GET | Model metadata |
| `/docs` | GET | Swagger documentation |

## Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -F "file=@fundus_image.jpg" \
  -F "include_gradcam=true"

# Or use the Swagger UI at http://localhost:8000/docs
```

## Monitoring
- Use the `/health` endpoint for uptime monitoring
- Check container logs: `docker logs <container_id>`
- Set up alerts on HTTP 5xx responses
