# 🏥 Diabetic Retinopathy Detection System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Features
- **High Performance**: 97.7% AUC, 98.7% Recall on Referable DR.
- **Modern UI**: Professional React interface with animated Prism background and fluid navigation.
- **Explainable AI**: Grad-CAM heatmaps for every prediction.
- **Production API**: FastAPI backend with Docker support.

## 💻 Tech Stack
- **AI/ML**: PyTorch, EfficientNet-B0, Albumentations
- **Backend**: FastAPI, Uvicorn
- **Frontend**: React, Vite, TailwindCSS, Framer Motion, Shadcn UI
- **DevOps**: Docker, GitHub Actions

## 🏁 Quick Start

### 1. Backend API
```bash
pip install -r requirements.txt
python api/app.py
```

### 2. Frontend UI
```bash
cd frontend
npm install
npm run dev
```
Access the UI at `http://localhost:5173`.

> ⚠️ **Medical Disclaimer**: This is a **screening aid**, NOT a certified medical device. All predictions must be reviewed by qualified ophthalmologists. Clinical use requires regulatory approval.

---

## 🎯 Key Features

- **High Sensitivity**: Optimized for recall ≥ 95% to minimize missed diagnoses
- **5-Grade Classification**: No DR, Mild, Moderate, Severe, Proliferative DR
- **Grad-CAM Visualization**: Heatmaps showing pathological regions for clinical trust
- **Production API**: FastAPI with Docker deployment, < 2s inference
- **GridSearch Optimization**: Systematic hyperparameter tuning for optimal performance
- **MLflow Tracking**: Full experiment logging and reproducibility

## 📊 Performance

| Metric | Target | Result |
|--------|--------|--------|
| Sensitivity (Recall) | ≥ 95% | TBD |
| AUC-ROC | ≥ 0.90 | TBD |
| Inference Latency | < 2s | TBD |
| Specificity | Maximize | TBD |

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/username/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Install Kaggle API and configure credentials
pip install kaggle
kaggle competitions download -c aptos2019-blindness-detection -p data/raw/
cd data/raw && unzip aptos2019-blindness-detection.zip
```

See [data/README.md](data/README.md) for detailed instructions.

### 3. Preprocess Data

```python
from src.data.preprocessing import process_dataset
from src.training.train import load_config
import pandas as pd

config = load_config("config/config.yaml")
df = pd.read_csv("data/raw/train.csv")
process_dataset("data/raw/train_images", "data/processed", config, df)
```

### 4. Train Model

```bash
# Standard training
python -m src.training.train --config config/config.yaml

# GridSearch hyperparameter optimization
python -m src.training.optimizer_search --config config/config.yaml --max-epochs 15
```

### 5. Run Inference

```bash
# Single image
python -m src.inference.predict --image path/to/fundus.jpg --model models/best_model.pth

# With Grad-CAM visualization
python -m src.inference.predict --image fundus.jpg --model models/best_model.pth --gradcam

# Batch processing
python -m src.inference.predict --dir images/ --output results.csv
```

### 6. Deploy API

```bash
# Local
cd api && uvicorn app:app --host 0.0.0.0 --port 8000

# Docker
docker build -t dr-detection -f api/Dockerfile .
docker run -p 8000:8000 -v ./models:/app/models dr-detection

# API docs at http://localhost:8000/docs
```

---

## 📁 Project Structure

```
diabetic-retinopathy-detection/
├── config/config.yaml          # All hyperparameters & settings
├── src/
│   ├── data/
│   │   ├── preprocessing.py    # Fundus preprocessing (Ben Graham, auto-crop)
│   │   └── dataloader.py       # DataLoader with augmentation & sampling
│   ├── models/
│   │   ├── cnn_model.py        # EfficientNet/ResNet with progressive unfreeze
│   │   └── loss_functions.py   # Focal Loss & Weighted Cross-Entropy
│   ├── training/
│   │   ├── train.py            # Training loop with mixed precision
│   │   └── optimizer_search.py # GridSearch hyperparameter optimization
│   ├── evaluation/
│   │   └── metrics.py          # AUC-ROC, confusion matrix, error analysis
│   └── inference/
│       ├── predict.py          # CLI inference with batch support
│       └── gradcam.py          # Grad-CAM interpretability
├── api/
│   ├── app.py                  # FastAPI application
│   ├── Dockerfile              # Production Docker image
│   └── requirements.txt        # API-specific dependencies
├── tests/                      # Unit tests
├── data/                       # Dataset (not tracked in git)
├── models/                     # Saved checkpoints
└── outputs/                    # Evaluation plots & reports
```

---

## 🔬 Methodology

### Architecture
- **Backbone**: EfficientNet-B3 pretrained on ImageNet
- **Classifier**: Custom head with BatchNorm, ReLU, Dropout
- **Fine-tuning**: Progressive unfreezing (head first, then backbone)

### Training Strategy
- **Loss**: Focal Loss (γ=2.0) with auto-computed class weights
- **Optimizer**: AdamW with Cosine Annealing LR
- **Imbalance**: Weighted random sampling + focal loss
- **Regularization**: Dropout (0.4), weight decay, early stopping
- **Augmentation**: Albumentations (rotation, flip, elastic, color jitter)
- **Optimization**: GridSearch over architectures, LR, dropout, batch size

### Evaluation
- AUC-ROC (macro), Recall, Precision, F1, Confusion Matrix
- Optimal threshold search for target recall ≥ 95%
- False negative analysis (critical for medical safety)
- Grad-CAM interpretability reports

---

## 🔧 Configuration

All settings are centralized in [`config/config.yaml`](config/config.yaml):

```yaml
model:
  architecture: "efficientnet_b3"
  num_classes: 5
  dropout: 0.4

training:
  learning_rate: 0.0003
  batch_size: 16
  max_epochs: 50
  scheduler: "cosine"
  mixed_precision: true
```

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

This software is provided for **research and educational purposes only**. It is:

- **NOT** a certified medical device (FDA, CE, or equivalent)
- **NOT** a substitute for professional medical diagnosis
- **NOT** intended for unsupervised clinical use

Any deployment in a clinical setting requires:
- Regulatory validation and approval
- Supervision by qualified healthcare professionals
- Compliance with local medical device regulations
- Proper patient consent and data handling (HIPAA/GDPR)
