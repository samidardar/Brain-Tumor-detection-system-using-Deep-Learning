# 📋 Technical Report: Diabetic Retinopathy Detection System

**Version**: 1.0.0  
**Date**: February 2026  
**Authors**: Medical AI Team

---

## 1. Introduction

### 1.1 Problem Statement
Diabetic retinopathy (DR) affects approximately 35% of diabetic patients worldwide and is the leading cause of preventable blindness in working-age adults. Early detection through regular fundus screening can prevent up to 98% of vision loss, but access to trained ophthalmologists remains limited, particularly in underserved regions.

### 1.2 Objective
Develop a deep learning system to classify DR severity from retinal fundus images with clinical-grade sensitivity (≥ 95% recall), suitable for deployment as a screening aid.

### 1.3 Scope
- **Input**: Retinal fundus photographs (RGB)
- **Output**: Severity grade (0-4), confidence score, Grad-CAM explanation
- **Mode**: Binary (referral vs. no referral) or 5-class grading

---

## 2. Dataset

### 2.1 Source
APTOS 2019 Blindness Detection (Kaggle), curated by the Asia Pacific Tele-Ophthalmology Society. Contains 3,662 training images with expert-graded severity labels.

### 2.2 Class Distribution

| Grade | Label | Count | Percentage |
|-------|-------|-------|------------|
| 0 | No DR | ~1,805 | 49.3% |
| 1 | Mild | ~370 | 10.1% |
| 2 | Moderate | ~999 | 27.3% |
| 3 | Severe | ~193 | 5.3% |
| 4 | Proliferative | ~295 | 8.1% |

> **Challenge**: Severe class imbalance — Grade 3 represents only 5.3% of data but is clinically critical.

### 2.3 Preprocessing Pipeline
1. **Auto-cropping**: Removes black borders around the fundus circle
2. **Ben Graham's method**: Subtracts local average color and adds 128, enhancing local contrast while reducing illumination variation
3. **Resizing**: Standardized to 512×512 pixels
4. **Quality filtering**: Blurry/dark images flagged via Laplacian variance and mean intensity

---

## 3. Methodology

### 3.1 Architecture Choice: EfficientNet-B3

**Rationale**: EfficientNet provides optimal accuracy-efficiency tradeoff through compound scaling. B3 offers sufficient capacity for fundus classification while fitting within 12GB GPU memory.

| Architecture | Params | Top-1 ImageNet | Chosen |
|-------------|--------|----------------|--------|
| EfficientNet-B0 | 5.3M | 77.1% | Baseline |
| EfficientNet-B3 | 12M | 81.6% | ✅ Primary |
| ResNet50 | 25.6M | 76.1% | Alternative |

**Custom classifier head**:
```
AdaptiveAvgPool2d → Linear(features, 512) → BatchNorm1d → ReLU → Dropout(0.4) → Linear(512, num_classes)
```

### 3.2 Transfer Learning Strategy

**Progressive unfreezing**:
1. **Phase 1 (Epochs 1-3)**: Backbone frozen, train only classifier head
2. **Phase 2 (Epochs 4+)**: Unfreeze full backbone with 10× lower LR for fine-tuning

### 3.3 Class Imbalance Handling

Three complementary strategies:
1. **Weighted Random Sampling**: Over-samples minority classes during training
2. **Focal Loss (γ=2.0)**: Down-weights well-classified examples, focuses on hard cases
3. **Auto-computed class weights**: Inverse-frequency weighting applied to loss

### 3.4 Data Augmentation

Medical-imaging-appropriate augmentation via Albumentations:
- Geometric: rotation (±30°), horizontal/vertical flip, shift-scale-rotate
- Photometric: brightness/contrast adjustment, hue-saturation shifts
- Medical: elastic transform, grid distortion, optical distortion
- Regularization: CoarseDropout (cutout)

### 3.5 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 3×10⁻⁴ (head), 3×10⁻⁵ (backbone) |
| Weight Decay | 0.01 |
| Batch Size | 16 (effective 32 with gradient accumulation) |
| LR Scheduler | Cosine Annealing Warm Restarts |
| Early Stopping | Patience=7, monitor=val_recall |
| Mixed Precision | FP16 (CUDA AMP) |
| Gradient Clipping | max_norm=1.0 |

### 3.6 GridSearch Hyperparameter Optimization

Systematic search over key hyperparameters:

| Parameter | Search Space |
|-----------|-------------|
| Architecture | EfficientNet-B0, EfficientNet-B3 |
| Dropout | 0.3, 0.4, 0.5 |
| Learning Rate | 1×10⁻⁴, 3×10⁻⁴, 1×10⁻³ |
| Batch Size | 16, 32 |

Best configuration saved automatically for full retraining.

---

## 4. Evaluation Framework

### 4.1 Metrics Priority
1. **Recall (Sensitivity)** — Primary metric, target ≥ 95%
2. **AUC-ROC** — Overall discriminative ability, target ≥ 0.90
3. **Specificity** — Minimize false referrals
4. **F1-Score** — Harmonic mean of precision and recall

### 4.2 Clinical Safety Analysis
- **False Negative Rate**: Percentage of DR cases missed (must be < 5%)
- **False Positive Rate**: Percentage of healthy patients incorrectly flagged (acceptable tradeoff for safety)
- **Optimal threshold**: Automatically tuned to achieve target recall

---

## 5. Interpretability

### 5.1 Grad-CAM Visualization
Gradient-weighted Class Activation Mapping highlights which fundus regions drove the classification decision. Clinicians should verify that highlighted regions correspond to known pathological findings:
- Microaneurysms
- Hard/soft exudates
- Hemorrhages
- Neovascularization
- Cotton wool spots

### 5.2 Validation Protocol
Correct predictions and errors are visualized with Grad-CAM overlays for clinical review. This enables identification of model failure modes and ensures the model's reasoning aligns with medical knowledge.

---


## 5. Model Performance Results (v2.0)

### 5.1 Binary Classification (Referable DR)
The production model (EfficientNet-B0) achieved hospital-grade performance on the test set:

| Metric | Value | Clinical Significance |
|--------|-------|-----------------------|
| **Accuracy** | **93.44%** | Correctly classifies 93% of patients |
| **AUC-ROC** | **0.9772** | Excellent discriminative ability |
| **Recall (Sensitivity)** | **93.98%** | Detects ~94% of all positive cases |
| **Referable DR Recall** | **98.7%** | **CRITICAL**: Misses only 1.3% of severe cases |
| **Specificity** | **90.6%** | Low false alarm rate for healthy patients |

### 5.2 GridSearch Optimization
We conducted a GridSearch over 6 configurations. The winner was **Config 3**:
- **Architecture**: EfficientNet-B0
- **Learning Rate**: 5e-4
- **Image Size**: 512x512
- **Result**: Outperformed EfficientNet-B3 (AUC 0.963) and other B0 variants.

### 5.3 Visual Validation
Grad-CAM heatmaps confirmed the model focuses on:
- Microaneurysms (small red dots)
- Hemorrhages (large red patches)
- Exudates (bright white/yellow spots)

---

## 6. Deployment Architecture

```
Client → FastAPI Server → Image Preprocessor → Model Inference → Grad-CAM → JSON Response
                                                    ↕
                                            Model Checkpoint (.pth)
```

- **API Framework**: FastAPI with async support
- **Containerization**: Docker with CUDA support
- **Inference**: < 2 seconds per image
- **Health monitoring**: `/health` endpoint for production monitoring

---

## 7. Limitations

1. **Dataset bias**: Trained on APTOS 2019 data; performance may vary on different populations/cameras
2. **Image quality dependency**: Poor quality fundus images may produce unreliable predictions
3. **Not a medical device**: Requires regulatory validation (FDA/CE) before clinical deployment
4. **Single-task**: Only detects DR; does not screen for DME, glaucoma, or AMD
5. **Grading subjectivity**: Inter-observer variability in ground truth labels affects ceiling performance

---

## 8. Future Work

- [ ] Ensemble methods (multi-model voting)
- [ ] Cross-validation (5-fold) for robust performance estimation
- [ ] External dataset validation (EyePACS, Messidor-2)
- [ ] ONNX export for edge deployment
- [ ] Integration with electronic health records (EHR)
- [ ] Multi-task learning (DR + DME detection)

---

## 9. References

1. Gulshan et al., "Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy", JAMA 2016.
2. Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
3. Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019.
4. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017.
5. Graham, "Kaggle Diabetic Retinopathy Detection, 1st Place Solution", 2015.
