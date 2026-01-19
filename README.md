# Brain Tumor Segmentation (U-Net)

![Project Banner](outputs/training_curves.png)

## 📌 Overview
This project implements a **Brain Tumor Segmentation** system using a custom **U-Net** architecture in PyTorch. It detects and segments brain tumors in MRI scans with ~90% precision.

The system is optimized for consumer hardware (e.g., GTX 1650 4GB VRAM) and includes a **Streamlit** web interface for easy interaction.

## 🚀 Features
- **High Performance:** ~90% Precision, ~86% Dice Score
- **Resource Efficient:** Optimized for 4GB VRAM GPUs (Batch Size 4, Float32, GroupNorm)
- **Interactive UI:** Streamlit-based web app for real-time inference
- **Robust:** Handles class imbalance with Dice+BCE Loss
- **Medical Grade:** Patient-level split to prevent data leakage

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/brain-tumor-segmentation.git
   cd brain-tumor-segmentation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📂 Dataset
The project uses the **LGG MRI Segmentation Dataset** from The Cancer Genome Atlas (TCGA).
- **Download:** [Kaggle Link](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- **Structure:**
  ```
  dataset/
  ├── kaggle_3m/
  │   ├── TCGA_CS_4941_19960909/
  │   │   ├── ..._1.tif
  │   │   ├── ..._1_mask.tif
  │   └── ...
  ```

## 🚅 Training
To train the model from scratch:

```bash
python brain_tumor_segmentation.py --data_path "/path/to/dataset" --epochs 50 --batch_size 4 --no_amp
```

## 🖥️ Usage
Run the web interface to analyze new images:

```bash
streamlit run streamlit_app.py
```

## 📊 Results

| Metric | Value |
|--------|-------|
| Precision | 90% |
| Recall | 85% |
| Dice Score | 86% |
| IoU | 76% |

## 🏗️ Model Architecture
- **Type:** U-Net (Encoder-Decoder)
- **Input:** 3-channel RGB MRI slices (256x256)
- **Output:** Binary segmentation mask
- **Key Components:** GroupNorm, ReLU, Sigmoid, Skip Connections

## 📄 License
This project is licensed under the MIT License.
