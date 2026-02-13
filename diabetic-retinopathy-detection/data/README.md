# Data Directory

## Dataset: APTOS 2019 Blindness Detection

This project uses the [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) dataset from Kaggle.

### Download Instructions

1. **Install Kaggle CLI:**
   ```bash
   pip install kaggle
   ```

2. **Configure API key:**
   - Go to https://www.kaggle.com/settings → Create New Token
   - Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<user>\.kaggle\` (Windows)

3. **Download dataset:**
   ```bash
   kaggle competitions download -c aptos2019-blindness-detection -p data/raw/
   cd data/raw && unzip aptos2019-blindness-detection.zip
   ```

### Dataset Structure

After extraction:
```
data/raw/
├── train.csv           # Labels: id_code, diagnosis (0-4)
├── test.csv            # Test set IDs
├── train_images/       # 3,662 training fundus images
└── test_images/        # Test images
```

### Severity Grades

| Grade | Label             | Description                                    |
|-------|-------------------|------------------------------------------------|
| 0     | No DR             | No signs of diabetic retinopathy               |
| 1     | Mild              | Few microaneurysms                             |
| 2     | Moderate          | Microaneurysms, hemorrhages, hard exudates     |
| 3     | Severe            | Extensive hemorrhages, venous beading           |
| 4     | Proliferative DR  | Neovascularization, vitreous hemorrhage        |

### Preprocessing

Run the preprocessing pipeline to generate optimized images:
```bash
python -c "
from src.data.preprocessing import process_dataset
from src.training.train import load_config
import pandas as pd

config = load_config('config/config.yaml')
df = pd.read_csv('data/raw/train.csv')
process_dataset('data/raw/train_images', 'data/processed', config, df)
"
```

### ⚠️ Data Privacy

- This dataset uses de-identified images
- Do NOT add patient-identifiable data to this directory
- All data handling must comply with HIPAA / GDPR requirements
