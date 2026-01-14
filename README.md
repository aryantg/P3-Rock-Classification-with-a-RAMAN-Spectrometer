# Raman Mineral Classification

A comprehensive machine learning project for classifying minerals using Raman spectroscopy data with multiple deep learning and classical approaches.

**Classification Targets:** 7 mineral types (Albit, Calcite, Dolomit, Feldspat, Quarz, Rhodocrosite, Tile)

**Data:** Available at both 1 FPS and 30 FPS sampling rates for comprehensive evaluation

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Project Structure](#project-structure)
4. [Execution Pipeline](#execution-pipeline)
5. [Models Overview](#models-overview)
6. [Results](#results)
7. [Advanced Usage](#advanced-usage)

---

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional)

### Setup

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Configure paths and hyperparameters** (see [Configuration](#configuration) section)

3. **Run the complete pipeline:**

```bash
# Execute notebooks in order (from src/notebooks/)
jupyter notebook 01_eda_preprocessing.ipynb
jupyter notebook 02_supervised_resnet.ipynb
jupyter notebook 03_unsupervised_autoencoder.ipynb
jupyter notebook 04_profiles_classification.ipynb
jupyter notebook 05_testing_and_comparison.ipynb
```

## Inference

After training, you can run inference on a single Raman spectral image using the script:

```bash
python src/scripts/inference.py
```

At the top of the script, set:

# Path to the BMP image for inference

IMAGE_PATH

# Model type: "resnet" or "autoencoder"

MODEL_TYPE = "resnet"

# Experiment frame rate: "1fps" or "30fps"

FPS_TYPE = "30fps"

---

## Configuration

### Main Configuration File: `src/notebooks/config.py`

This is the **central configuration file** where you control all paths, hyperparameters, and settings.

#### Paths to Update (Critical)

```python
# Line 11: Update to your data directory
RAW_DATA_DIR = Path("R:/MUL/Courses/Applied ML/raman-mineral-classification/data/raw")

# Lines 14-15: Experiment and results output directories (auto-created)
EXP_DIR_ROOT = BASE_DIR / "experiments"
RESULTS_DIR = BASE_DIR / "results"
```

#### Hyperparameters (Tunable)

```python
# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-detects GPU
SEED = 42  # For reproducibility

# Data splitting
VAL_RATIO = 0.1    # 10% validation
TEST_RATIO = 0.1   # 10% test

# Training
BATCH_SIZE = 32        # Training batch size
TEST_BATCH_SIZE = 16   # Evaluation batch size
EPOCHS = 5             # Number of training epochs
LR = 3e-4              # Learning rate
DROPOUT = 0.4          # Dropout rate

# Model-specific
LATENT_DIM = 64              # Autoencoder latent dimension
RF_ESTIMATORS = 200          # Random Forest trees (supervised)
RF_PROFILES_ESTIMATORS = 300 # Random Forest trees (profile-based)

# Early stopping
EARLY_STOPPING_PATIENCE = 5   # Patience for early stopping
EARLY_STOPPING_DELTA = 1e-4   # Minimum improvement threshold

# FPS modes (do not change unless data structure changes)
FPS_LIST = ["1fps", "30fps"]
```

#### Flags

```python
SAVE_MODEL = True   # Save trained models
SAVE_PLOTS = True   # Save visualizations
VERBOSE = True      # Detailed logging
```

---

## Project Structure

```
raman-mineral-classification/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── src/
│   ├── __init__.py
│   ├── notebooks/
│   │   ├── config.py                  # ⚙️ MAIN CONFIG FILE
│   │   ├── 01_eda_preprocessing.ipynb          # Data exploration & preprocessing
│   │   ├── 02_supervised_resnet.ipynb          # ResNet18 supervised learning
│   │   ├── 03_unsupervised_autoencoder.ipynb   # Autoencoder + Random Forest
│   │   ├── 04_profiles_classification.ipynb    # Profile-based classification
│   │   └── 05_testing_and_comparison.ipynb     # Final model comparison
│   ├── datasets/
│   │   └── raman_dataset.py           # Dataset loading utilities
│   ├── models/
│   │   └── autoencoder.py             # Autoencoder architecture
│   ├── utils/
│   │   ├── metrics.py                 # Evaluation metrics
│   │   ├── plots.py                   # Visualization utilities
│   │   └── seed.py                    # Reproducibility helpers
│   └── scripts/
│       ├── inference.py               # Single image inference
│       ├── test_resnet_models.py      # ResNet evaluation script
│       ├── test_autoencoder_unsupervised.py  # AE evaluation script
│       └── test_profiles_rf.py        # Profile-based evaluation script
├── data/
│   └── raw/
│       ├── Albit/          {1fps/, 30fps/} with .bmp images
│       ├── Calcite/        {1fps/, 30fps/} with .bmp images
│       ├── Dolomit/        {1fps/, 30fps/} with .bmp images
│       ├── Feldspat/       {1fps/, 30fps/} with .bmp images
│       ├── Quarz/          {1fps/, 30fps/} with .bmp images
│       ├── Rhodocrosite/   {1fps/, 30fps/} with .bmp images
│       └── Tile/           {1fps/, 30fps/} with .bmp images
├── experiments/
│   ├── resnet/
│   │   ├── 1fps/  {resnet.pt, metrics.json}
│   │   └── 30fps/ {resnet.pt, metrics.json}
│   ├── autoencoder_rf/
│   │   ├── 1fps/  {autoencoder_best.pt, rf.pkl, metrics.json}
│   │   └── 30fps/ {autoencoder_best.pt, rf.pkl, metrics.json}
│   └── profiles_random_forest/
│       ├── 1fps/  {rf.pkl, metrics.json}
│       └── 30fps/ {rf.pkl, metrics.json}
└── results/
    ├── resnet/
    │   ├── test_results_1fps.json
    │   ├── test_results_30fps.json
    │   ├── confusion_matrix_1fps.png
    │   └── confusion_matrix_30fps.png
    ├── autoencoder_rf/
    │   ├── test_results_1fps.json
    │   ├── test_results_30fps.json
    │   ├── confusion_matrix_1fps.json
    │   └── confusion_matrix_30fps.json
    └── profiles_rf/
        ├── test_results_1fps.json
        ├── test_results_30fps.json
        └── test_summary.json
```

---

## Execution Pipeline

### Recommended Order (Sequential)

```
1️⃣  01_eda_preprocessing.ipynb
    └─ Explore data, visualize distributions, check data quality
    └─ Outputs: Summary statistics, visualization plots

2️⃣  02_supervised_resnet.ipynb
    └─ Train ResNet18 for both FPS settings
    └─ Outputs: resnet.pt models, metrics.json, confusion matrix images
    └─ Expected Accuracy: ~95-100% (high quality spectral data)

3️⃣  03_unsupervised_autoencoder.ipynb
    └─ Train autoencoder for unsupervised feature learning
    └─ Apply Random Forest on latent features
    └─ Outputs: autoencoder_best.pt, rf.pkl, test results
    └─ Expected Accuracy: ~95-100%

4️⃣  04_profiles_classification.ipynb
    └─ Extract Raman profile statistics (mean, std, max, etc.)
    └─ Train Random Forest on profile features
    └─ Outputs: rf.pkl, test results
    └─ Expected Accuracy: ~90-95%

5️⃣  05_testing_and_comparison.ipynb
    └─ Load all model results from results/ directory
    └─ Create comparison visualizations
    └─ Generate summary report
    └─ Outputs: model_comparison.png, results_summary.csv
```

---

## Models Overview

### Model 1: ResNet18 (Supervised)

- **Architecture:** ResNet18 pretrained on ImageNet, fine-tuned
- **Input:** 224×224 BMP images
- **Approach:** Direct image classification
- **Advantages:** Fast inference, simple pipeline
- **Configuration:** `src/notebooks/02_supervised_resnet.ipynb`

### Model 2: Autoencoder + Random Forest (Unsupervised)

- **Architecture:** CNN Autoencoder → Latent Features → Random Forest
- **Input:** 224×224 BMP images
- **Approach:** Unsupervised feature learning, then RF classifier
- **Advantages:** Captures underlying data distribution
- **Configuration:** `src/notebooks/03_unsupervised_autoencoder.ipynb`

### Model 3: Random Forest on Raman Profiles (Classical)

- **Architecture:** Profile statistics extraction → Random Forest
- **Input:** Extracted Raman spectral profiles (CSV format)
- **Approach:** Traditional ML on handcrafted features
- **Advantages:** Interpretable, fast
- **Configuration:** `src/notebooks/04_profiles_classification.ipynb`

---

## Results

All results are automatically saved to the `results/` directory after running each notebook.

### Output Files

- **test*results*\*.json** – Accuracy, confusion matrix, class info
- **confusion*matrix*\*.png** – Heatmap visualization
- **model_comparison.png** – Side-by-side model comparison
- **results_summary.csv** – CSV summary of all models

### Expected Performance (7-class classification)

- ResNet: **100% accuracy** (1fps & 30fps)
- Autoencoder+RF: **100% accuracy** (1fps & 30fps)
- Profile RF: **~95% accuracy** (1fps & 30fps)

---

## Advanced Usage

### Running Individual Models

```bash
# Python scripts (alternative to notebooks)
python src/scripts/test_resnet_models.py
python src/scripts/test_autoencoder_unsupervised.py
python src/scripts/test_profiles_rf.py
```

### Single Image Inference

```bash
python src/scripts/inference.py \
    --image path/to/image.bmp \
    --experiment resnet \
    --fps 1fps
```

### Modifying Hyperparameters

**Before running notebooks:**

1. Edit `src/notebooks/config.py`
2. Change desired parameters (EPOCHS, LR, BATCH_SIZE, etc.)
3. Run notebooks as usual – they auto-load from config.py

**Example: Training for 10 epochs with higher learning rate**

```python
# src/notebooks/config.py
EPOCHS = 10           # Changed from 5
LR = 5e-4             # Changed from 3e-4
BATCH_SIZE = 64       # Changed from 32
```

### Reproducibility

- All training uses fixed `SEED = 42`
- Same splits across all FPS settings
- Deterministic PyTorch operations (`torch.manual_seed()`)

---

## Troubleshooting

### GPU Not Detected

```python
# In config.py, manually override:
DEVICE = "cpu"  # Force CPU
```

### Out of Memory Errors

```python
# In config.py, reduce batch size:
BATCH_SIZE = 16  # Smaller batches
```

### Data Path Issues

```python
# Ensure data directory exists and is correctly set in config.py
# Windows example:
RAW_DATA_DIR = Path("C:/path/to/data/raw")
```

### Model Not Found Errors

- Ensure previous notebooks have run successfully
- Check that `experiments/` directory exists with trained models

---

## Dependencies

See `requirements.txt` for full list:

- **Deep Learning:** torch, torchvision
- **ML:** scikit-learn, numpy, pandas
- **Visualization:** matplotlib, seaborn
- **Utils:** Pillow, opencv-python, joblib, tqdm

---

## License & Attribution

Applied ML Course Project - Raman Mineral Classification

---

**Last Updated:** January 2026
