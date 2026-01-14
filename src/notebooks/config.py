
from pathlib import Path
import torch

# ------------------------
# Project Paths
# ------------------------
# BASE_DIR points to the project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Dataset location
RAW_DATA_DIR = Path("R:/MUL/Courses/Applied ML/raman-mineral-classification/data/raw")

# Output folders (absolute paths)
EXP_DIR_ROOT = BASE_DIR / "experiments"
RESULTS_DIR = BASE_DIR / "results"

# ------------------------
# Device & Seed
# ------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Random Splits
# ------------------------
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ------------------------
# Hyperparameters
# ------------------------
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 5
DROPOUT = 0.3
FREEZE_UNTIL_LAYER = 1   # freeze conv1 + layer1
NUM_WORKERS = 0

# Autoencoder
LATENT_DIM = 64  # latent vector size

# Random Forest
RF_ESTIMATORS = 200
RF_PROFILES_ESTIMATORS = 300

# FPS Options
FPS_LIST = ["1fps", "30fps"]


# Early stopping
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_DELTA = 1e-4

# ------------------------
# Misc Settings
# ------------------------
SAVE_MODEL = True
SAVE_PLOTS = True
VERBOSE = True

# Ensure directories exist
EXP_DIR_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

