"""
Standalone Testing Script for Unsupervised Autoencoder + Random Forest Pipeline
================================================================================
Tests the trained autoencoder and random forest models on the same test split.
Generates confusion matrices, classification reports, and accuracy metrics.

Results Storage Location (from notebook):
  - Models: experiments/autoencoder_rf/{1fps,30fps}/
    * autoencoder_best.pt: Trained autoencoder weights
    * rf.pkl: Trained random forest classifier
    * metrics.json: Test accuracy metrics
  
  - Latent representations (temporary):
    * Z_train.dat, Y_train.dat: Training latents
    * Z_test.dat, Y_test.dat: Test latents
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
from sklearn.model_selection import train_test_split

# =====================================
# Setup & Imports
# =====================================
SRC_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = SRC_DIR / "notebooks"
sys.path.insert(0, str(NOTEBOOKS_DIR))
import config

# Set seeds for reproducibility
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

device = config.DEVICE
EXP_DIR_ROOT = config.EXP_DIR_ROOT / "autoencoder_rf"

print(f"Device: {device}")
print(f"Experiments stored in: {EXP_DIR_ROOT}")
print(f"Results will be saved to: {config.RESULTS_DIR}")


# =====================================
# Dataset Class (matches notebook exactly)
# =====================================
class RamanAEDataset(Dataset):
    """
    Loads Raman spectral images with same train/val/test split as training.
    Uses same random state to ensure identical test set.
    """
    def __init__(self, split="test", fps_filter=None):
        all_rows = []

        for mineral_folder in config.RAW_DATA_DIR.iterdir():
            if not mineral_folder.is_dir():
                continue
            mineral_name = mineral_folder.name

            for fps_folder in mineral_folder.iterdir():
                if not fps_folder.is_dir():
                    continue
                fps_label = fps_folder.name
                if fps_filter and fps_label != fps_filter:
                    continue
                images = sorted(fps_folder.glob("*.bmp"))

                # On-the-fly split (CRITICAL: use same random state as training)
                train_val, test_imgs = train_test_split(
                    images, test_size=config.TEST_RATIO, random_state=config.SEED
                )
                train_imgs, val_imgs = train_test_split(
                    train_val, test_size=config.VAL_RATIO / (1 - config.TEST_RATIO), random_state=config.SEED
                )
                splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

                for img_path in splits[split]:
                    all_rows.append({"image": img_path, "mineral": mineral_name, "fps": fps_label})

        self.meta = pd.DataFrame(all_rows)
        self.labels = sorted(self.meta.mineral.unique())
        self.label_map = {l: i for i, l in enumerate(self.labels)}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img = Image.open(row.image).convert("RGB")
        img = self.transform(img)
        label = self.label_map[row.mineral]
        return img, label


# =====================================
# Autoencoder Model (matches notebook)
# =====================================
class AutoEncoder(nn.Module):
    """Simple 2-layer convolutional autoencoder."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        z = self.enc(x)
        return self.dec(z), z


# =====================================
# Testing Function
# =====================================
def test_autoencoder_rf(fps_label, save_results=True):
    """
    Test the trained autoencoder + RF pipeline on a specific FPS setting.
    
    Args:
        fps_label (str): "1fps" or "30fps"
        save_results (bool): Save plots and metrics to results directory
        
    Returns:
        tuple: (accuracy, confusion_matrix)
    """
    print(f"\n{'='*70}")
    print(f"📌 Testing Autoencoder + RF Pipeline for {fps_label.upper()}")
    print(f"{'='*70}")

    exp_dir = EXP_DIR_ROOT / fps_label
    
    # Verify models exist
    ae_path = exp_dir / "autoencoder_best.pt"
    rf_path = exp_dir / "rf.pkl"
    
    if not ae_path.exists():
        print(f"❌ Error: Autoencoder model not found at {ae_path}")
        return None, None
    if not rf_path.exists():
        print(f"❌ Error: Random Forest model not found at {rf_path}")
        return None, None

    print(f"✅ Loading models from {exp_dir}")
    
    # Load Autoencoder
    ae_model = AutoEncoder().to(device)
    ae_model.load_state_dict(torch.load(ae_path, map_location=device))
    ae_model.eval()
    print(f"   - Autoencoder: {ae_path}")

    # Load Random Forest
    rf_model = joblib.load(rf_path)
    print(f"   - Random Forest: {rf_path}")

    # Load Test Dataset (same split as training)
    test_dataset = RamanAEDataset("test", fps_filter=fps_label)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"   - Test samples: {len(test_dataset)}")

    # Get labels
    labels = test_dataset.labels
    print(f"   - Classes: {labels}")

    # Extract Latent Representations
    print(f"\n📊 Extracting latent representations from AE...")
    Z_test_list = []
    Y_test_list = []

    ae_model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            _, z = ae_model(x)
            z_flat = z.flatten(1).cpu().numpy()
            Z_test_list.append(z_flat)
            Y_test_list.append(y.numpy())
            if (batch_idx + 1) % 10 == 0:
                print(f"   - Processed {batch_idx + 1}/{len(test_loader)} batches")

    Z_test = np.vstack(Z_test_list)
    Y_test = np.concatenate(Y_test_list)
    
    print(f"   - Latent shape: {Z_test.shape}")
    print(f"   - Labels shape: {Y_test.shape}")

    # Predictions using Random Forest
    print(f"\n🤖 Generating RF predictions...")
    y_pred = rf_model.predict(Z_test)
    y_true = Y_test

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"   ✅ Test Accuracy: {acc:.4f}")

    # Classification Report
    print(f"\n📄 Classification Report:")
    print("─" * 70)
    report_str = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    print(report_str)

    # Confusion Matrix
    print(f"\n📊 Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(f"Confusion Matrix - Autoencoder + RF ({fps_label.upper()})", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_results:
        results_dir = config.RESULTS_DIR / "autoencoder_rf"
        results_dir.mkdir(parents=True, exist_ok=True)
        cm_path = results_dir / f"confusion_matrix_{fps_label}.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {cm_path}")
    
    plt.show()

    # Save Test Results JSON
    if save_results:
        results_json = {
            "fps": fps_label,
            "accuracy": float(acc),
            "num_test_samples": int(len(Y_test)),
            "num_classes": int(len(labels)),
            "classes": list(labels),
            "confusion_matrix": cm.tolist()
        }
        results_file = results_dir / f"test_results_{fps_label}.json"
        with open(results_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"   ✅ Saved: {results_file}")

    return acc, cm


# =====================================
# Main Execution
# =====================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("AUTOENCODER + RANDOM FOREST - TESTING SCRIPT")
    print("="*70)
    print(f"Device: {device}")
    print(f"Seed: {config.SEED}")
    print(f"Test ratio: {config.TEST_RATIO}")
    print(f"Batch size: {config.BATCH_SIZE}")

    # Test for both FPS settings
    results = {}
    for fps in config.FPS_LIST:
        acc, cm = test_autoencoder_rf(fps, save_results=True)
        if acc is not None:
            results[fps] = {"accuracy": acc, "confusion_matrix": cm}

    # Summary
    print(f"\n{'='*70}")
    print("📊 FINAL TEST SUMMARY")
    print(f"{'='*70}")
    for fps, data in sorted(results.items()):
        print(f"{fps:>6} FPS  →  Accuracy: {data['accuracy']:.4f}")
    print(f"{'='*70}\n")
