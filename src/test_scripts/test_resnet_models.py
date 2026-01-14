# test_resnet_standalone.py
import sys
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ----------------------------
# Config
# ----------------------------
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC_DIR))

from notebooks.config import RAW_DATA_DIR, EXP_DIR_ROOT, BATCH_SIZE, DEVICE, SEED, TEST_RATIO, VAL_RATIO

device = torch.device(DEVICE)
print(f"Using device: {device}")

# ----------------------------
# Dataset
# ----------------------------
class RamanDataset(Dataset):
    def __init__(self, fps_label, split="test"):
        rows = []
        for mineral_dir in RAW_DATA_DIR.iterdir():
            if not mineral_dir.is_dir():
                continue
            fps_dir = mineral_dir / fps_label
            if not fps_dir.exists():
                continue
            images = sorted(fps_dir.glob("*.bmp"))
            
            from sklearn.model_selection import train_test_split
            train_val, test = train_test_split(images, test_size=TEST_RATIO, random_state=SEED)
            train, val = train_test_split(train_val, test_size=VAL_RATIO / (1 - TEST_RATIO), random_state=SEED)
            
            split_map = {"train": train, "val": val, "test": test}
            for img in split_map[split]:
                rows.append({"path": img, "label": mineral_dir.name})
        
        self.df = pd.DataFrame(rows)
        self.labels = sorted(self.df.label.unique())
        self.label_map = {l: i for i, l in enumerate(self.labels)}
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert("RGB")
        img = self.transform(img)
        label = self.label_map[row.label]
        return img, label

# ----------------------------
# Model loader
# ----------------------------
def load_trained_resnet(num_classes, model_path):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ----------------------------
# Evaluate function
# ----------------------------
def evaluate_model(fps_label):
    print(f"\n📌 Evaluating ResNet on {fps_label}")
    
    test_ds = RamanDataset(fps_label, split="test")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, pin_memory=True)
    
    model_path = EXP_DIR_ROOT / "resnet" / fps_label / "resnet.pt"
    if not model_path.exists():
        print(f"⚠️ Model not found at {model_path}")
        return None, None
    
    model = load_trained_resnet(len(test_ds.labels), model_path)
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.cpu().numpy())
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Test Accuracy ({fps_label}): {acc:.4f}")
    
    # Classification report
    print("\n📄 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=test_ds.labels, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=test_ds.labels, yticklabels=test_ds.labels, cmap="Blues")
    plt.title(f"Confusion Matrix – {fps_label}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    return acc, cm

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    acc_1fps, cm_1fps = evaluate_model("1fps")
    acc_30fps, cm_30fps = evaluate_model("30fps")

    print("\n📊 Final Test Accuracies")
    print(f"1 FPS  → {acc_1fps:.4f}" if acc_1fps is not None else "1 FPS → Not tested")
    print(f"30 FPS → {acc_30fps:.4f}" if acc_30fps is not None else "30 FPS → Not tested")
