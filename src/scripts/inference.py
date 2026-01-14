import sys
from pathlib import Path
import torch
import numpy as np
import joblib
from PIL import Image
from torchvision import models, transforms


# -------------------------------------------------
# USER CONFIGURATION (EDIT ONLY THIS)
# -------------------------------------------------
# IMAGE_PATH = r"R:/MUL/Courses/Applied ML/raman-mineral-classification/experiments/sample_for_inference/SPECTRAL_Albit113_planar1_30FPS_395mW_30FPS_42_32986us_2025_09_05-08_39_02_609.bmp"
IMAGE_PATH = r"R:/MUL/Courses/Applied ML/raman-mineral-classification/experiments/sample_for_inference/SPECTRAL_Rhodochrosite_planar1_30FPS_396mW_30FPS_42_32986us_2025_08_20-11_45_49_980.bmp"

MODEL_TYPE = "resnet"       # "resnet" or "autoencoder"
FPS_TYPE   = "30fps"         # "1fps" or "30fps"
# -------------------------------------------------


# -------------------------------------------------
# Path setup
# -------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))


# -------------------------------------------------
# Config import
# -------------------------------------------------
from notebooks.config import DEVICE, EXP_DIR_ROOT, LATENT_DIM, DROPOUT


device = torch.device(DEVICE)
print(f"Using device: {device}")


# -------------------------------------------------
# Class labels (MUST match training order)
# -------------------------------------------------
MINERALS = [
    "Albit",
    "Calcite",
    "Dolomit",
    "Feldspat",
    "Quarz",
    "Rhodocrosite",
    "Tile"
]


# -------------------------------------------------
# Image preprocessing
# -------------------------------------------------
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


# =================================================
# ResNet inference
# =================================================
def infer_resnet(image_path):
    # Path to trained model
    model_path = EXP_DIR_ROOT / "resnet" / FPS_TYPE / "resnet.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"ResNet model not found: {model_path}")

    # ----------------------------
    # Load model architecture
    # ----------------------------
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # The head must match exactly how it was trained
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(DROPOUT),
        torch.nn.Linear(model.fc.in_features, len(MINERALS))
    )

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # ----------------------------
    # Load & preprocess image
    # ----------------------------
    x = load_image(image_path)

    # ----------------------------
    # Forward pass & prediction
    # ----------------------------
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

    return MINERALS[pred.item()], conf.item() * 100



# =================================================
# AutoEncoder + Random Forest inference
# =================================================
def infer_autoencoder_rf(image_path):
    from models.autoencoder import AutoEncoder

    exp_dir = EXP_DIR_ROOT / "autoencoder_rf" / FPS_TYPE
    ae_path = exp_dir / "autoencoder_best.pt"
    rf_path = exp_dir / "rf.pkl"

    if not ae_path.exists() or not rf_path.exists():
        raise FileNotFoundError(f"AE or RF model missing in {exp_dir}")

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(device)
    ae.load_state_dict(torch.load(ae_path, map_location=device))
    ae.eval()

    rf = joblib.load(rf_path)

    x = load_image(image_path)

    with torch.no_grad():
        _, z = ae(x)

    z = z.view(1, -1).cpu().numpy()

    probs = rf.predict_proba(z)[0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx] * 100

    return MINERALS[pred_idx], confidence


# =================================================
# Main execution
# =================================================
if __name__ == "__main__":
    image_path = Path(IMAGE_PATH)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("\n🔍 Running Mineral Inference")
    print(f"Model : {MODEL_TYPE}")
    print(f"FPS   : {FPS_TYPE}")
    print(f"Image : {image_path.name}\n")

    if MODEL_TYPE == "resnet":
        pred, conf = infer_resnet(image_path)
    elif MODEL_TYPE == "autoencoder":
        pred, conf = infer_autoencoder_rf(image_path)
    else:
        raise ValueError("MODEL_TYPE must be 'resnet' or 'autoencoder'")

    print("✅ Prediction Result")
    print(f"Predicted Mineral : {pred}")
    print(f"Confidence        : {conf:.2f}%")
