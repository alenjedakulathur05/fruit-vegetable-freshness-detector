"""
utils.py
--------
Shared settings and small helper functions used by every other script.

Keeping these in ONE place guarantees that training, evaluation, prediction
and the web app all use exactly the same image size, normalisation values,
file paths and label names.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# 1. Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"                 # original downloaded images
PROCESSED_DIR = DATASET_DIR / "processed"     # cleaned + resized images
PROCESSED_IMAGES_DIR = PROCESSED_DIR / "images"
SPLITS_CSV = PROCESSED_DIR / "splits.csv"     # one row per image: path, label, split
DATASET_SUMMARY_JSON = PROCESSED_DIR / "dataset_summary.json"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "fruit_vegetable_freshness_model.pt"
CLASS_LABELS_PATH = MODELS_DIR / "class_labels.json"

RESULTS_DIR = PROJECT_ROOT / "results"
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "sample_images"

# Kaggle dataset used by this project (see DATASET.md)
KAGGLE_DATASET = "muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten"

# ---------------------------------------------------------------------------
# 2. Image / model settings
# ---------------------------------------------------------------------------
IMG_SIZE = 224  # MobileNetV2 was pre-trained on 224 x 224 ImageNet images

# ImageNet channel statistics. The pretrained backbone expects inputs that
# were normalised with exactly these numbers: (pixel/255 - mean) / std
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# If the model's top probability is below this value we warn the user that
# the image may be outside the supported categories.
CONFIDENCE_THRESHOLD = 0.70

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SEED = 42


# ---------------------------------------------------------------------------
# 3. Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    """Make random operations repeatable (splits, augmentation order, init)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Use the NVIDIA GPU if PyTorch can see it, otherwise fall back to CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 4. Image loading (shared by preprocessing AND prediction)
# ---------------------------------------------------------------------------
def load_and_resize(image_source, size: int = IMG_SIZE) -> Image.Image:
    """
    Open an image, fix phone-camera rotation, convert to RGB and resize it to
    size x size pixels.

    `image_source` can be a file path or an open file-like object (the web app
    passes the uploaded bytes). The SAME function is used when building the
    processed training set and when predicting, so a new image goes through
    exactly the same resizing as the training images did.
    """
    img = Image.open(image_source)
    # For large JPEGs, let the decoder downscale while decoding (much faster).
    img.draft("RGB", (size * 2, size * 2))
    img = ImageOps.exif_transpose(img)  # respect the EXIF "orientation" tag
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        # transparent PNG: put it on a white background instead of letting the
        # hidden (often black) pixels behind the transparency show up
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
    img = img.convert("RGB")            # greyscale / RGBA / palette -> 3 channels
    img = img.resize((size, size), Image.Resampling.BICUBIC)
    return img


def is_valid_image(image_source) -> bool:
    """Return True if PIL can fully decode the file (detects corrupted images)."""
    try:
        with Image.open(image_source) as img:
            img.verify()  # checks file integrity without decoding pixels
        if hasattr(image_source, "seek"):
            image_source.seek(0)
        with Image.open(image_source) as img:
            img.load()    # actually decode the pixels (catches truncated files)
        if hasattr(image_source, "seek"):
            image_source.seek(0)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 5. Label helpers
# ---------------------------------------------------------------------------
# The dataset folders are named like "Apple__Healthy" or "Bellpepper__Rotten".
_FRESH_WORDS = {"healthy", "fresh", "good"}
_SPOILED_WORDS = {"rotten", "stale", "spoiled", "bad"}
_PRETTY_NAMES = {"Bellpepper": "Bell Pepper"}


def parse_label(class_name: str) -> dict:
    """
    Split a class folder name into (produce, freshness).

    "Apple__Healthy"   -> {"produce": "Apple",       "freshness": "Fresh"}
    "Bellpepper__Rotten" -> {"produce": "Bell Pepper", "freshness": "Spoiled"}
    """
    parts = [p for p in class_name.replace("__", "_").split("_") if p]
    freshness = "Unknown"
    produce_parts = []
    for part in parts:
        word = part.lower()
        if word in _FRESH_WORDS:
            freshness = "Fresh"
        elif word in _SPOILED_WORDS:
            freshness = "Spoiled"
        else:
            produce_parts.append(part)
    produce = " ".join(produce_parts) if produce_parts else class_name
    produce = _PRETTY_NAMES.get(produce, produce)
    return {"produce": produce, "freshness": freshness}


def display_name(class_name: str) -> str:
    """'Apple__Rotten' -> 'Spoiled Apple' (used in reports and plots)."""
    info = parse_label(class_name)
    return f"{info['freshness']} {info['produce']}"


# ---------------------------------------------------------------------------
# 6. Small file helpers
# ---------------------------------------------------------------------------
def save_json(data, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, SAMPLE_IMAGES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def default_num_workers() -> int:
    """Number of background processes that load images during training."""
    cpus = os.cpu_count() or 2
    return max(0, min(4, cpus - 2))


# ---------------------------------------------------------------------------
# 7. Training-curve plots (used by train.py and evaluate.py)
# ---------------------------------------------------------------------------
def plot_training_history(history: list, out_dir=RESULTS_DIR) -> None:
    """Save training/validation accuracy and loss curves as PNG files."""
    import matplotlib

    matplotlib.use("Agg")  # draw to files, no window needed
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    # Epoch where fine-tuning started (for a vertical marker line)
    ft_start = next((h["epoch"] for h in history if h["phase"] == "fine_tune"), None)

    for metric, ylabel, fname in [
        ("acc", "Accuracy", "training_accuracy.png"),
        ("loss", "Loss (cross-entropy)", "training_loss.png"),
    ]:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [h[f"train_{metric}"] for h in history], "o-", label=f"Training {ylabel.split()[0].lower()}")
        plt.plot(epochs, [h[f"val_{metric}"] for h in history], "s-", label=f"Validation {ylabel.split()[0].lower()}")
        if ft_start is not None:
            plt.axvline(ft_start - 0.5, color="gray", linestyle="--", label="Fine-tuning starts")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"Training vs validation {ylabel.split()[0].lower()}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=120)
        plt.close()
