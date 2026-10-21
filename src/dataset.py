"""
dataset.py
----------
Feeds the preprocessed images to PyTorch during training/evaluation.

  * FreshnessDataset  - reads dataset/processed/splits.csv and returns (image_tensor, label_index)
  * get_train_transform() - data augmentation + normalisation (training only)
  * get_eval_transform()  - normalisation only (validation, test, prediction, web app)
  * build_dataloaders()   - creates the train / val / test DataLoaders
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, PROJECT_ROOT, SEED, SPLITS_CSV, default_num_workers,
)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_train_transform() -> transforms.Compose:
    """
    Realistic augmentation: each epoch the model sees slightly different
    versions of every training image, which reduces over-fitting.
    We deliberately do NOT change hue/saturation, because colour changes
    (browning, dark spots, mould) are exactly what indicates spoilage.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),         # mirror image
        transforms.RandomAffine(
            degrees=15,                                  # small rotation (+/- 15 degrees)
            translate=(0.10, 0.10),                      # shift up to 10% sideways/up-down
            scale=(0.90, 1.10),                          # small zoom in/out
        ),
        transforms.ColorJitter(brightness=0.20),        # moderate lighting change
        transforms.ToTensor(),                           # pixels 0..255 -> 0..1, shape (3, H, W)
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transform() -> transforms.Compose:
    """No randomness: used for validation, test, predict.py and the web app."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FreshnessDataset(Dataset):
    """One item = (normalised image tensor, integer class label)."""

    def __init__(self, frame: pd.DataFrame, transform):
        self.paths = frame["processed_path"].tolist()
        self.labels = frame["label_index"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(PROJECT_ROOT / self.paths[idx]) as img:
            image = self.transform(img.convert("RGB"))
        return image, self.labels[idx]


def load_splits(max_per_class=None) -> pd.DataFrame:
    """Read splits.csv. `max_per_class` keeps only N images per class+split (quick tests)."""
    if not SPLITS_CSV.exists():
        sys.exit(f"{SPLITS_CSV} not found. Run: python src/data_preprocessing.py")
    frame = pd.read_csv(SPLITS_CSV)
    if max_per_class:
        # shuffle, then keep the first N rows of every (class, split) pair
        frame = frame.sample(frac=1.0, random_state=SEED).groupby(["label", "split"]).head(max_per_class)
    return frame


def build_dataloaders(batch_size=32, num_workers=None, max_per_class=None, pin_memory=False):
    """Return {'train': DataLoader, 'val': DataLoader, 'test': DataLoader} and the frame."""
    frame = load_splits(max_per_class)
    if num_workers is None:
        num_workers = default_num_workers()
    loaders = {}
    for split in ("train", "val", "test"):
        part = frame[frame["split"] == split]
        dataset = FreshnessDataset(part, get_train_transform() if split == "train" else get_eval_transform())
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),          # shuffle only the training data
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            generator=torch.Generator().manual_seed(SEED) if split == "train" else None,
        )
    return loaders, frame
