"""Flat 100-class data loaders used by the non-hierarchical baseline.

The hierarchical router/specialist pipeline uses `hierarchy_data.py` instead.
"""

from __future__ import annotations

import glob
import os
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, models
from torchvision.transforms import v2


class UnlabeledTestDataset(Dataset):
    """Returns (image_tensor, filename) for every .jpg in a flat directory."""

    def __init__(self, root_dir: str = "./data/test", transform: v2.Compose | None = None):
        self.image_paths = sorted(
            glob.glob(os.path.join(root_dir, "*.jpg")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, os.path.basename(path)


def _default_transforms() -> Tuple[v2.Compose, v2.Compose]:
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    eval_tf = weights.transforms()
    train_tf = v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.85, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def get_dataloaders(
    train_dir: str = "./data/train",
    test_dir: str = "./data/test",
    batch_size: int = 32,
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Build train/val/test loaders for the flat 100-class problem.

    Returns:
        (train_loader, val_loader, test_loader, idx_to_class) where
        `idx_to_class` maps ImageFolder's contiguous class index back to the
        original integer class id stored as a folder name (e.g. "0".."99").
    """
    train_tf, eval_tf = _default_transforms()

    full_train_aug = datasets.ImageFolder(root=train_dir, transform=train_tf)
    full_train_eval = datasets.ImageFolder(root=train_dir, transform=eval_tf)

    # Index-level split keeps the two ImageFolder views (augmented vs. eval) aligned.
    n_total = len(full_train_aug)
    n_val = int(round(n_total * val_split))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset_idx, val_subset_idx = random_split(
        range(n_total), [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(
        Subset(full_train_aug, list(train_subset_idx)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(full_train_eval, list(val_subset_idx)),
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        UnlabeledTestDataset(test_dir, transform=eval_tf),
        batch_size=batch_size,
        shuffle=False,
    )

    idx_to_class = {v: k for k, v in full_train_aug.class_to_idx.items()}
    return train_loader, val_loader, test_loader, idx_to_class
