"""Dataset and DataLoader builders for the hierarchical pipeline.

Two views of the same `data/train` directory are exposed:

    Router loaders       relabel every image to its 4-way superclass id.
    Specialist loaders   keep only one superclass and relabel to local [0, 24].

Both use a stratified train/val split over the *original* 100 classes so every
fine class appears in both splits.
"""

import glob
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models
from torchvision.transforms import v2

from hierarchy import (
    SUPERCLASS_RANGES,
    original_id_to_local,
    original_id_to_router_label,
)


def get_router_transforms():
    """Eval transform from ConvNeXt-Tiny weights + a horizontal flip for training."""
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    eval_tf = weights.transforms()
    train_tf = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        eval_tf,
    ])
    return train_tf, eval_tf


def get_specialist_transforms(superclass_name: str):
    """Per-superclass augmentation recipes.

    Cars use plain center-crop because aggressive crops hurt make/model cues;
    flowers and planes get color jitter and rotation; food gets a mild resized
    crop. The eval transform is shared across all specialists.
    """
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    eval_tf = weights.transforms()

    aug_map = {
        "food": v2.Compose([
            v2.RandomResizedCrop(224, scale=(0.9, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        "flowers": v2.Compose([
            v2.RandomResizedCrop(224, scale=(0.82, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.12, hue=0.02),
            v2.RandomRotation(8),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.RandomErasing(p=0.03, scale=(0.02, 0.04)),
        ]),
        "cars": v2.Compose([
            v2.Resize(232),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        "planes": v2.Compose([
            v2.RandomResizedCrop(224, scale=(0.88, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.06, contrast=0.06, saturation=0.06, hue=0.01),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }

    return aug_map[superclass_name], eval_tf


class RouterDataset(Dataset):
    """Wraps an ImageFolder and relabels every sample to its 4-way superclass id."""

    def __init__(self, base_dataset: datasets.ImageFolder):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label_idx = self.base_dataset[idx]
        original_id = int(self.base_dataset.classes[int(label_idx)])
        return image, original_id_to_router_label(original_id)


class SpecialistDataset(Dataset):
    """Wraps an ImageFolder and keeps only one superclass, relabeled to [0, 24]."""

    def __init__(self, base_dataset: datasets.ImageFolder, superclass_name: str):
        self.base_dataset = base_dataset
        self.superclass_name = superclass_name

        start, end = SUPERCLASS_RANGES[superclass_name]
        self.indices = []
        for i, (_, label_idx) in enumerate(base_dataset.samples):
            original_id = int(base_dataset.classes[int(label_idx)])
            if start <= original_id <= end:
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label_idx = self.base_dataset[real_idx]
        original_id = int(self.base_dataset.classes[int(label_idx)])
        _, local_label = original_id_to_local(original_id)
        return image, local_label


def _stratified_split_indices(labels, val_ratio: float = 0.15, seed: int = 42):
    """Class-stratified split: each class gets `ceil(n * val_ratio)` val samples."""
    generator = torch.Generator().manual_seed(seed)

    class_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        class_to_indices.setdefault(int(label), []).append(idx)

    train_indices, val_indices = [], []
    for inds in class_to_indices.values():
        perm = torch.randperm(len(inds), generator=generator).tolist()
        shuffled = [inds[i] for i in perm]
        val_count = max(1, int(round(len(shuffled) * val_ratio)))
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    return sorted(train_indices), sorted(val_indices)


def _make_loaders_from_indices(train_ds, eval_ds, train_indices, val_indices, batch_size=32):
    train_loader = DataLoader(Subset(train_ds, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(eval_ds, val_indices), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _make_loaders_from_dataset(train_ds, eval_ds, batch_size=32, val_ratio=0.15, seed=42):
    labels = [label for _, label in eval_ds]
    train_indices, val_indices = _stratified_split_indices(labels, val_ratio=val_ratio, seed=seed)
    return _make_loaders_from_indices(train_ds, eval_ds, train_indices, val_indices, batch_size)


def get_router_dataloaders(train_dir: str = "./data/train", batch_size: int = 32, val_ratio: float = 0.15, seed: int = 42):
    """Loaders for the 4-way router. Split is stratified over the original 100 classes
    so each fine class is represented in both train and val before the relabel-to-4."""
    train_tf, eval_tf = get_router_transforms()

    base_train_aug = datasets.ImageFolder(root=train_dir, transform=train_tf)
    base_train_eval = datasets.ImageFolder(root=train_dir, transform=eval_tf)

    train_indices, val_indices = _stratified_split_indices(
        base_train_eval.targets, val_ratio=val_ratio, seed=seed
    )

    return _make_loaders_from_indices(
        RouterDataset(base_train_aug),
        RouterDataset(base_train_eval),
        train_indices,
        val_indices,
        batch_size=batch_size,
    )


def get_specialist_dataloaders(superclass_name: str, train_dir: str = "./data/train", batch_size: int = 32, val_ratio: float = 0.15, seed: int = 42):
    """Loaders for a single specialist (one superclass, 25 fine classes)."""
    train_tf, eval_tf = get_specialist_transforms(superclass_name)

    base_train_aug = datasets.ImageFolder(root=train_dir, transform=train_tf)
    base_train_eval = datasets.ImageFolder(root=train_dir, transform=eval_tf)

    return _make_loaders_from_dataset(
        SpecialistDataset(base_train_aug, superclass_name),
        SpecialistDataset(base_train_eval, superclass_name),
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
    )


class HierarchicalTestDataset(Dataset):
    """Iterates the unlabeled test set in numeric-filename order."""

    def __init__(self, root_dir: str = "./data/test", transform=None):
        self.image_paths = sorted(
            glob.glob(os.path.join(root_dir, "*.jpg")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, os.path.basename(path)
