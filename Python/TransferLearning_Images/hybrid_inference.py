"""Hierarchical ensemble inference for the 100-class problem.

For each test image:
    1. The router picks one of four superclasses.
    2. If a specialist ensemble exists for that superclass, average its logits
       (with TTA) and pick the local argmax, then map back to a global id.
    3. Otherwise fall back to the flat 100-class model with logits masked to
       only the predicted superclass's range.

Edit `CONFIG` below to point at your trained checkpoints before running.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.transforms import v2

from hierarchy import ROUTER_LABEL_TO_SUPERCLASS, SUPERCLASS_RANGES, local_to_original_id


# =========================
# Model definitions
# =========================
class RouterConvNeXt(nn.Module):
    """ConvNeXt-Tiny re-headed to 4-way superclass classification."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = models.convnext_tiny(weights=None)
        self.backbone.classifier[2] = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SpecialistConvNeXt(nn.Module):
    """ConvNeXt-Tiny re-headed to 25-way local-class classification."""

    def __init__(self, num_classes: int = 25):
        super().__init__()
        self.backbone = models.convnext_tiny(weights=None)
        self.backbone.classifier[2] = nn.Sequential(nn.Linear(768, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class FlatConvNeXt(nn.Module):
    """ConvNeXt-Tiny re-headed to all 100 fine classes (used as a fallback)."""

    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.backbone = models.convnext_tiny(weights=None)
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(768, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# =========================
# Config
# =========================
@dataclass
class HybridConfig:
    router_ckpt: str
    flat_ckpt: str
    food_specialist_ckpts: Sequence[str]
    flowers_specialist_ckpts: Sequence[str]
    cars_specialist_ckpts: Sequence[str] | None = None
    planes_specialist_ckpts: Sequence[str] | None = None
    use_tta_router: bool = False
    use_tta_specialists: bool = True
    use_tta_flat: bool = True


# Cars / planes specialists were never trained, so those routes fall back to
# the flat model masked to the predicted superclass's id range.
CONFIG = HybridConfig(
    router_ckpt="./checkpoints/best_router_convnext.pt",
    flat_ckpt="./checkpoints/best_convnext.pt",
    food_specialist_ckpts=[
        "./checkpoints/phase1_specialist_food_convnext_seed42.pt",
        "./checkpoints/phase1_specialist_food_convnext_seed123.pt",
        "./checkpoints/phase1_specialist_food_convnext_seed999.pt",
    ],
    flowers_specialist_ckpts=[
        "./checkpoints/phase1_specialist_flowers_convnext_seed42.pt",
        "./checkpoints/phase1_specialist_flowers_convnext_seed123.pt",
        "./checkpoints/phase1_specialist_flowers_convnext_seed999.pt",
    ],
    cars_specialist_ckpts=None,
    planes_specialist_ckpts=None,
)


# =========================
# Transforms / TTA
# =========================
def get_eval_transform() -> v2.Compose:
    return models.ConvNeXt_Tiny_Weights.DEFAULT.transforms()


def get_tta_transforms() -> List[v2.Compose]:
    """Deterministic eval transforms only — never use random transforms at inference."""
    return [
        models.ConvNeXt_Tiny_Weights.DEFAULT.transforms(),
        v2.Compose([
            v2.Resize(236),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        v2.Compose([
            v2.Resize(232),
            v2.CenterCrop(224),
            v2.RandomHorizontalFlip(p=1.0),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    ]


# =========================
# Loading helpers
# =========================
def _load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device) -> nn.Module:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_router(ckpt_path: str, device: torch.device) -> RouterConvNeXt:
    return _load_checkpoint(RouterConvNeXt(num_classes=4), ckpt_path, device)


def load_specialist_ensemble(ckpt_paths: Sequence[str], device: torch.device) -> List[SpecialistConvNeXt]:
    models_list: List[SpecialistConvNeXt] = []
    for ckpt in ckpt_paths:
        if ckpt and os.path.exists(ckpt):
            models_list.append(_load_checkpoint(SpecialistConvNeXt(num_classes=25), ckpt, device))
    if not models_list:
        raise ValueError(f"No valid specialist checkpoints found in: {ckpt_paths}")
    return models_list


def load_flat_model(ckpt_path: str, device: torch.device) -> FlatConvNeXt:
    return _load_checkpoint(FlatConvNeXt(num_classes=100), ckpt_path, device)


# =========================
# Inference helpers
# =========================
@torch.no_grad()
def predict_logits_single(model: nn.Module, pil_image: Image.Image, transform: v2.Compose, device: torch.device) -> torch.Tensor:
    x = transform(pil_image).unsqueeze(0).to(device)
    return model(x).squeeze(0)


@torch.no_grad()
def predict_logits_tta(model: nn.Module, pil_image: Image.Image, transforms: Sequence[v2.Compose], device: torch.device) -> torch.Tensor:
    logits_sum = None
    for tf in transforms:
        logits = predict_logits_single(model, pil_image, tf, device)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    return logits_sum / len(transforms)


@torch.no_grad()
def ensemble_logits(models_list: Sequence[nn.Module], pil_image: Image.Image, device: torch.device, use_tta: bool) -> torch.Tensor:
    transforms = get_tta_transforms() if use_tta else [get_eval_transform()]
    logits_sum = None
    for model in models_list:
        logits = predict_logits_tta(model, pil_image, transforms, device)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    return logits_sum / len(models_list)


def mask_logits_to_superclass(flat_logits: torch.Tensor, superclass_name: str) -> torch.Tensor:
    """Restrict a 100-way logit vector to one superclass's id range."""
    masked = torch.full_like(flat_logits, float("-inf"))
    start, end = SUPERCLASS_RANGES[superclass_name]
    masked[start:end + 1] = flat_logits[start:end + 1]
    return masked


# =========================
# Hybrid predictor
# =========================
class HybridPredictor:
    def __init__(self, config: HybridConfig, device: str | None = None):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.config = config

        self.router = load_router(config.router_ckpt, self.device)
        self.flat_model = load_flat_model(config.flat_ckpt, self.device)

        self.specialists: Dict[str, List[SpecialistConvNeXt]] = {
            "food": load_specialist_ensemble(config.food_specialist_ckpts, self.device),
            "flowers": load_specialist_ensemble(config.flowers_specialist_ckpts, self.device),
        }
        if config.cars_specialist_ckpts:
            self.specialists["cars"] = load_specialist_ensemble(config.cars_specialist_ckpts, self.device)
        if config.planes_specialist_ckpts:
            self.specialists["planes"] = load_specialist_ensemble(config.planes_specialist_ckpts, self.device)

    @torch.no_grad()
    def predict(self, pil_image: Image.Image) -> dict:
        pil_image = pil_image.convert("RGB")

        router_logits = (
            predict_logits_tta(self.router, pil_image, get_tta_transforms(), self.device)
            if self.config.use_tta_router
            else predict_logits_single(self.router, pil_image, get_eval_transform(), self.device)
        )
        router_label = int(router_logits.argmax().item())
        superclass = ROUTER_LABEL_TO_SUPERCLASS[router_label]

        # Specialist route for superclasses where one was trained.
        if superclass in self.specialists and superclass in {"food", "flowers"}:
            spec_logits = ensemble_logits(
                self.specialists[superclass],
                pil_image,
                self.device,
                use_tta=self.config.use_tta_specialists,
            )
            local_pred = int(spec_logits.argmax().item())
            return {
                "router_superclass": superclass,
                "prediction": local_to_original_id(superclass, local_pred),
                "path": "specialist",
                "local_prediction": local_pred,
            }

        # Fallback: flat 100-way classifier with logits masked to the superclass.
        flat_logits = (
            predict_logits_tta(self.flat_model, pil_image, get_tta_transforms(), self.device)
            if self.config.use_tta_flat
            else predict_logits_single(self.flat_model, pil_image, get_eval_transform(), self.device)
        )
        masked_logits = mask_logits_to_superclass(flat_logits, superclass)
        return {
            "router_superclass": superclass,
            "prediction": int(masked_logits.argmax().item()),
            "path": "flat_masked",
            "local_prediction": None,
        }

    def predict_folder(self, test_dir: str, output_csv: str = "submission.csv") -> pd.DataFrame:
        image_paths = sorted(
            glob.glob(os.path.join(test_dir, "*.jpg")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
        )

        rows = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            result = self.predict(img)
            rows.append({
                "id": os.path.basename(path),
                "label": result["prediction"],
                "router_superclass": result["router_superclass"],
                "path": result["path"],
            })

        df = pd.DataFrame(rows)
        df["sort_key"] = df["id"].str.replace(".jpg", "", regex=False).astype(int)
        df = df.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)
        df[["id", "label"]].to_csv(output_csv, index=False)
        return df


def main() -> None:
    predictor = HybridPredictor(CONFIG)
    print(f"Using device: {predictor.device}")

    df = predictor.predict_folder("./data/test", output_csv="submission.csv")
    print(df.head())
    print("Saved submission to submission.csv")


if __name__ == "__main__":
    main()
