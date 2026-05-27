"""Standalone Test-Time Augmentation helper.

Used by the flat-classifier baseline. The hierarchical pipeline rolls TTA into
`hybrid_inference.HybridPredictor` directly.
"""

import torch
from torchvision import models
from torchvision.transforms import v2


def get_tta_transforms():
    """Three deterministic eval transforms whose logits are averaged at inference."""
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


@torch.no_grad()
def predict_with_tta(model, pil_image, tta_transforms, device):
    """Mean of model logits across each TTA transform."""
    model.eval()
    logits_sum = None
    for tf in tta_transforms:
        x = tf(pil_image).unsqueeze(0).to(device)
        logits = model(x)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    return logits_sum / len(tta_transforms)
