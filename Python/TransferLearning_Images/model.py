"""Flat ResNet-18 baseline (early experiment, kept for reference).

The current best pipeline lives in `main_router.py` + `main_specialist.py` +
`hybrid_inference.py`, all of which use ConvNeXt-Tiny.
"""

import torch.nn as nn
from torchvision import models


def create_model(num_classes: int = 100, fine_tune: bool = False) -> nn.Module:
    """Load ImageNet-pretrained ResNet-18 and swap the head for `num_classes`.

    If `fine_tune` is False, the backbone is frozen and only the new head trains.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


if __name__ == "__main__":
    test_model = create_model(num_classes=100, fine_tune=False)
    print(test_model)
