"""Train a 25-way specialist classifier for one superclass.

Each specialist is a ConvNeXt-Tiny fine-tuned only on images from a single
superclass (food / flowers / cars / planes). Phase 1 trains the head; Phase 2
unfreezes the last two ConvNeXt stages and uses layer-wise learning rates.

Change `SUPERCLASS_NAME` in `main()` to train a different specialist.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import engine
import hierarchy_data
import utils


class SpecialistConvNeXt(nn.Module):
    """ConvNeXt-Tiny re-headed to 25 classes (one superclass's fine labels)."""

    def __init__(self, num_classes: int = 25):
        super().__init__()
        self.backbone = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.classifier[2] = nn.Sequential(nn.Linear(768, num_classes))

    def forward(self, x):
        return self.backbone(x)


def plot_training_history(history: dict, phase_name: str = "Training") -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="o")
    plt.title(f"{phase_name} Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    plt.plot(epochs, history["val_acc"], label="Val Acc", marker="o")
    plt.title(f"{phase_name} Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    filename = f"{phase_name.replace(' ', '_')}_curves.png"
    plt.savefig(filename)
    print(f"--> Saved training curves to {filename}")
    plt.show()


# Per-superclass training recipes. Different superclasses benefit from
# different amounts of regularization and batch-level augmentation.
SPECIALIST_CFG = {
    "food": {
        "phase1_epochs": 20,
        "phase2_epochs": 10,
        "label_smoothing": 0.00,
        "batch_aug_mode": "none",
        "mixup_alpha": 0.15,
        "cutmix_alpha": 1.0,
    },
    "flowers": {
        "phase1_epochs": 18,
        "phase2_epochs": 12,
        "label_smoothing": 0.05,
        "batch_aug_mode": "mixup_cutmix",
        "mixup_alpha": 0.20,
        "cutmix_alpha": 1.0,
    },
    "cars": {
        "phase1_epochs": 16,
        "phase2_epochs": 12,
        "label_smoothing": 0.05,
        "batch_aug_mode": "cutmix",
        "mixup_alpha": 0.20,
        "cutmix_alpha": 1.0,
    },
    "planes": {
        "phase1_epochs": 16,
        "phase2_epochs": 12,
        "label_smoothing": 0.05,
        "batch_aug_mode": "cutmix",
        "mixup_alpha": 0.20,
        "cutmix_alpha": 1.0,
    },
}


def main() -> None:
    BATCH_SIZE = 32
    SEED = 999

    SUPERCLASS_NAME = "food"  # one of: "food", "flowers", "cars", "planes"

    cfg = SPECIALIST_CFG[SUPERCLASS_NAME]
    PHASE1_EPOCHS = cfg["phase1_epochs"]
    PHASE2_EPOCHS = cfg["phase2_epochs"]

    RUN_NAME = f"specialist_{SUPERCLASS_NAME}_convnext"
    PHASE1_CKPT = f"./checkpoints/phase1_{RUN_NAME}.pt"
    BEST_CKPT = f"./checkpoints/best_{RUN_NAME}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training specialist for superclass: {SUPERCLASS_NAME}")

    print("\nLoading specialist datasets...")
    train_loader, val_loader = hierarchy_data.get_specialist_dataloaders(
        superclass_name=SUPERCLASS_NAME,
        batch_size=BATCH_SIZE,
        val_ratio=0.10,
        seed=SEED,
    )

    print("\nInitializing specialist model...")
    model = SpecialistConvNeXt(num_classes=25).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    # Phase 1: head only.
    print(f"\n--- PHASE 1: Warming up {SUPERCLASS_NAME} specialist head ---")
    optimizer_phase1 = optim.AdamW(
        model.backbone.classifier.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    scheduler_phase1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase1, T_max=PHASE1_EPOCHS
    )

    history_phase1 = engine.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_phase1,
        device=device,
        num_classes=25,
        epochs=PHASE1_EPOCHS,
        ckpt_path=PHASE1_CKPT,
        scheduler=scheduler_phase1,
        batch_aug_mode=cfg["batch_aug_mode"],
        mixup_alpha=cfg["mixup_alpha"],
        cutmix_alpha=cfg["cutmix_alpha"],
    )
    plot_training_history(history_phase1, f"{SUPERCLASS_NAME} Specialist Phase 1")

    # Phase 2: unfreeze the last two ConvNeXt stages with layer-wise LRs
    # (highest at the head, lowest deepest in the backbone).
    print(f"\n--- PHASE 2: Fine-tuning {SUPERCLASS_NAME} specialist final stages ---")
    checkpoint = torch.load(PHASE1_CKPT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    for i in (6, 7):
        for param in model.backbone.features[i].parameters():
            param.requires_grad = True

    optimizer_phase2 = optim.AdamW(
        [
            {"params": model.backbone.classifier.parameters(), "lr": 1e-4},
            {"params": model.backbone.features[7].parameters(), "lr": 3e-5},
            {"params": model.backbone.features[6].parameters(), "lr": 1e-5},
        ],
        weight_decay=1e-4,
    )
    scheduler_phase2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase2, T_max=PHASE2_EPOCHS
    )

    history_phase2 = engine.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_phase2,
        device=device,
        num_classes=25,
        epochs=PHASE2_EPOCHS,
        ckpt_path=BEST_CKPT,
        scheduler=scheduler_phase2,
        batch_aug_mode=cfg["batch_aug_mode"],
        mixup_alpha=cfg["mixup_alpha"],
        cutmix_alpha=cfg["cutmix_alpha"],
    )
    plot_training_history(history_phase2, f"{SUPERCLASS_NAME} Specialist Phase 2")

    print(f"\nDone training {SUPERCLASS_NAME} specialist.")
    print(f"Best checkpoint saved to: {BEST_CKPT}")


if __name__ == "__main__":
    utils.set_seed(999)
    main()
