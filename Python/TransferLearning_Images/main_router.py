"""Train the 4-way superclass router (food / flowers / cars / planes).

Backbone: ImageNet-pretrained ConvNeXt-Tiny.
Phase 1 trains only the new classifier head; Phase 2 unfreezes the final
ConvNeXt stage and fine-tunes at a much lower learning rate.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import engine
import hierarchy_data
import utils


class RouterConvNeXt(nn.Module):
    """ConvNeXt-Tiny with the head replaced by a 4-way classifier."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.classifier[2] = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.backbone(x)


def plot_training_history(history: dict, phase_name: str = "Training") -> None:
    """Save a side-by-side loss/accuracy plot for one training phase."""
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


def main() -> None:
    BATCH_SIZE = 32
    SEED = 42
    PHASE1_EPOCHS = 3
    PHASE2_EPOCHS = 8

    RUN_NAME = "router_convnext"
    PHASE1_CKPT = f"./checkpoints/phase1_{RUN_NAME}.pt"
    BEST_CKPT = f"./checkpoints/best_{RUN_NAME}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading router datasets...")
    train_loader, val_loader = hierarchy_data.get_router_dataloaders(
        batch_size=BATCH_SIZE,
        val_ratio=0.15,
        seed=SEED,
    )

    print("\nInitializing router model...")
    model = RouterConvNeXt(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: train the new classifier head only.
    print("\n--- PHASE 1: Warming up the router head ---")
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
        num_classes=4,
        epochs=PHASE1_EPOCHS,
        ckpt_path=PHASE1_CKPT,
        scheduler=scheduler_phase1,
        use_mixup=False,
    )
    plot_training_history(history_phase1, "Router Phase 1")

    # Phase 2: unfreeze the final ConvNeXt stage and fine-tune at a lower LR.
    print("\n--- PHASE 2: Fine-tuning router final stage ---")
    checkpoint = torch.load(PHASE1_CKPT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    for param in model.backbone.features[7].parameters():
        param.requires_grad = True

    optimizer_phase2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-5,
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
        num_classes=4,
        epochs=PHASE2_EPOCHS,
        ckpt_path=BEST_CKPT,
        scheduler=scheduler_phase2,
        use_mixup=False,
    )
    plot_training_history(history_phase2, "Router Phase 2")

    print(f"\nDone training router. Best checkpoint saved to: {BEST_CKPT}")


if __name__ == "__main__":
    utils.set_seed(42)
    main()
