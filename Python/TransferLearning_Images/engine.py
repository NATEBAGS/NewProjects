import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm.auto import tqdm


def _resolve_batch_aug(
    use_mixup: bool,
    batch_aug_mode: Optional[str],
    num_classes: int,
    mixup_alpha: float,
    cutmix_alpha: float,
):
    """Return a single torchvision batch-augmentation transform, or None.

    Supports both call styles used by the training scripts:
      - older `use_mixup=True/False` (router scripts)
      - newer `batch_aug_mode in {"none", "mixup", "cutmix", "mixup_cutmix"}` (specialists)
    """
    if batch_aug_mode is None:
        batch_aug_mode = "mixup" if use_mixup else "none"

    if batch_aug_mode == "none":
        return None
    if batch_aug_mode == "mixup":
        return v2.MixUp(num_classes=num_classes, alpha=mixup_alpha)
    if batch_aug_mode == "cutmix":
        return v2.CutMix(num_classes=num_classes, alpha=cutmix_alpha)
    if batch_aug_mode == "mixup_cutmix":
        # Randomly pick MixUp or CutMix per batch.
        return v2.RandomChoice([
            v2.MixUp(num_classes=num_classes, alpha=mixup_alpha),
            v2.CutMix(num_classes=num_classes, alpha=cutmix_alpha),
        ])

    raise ValueError(f"Unknown batch_aug_mode: {batch_aug_mode!r}")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    use_mixup: bool = True,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    batch_aug_mode: Optional[str] = None,
):
    """Run one training epoch and return (avg_loss, accuracy_percent)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    batch_aug = _resolve_batch_aug(
        use_mixup=use_mixup,
        batch_aug_mode=batch_aug_mode,
        num_classes=num_classes,
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
    )

    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        if batch_aug is not None:
            inputs, targets = batch_aug(inputs, targets)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += inputs.size(0)

        # MixUp/CutMix turn integer targets into soft (one-hot-ish) targets.
        true_classes = targets.argmax(dim=1) if targets.ndim == 2 else targets
        correct += (predicted == true_classes).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
):
    """Run one validation pass and return (avg_loss, accuracy_percent)."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, desc="Validating", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    return val_loss / total, 100.0 * correct / total


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    epochs: int = 10,
    ckpt_path: str = "./checkpoints/best_model.pt",
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    use_mixup: bool = True,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    batch_aug_mode: Optional[str] = None,
):
    """Train `model` for `epochs`, saving the best-validation checkpoint to `ckpt_path`."""
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch = -1

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            batch_aug_mode=batch_aug_mode,
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": best_val_acc,
                },
                ckpt_path,
            )
            print(f"--> Best model saved to {ckpt_path}")

        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Learning rate stepped to: {current_lr:.6f}")

    print(
        f"\nTraining complete. Best validation accuracy: "
        f"{best_val_acc:.2f}% at epoch {best_epoch}."
    )
    return history
