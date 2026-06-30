import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CLASSES, ChestXray14Dataset, get_transforms
from src.model import ChestXrayClassifier


class FocalLoss(nn.Module):
    """Multi-label focal loss for imbalanced chest X-ray classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        prob = torch.sigmoid(inputs)
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma

        if self.alpha >= 0.0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            bce_loss = alpha_t * bce_loss

        return (focal_weight * bce_loss).mean()


def encode_finding_labels(finding_labels: str, class_to_idx: dict[str, int]) -> np.ndarray:
    label_vector = np.zeros(len(CLASSES), dtype=np.float32)

    if finding_labels is None or (isinstance(finding_labels, float) and np.isnan(finding_labels)):
        return label_vector

    for finding in str(finding_labels).split("|"):
        finding = finding.strip()
        if finding and finding != "No Finding" and finding in class_to_idx:
            label_vector[class_to_idx[finding]] = 1.0

    return label_vector


def compute_sample_weights(dataset: ChestXray14Dataset) -> torch.DoubleTensor:
    """Assign higher sampling weight to images containing rare findings."""
    class_counts = np.zeros(len(CLASSES), dtype=np.float64)

    for finding_labels in dataset.df["Finding Labels"]:
        class_counts += encode_finding_labels(finding_labels, dataset.class_to_idx)

    class_weights = len(dataset) / (class_counts + 1.0)

    sample_weights = []
    for finding_labels in dataset.df["Finding Labels"]:
        label_vector = encode_finding_labels(finding_labels, dataset.class_to_idx)
        if label_vector.sum() == 0:
            sample_weights.append(1.0)
        else:
            positive_weights = class_weights[label_vector == 1]
            sample_weights.append(float(positive_weights.max()))

    return torch.DoubleTensor(sample_weights)


def compute_multilabel_auc(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, dict[str, float | None]]:
    """Compute mean and per-class AUC-ROC, skipping classes with a single label value."""
    per_class_auc: dict[str, float | None] = {}
    valid_scores: list[float] = []

    for class_idx, class_name in enumerate(CLASSES):
        y_true = targets[:, class_idx]
        y_score = probabilities[:, class_idx]

        if len(np.unique(y_true)) < 2:
            per_class_auc[class_name] = None
            continue

        auc = roc_auc_score(y_true, y_score)
        per_class_auc[class_name] = float(auc)
        valid_scores.append(float(auc))

    mean_auc = float(np.mean(valid_scores)) if valid_scores else 0.0
    return mean_auc, per_class_auc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, dict[str, float | None]]:
    model.eval()
    running_loss = 0.0
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        all_probabilities.append(probabilities)
        all_targets.append(labels.cpu().numpy())

    avg_loss = running_loss / len(data_loader.dataset)
    targets = np.concatenate(all_targets, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)
    mean_auc, per_class_auc = compute_multilabel_auc(targets, probabilities)

    return avg_loss, mean_auc, per_class_auc


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(data_loader.dataset)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_auc: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_auc": best_auc,
            "classes": CLASSES,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-label chest X-ray classifier.")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing PNG images.")
    parser.add_argument("--train-csv", type=Path, required=True, help="Training CSV path.")
    parser.add_argument("--val-csv", type=Path, required=True, help="Validation CSV path.")
    parser.add_argument("--checkpoint-path", type=Path, default=ROOT / "models" / "best_model.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not args.no_amp

    train_dataset = ChestXray14Dataset(
        image_dir=args.image_dir,
        csv_path=args.train_csv,
        transform=get_transforms("train"),
        mode="train",
    )
    val_dataset = ChestXray14Dataset(
        image_dir=args.image_dir,
        csv_path=args.val_csv,
        transform=get_transforms("val"),
        mode="val",
    )

    sample_weights = compute_sample_weights(train_dataset)
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ChestXrayClassifier(num_classes=len(CLASSES), pretrained=not args.no_pretrained)
    model.to(device)

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)

    best_auc = -1.0
    epochs_without_improvement = 0

    print(f"Training on {device} | AMP: {use_amp}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_loss, val_auc, per_class_auc = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_auc={val_auc:.4f} | lr={current_lr:.2e}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_without_improvement = 0
            save_checkpoint(
                path=args.checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_auc=best_auc,
            )
            print(f"Saved new best model to {args.checkpoint_path} (AUC={best_auc:.4f})")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement}/{args.patience} epoch(s) "
                f"(best AUC={best_auc:.4f})"
            )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered after epoch {epoch}.")
            break

    print("Training complete.")
    print(f"Best validation AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
