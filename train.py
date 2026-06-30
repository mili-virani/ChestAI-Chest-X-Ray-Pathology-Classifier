import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/kaggle/working')
from dataset import ChestXray14Dataset, build_image_lookup, CLASSES
from model import ChestXrayClassifier

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

def get_sampler(dataset):
    import pandas as pd
    df = dataset.df
    class_counts = np.zeros(len(CLASSES))
    for labels in df['Finding Labels']:
        if pd.isna(labels): continue
        for finding in labels.split('|'):
            finding = finding.strip()
            if finding in dataset.class_to_idx:
                class_counts[dataset.class_to_idx[finding]] += 1
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = []
    for labels in df['Finding Labels']:
        if pd.isna(labels):
            sample_weights.append(class_weights.min())
            continue
        findings = [f.strip() for f in labels.split('|') if f.strip() in dataset.class_to_idx]
        w = max((class_weights[dataset.class_to_idx[f]] for f in findings), default=class_weights.min())
        sample_weights.append(w)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device).float()
        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        if i % 200 == 0:
            print(f"  step {i}/{len(loader)}  loss={loss.item():.4f}", flush=True)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    try:
        auc = roc_auc_score(labels, preds, average='macro')
    except:
        auc = 0.0
    return total_loss / len(loader), auc

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float()
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    try:
        auc = roc_auc_score(labels, preds, average='macro')
        per_class = roc_auc_score(labels, preds, average=None)
    except:
        auc, per_class = 0.0, [0.0]*len(CLASSES)
    return total_loss / len(loader), auc, per_class

def main():
    IMAGE_BASE  = '/kaggle/input/datasets/organizations/nih-chest-xrays/data'
    TRAIN_CSV   = '/kaggle/working/train.csv'
    VAL_CSV     = '/kaggle/working/val.csv'
    OUTPUT_DIR  = '/kaggle/working/models'
    EPOCHS      = 30        # more epochs
    BATCH_SIZE  = 32
    LR          = 3e-4      # higher LR with warmup
    PATIENCE    = 8         # more patience
    NUM_WORKERS = 2
    WARMUP_EPOCHS = 3       # warmup for 3 epochs

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Building image lookup...", flush=True)
    image_lookup = build_image_lookup(IMAGE_BASE)
    print(f"Found {len(image_lookup)} images")

    train_ds = ChestXray14Dataset(TRAIN_CSV, image_lookup, mode='train')
    val_ds   = ChestXray14Dataset(VAL_CSV,   image_lookup, mode='val')
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    sampler = get_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model     = ChestXrayClassifier(num_classes=14, pretrained=True).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Warmup then cosine decay
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler('cuda')

    best_auc   = 0.0
    no_improve = 0
    history    = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*50}\nEpoch {epoch}/{EPOCHS}  LR={optimizer.param_groups[0]['lr']:.2e}")
        t0 = time.time()

        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_auc, per_class_auc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  train_loss={train_loss:.4f}  train_auc={train_auc:.4f}")
        print(f"  val_loss={val_loss:.4f}    val_auc={val_auc:.4f}  ({elapsed:.0f}s)")
        print("  Per-class AUC:")
        for cls, auc in zip(CLASSES, per_class_auc):
            print(f"    {cls:<22} {auc:.4f}")

        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'train_auc': train_auc, 'val_loss': val_loss, 'val_auc': val_auc})

        if val_auc > best_auc:
            best_auc = val_auc
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': float(val_auc),                                          # ← cast this too
                'per_class_auc': {cls: float(auc) for cls, auc in zip(CLASSES, per_class_auc)},  # ← fix here
            }, os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"  ✓ Saved best model (val_auc={best_auc:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement {no_improve}/{PATIENCE}")
            if no_improve >= PATIENCE:
                print("Early stopping.")
                break

    with open(os.path.join(OUTPUT_DIR, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nDone! Best val AUC: {best_auc:.4f}")

if __name__ == '__main__':
    main()
