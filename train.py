
import csv
import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import build_model
from utils import set_seed


AMP_ENABLED = config.USE_AMP and config.DEVICE.type == "cuda"


def run_epoch(model, loader, criterion, optimizer, scaler, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=config.DEVICE.type, enabled=AMP_ENABLED):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1


def main():
    set_seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    train_loader, val_loader, _, class_weights = get_dataloaders()

    model = build_model()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.DEVICE))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=config.LR_PATIENCE
    )
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    print(
        f"Training {config.BACKBONE} on {config.DEVICE} | "
        f"img_size={config.IMG_SIZE} batch_size={config.BATCH_SIZE} amp={AMP_ENABLED}"
    )

    for epoch in range(1, config.EPOCHS + 1):
        start = time.time()

        train_loss, train_f1 = run_epoch(model, train_loader, criterion, optimizer, scaler, train=True)
        val_loss, val_f1 = run_epoch(model, val_loader, criterion, optimizer, scaler, train=False)

        scheduler.step(val_f1)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch:02d}/{config.EPOCHS} | "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} | {elapsed:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_f1": val_f1,
            }
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": config.BACKBONE,
                    "class_names": config.CLASS_NAMES,
                    "img_size": config.IMG_SIZE,
                    "val_f1": val_f1,
                },
                config.CHECKPOINT_PATH,
            )
            print(f"  -> new best model saved (val_f1={val_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(f"No improvement for {config.EARLY_STOPPING_PATIENCE} epochs -- stopping early.")
                break

    with open(config.HISTORY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"Done. Best val macro-F1: {best_val_f1:.4f}. Checkpoint saved to: {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
