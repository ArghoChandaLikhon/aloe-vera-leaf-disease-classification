"""
Evaluate the best saved checkpoint on the held-out test set.

Run with:  python evaluate.py

Writes a classification report (txt) and a confusion matrix figure (png) to
outputs/ -- both go straight into the paper's Results section.
"""

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import config
from dataset import get_dataloaders
from model import build_model


def plot_confusion_matrix(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix -- Test Set")

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)

    _, _, test_loader, _ = get_dataloaders()

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test accuracy: {acc:.4f}\n")
    print(report)

    with open(config.REPORT_PATH, "w") as f:
        f.write(f"Backbone: {checkpoint.get('backbone', config.BACKBONE)}\n")
        f.write(f"Test accuracy: {acc:.4f}\n\n")
        f.write(report)

    plot_confusion_matrix(cm, config.CLASS_NAMES, config.CONFUSION_MATRIX_PATH)
    print(f"\nSaved report to {config.REPORT_PATH}")
    print(f"Saved confusion matrix to {config.CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
