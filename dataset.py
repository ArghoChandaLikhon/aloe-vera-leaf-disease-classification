"""
Dataset loading for the Aloe Vera Healthy / Rot / Rust classifier.

Expects config.DATA_DIR laid out as:
    data/aloe_vera/
        healthy/  *.jpg
        rot/      *.jpg
        rust/     *.jpg

Uses two ImageFolder instances over the SAME directory (one with training
augmentation, one without) and slices both with matching Subset indices, so
the same physical image is never used for both training and evaluation.
"""

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import config
from utils import compute_class_weights, stratified_split_indices

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


def get_dataloaders():
    train_transform, eval_transform = build_transforms(config.IMG_SIZE)

    train_view = datasets.ImageFolder(config.DATA_DIR, transform=train_transform)
    eval_view = datasets.ImageFolder(config.DATA_DIR, transform=eval_transform)

    if train_view.classes != config.CLASS_NAMES:
        raise ValueError(
            f"Found classes {train_view.classes} in {config.DATA_DIR}, "
            f"expected {config.CLASS_NAMES}. Check your folder names match exactly."
        )

    targets = [label for _, label in train_view.samples]
    train_idx, val_idx, test_idx = stratified_split_indices(
        targets, config.VAL_FRACTION, config.TEST_FRACTION, config.SEED
    )

    train_ds = Subset(train_view, train_idx)
    val_ds = Subset(eval_view, val_idx)
    test_ds = Subset(eval_view, test_idx)

    train_targets = [targets[i] for i in train_idx]
    class_weights = compute_class_weights(train_targets, config.NUM_CLASSES)

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    print(f"Dataset split -> train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")
    print(f"Classes (index order): {train_view.classes}")

    return train_loader, val_loader, test_loader, class_weights
