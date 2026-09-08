
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split_indices(targets, val_fraction: float, test_fraction: float, seed: int):
    targets = np.asarray(targets)
    all_idx = np.arange(len(targets))
    holdout_fraction = val_fraction + test_fraction

    train_idx, holdout_idx = train_test_split(
        all_idx,
        test_size=holdout_fraction,
        stratify=targets,
        random_state=seed,
    )

    holdout_targets = targets[holdout_idx]
    relative_test_fraction = test_fraction / holdout_fraction

    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=relative_test_fraction,
        stratify=holdout_targets,
        random_state=seed,
    )

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def compute_class_weights(targets, num_classes: int) -> torch.Tensor:
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.asarray(targets))
    return torch.tensor(weights, dtype=torch.float32)
