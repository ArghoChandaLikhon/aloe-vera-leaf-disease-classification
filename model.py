"""Model builder. Swap config.BACKBONE to change architectures -- nothing
else in the codebase needs to change (timm handles the classifier head
resizing for any backbone name)."""

import timm
import torch.nn as nn

import config


def build_model() -> nn.Module:
    model = timm.create_model(
        config.BACKBONE,
        pretrained=config.PRETRAINED,
        num_classes=config.NUM_CLASSES,
    )
    return model.to(config.DEVICE)
