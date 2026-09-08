
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
