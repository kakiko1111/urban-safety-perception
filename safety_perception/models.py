"""Model builders: ResNet50 (torchvision) and ViT-B/16 (timm), both with a 2-class head."""

import timm
import torch.nn as nn
from torchvision import models


def build_model(arch: str, pretrained: bool = True) -> nn.Module:
    if arch == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if arch == "vit_b16":
        return timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=2)
    raise ValueError(f"Unknown architecture: {arch}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
