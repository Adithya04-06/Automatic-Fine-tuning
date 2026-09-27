"""Model factory for pretrained deep learning backbones with customizable classification heads."""

from typing import Optional
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    efficientnet_b0,
    resnet18,
    resnet34,
    resnet50,
    vit_b_16,
    vit_b_32,
)


def build_model(
    backbone_name: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    """Builds a backbone model (ResNet-18, EfficientNet-B0, ViT, etc.) and replaces the classification head.
    
    Args:
        backbone_name: Name of the architecture ('resnet18', 'efficientnet_b0', 'vit_b_16', etc.).
        num_classes: Number of output target classes.
        pretrained: Whether to load ImageNet pretrained weights.
        dropout: Optional dropout rate before the final linear layer.

    Returns:
        A PyTorch nn.Module instance ready for fine-tuning.
    """
    name = backbone_name.lower().replace("-", "_").strip()

    # 1. ResNet Family
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        if dropout > 0.0:
            model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
        else:
            model.fc = nn.Linear(in_features, num_classes)
        return model

    elif name == "resnet34":
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        model = resnet34(weights=weights)
        in_features = model.fc.in_features
        if dropout > 0.0:
            model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
        else:
            model.fc = nn.Linear(in_features, num_classes)
        return model

    elif name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        if dropout > 0.0:
            model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
        else:
            model.fc = nn.Linear(in_features, num_classes)
        return model

    # 2. EfficientNet Family
    elif name in ("efficientnet_b0", "efficientnet"):
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        drop_p = dropout if dropout > 0.0 else 0.2
        model.classifier = nn.Sequential(
            nn.Dropout(p=drop_p),
            nn.Linear(in_features, num_classes),
        )
        return model

    # 3. Vision Transformer (ViT) Family
    elif name in ("vit_b_16", "vit", "vision_transformer"):
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        if dropout > 0.0:
            model.heads = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            model.heads.head = nn.Linear(in_features, num_classes)
        return model

    elif name == "vit_b_32":
        weights = ViT_B_32_Weights.DEFAULT if pretrained else None
        model = vit_b_32(weights=weights)
        in_features = model.heads.head.in_features
        if dropout > 0.0:
            model.heads = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            model.heads.head = nn.Linear(in_features, num_classes)
        return model

    else:
        supported = ["resnet18", "resnet34", "resnet50", "efficientnet_b0", "vit_b_16", "vit_b_32"]
        raise ValueError(f"Unsupported backbone: '{backbone_name}'. Supported backbones: {supported}")
