# models/resnet18.py

import torch
from torch import nn
from torchvision.models import resnet18 as tv_resnet18


def resnet18(num_classes: int = 10, in_channels: int = 3) -> nn.Module:
    """
    Standard torchvision ResNet18, with:
      - first conv adjusted for in_ch
      - final fc adjusted for num_classes
    """
    # No pretrained weights by default
    m = tv_resnet18(weights=None)

    # Adjust first conv for channel count if needed
    if in_channels != 3:
        old_conv = m.conv1
        m.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

    # Adjust final classifier for num_classes
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)

    return m
