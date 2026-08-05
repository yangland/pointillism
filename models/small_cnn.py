"""Compact convolutional network for 32x32 image classification."""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Four-convolution CNN with global average pooling."""

    def __init__(
        self,
        num_classes: int = 10,
        in_ch: int = 3,
        width: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 4, width * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(width * 4, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


def small_cnn(
    num_classes: int = 10,
    in_channels: int = 3,
    width: int = 16,
    dropout: float = 0.0,
) -> SmallCNN:
    return SmallCNN(
        num_classes=num_classes,
        in_ch=in_channels,
        width=width,
        dropout=dropout,
    )
