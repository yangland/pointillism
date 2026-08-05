from __future__ import annotations

import torch
from torch import nn


class AudioCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_ch: int = 1, width: int = 32, dropout: float = 0.2):
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
        z = self.features(x)
        z = self.pool(z).flatten(1)
        z = self.dropout(z)
        return self.classifier(z)


def audio_cnn(num_classes: int = 10, in_ch: int = 1, width: int = 32, dropout: float = 0.2) -> AudioCNN:
    return AudioCNN(num_classes=num_classes, in_ch=in_ch, width=width, dropout=dropout)
