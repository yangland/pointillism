# models/mobilenetv2.py

import torch
from torch import nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            # pointwise
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        # depthwise
        layers.append(
            ConvBNReLU(
                hidden_dim,
                hidden_dim,
                stride=stride,
                groups=hidden_dim,
                kernel_size=3,
            )
        )
        # pointwise-linear
        layers.append(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        )
        layers.append(nn.BatchNorm2d(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2CIFAR(nn.Module):
    """
    MobileNetV2 adapted for CIFAR-style inputs (C x 32 x 32).

    width_mult controls channel scaling.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_ch: int = 3,
        width_mult: float = 1.0,
        round_nearest: int = 8,
    ):
        super().__init__()

        block = InvertedResidual

        # (t, c, n, s) from MobileNetV2; strides adjusted for 32x32
        # s=1 on second stage to avoid too aggressive shrinking early.
        inverted_residual_setting = [
            # expand, out_channels, num_blocks, stride
            (1, 16, 1, 1),
            (6, 24, 2, 1),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        input_channel = _make_divisible(32 * width_mult, round_nearest)
        last_channel = _make_divisible(1280 * max(1.0, width_mult), round_nearest)

        # first layer: keep stride=1 for 32x32 inputs
        features = [ConvBNReLU(in_ch, input_channel, kernel_size=3, stride=1)]

        # inverted residual stages
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # last conv
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_v2_cifar(
    num_classes: int = 10,
    in_channels: int = 3,
    width_mult: float = 1.0,
) -> MobileNetV2CIFAR:
    return MobileNetV2CIFAR(
        num_classes=num_classes,
        in_ch=in_channels,
        width_mult=width_mult,
    )
