# CIFAR-style ResNet-20 for 32x32 inputs, adapted for 10 classes
import torch
from torch import nn
import torch.nn.functional as F

def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_blocks=[3,3,3], num_classes=10, in_ch=3):
        super().__init__()
        self.in_planes = 16
        self.conv1 = _conv3x3(in_ch, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out).flatten(1)
        out = self.fc(out)
        return out


def resnet20(num_classes=10, in_channels=3):
    return ResNet([3,3,3], num_classes=num_classes, in_ch=in_channels)

def resnet8(num_classes=10, in_channels=3):
    # depth = 6*1 + 2 = 8
    return ResNet([1,1,1], num_classes=num_classes, in_ch=in_channels)

def resnet32(num_classes=10, in_channels=3):
    return ResNet([5,5,5], num_classes=num_classes, in_ch=in_channels)

def resnet44(num_classes=10, in_channels=3):
    return ResNet([7,7,7], num_classes=num_classes, in_ch=in_channels)

def resnet56(num_classes=10, in_channels=3):
    return ResNet([9,9,9], num_classes=num_classes, in_ch=in_channels)