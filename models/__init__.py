# models/__init__.py

from models.resnet18 import resnet18
from models.resnet20 import resnet20
from models.mobilenetv2 import mobilenet_v2_cifar
from models.small_cnn import small_cnn


def build_model(*, arch: str, num_classes: int, in_channels: int = 3):
    arch = arch.lower()

    if arch == "resnet18":
        return resnet18(num_classes=num_classes, in_channels=in_channels)

    if arch == "resnet20":
        return resnet20(num_classes=num_classes, in_channels=in_channels)

    if arch == "mobilenetv2":
        return mobilenet_v2_cifar(num_classes=num_classes, in_channels=in_channels)

    if arch in ("smallcnn", "small_cnn"):
        return small_cnn(num_classes=num_classes, in_channels=in_channels)

    raise ValueError(f"Unknown model architecture: {arch}")
