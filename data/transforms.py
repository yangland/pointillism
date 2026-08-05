# data/transforms.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import torchvision.transforms as T
except Exception as e:
    raise RuntimeError("torchvision required for transforms") from e


@dataclass(frozen=True)
class TransformCfg:
    dataset: str  # e.g. "cifar10" | "cifar100" | "mnist" | "fmnist" | "svhn" | "emnist"
    train: bool
    blur: bool = False
    blur_kernel: int = 5
    blur_sigma: Tuple[float, float] = (0.5, 1.2)


def _canonical_dataset_name(dataset: str) -> str:
    ds = str(dataset).lower().strip()

    if ds in {"fmnist", "fashion", "fashionmnist", "fashion-mnist"}:
        return "fmnist"
    if ds.startswith("svhn"):
        return "svhn"
    if ds.startswith("cifar100"):
        return "cifar100"
    if ds.startswith("cifar10"):
        return "cifar10"
    return ds


def _norm_cfg(dataset: str):
    ds = _canonical_dataset_name(dataset)

    # -------- grayscale --------
    if ds in ["speech_commands", "speechcommands", "gsc"]:
        return (0.0,), (1.0,)

    if ds in ["mnist", "emnist"]:
        mean = (0.1307,)
        std  = (0.3081,)
        return mean, std

    if ds in ["fmnist", "fashion", "fashionmnist"]:
        mean = (0.2860,)
        std  = (0.3530,)
        return mean, std

    # -------- RGB --------
    if ds == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)
        return mean, std

    if ds == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std  = (0.2675, 0.2565, 0.2761)
        return mean, std

    if ds == "svhn":
        mean = (0.4377, 0.4438, 0.4728)
        std  = (0.1980, 0.2010, 0.1970)
        return mean, std

    if ds == "gtsrb":
        # Commonly used GTSRB normalization (RGB)
        mean = (0.3403, 0.3121, 0.3214)
        std  = (0.2724, 0.2608, 0.2669)
        return mean, std

    if ds == "imagenette":
        # ImageNet normalization (Imagenette ⊂ ImageNet)
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
        return mean, std

    raise ValueError(f"Unknown dataset for normalization: {dataset}")



def build_transform(cfg: TransformCfg):
    mean, std = _norm_cfg(cfg.dataset)

    ops = []

    # PIL-level blur before tensor conversion
    if cfg.blur:
        ops.append(T.GaussianBlur(kernel_size=cfg.blur_kernel, sigma=cfg.blur_sigma))

    # standard
    ops.append(T.ToTensor())
    ops.append(T.Normalize(mean=mean, std=std))

    return T.Compose(ops)

def norm_cfg(dataset: str):
    """
    Public accessor for normalization config used by build_transform().
    Keeps mean/std defined in one place.
    """
    return _norm_cfg(dataset)
