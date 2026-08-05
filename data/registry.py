# data/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch
from torch.utils.data import Dataset

try:
    from torchvision import datasets
except Exception as e:
    raise RuntimeError("torchvision required for CIFAR10/CIFAR100/FMNIST/MNIST/SVHN/EMNIST datasets") from e


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    in_channels: int
    default_size: int  # (H=W), for convenience


class _TargetRemapDataset(Dataset):
    def __init__(self, base: Dataset, *, map_fn, classes=None):
        self.base = base
        self._map_fn = map_fn

        base_targets = getattr(base, "targets", None)
        if base_targets is not None:
            if torch.is_tensor(base_targets):
                self.targets = torch.tensor(
                    [int(map_fn(int(t))) for t in base_targets.tolist()],
                    dtype=base_targets.dtype,
                )
            else:
                self.targets = [int(map_fn(int(t))) for t in base_targets]

        if classes is not None:
            self.classes = list(classes)
        elif hasattr(base, "classes"):
            self.classes = list(getattr(base, "classes"))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        x, y = self.base[index]
        return x, int(self._map_fn(int(y)))

    def __getattr__(self, name):
        return getattr(self.base, name)


def _build_cifar10(root: str, train: bool, transform) -> Dataset:
    return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)


def _build_cifar100(root: str, train: bool, transform) -> Dataset:
    return datasets.CIFAR100(root=root, train=train, download=True, transform=transform)


def _build_fmnist(root: str, train: bool, transform) -> Dataset:
    return datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)


def _build_mnist(root: str, train: bool, transform) -> Dataset:
    return datasets.MNIST(root=root, train=train, download=True, transform=transform)


def _build_svhn(root: str, train: bool, transform) -> Dataset:
    split = "train" if bool(train) else "test"
    return datasets.SVHN(root=root, split=split, download=True, transform=transform)


def _build_emnist(root: str, train: bool, transform) -> Dataset:
    ds = datasets.EMNIST(root=root, split="letters", train=train, download=True, transform=transform)
    # Torchvision's letters split uses labels 1..26 with a dummy "N/A" class at index 0.
    # Remap to 0..25 so the rest of the codebase can treat it as a normal 26-class dataset.
    return _TargetRemapDataset(ds, map_fn=lambda y: int(y) - 1, classes=list("abcdefghijklmnopqrstuvwxyz"))


def _canonical_dataset_name(name: str) -> str:
    key = str(name).lower().strip()
    aliases = {
        "fashion": "fmnist",
        "fashionmnist": "fmnist",
        "fashion-mnist": "fmnist",
    }
    return aliases.get(key, key)

DATASET_SPECS: Dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(name="cifar10", num_classes=10, in_channels=3, default_size=32),
    "cifar100": DatasetSpec(name="cifar100", num_classes=100, in_channels=3, default_size=32),
    "mnist": DatasetSpec(name="mnist", num_classes=10, in_channels=1, default_size=28),
    "fmnist": DatasetSpec(name="fmnist", num_classes=10, in_channels=1, default_size=28),
    "svhn": DatasetSpec(name="svhn", num_classes=10, in_channels=3, default_size=32),
    "emnist": DatasetSpec(name="emnist", num_classes=26, in_channels=1, default_size=28),
}


DATASET_BUILDERS: Dict[str, Callable[[str, bool, object], Dataset]] = {
    "cifar10": _build_cifar10,
    "cifar100": _build_cifar100,
    "mnist": _build_mnist,
    "fmnist": _build_fmnist,
    "svhn": _build_svhn,
    "emnist": _build_emnist,
}


def get_dataset_spec(name: str) -> DatasetSpec:
    key = _canonical_dataset_name(name)
    if key not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_SPECS.keys())}")
    return DATASET_SPECS[key]


def build_dataset(
    name: str,
    *,
    root: str,
    train: bool,
    transform,
) -> Dataset:
    key = _canonical_dataset_name(name)
    if key not in DATASET_BUILDERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_BUILDERS.keys())}")
    return DATASET_BUILDERS[key](root, train, transform)
