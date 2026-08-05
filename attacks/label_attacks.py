from __future__ import annotations

from typing import Dict

import torch


class LabelMappingWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, label_map: Dict[int, int]):
        self.base = base_dataset
        self.label_map = {int(k): int(v) for k, v in dict(label_map).items()}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        y = int(y)
        return x, self.label_map.get(y, y)


class SelectiveLabelMappingWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, label_map: Dict[int, int], poison_indices):
        self.base = base_dataset
        self.label_map = {int(k): int(v) for k, v in dict(label_map).items()}
        self.poison_set = set(int(i) for i in poison_indices)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        y = int(y)
        if int(idx) in self.poison_set:
            y = self.label_map.get(y, y)
        return x, y


def build_pair_flip_map(pair) -> Dict[int, int]:
    if pair is None or len(pair) != 2:
        raise ValueError("Label flipping requires attack.flip_pair with exactly two labels, e.g. [1, 7].")
    a, b = int(pair[0]), int(pair[1])
    if a == b:
        raise ValueError("Label flipping requires two distinct labels.")
    return {a: b, b: a}


def build_targeted_label_flip_map(source_labels, target_label: int) -> Dict[int, int]:
    tgt = int(target_label)
    if source_labels == "all":
        raise ValueError("build_targeted_label_flip_map requires explicit source labels, not 'all'.")

    srcs = sorted({int(s) for s in source_labels if int(s) != tgt})
    if not srcs:
        raise ValueError("Label flipping requires at least one source label distinct from target_label.")
    return {src: tgt for src in srcs}


def build_label_perturbation_map(num_classes: int) -> Dict[int, int]:
    n = int(num_classes)
    if n <= 1:
        raise ValueError(f"Label perturbation requires num_classes > 1, got {n}.")
    return {c: (c + 1) % n for c in range(n)}
