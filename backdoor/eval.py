# backdoor/eval.py
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .badnet import resolve_origin_from_pos, add_pattern_trigger, resolve_pattern_size


class AttackEvalDataset(Dataset):
    """
    Wrap a clean test set and return poisoned samples for ASR evaluation.

    For each item: (poisoned_x, target_label, original_label)
    """
    def __init__(self, base_dataset: Dataset, pick_indices, attack_cfg: dict):
        self.base = base_dataset
        self.pick = list(pick_indices)
        self.cfg = dict(attack_cfg)

        self.target_label = int(self.cfg.get("target_label", -1))
        self.value = float(self.cfg.get("value", 1.0))

        self.pattern_type = (self.cfg.get("pattern_type") or "badnet_corner").lower()
        self.pattern_pos = self.cfg.get("pattern_pos", "bottom_right")
        self.pattern_padding = int(self.cfg.get("pattern_padding", 0))
        self.pattern_offsets = self.cfg.get("pattern_offsets", None)

        self.pattern_size = resolve_pattern_size(
            pattern_type=self.pattern_type,
            pattern_size=self.cfg.get("pattern_size", None),
            pattern_offsets=self.pattern_offsets,
        )

        self._resolved_origin = None
        if self.cfg.get("pattern_origin", None) is not None:
            o = self.cfg["pattern_origin"]
            self._resolved_origin = (int(o[0]), int(o[1]))

    def __len__(self):
        return len(self.pick)

    def _resolve_origin_if_needed(self):
        if self._resolved_origin is not None:
            return self._resolved_origin
        sample_img, _ = self.base[0]
        self._resolved_origin = resolve_origin_from_pos(
            sample_img,
            pattern_size=self.pattern_size,
            pattern_pos=self.pattern_pos,
            padding=self.pattern_padding,
        )
        return self._resolved_origin

    def __getitem__(self, idx):
        base_idx = self.pick[idx]
        x, y = self.base[base_idx]
        origin = self._resolve_origin_if_needed()
        xt = add_pattern_trigger(
            x,
            origin_x=origin[0],
            origin_y=origin[1],
            offsets=self.pattern_offsets,
            value=self.value,
            pattern_type=self.pattern_type,
        )
        return xt, self.target_label, int(y)


class LabelFlipEvalDataset(Dataset):
    """
    Wrap a clean test set for targeted label-flip ASR evaluation.

    For each item: (clean_x, target_label, original_label)
    """

    def __init__(self, base_dataset: Dataset, pick_indices, attack_cfg: dict):
        self.base = base_dataset
        self.pick = list(pick_indices)
        self.cfg = dict(attack_cfg)
        self.target_label = int(self.cfg.get("target_label", -1))

    def __len__(self):
        return len(self.pick)

    def __getitem__(self, idx):
        base_idx = self.pick[idx]
        x, y = self.base[base_idx]
        return x, self.target_label, int(y)


def _extract_labels_fast(ds: Dataset) -> np.ndarray:
    """
    Fast label extraction for datasets used in ASR eval selection.

    Prefer metadata fields that do not require loading/transforms every image.
    """
    if hasattr(ds, "targets"):
        return np.asarray(getattr(ds, "targets"), dtype=int)
    if hasattr(ds, "labels"):
        return np.asarray(getattr(ds, "labels"), dtype=int)
    if hasattr(ds, "_samples"):
        samples = getattr(ds, "_samples")
        return np.asarray([int(y) for _, y in samples], dtype=int)
    return np.asarray([int(ds[i][1]) for i in range(len(ds))], dtype=int)


def build_backdoor_evalset(test_set: Dataset, attack_cfg: dict, per_class: int = 50) -> AttackEvalDataset:
    labels = _extract_labels_fast(test_set)

    target_label = int(attack_cfg["target_label"])

    src = attack_cfg.get("source_labels", "all")
    if src == "all":
        classes = sorted(set(labels.tolist()))
        classes = [c for c in classes if c != target_label]  # key change
    else:
        classes = [int(s) for s in src if int(s) != target_label]

    rng = np.random.default_rng(int(attack_cfg.get("seed", 0)))
    pick = []
    for c in classes:
        idxs = np.where(labels == c)[0]
        if idxs.size == 0:
            continue
        if idxs.size > per_class:
            idxs = rng.choice(idxs, size=per_class, replace=False)
        pick.extend(idxs.tolist())

    return AttackEvalDataset(test_set, pick, attack_cfg)


def build_label_flip_evalset(test_set: Dataset, attack_cfg: dict, per_class: int = 50) -> LabelFlipEvalDataset:
    labels = _extract_labels_fast(test_set)

    target_label = int(attack_cfg["target_label"])

    src = attack_cfg.get("source_labels", None)
    if src is None:
        flip_pair = attack_cfg.get("flip_pair", attack_cfg.get("label_pair", None))
        if flip_pair is None or len(flip_pair) != 2:
            raise ValueError(
                "Label flipping ASR evaluation requires attack.source_labels "
                "or a valid attack.flip_pair/label_pair."
            )
        classes = [int(c) for c in flip_pair if int(c) != target_label]
    elif src == "all":
        classes = sorted(set(labels.tolist()))
        classes = [c for c in classes if c != target_label]
    else:
        classes = [int(s) for s in src if int(s) != target_label]

    rng = np.random.default_rng(int(attack_cfg.get("seed", 0)))
    pick = []
    for c in classes:
        idxs = np.where(labels == c)[0]
        if idxs.size == 0:
            continue
        if idxs.size > per_class:
            idxs = rng.choice(idxs, size=per_class, replace=False)
        pick.extend(idxs.tolist())

    return LabelFlipEvalDataset(test_set, pick, attack_cfg)
