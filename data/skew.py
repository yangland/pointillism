# data/skew.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def _extract_labels(ds: Dataset) -> np.ndarray:
    """
    Returns labels as numpy array for torchvision datasets or generic datasets.
    """
    if hasattr(ds, "targets"):
        y = np.array(getattr(ds, "targets"))
        return y.astype(int)

    # fallback: iterate once (slower, but robust)
    labels: List[int] = []
    for i in range(len(ds)):
        _, yi = ds[i]
        labels.append(int(yi))
    return np.array(labels, dtype=int)


@dataclass(frozen=True)
class LabelSkewCfg:
    target_label: int
    alpha: float                 # fraction for target label
    seed: int = 0
    total_size: Optional[int] = None  # if None, use full dataset size


def build_label_skew_subset(
    base_dataset: Dataset,
    *,
    num_classes: int,
    cfg: LabelSkewCfg,
) -> Subset:
    """
    Build a Subset with label histogram controlled by cfg.alpha for cfg.target_label.

    target fraction = alpha
    other classes share remaining equally.

    Notes:
    - If requested counts exceed available class counts, we clip per-class selection.
      The resulting subset may be smaller than total_size in extreme cases.
    """
    assert 0.0 < cfg.alpha < 1.0
    t = int(cfg.target_label)
    K = int(num_classes)

    labels = _extract_labels(base_dataset)
    N = len(labels)

    M = int(cfg.total_size) if cfg.total_size is not None else N
    M = min(M, N)

    # desired counts
    tgt_cnt = int(round(M * float(cfg.alpha)))
    other_cnt_each = int(round((M - tgt_cnt) / float(K - 1)))

    # gather indices per class
    rng = np.random.default_rng(int(cfg.seed))
    idx_by_c: Dict[int, np.ndarray] = {}
    for c in range(K):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        idx_by_c[c] = idx

    pick: List[int] = []

    # pick target
    tgt_idx = idx_by_c.get(t, np.array([], dtype=int))
    pick.extend(tgt_idx[: min(tgt_cnt, tgt_idx.size)].tolist())

    # pick others
    for c in range(K):
        if c == t:
            continue
        idx = idx_by_c.get(c, np.array([], dtype=int))
        pick.extend(idx[: min(other_cnt_each, idx.size)].tolist())

    rng.shuffle(pick)
    return Subset(base_dataset, pick)
