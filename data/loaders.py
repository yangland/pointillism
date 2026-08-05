# data/loaders.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from .registry import build_dataset, get_dataset_spec
from .transforms import TransformCfg, build_transform
from .skew import LabelSkewCfg, build_label_skew_subset


def _seed_worker(worker_id: int):
    # deterministic worker seeds across DataLoader workers
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass(frozen=True)
class LoaderCfg:
    dataset: str
    data_root: str = "./data"
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    seed: int = 0
    data_fraction: float = 1.0

def _apply_data_fraction(ds, *, frac: float, seed: int):
    frac = float(frac)
    if frac >= 1.0:
        return ds
    if frac <= 0.0:
        raise ValueError(f"data_fraction must be in (0,1], got {frac}")

    n = len(ds)
    m = max(1, int(round(n * frac)))

    rng = np.random.default_rng(int(seed))
    idx = np.arange(n)
    rng.shuffle(idx)
    idx = idx[:m].tolist()
    return Subset(ds, idx)

def build_train_loader(cfg: LoaderCfg, *, label_skew: LabelSkewCfg | None = None) -> DataLoader:
    tfm = build_transform(TransformCfg(dataset=cfg.dataset, train=True, blur=False))
    ds = build_dataset(cfg.dataset, root=cfg.data_root, train=True, transform=tfm)

    if label_skew is not None:
        spec = get_dataset_spec(cfg.dataset)
        ds = build_label_skew_subset(ds, num_classes=spec.num_classes, cfg=label_skew)

    # apply fraction after skew
    ds = _apply_data_fraction(ds, frac=cfg.data_fraction, seed=cfg.seed)

    g = torch.Generator()
    g.manual_seed(int(cfg.seed))

    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        persistent_workers=bool(cfg.persistent_workers) and int(cfg.num_workers) > 0,
        worker_init_fn=_seed_worker,
        generator=g,
    )



def build_test_loaders(
    cfg: LoaderCfg,
    *,
    blur_levels: Optional[Dict[str, Dict]] = None,
) -> Dict[str, DataLoader]:
    """
    Returns dict:
      - "clean": DataLoader
      - plus keys from blur_levels (e.g., "blur1", "blur2")
    blur_levels example:
      {
        "blur1": {"blur_kernel": 3, "blur_sigma": (0.5, 0.8)},
        "blur2": {"blur_kernel": 7, "blur_sigma": (1.0, 1.5)}
      }
    """
    loaders: Dict[str, DataLoader] = {}

    def _make_loader(blur: bool, blur_kernel: int = 5, blur_sigma=(0.5, 1.2)) -> DataLoader:
        tfm = build_transform(
            TransformCfg(
                dataset=cfg.dataset,
                train=False,
                blur=blur,
                blur_kernel=int(blur_kernel),
                blur_sigma=tuple(blur_sigma),
            )
        )
        ds = build_dataset(cfg.dataset, root=cfg.data_root, train=False, transform=tfm)

        return DataLoader(
            ds,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=int(cfg.num_workers),
            pin_memory=bool(cfg.pin_memory),
            persistent_workers=bool(cfg.persistent_workers) and int(cfg.num_workers) > 0,
            worker_init_fn=_seed_worker,
        )

    loaders["clean"] = _make_loader(blur=False)

    if blur_levels:
        for name, bcfg in blur_levels.items():
            loaders[name] = _make_loader(
                blur=True,
                blur_kernel=bcfg.get("blur_kernel", 5),
                blur_sigma=bcfg.get("blur_sigma", (0.5, 1.2)),
            )

    return loaders
