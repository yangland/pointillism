#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import yaml
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torchvision.transforms as T

from backdoor.badnet import PatternBackdoorWrapper
from backdoor.eval import build_backdoor_evalset
from build_data_labelskew import _as_numpy_targets, _load_dataset
from data.loaders import _seed_worker
from data.registry import get_dataset_spec
from data.transforms import norm_cfg
from pointillism.pointillism_signature import run_pointillism_search
from pointillism.mcp_analysis import (
    compute_all_pairs_edge_consistency_rows,
    compute_all_pairs_topk_consistency_rows,
    compute_edge_transfer_pair_rows,
    compute_edge_consistency_pair_rows,
    compute_focus_pair_edge_consistency_rows,
    compute_focus_pair_topk_consistency_rows,
    compute_topl_transfer_pair_rows,
    compute_topk_consistency_pair_rows,
)
from utils.norm import _norm_cfg

from case_study_exp.common import (
    RunLogger,
    build_random_init_model,
    ensure_dir,
    now_tag,
    resolve_device,
    save_signature_outputs,
    set_seed,
    write_csv,
)


class NormalizeWrapper(torch.utils.data.Dataset):
    def __init__(self, base: torch.utils.data.Dataset, *, dataset: str):
        self.base = base
        mean, std = norm_cfg(dataset)
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        x = (x - self.mean) / self.std
        return x, y


class NormalizeAttackEvalWrapper(torch.utils.data.Dataset):
    def __init__(self, base: torch.utils.data.Dataset, *, dataset: str):
        self.base = base
        mean, std = norm_cfg(dataset)
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, tgt, orig = self.base[idx]
        x = (x - self.mean) / self.std
        return x, tgt, orig


def _make_loader(
    ds,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int,
):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers) and int(num_workers) > 0,
        worker_init_fn=_seed_worker,
        generator=g,
        drop_last=False,
    )


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * x.shape[0]
        count += x.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = torch.argmax(model(x), dim=1)
        correct += int((pred == y).sum().cpu())
        total += y.numel()
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def eval_asr(model, loader, device):
    model.eval()
    hit = 0
    total = 0
    for x, tgt, _orig in loader:
        x = x.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        pred = torch.argmax(model(x), dim=1)
        hit += int((pred == tgt).sum().cpu())
        total += tgt.numel()
    return 100.0 * hit / max(total, 1)


def _sample_fixed_size_subset(dataset, *, sample_size: int, seed: int) -> Subset:
    n = len(dataset)
    m = int(sample_size)
    if m <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    if m > n:
        raise ValueError(f"sample_size={m} exceeds dataset size={n}")

    rng = np.random.default_rng(int(seed))
    idx = np.arange(n)
    rng.shuffle(idx)
    chosen = idx[:m].tolist()
    return Subset(dataset, chosen)


def _sample_disjoint_subsets(
    dataset,
    *,
    num_splits: int,
    subset_size: int,
    seed: int,
) -> List[Subset]:
    n = len(dataset)
    k = int(num_splits)
    m = int(subset_size)
    if k <= 0 or m <= 0:
        raise ValueError(f"Need positive num_splits and subset_size, got {num_splits}, {subset_size}")
    total = k * m
    if total > n:
        raise ValueError(
            f"Requested {k} disjoint subsets of size {m} (total {total}) but dataset has only {n} samples."
        )

    rng = np.random.default_rng(int(seed))
    idx = np.arange(n)
    rng.shuffle(idx)

    out: List[Subset] = []
    for split_idx in range(k):
        start = split_idx * m
        end = (split_idx + 1) * m
        out.append(Subset(dataset, idx[start:end].tolist()))
    return out


def _effective_label_counts(dataset, *, num_classes: int) -> List[int]:
    if isinstance(dataset, NormalizeWrapper):
        return _effective_label_counts(dataset.base, num_classes=int(num_classes))

    if isinstance(dataset, Subset):
        labels = _as_numpy_targets(dataset)
        counts = np.bincount(labels, minlength=int(num_classes))
        return [int(x) for x in counts.tolist()]

    if isinstance(dataset, PatternBackdoorWrapper):
        counts = [0 for _ in range(int(num_classes))]
        for _x, y in dataset:
            counts[int(y)] += 1
        return counts

    labels = _as_numpy_targets(dataset)
    counts = np.bincount(labels, minlength=int(num_classes))
    return [int(x) for x in counts.tolist()]


def _safe_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if not np.isnan(float(v))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _load_domain_datasets(*, domain: str, data_root: str):
    spec = get_dataset_spec(domain)
    out_hw = int(spec.default_size)
    raw_transform = T.Compose([
        T.Resize((out_hw, out_hw)),
        T.ToTensor(),
    ])
    train_raw, test_raw, inferred_num_classes = _load_dataset(
        ds_name=domain,
        data_root=data_root,
        transform=raw_transform,
    )
    return train_raw, test_raw, int(inferred_num_classes), out_hw


def _canonical_pair_key(case_a: str, case_b: str) -> Tuple[str, str]:
    a = str(case_a)
    b = str(case_b)
    return (a, b) if a <= b else (b, a)


def _extract_case_support_labels(case_payload: Dict[str, Any]) -> List[int]:
    q1 = dict(case_payload.get("q1", {}))
    labels = q1.get("effective_support_labels", None)
    if labels is None:
        labels = q1.get("support_labels", None)
    if labels is None:
        topl_by_label = dict(case_payload.get("topl_by_label", {}))
        labels = sorted(int(k) for k in topl_by_label.keys())
    return sorted({int(x) for x in labels})


def build_shared_support_lookup(
    *,
    case_names: List[str],
    case_results: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, List[int]], Dict[Tuple[str, str], List[int]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    support_by_case: Dict[str, List[int]] = {}
    case_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    for case_name in case_names:
        labels = _extract_case_support_labels(case_results[str(case_name)])
        support_by_case[str(case_name)] = [int(x) for x in labels]
        q1 = dict(case_results[str(case_name)].get("q1", {}))
        case_rows.append({
            "case": str(case_name),
            "support_size": int(len(labels)),
            "support_labels": ",".join(str(x) for x in labels),
            "R_f": float(q1.get("R_f", float("nan"))),
            "tau_R": q1.get("tau_R", None),
            "is_discriminative": q1.get("is_discriminative", None),
        })

    shared_by_pair: Dict[Tuple[str, str], List[int]] = {}
    ordered_names = [str(x) for x in case_names]
    for idx_a, case_a in enumerate(ordered_names):
        labels_a = set(support_by_case.get(str(case_a), []))
        for idx_b in range(idx_a, len(ordered_names)):
            case_b = ordered_names[idx_b]
            labels_b = set(support_by_case.get(str(case_b), []))
            shared = sorted(labels_a & labels_b)
            key = _canonical_pair_key(case_a, case_b)
            shared_by_pair[key] = [int(x) for x in shared]
            pair_rows.append({
                "case_a": str(case_a),
                "case_b": str(case_b),
                "shared_support_size": int(len(shared)),
                "shared_support_labels": ",".join(str(x) for x in shared),
            })

    return support_by_case, shared_by_pair, case_rows, pair_rows


def _shared_labels_for_pair(
    *,
    shared_labels_by_pair: Optional[Dict[Tuple[str, str], List[int]]],
    case_a: str,
    case_b: str,
    num_classes: int,
) -> List[int]:
    if shared_labels_by_pair is None:
        return [int(x) for x in range(int(num_classes))]
    return [
        int(x)
        for x in shared_labels_by_pair.get(_canonical_pair_key(case_a, case_b), [])
    ]


def _display_name_map(cross_domain_case: str) -> Dict[str, str]:
    return {
        "benign1": "benign1",
        "benign2": "benign2",
        "backdoor": "backdoored",
        str(cross_domain_case): "diff domain",
    }


def _get_plot_cmap(name: str):
    key = str(name).lower()
    if key == "parula":
        # Compact parula-style approximation for publication heatmaps.
        colors = [
            "#352a87", "#2f3594", "#2941a1", "#2450ad", "#1f5eb8", "#1d6cc0",
            "#1f79c5", "#2687c7", "#3093c7", "#3c9fc5", "#49aac1", "#57b4bb",
            "#67bdb2", "#78c4a7", "#8acb9b", "#9dd08e", "#b1d482", "#c5d777",
            "#d8d96d", "#e9da67", "#f6db66", "#f9df72", "#f9e78a", "#f6efab",
        ]
        return LinearSegmentedColormap.from_list("parula_custom", colors, N=256)
    return plt.get_cmap(str(name))


def compute_symmetric_topk_consensus_rows(
    *,
    raw_rows: List[Dict[str, Any]],
    focus_pairs: List[Dict[str, str]],
    num_classes: int,
    shared_labels_by_pair: Optional[Dict[Tuple[str, str], List[int]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    directed: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for row in raw_rows:
        key = (
            str(row["source_model"]),
            str(row["eval_model"]),
            int(row["label"]),
        )
        directed[key] = row

    out_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for spec in focus_pairs:
        case_a = str(spec["case_a"])
        case_b = str(spec["case_b"])
        group = str(spec["group"])
        display_name = str(spec["display_name"])

        vals_bar_t: List[float] = []
        num_available = 0
        shared_labels = set(
            _shared_labels_for_pair(
                shared_labels_by_pair=shared_labels_by_pair,
                case_a=case_a,
                case_b=case_b,
                num_classes=int(num_classes),
            )
        )

        for c in range(int(num_classes)):
            if int(c) not in shared_labels:
                continue
            row_ab = directed.get((case_a, case_b, int(c)))
            row_ba = directed.get((case_b, case_a, int(c)))
            if row_ab is None or row_ba is None:
                continue

            t_ab = float(row_ab["avg_target_softmax"])
            t_ba = float(row_ba["avg_target_softmax"])
            bar_t = 0.5 * (t_ab + t_ba)
            vals_bar_t.append(bar_t)
            num_available += 1

            out_rows.append({
                "comparison_group": group,
                "display_name": display_name,
                "case_a": case_a,
                "case_b": case_b,
                "domain_a": str(row_ab.get("source_domain", "")),
                "domain_b": str(row_ab.get("eval_domain", "")),
                "class_label": int(c),
                "t_a_to_b": float(t_ab),
                "t_b_to_a": float(t_ba),
                "bar_T_c": float(bar_t),
                "num_examples_a_to_b": int(row_ab.get("num_examples", 0)),
                "num_examples_b_to_a": int(row_ba.get("num_examples", 0)),
            })

        summary_rows.append({
            "comparison_group": group,
            "display_name": display_name,
            "case_a": case_a,
            "case_b": case_b,
            "num_classes_available": int(num_available),
            "mean_bar_T_c": float(_safe_mean(vals_bar_t)),
        })

    return out_rows, summary_rows


def compute_all_pairs_symmetric_topk_rows(
    *,
    raw_rows: List[Dict[str, Any]],
    case_names: List[str],
    display_name_map: Dict[str, str],
    num_classes: int,
    shared_labels_by_pair: Optional[Dict[Tuple[str, str], List[int]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    directed: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for row in raw_rows:
        directed[(
            str(row["source_model"]),
            str(row["eval_model"]),
            int(row["label"]),
        )] = row

    per_class_rows: List[Dict[str, Any]] = []
    mean_rows: List[Dict[str, Any]] = []

    ordered_names = [str(x) for x in case_names]
    for idx_a, case_a in enumerate(ordered_names):
        for idx_b in range(idx_a, len(ordered_names)):
            case_b = ordered_names[idx_b]
            vals: List[float] = []
            num_available = 0
            shared_labels = set(
                _shared_labels_for_pair(
                    shared_labels_by_pair=shared_labels_by_pair,
                    case_a=case_a,
                    case_b=case_b,
                    num_classes=int(num_classes),
                )
            )

            for c in range(int(num_classes)):
                if int(c) not in shared_labels:
                    continue
                row_ab = directed.get((case_a, case_b, int(c)))
                row_ba = directed.get((case_b, case_a, int(c)))
                if row_ab is None or row_ba is None:
                    continue

                if case_a == case_b:
                    bar_t = float(row_ab["avg_target_softmax"])
                else:
                    bar_t = 0.5 * (
                        float(row_ab["avg_target_softmax"]) + float(row_ba["avg_target_softmax"])
                    )
                vals.append(bar_t)
                num_available += 1

                per_class_rows.append({
                    "case_a": display_name_map.get(case_a, case_a),
                    "case_b": display_name_map.get(case_b, case_b),
                    "raw_case_a": case_a,
                    "raw_case_b": case_b,
                    "class_label": int(c),
                    "bar_T_c": float(bar_t),
                })

            mean_rows.append({
                "case_a": display_name_map.get(case_a, case_a),
                "case_b": display_name_map.get(case_b, case_b),
                "raw_case_a": case_a,
                "raw_case_b": case_b,
                "num_classes_available": int(num_available),
                "mean_bar_T_c": float(_safe_mean(vals)),
            })

    return per_class_rows, mean_rows


def _edge_pair_preservation_value(row: Dict[str, Any]) -> float:
    if "avg_pair_ambiguity" in row:
        return float(row["avg_pair_ambiguity"])
    return float(row["pair_top2_acc"])


def _compute_directed_class_edge_preservation_rows(
    *,
    raw_rows: List[Dict[str, Any]],
    num_classes: int,
    reduction: str = "mean",
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str, int], List[Dict[str, Any]]] = {}

    for row in raw_rows:
        src = str(row["source_model"])
        src_domain = str(row.get("source_domain", ""))
        tgt = str(row["eval_model"])
        tgt_domain = str(row.get("eval_domain", ""))
        pair_type = str(row.get("pair_type", ""))
        pair_i = int(row["pair_i"])
        pair_j = int(row["pair_j"])

        for c in (pair_i, pair_j):
            key = (src, src_domain, tgt, tgt_domain, pair_type, int(c))
            grouped.setdefault(key, []).append(row)

    out_rows: List[Dict[str, Any]] = []
    for (src, src_domain, tgt, tgt_domain, pair_type, c), rows_c in grouped.items():
        p_vals = [_edge_pair_preservation_value(r) for r in rows_c]
        out_rows.append({
            "source_model": src,
            "source_domain": src_domain,
            "eval_model": tgt,
            "eval_domain": tgt_domain,
            "pair_type": pair_type,
            "class_label": int(c),
            "num_edge_pairs": int(len(rows_c)),
            "P_c": float(_edge_reduction(p_vals, reduction)),
            "mean_pair_ambiguity": float(_safe_mean(p_vals)),
        })

    return out_rows


def compute_symmetric_edge_consensus_rows(
    *,
    raw_rows: List[Dict[str, Any]],
    focus_pairs: List[Dict[str, str]],
    num_classes: int,
    reduction: str = "mean",
    shared_labels_by_pair: Optional[Dict[Tuple[str, str], List[int]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    directed_class_rows = _compute_directed_class_edge_preservation_rows(
        raw_rows=raw_rows,
        num_classes=int(num_classes),
        reduction=str(reduction),
    )

    directed: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for row in directed_class_rows:
        key = (
            str(row["source_model"]),
            str(row["eval_model"]),
            int(row["class_label"]),
        )
        directed[key] = row

    out_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for spec in focus_pairs:
        case_a = str(spec["case_a"])
        case_b = str(spec["case_b"])
        group = str(spec["group"])
        display_name = str(spec["display_name"])

        vals_bar_p: List[float] = []
        num_available = 0
        shared_labels = set(
            _shared_labels_for_pair(
                shared_labels_by_pair=shared_labels_by_pair,
                case_a=case_a,
                case_b=case_b,
                num_classes=int(num_classes),
            )
        )

        for c in range(int(num_classes)):
            if int(c) not in shared_labels:
                continue
            row_ab = directed.get((case_a, case_b, int(c)))
            row_ba = directed.get((case_b, case_a, int(c)))
            if row_ab is None or row_ba is None:
                continue

            p_ab = float(row_ab["P_c"])
            p_ba = float(row_ba["P_c"])
            bar_p = 0.5 * (p_ab + p_ba)
            vals_bar_p.append(bar_p)
            num_available += 1

            out_rows.append({
                "comparison_group": group,
                "display_name": display_name,
                "case_a": case_a,
                "case_b": case_b,
                "domain_a": str(row_ab.get("source_domain", "")),
                "domain_b": str(row_ab.get("eval_domain", "")),
                "class_label": int(c),
                "p_a_to_b": float(p_ab),
                "p_b_to_a": float(p_ba),
                "bar_P_c": float(bar_p),
                "num_edge_pairs_a_to_b": int(row_ab.get("num_edge_pairs", 0)),
                "num_edge_pairs_b_to_a": int(row_ba.get("num_edge_pairs", 0)),
            })

        summary_rows.append({
            "comparison_group": group,
            "display_name": display_name,
            "case_a": case_a,
            "case_b": case_b,
            "num_classes_available": int(num_available),
            "mean_bar_P_c": float(_safe_mean(vals_bar_p)),
        })

    return directed_class_rows, out_rows, summary_rows


def compute_all_pairs_symmetric_edge_rows(
    *,
    raw_rows: List[Dict[str, Any]],
    case_names: List[str],
    display_name_map: Dict[str, str],
    num_classes: int,
    reduction: str = "mean",
    shared_labels_by_pair: Optional[Dict[Tuple[str, str], List[int]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    directed_class_rows = _compute_directed_class_edge_preservation_rows(
        raw_rows=raw_rows,
        num_classes=int(num_classes),
        reduction=str(reduction),
    )

    directed: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for row in directed_class_rows:
        directed[(
            str(row["source_model"]),
            str(row["eval_model"]),
            int(row["class_label"]),
        )] = row

    per_class_rows: List[Dict[str, Any]] = []
    mean_rows: List[Dict[str, Any]] = []

    ordered_names = [str(x) for x in case_names]
    for idx_a, case_a in enumerate(ordered_names):
        for idx_b in range(idx_a, len(ordered_names)):
            case_b = ordered_names[idx_b]
            vals: List[float] = []
            num_available = 0
            shared_labels = set(
                _shared_labels_for_pair(
                    shared_labels_by_pair=shared_labels_by_pair,
                    case_a=case_a,
                    case_b=case_b,
                    num_classes=int(num_classes),
                )
            )

            for c in range(int(num_classes)):
                if int(c) not in shared_labels:
                    continue
                row_ab = directed.get((case_a, case_b, int(c)))
                row_ba = directed.get((case_b, case_a, int(c)))
                if row_ab is None or row_ba is None:
                    continue

                if case_a == case_b:
                    bar_p = float(row_ab["P_c"])
                else:
                    bar_p = 0.5 * (float(row_ab["P_c"]) + float(row_ba["P_c"]))
                vals.append(bar_p)
                num_available += 1

                per_class_rows.append({
                    "case_a": display_name_map.get(case_a, case_a),
                    "case_b": display_name_map.get(case_b, case_b),
                    "raw_case_a": case_a,
                    "raw_case_b": case_b,
                    "class_label": int(c),
                    "bar_P_c": float(bar_p),
                })

            mean_rows.append({
                "case_a": display_name_map.get(case_a, case_a),
                "case_b": display_name_map.get(case_b, case_b),
                "raw_case_a": case_a,
                "raw_case_b": case_b,
                "num_classes_available": int(num_available),
                "mean_bar_P_c": float(_safe_mean(vals)),
            })

    return directed_class_rows, per_class_rows, mean_rows


def _rows_to_metric_vector(
    *,
    rows: List[Dict[str, Any]],
    metric_key: str,
    num_classes: int,
) -> np.ndarray:
    vec = np.full(int(num_classes), np.nan, dtype=np.float64)
    for row in rows:
        c = int(row["class_label"])
        if 0 <= c < int(num_classes):
            vec[c] = float(row[metric_key])
    return vec


def _plot_consensus_panel(
    ax,
    *,
    values: np.ndarray,
    title: str,
    ylabel: str,
    color: str,
    ymax: float = 1.0,
):
    x = np.arange(values.shape[0], dtype=np.int64)
    finite_mask = np.isfinite(values)
    vals = np.where(finite_mask, values, 0.0)

    ax.bar(x, vals, color=str(color), alpha=0.9, width=0.72)
    for idx, ok in enumerate(finite_mask.tolist()):
        if not ok:
            ax.bar(idx, 0.0, color="#D9D9D9", alpha=0.6, width=0.72)
            ax.text(idx, 0.02, "NA", ha="center", va="bottom", fontsize=8, rotation=90)

    mean_val = float(np.nanmean(values)) if np.any(finite_mask) else float("nan")
    if np.isfinite(mean_val):
        ax.axhline(mean_val, color="#333333", linestyle="--", linewidth=1.0)
        ax.text(
            0.98,
            0.96,
            f"mean={mean_val:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Class label")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_ylim(0.0, float(ymax))
    ax.grid(True, axis="y", alpha=0.25)


def _build_symmetric_matrix(
    *,
    case_names: List[str],
    rows: List[Dict[str, Any]],
    metric_key: str,
) -> np.ndarray:
    name_to_idx = {str(name): i for i, name in enumerate(case_names)}
    mat = np.full((len(case_names), len(case_names)), np.nan, dtype=np.float64)

    for row in rows:
        case_a = str(row["case_a"])
        case_b = str(row["case_b"])
        if case_a not in name_to_idx or case_b not in name_to_idx:
            continue
        i = name_to_idx[case_a]
        j = name_to_idx[case_b]
        val = float(row[metric_key])
        mat[i, j] = val
        mat[j, i] = val
    return mat


def _save_consensus_heatmap(
    *,
    case_names: List[str],
    rows: List[Dict[str, Any]],
    metric_key: str,
    title: Optional[str],
    out_path: Path,
    cbar_label: str,
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = 1.0,
    cmap: str = "viridis",
    norm_mode: str = "linear",
    figsize: Tuple[float, float] = (4.8, 4.1),
    title_fontsize: int = 11,
    tick_fontsize: int = 9,
    annot_fontsize: int = 8,
) -> None:
    ensure_dir(out_path.parent)
    mat = _build_symmetric_matrix(
        case_names=case_names,
        rows=rows,
        metric_key=metric_key,
    )

    cmap_obj = _get_plot_cmap(cmap)
    norm = None
    plot_mat = np.array(mat, copy=True)
    if str(norm_mode).lower() == "log":
        positive_vals = mat[np.isfinite(mat) & (mat > 0.0)]
        if positive_vals.size > 0:
            local_vmin = float(np.min(positive_vals))
            local_vmax = float(np.max(positive_vals))
            if vmin is not None:
                local_vmin = max(local_vmin, float(vmin))
            if vmax is not None:
                local_vmax = min(local_vmax, float(vmax))
            if local_vmax <= local_vmin:
                local_vmax = max(local_vmin * 1.01, 1.0)
            norm = LogNorm(vmin=local_vmin, vmax=local_vmax)
            zero_mask = np.isfinite(plot_mat) & (plot_mat <= 0.0)
            plot_mat[zero_mask] = local_vmin
    if norm is None:
        norm = Normalize(vmin=vmin, vmax=vmax)

    plt.figure(figsize=figsize)
    im = plt.imshow(plot_mat, cmap=cmap_obj, aspect="auto", norm=norm)
    plt.xticks(np.arange(len(case_names)), case_names, rotation=45, ha="right", fontsize=tick_fontsize)
    plt.yticks(np.arange(len(case_names)), case_names, fontsize=tick_fontsize)
    if title:
        plt.title(str(title), fontsize=title_fontsize)

    for i in range(len(case_names)):
        for j in range(len(case_names)):
            val = mat[i, j]
            text = "NA" if np.isnan(val) else f"{val:.3f}"
            if np.isnan(val):
                text_color = "white"
            else:
                norm_val = float(plot_mat[i, j])
                rgba = cmap_obj(norm(norm_val))
                r, g, b = rgba[:3]
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                text_color = "black" if luminance >= 0.58 else "white"
            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=annot_fontsize,
            )

    cbar = plt.colorbar(im)
    if str(cbar_label or ""):
        cbar.ax.set_ylabel(cbar_label, rotation=90, fontsize=tick_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_consensus_per_class_heatmaps(
    *,
    case_names: List[str],
    per_class_rows: List[Dict[str, Any]],
    metric_key: str,
    metric_family: str,
    out_dir: Path,
    cmap: str,
    norm_mode: str = "linear",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    ensure_dir(out_dir)
    class_labels = sorted({int(r["class_label"]) for r in per_class_rows})
    for c in class_labels:
        rows_c = [r for r in per_class_rows if int(r["class_label"]) == int(c)]
        class_dir = out_dir / f"class_{int(c)}"
        ensure_dir(class_dir)
        if str(metric_family) == "T_c":
            file_name = f"class_{int(c)}_T_c.png"
            cbar_label = rf"$T_{{c={int(c)}}}$"
        elif str(metric_family) == "P_c":
            file_name = f"class_{int(c)}_P_c.png"
            cbar_label = rf"$P_{{c={int(c)}}}$"
        else:
            file_name = f"{metric_key}.png"
            cbar_label = metric_key
        _save_consensus_heatmap(
            case_names=case_names,
            rows=rows_c,
            metric_key=metric_key,
            title=None,
            out_path=class_dir / file_name,
            cbar_label=cbar_label,
            vmin=float(vmin),
            vmax=float(vmax),
            cmap=cmap,
            norm_mode=norm_mode,
        )


def save_consensus_mean_heatmap(
    *,
    case_names: List[str],
    mean_rows: List[Dict[str, Any]],
    metric_key: str,
    metric_family: str,
    out_path: Path,
    cmap: str,
    norm_mode: str = "linear",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    cbar_label = ""
    _save_consensus_heatmap(
        case_names=case_names,
        rows=mean_rows,
        metric_key=metric_key,
        title=None,
        out_path=out_path,
        cbar_label=cbar_label,
        vmin=float(vmin),
        vmax=float(vmax),
        cmap=cmap,
        norm_mode=norm_mode,
    )


def _extract_pair_metric_vector(
    *,
    rows: List[Dict[str, Any]],
    case_a: str,
    case_b: str,
    metric_key: str,
    num_classes: int,
) -> np.ndarray:
    vec = np.full(int(num_classes), np.nan, dtype=np.float64)
    pair_rows = [
        r for r in rows
        if (
            str(r.get("raw_case_a", r.get("case_a", ""))) == str(case_a)
            and str(r.get("raw_case_b", r.get("case_b", ""))) == str(case_b)
        )
    ]
    for row in pair_rows:
        c = int(row["class_label"])
        if 0 <= c < int(num_classes):
            vec[c] = float(row[metric_key])
    return vec


def _plot_vector_heatmap(
    ax,
    *,
    values: np.ndarray,
    panel_title: str,
    cmap: str,
    norm_mode: str = "linear",
    vmin: float = 0.0,
    vmax: float = 1.0,
    tick_fontsize: int = 9,
    annot_fontsize: int = 8,
):
    mat = values.reshape(1, -1)
    plot_mat = np.array(mat, copy=True)
    cmap_obj = _get_plot_cmap(cmap)
    if str(norm_mode).lower() == "log":
        positive_vals = mat[np.isfinite(mat) & (mat > 0.0)]
        if positive_vals.size > 0:
            local_vmin = max(float(vmin), float(np.min(positive_vals)))
            local_vmax = min(float(vmax), float(np.max(positive_vals)))
            if local_vmax <= local_vmin:
                local_vmax = max(local_vmin * 1.01, 1.0)
            zero_mask = np.isfinite(plot_mat) & (plot_mat <= 0.0)
            plot_mat[zero_mask] = local_vmin
            norm = LogNorm(vmin=local_vmin, vmax=local_vmax)
        else:
            norm = Normalize(vmin=float(vmin), vmax=float(vmax))
    else:
        norm = Normalize(vmin=float(vmin), vmax=float(vmax))

    im = ax.imshow(plot_mat, aspect="auto", cmap=cmap_obj, norm=norm)
    ax.set_title(panel_title, fontsize=10)
    ax.set_yticks([])
    ax.set_xticks(np.arange(values.shape[0]))
    ax.set_xticklabels([str(i) for i in range(values.shape[0])], fontsize=tick_fontsize)
    ax.set_xlabel("Class label", fontsize=tick_fontsize)

    for j in range(values.shape[0]):
        val = float(values[j])
        if np.isnan(val):
            text = "NA"
            color = "white"
        else:
            rgba = cmap_obj(norm(float(plot_mat[0, j])))
            r, g, b = rgba[:3]
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            color = "black" if luminance >= 0.58 else "white"
            text = f"{val:.3f}"
        ax.text(j, 0, text, ha="center", va="center", fontsize=annot_fontsize, color=color)
    return im


def save_focus_pair_class_heatmaps(
    *,
    topl_rows: List[Dict[str, Any]],
    edge_rows: List[Dict[str, Any]],
    focus_pairs: List[Dict[str, str]],
    num_classes: int,
    out_dir: Path,
    topl_metric_key: str = "bar_T_c",
    edge_metric_key: str = "bar_P_c",
    topl_title: str = r"$T_c$",
    edge_title: str = r"$P_c$",
    topl_cbar_label: str = "",
    edge_cbar_label: str = "",
    topl_norm_mode: str = "linear",
    edge_norm_mode: str = "log",
    topl_vmin: float = 0.0,
    topl_vmax: float = 1.0,
    edge_vmin: float = 0.0,
    edge_vmax: float = 1.0,
) -> None:
    ensure_dir(out_dir)

    fig, axes = plt.subplots(
        nrows=len(focus_pairs),
        ncols=2,
        figsize=(10.6, 5.6),
        gridspec_kw={"width_ratios": [1.0, 1.0], "hspace": 0.7, "wspace": 0.4},
    )
    if len(focus_pairs) == 1:
        axes = np.asarray([axes], dtype=object)

    topl_im = None
    edge_im = None

    for row_idx, spec in enumerate(focus_pairs):
        case_a = str(spec["case_a"])
        case_b = str(spec["case_b"])
        display = str(spec["display_name"])

        t_vec = _extract_pair_metric_vector(
            rows=topl_rows,
            case_a=case_a,
            case_b=case_b,
            metric_key=topl_metric_key,
            num_classes=int(num_classes),
        )
        p_vec = _extract_pair_metric_vector(
            rows=edge_rows,
            case_a=case_a,
            case_b=case_b,
            metric_key=edge_metric_key,
            num_classes=int(num_classes),
        )

        topl_im = _plot_vector_heatmap(
            axes[row_idx, 0],
            values=t_vec,
            panel_title=f"{display}: {topl_title}",
            cmap="parula",
            norm_mode=topl_norm_mode,
            vmin=float(topl_vmin),
            vmax=float(topl_vmax),
        )
        edge_im = _plot_vector_heatmap(
            axes[row_idx, 1],
            values=p_vec,
            panel_title=f"{display}: {edge_title}",
            cmap="copper",
            norm_mode=edge_norm_mode,
            vmin=float(edge_vmin),
            vmax=float(edge_vmax),
        )

    if topl_im is not None:
        cbar_t = fig.colorbar(topl_im, ax=axes[:, 0], fraction=0.028, pad=0.03)
        if str(topl_cbar_label or ""):
            cbar_t.ax.set_ylabel(topl_cbar_label, rotation=90)
    if edge_im is not None:
        cbar_p = fig.colorbar(edge_im, ax=axes[:, 1], fraction=0.028, pad=0.03)
        if str(edge_cbar_label or ""):
            cbar_p.ax.set_ylabel(edge_cbar_label, rotation=90)

    fig.tight_layout()
    fig.savefig(out_dir / "mnist_only_focus_pairs_Tc_Pc_med_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_focus_pair_metric_heatmap(
    *,
    rows: List[Dict[str, Any]],
    focus_pairs: List[Dict[str, str]],
    num_classes: int,
    metric_key: str,
    out_path: Path,
    cmap: str,
    cbar_label: str,
    norm_mode: str = "linear",
    text_color_threshold: Optional[float] = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    ensure_dir(out_path.parent)

    pair_vectors: List[np.ndarray] = []
    row_labels: List[str] = []
    row_means: List[float] = []
    for spec in focus_pairs:
        case_a = str(spec["case_a"])
        case_b = str(spec["case_b"])
        display = str(spec["display_name"])
        vec = _extract_pair_metric_vector(
            rows=rows,
            case_a=case_a,
            case_b=case_b,
            metric_key=metric_key,
            num_classes=int(num_classes),
        )
        pair_vectors.append(vec)
        mean_val = float(np.nanmean(vec)) if np.any(np.isfinite(vec)) else float("nan")
        row_means.append(mean_val)
        row_labels.append(display)

    if not pair_vectors:
        return

    mat = np.stack(pair_vectors, axis=0)
    avg_col = np.asarray(row_means, dtype=np.float64).reshape(-1, 1)
    mat = np.concatenate([mat, avg_col], axis=1)
    plot_mat = np.array(mat, copy=True)
    cmap_obj = _get_plot_cmap(cmap)

    if str(norm_mode).lower() == "log":
        positive_vals = mat[np.isfinite(mat) & (mat > 0.0)]
        if positive_vals.size > 0:
            local_vmin = max(float(vmin), float(np.min(positive_vals)))
            local_vmax = min(float(vmax), float(np.max(positive_vals)))
            if local_vmax <= local_vmin:
                local_vmax = max(local_vmin * 1.01, 1.0)
            zero_mask = np.isfinite(plot_mat) & (plot_mat <= 0.0)
            plot_mat[zero_mask] = local_vmin
            norm = LogNorm(vmin=local_vmin, vmax=local_vmax)
        else:
            norm = Normalize(vmin=float(vmin), vmax=float(vmax))
    else:
        norm = Normalize(vmin=float(vmin), vmax=float(vmax))

    ncols = int(num_classes) + 1
    nrows = len(focus_pairs)

    # Use a fixed paper-friendly canvas so T_c and P_c plots match each other,
    # while avoiding an overly stretched 3x11 matrix.
    fig_w = 7.9
    fig_h = 2.55

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.16, 0.22, 0.68, 0.64])
    im = ax.imshow(plot_mat, aspect="auto", cmap=cmap_obj, norm=norm)
    ax.set_xlabel("Class label")
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels([str(i) for i in range(int(num_classes))] + ["avg"])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = float(mat[i, j])
            if np.isnan(val):
                text = "NA"
                color = "white"
            else:
                if text_color_threshold is not None and str(norm_mode).lower() == "linear":
                    # Deterministic threshold so similarly rounded values do not flip text color.
                    color = "black" if val >= float(text_color_threshold) else "white"
                else:
                    rgba = cmap_obj(norm(float(plot_mat[i, j])))
                    r, g, b = rgba[:3]
                    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    color = "black" if luminance >= 0.58 else "white"
                text = f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    for idx, tick in enumerate(ax.get_xticklabels()):
        if idx == 7:
            tick.set_fontweight("bold")

    cax = fig.add_axes([0.855, 0.22, 0.018, 0.64])
    cbar = fig.colorbar(im, cax=cax)
    if str(cbar_label or ""):
        cbar.set_label(cbar_label, labelpad=2, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _edge_reduction(values: Sequence[float], mode: str) -> float:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    key = str(mode).lower()
    if key == "min":
        return float(np.min(arr))
    if key in {"med", "median"}:
        return float(np.median(arr))
    if key in {"avg", "mean"}:
        return float(np.mean(arr))
    raise ValueError(f"Unsupported edge reduction mode: {mode}")


def compute_all_pairs_symmetric_edge_rows_variant(
    *,
    raw_rows: List[Dict[str, Any]],
    case_names: List[str],
    display_name_map: Dict[str, str],
    num_classes: int,
    reduction: str,
    shared_labels_by_pair: Optional[Dict[Tuple[str, str], List[int]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str, str, str, str, int], List[Dict[str, Any]]] = {}

    for row in raw_rows:
        src = str(row["source_model"])
        src_domain = str(row.get("source_domain", ""))
        tgt = str(row["eval_model"])
        tgt_domain = str(row.get("eval_domain", ""))
        pair_type = str(row.get("pair_type", ""))
        pair_i = int(row["pair_i"])
        pair_j = int(row["pair_j"])
        for c in (pair_i, pair_j):
            key = (src, src_domain, tgt, tgt_domain, pair_type, int(c))
            grouped.setdefault(key, []).append(row)

    directed_rows: List[Dict[str, Any]] = []
    for (src, src_domain, tgt, tgt_domain, pair_type, c), rows_c in grouped.items():
        p_vals = [_edge_pair_preservation_value(r) for r in rows_c]
        directed_rows.append({
            "source_model": src,
            "source_domain": src_domain,
            "eval_model": tgt,
            "eval_domain": tgt_domain,
            "pair_type": pair_type,
            "class_label": int(c),
            "num_edge_pairs": int(len(rows_c)),
            "P_c": float(_edge_reduction(p_vals, reduction)),
            "mean_pair_ambiguity": float(_safe_mean(p_vals)),
        })

    directed: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for row in directed_rows:
        directed[(str(row["source_model"]), str(row["eval_model"]), int(row["class_label"]))] = row

    per_class_rows: List[Dict[str, Any]] = []
    mean_rows: List[Dict[str, Any]] = []

    ordered_names = [str(x) for x in case_names]
    for idx_a, case_a in enumerate(ordered_names):
        for idx_b in range(idx_a, len(ordered_names)):
            case_b = ordered_names[idx_b]
            vals: List[float] = []
            num_available = 0
            shared_labels = set(
                _shared_labels_for_pair(
                    shared_labels_by_pair=shared_labels_by_pair,
                    case_a=case_a,
                    case_b=case_b,
                    num_classes=int(num_classes),
                )
            )

            for c in range(int(num_classes)):
                if int(c) not in shared_labels:
                    continue
                row_ab = directed.get((case_a, case_b, int(c)))
                row_ba = directed.get((case_b, case_a, int(c)))
                if row_ab is None or row_ba is None:
                    continue
                if case_a == case_b:
                    bar_p = float(row_ab["P_c"])
                else:
                    bar_p = 0.5 * (float(row_ab["P_c"]) + float(row_ba["P_c"]))
                vals.append(bar_p)
                num_available += 1
                per_class_rows.append({
                    "case_a": display_name_map.get(case_a, case_a),
                    "case_b": display_name_map.get(case_b, case_b),
                    "raw_case_a": case_a,
                    "raw_case_b": case_b,
                    "class_label": int(c),
                    "bar_P_c": float(bar_p),
                })

            mean_rows.append({
                "case_a": display_name_map.get(case_a, case_a),
                "case_b": display_name_map.get(case_b, case_b),
                "raw_case_a": case_a,
                "raw_case_b": case_b,
                "num_classes_available": int(num_available),
                "mean_bar_P_c": float(_safe_mean(vals)),
            })

    return directed_rows, per_class_rows, mean_rows


def save_q2_feature_consensus_figure(
    *,
    topl_rows: List[Dict[str, Any]],
    edge_rows: List[Dict[str, Any]],
    focus_pairs: List[Dict[str, str]],
    num_classes: int,
    out_path: Path,
    topl_metric_key: str = "bar_T_c",
    edge_metric_key: str = "bar_P_c",
    topl_panel_title: str = "Top-K ($\\bar T_c$)",
    edge_panel_title: str = "Edge Preservation ($\\bar P_c$)",
    topl_ylabel: str = "$\\bar T_c$",
    edge_ylabel: str = "$\\bar P_c$",
    topl_ymax: float = 1.0,
    edge_ymax: float = 1.0,
    figure_title: str = "Q2 Feature Consensus under Transfer",
) -> None:
    ensure_dir(out_path.parent)

    topl_by_group = {
        str(spec["group"]): [r for r in topl_rows if str(r["comparison_group"]) == str(spec["group"])]
        for spec in focus_pairs
    }
    edge_by_group = {
        str(spec["group"]): [r for r in edge_rows if str(r["comparison_group"]) == str(spec["group"])]
        for spec in focus_pairs
    }

    fig, axes = plt.subplots(
        nrows=len(focus_pairs),
        ncols=2,
        figsize=(13.6, 10.8),
        sharex=False,
        sharey=True,
    )
    if len(focus_pairs) == 1:
        axes = np.asarray([axes], dtype=object)

    panel_letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    letter_idx = 0

    for row_idx, spec in enumerate(focus_pairs):
        group = str(spec["group"])
        display_name = str(spec["display_name"])

        topl_vec = _rows_to_metric_vector(
            rows=topl_by_group[group],
            metric_key=topl_metric_key,
            num_classes=int(num_classes),
        )
        edge_vec = _rows_to_metric_vector(
            rows=edge_by_group[group],
            metric_key=edge_metric_key,
            num_classes=int(num_classes),
        )

        _plot_consensus_panel(
            axes[row_idx, 0],
            values=topl_vec,
            title=f"{panel_letters[letter_idx]} {display_name}: {topl_panel_title}",
            ylabel=topl_ylabel,
            color="#4C78A8",
            ymax=float(topl_ymax),
        )
        letter_idx += 1

        _plot_consensus_panel(
            axes[row_idx, 1],
            values=edge_vec,
            title=f"{panel_letters[letter_idx]} {display_name}: {edge_panel_title}",
            ylabel=edge_ylabel,
            color="#E45756",
            ymax=float(edge_ymax),
        )
        letter_idx += 1

    fig.suptitle(str(figure_title), fontsize=14, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _default_focus_pairs(anchor_benign_case: str, cross_domain_case: str) -> List[Dict[str, str]]:
    anchor = str(anchor_benign_case)
    cross_case = str(cross_domain_case)
    return [
        {
            "group": "benign_benign",
            "display_name": f"{anchor} ↔ benign2",
            "case_a": anchor,
            "case_b": "benign2",
        },
        {
            "group": "benign_backdoor",
            "display_name": f"{anchor} ↔ backdoored",
            "case_a": anchor,
            "case_b": "backdoor",
        },
        {
            "group": "benign_cross_domain",
            "display_name": f"{anchor} ↔ diff domain",
            "case_a": anchor,
            "case_b": cross_case,
        },
    ]


def _parse_focus_pairs(
    *,
    analysis_cfg: Dict[str, Any],
    cross_domain_case: str,
) -> List[Dict[str, str]]:
    anchor_benign_case = str(analysis_cfg.get("anchor_benign_case", "benign1"))
    raw_pairs = analysis_cfg.get("focus_pairs", None)
    if raw_pairs is None:
        return _default_focus_pairs(anchor_benign_case=anchor_benign_case, cross_domain_case=cross_domain_case)

    out: List[Dict[str, str]] = []
    for item in raw_pairs:
        if not isinstance(item, dict):
            continue
        out.append({
            "group": str(item["group"]),
            "display_name": str(item.get("display_name", item["group"])),
            "case_a": str(item["case_a"]),
            "case_b": str(item["case_b"]),
        })
    if not out:
        raise ValueError("analysis.focus_pairs was provided but no valid pair specs were found.")
    return out


def _mnist_only_focus_pairs() -> List[Dict[str, str]]:
    return [
        {
            "display_name": "Ben1-Ben2",
            "case_a": "benign1",
            "case_b": "benign2",
        },
        {
            "display_name": "Ben1-BD",
            "case_a": "benign1",
            "case_b": "backdoor",
        },
        {
            "display_name": "Ben2-BD",
            "case_a": "benign2",
            "case_b": "backdoor",
        },
    ]


def _csv_path(csv_dir: Path, file_name: str) -> Path:
    ensure_dir(csv_dir)
    return csv_dir / str(file_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", required=True)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--name-suffix", default="")
    args = parser.parse_args()

    with open(args.yaml, "r") as f:
        cfg = yaml.safe_load(f)

    overrides_applied = False
    if args.max_epochs is not None:
        cfg.setdefault("train", {})["max_epochs"] = int(args.max_epochs)
        overrides_applied = True
    name_suffix = str(args.name_suffix or "").strip()
    if name_suffix:
        exp_override = cfg.setdefault("experiment", {})
        base_name = str(exp_override.get("name", "exp6_feature_consensus_transfer"))
        exp_override["name"] = f"{base_name}_{name_suffix}"
        overrides_applied = True

    exp_cfg = dict(cfg.get("experiment", {}) or {})
    data_cfg = dict(cfg.get("data", {}) or {})
    model_cfg = dict(cfg.get("model", {}) or {})
    train_cfg = dict(cfg.get("train", {}) or {})
    search_cfg = dict(cfg.get("search", {}) or {})
    optimizer_cfg = dict(cfg.get("optimizer", {}) or {})
    backdoor_cfg = dict(cfg.get("backdoor", {}) or {})
    analysis_cfg = dict(cfg.get("analysis", {}) or {})

    device = resolve_device(exp_cfg.get("device", "cuda"))
    seed = int(exp_cfg.get("seed", 123))

    mnist_domain = str(data_cfg.get("mnist_domain", "mnist")).lower()
    cross_domain = str(data_cfg.get("cross_domain", "fmnist")).lower()
    data_root = str(data_cfg.get("data_root", "./data"))

    out_dir = Path(exp_cfg["log_root"]) / f"{exp_cfg['name']}_{now_tag()}"
    ensure_dir(out_dir)
    csv_dir = out_dir / "csv"
    ensure_dir(csv_dir)
    logger = RunLogger(out_dir / "run.log")

    try:
        set_seed(seed)

        config_copy_path = out_dir / "config_used.yaml"
        if overrides_applied:
            with open(config_copy_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            logger.log(f"[Config] wrote_effective_yaml={config_copy_path}")
        else:
            shutil.copy2(args.yaml, config_copy_path)
            logger.log(f"[Config] copied_yaml={config_copy_path}")

        batch_size = int(data_cfg.get("batch_size", 256))
        num_workers = int(data_cfg.get("num_workers", 4))
        pin_memory = bool(data_cfg.get("pin_memory", True))
        persistent_workers = bool(data_cfg.get("persistent_workers", True))
        max_epochs = int(train_cfg.get("max_epochs", 25))
        mnist_client_fraction = float(data_cfg.get("mnist_client_fraction", 0.30))
        topl_transfer_batch_size = int(
            analysis_cfg.get("topl_eval_batch_size", search_cfg.get("batch_size", 4096))
        )
        edge_transfer_batch_size = int(
            analysis_cfg.get("edge_eval_batch_size", search_cfg.get("batch_size", 4096))
        )
        feature_consensus_variant = str(
            analysis_cfg.get("feature_consensus_variant", "discrepancy")
        ).lower()
        if feature_consensus_variant in {"new", "consistency"}:
            feature_consensus_variant = "discrepancy"
        if feature_consensus_variant in {"old", "legacy"}:
            feature_consensus_variant = "transfer"
        if feature_consensus_variant not in {"transfer", "discrepancy"}:
            raise ValueError(
                f"Unsupported analysis.feature_consensus_variant={feature_consensus_variant!r}. "
                "Expected 'discrepancy' or 'transfer'."
            )

        logger.log(f"[Output] out_dir={out_dir}")
        logger.log(f"[Output] csv_dir={csv_dir}")
        logger.log(f"[Analysis] feature_consensus_variant={feature_consensus_variant}")
        logger.log(
            "[TrainCfg] "
            f"batch_size={batch_size} max_epochs={max_epochs} "
            f"mnist_client_fraction={mnist_client_fraction:.4f} "
            f"lr={float(optimizer_cfg.get('lr', 0.1)):.6f} "
            f"momentum={float(optimizer_cfg.get('momentum', 0.9)):.4f} "
            f"weight_decay={float(optimizer_cfg.get('weight_decay', 5e-4)):.6f}"
        )

        train_mnist_raw, test_mnist_raw, num_classes_mnist, out_hw_mnist = _load_domain_datasets(
            domain=mnist_domain,
            data_root=data_root,
        )
        train_cross_raw, test_cross_raw, num_classes_cross, out_hw_cross = _load_domain_datasets(
            domain=cross_domain,
            data_root=data_root,
        )

        if int(num_classes_mnist) != int(num_classes_cross):
            raise ValueError(
                f"Class-count mismatch between {mnist_domain} ({num_classes_mnist}) and "
                f"{cross_domain} ({num_classes_cross})."
            )

        num_classes = int(model_cfg.get("num_classes", num_classes_mnist))
        mnist_client_size = int(round(float(len(train_mnist_raw)) * mnist_client_fraction))
        if mnist_client_size <= 0:
            raise ValueError(f"mnist_client_fraction produced non-positive client size: {mnist_client_size}")

        logger.log(f"[Domains] mnist_domain={mnist_domain} cross_domain={cross_domain}")
        logger.log(f"[Client size] mnist_each={mnist_client_size}")

        benign1_raw, benign2_raw, backdoor_base_raw = _sample_disjoint_subsets(
            train_mnist_raw,
            num_splits=3,
            subset_size=mnist_client_size,
            seed=seed + 101,
        )
        cross_domain_raw = _sample_fixed_size_subset(
            train_cross_raw,
            sample_size=mnist_client_size,
            seed=seed + 202,
        )

        cross_domain_case = "cross_domain"
        display_name_map = _display_name_map(cross_domain_case)
        focus_pairs = _parse_focus_pairs(
            analysis_cfg=analysis_cfg,
            cross_domain_case=cross_domain_case,
        )

        backdoor_raw = PatternBackdoorWrapper(
            backdoor_base_raw,
            target_label=int(backdoor_cfg.get("target_label", 7)),
            poison_frac=float(backdoor_cfg.get("poison_frac", 0.0)),
            pattern_pos=str(backdoor_cfg.get("pattern_pos", "bottom_right")),
            pattern_padding=int(backdoor_cfg.get("pattern_padding", 0)),
            pattern_size=tuple(backdoor_cfg.get("pattern_size", (3, 3))),
            pattern_offsets=backdoor_cfg.get("pattern_offsets", None),
            value=float(backdoor_cfg.get("value", 1.0)),
            seed=int(backdoor_cfg.get("seed", seed)),
            pattern_type=str(backdoor_cfg.get("pattern_type", "badnet_corner")),
        )

        case_raw_train: Dict[str, Dataset] = {
            "benign1": benign1_raw,
            "benign2": benign2_raw,
            "backdoor": backdoor_raw,
            cross_domain_case: cross_domain_raw,
        }
        case_domain = {
            "benign1": mnist_domain,
            "benign2": mnist_domain,
            "backdoor": mnist_domain,
            cross_domain_case: cross_domain,
        }
        case_role = {
            "benign1": "benign",
            "benign2": "benign",
            "backdoor": "backdoor",
            cross_domain_case: "cross_domain",
        }
        case_out_hw = {
            "benign1": out_hw_mnist,
            "benign2": out_hw_mnist,
            "backdoor": out_hw_mnist,
            cross_domain_case: out_hw_cross,
        }

        case_test_loader = {
            "benign1": _make_loader(
                NormalizeWrapper(test_mnist_raw, dataset=mnist_domain),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                seed=seed + 301,
            ),
            "benign2": _make_loader(
                NormalizeWrapper(test_mnist_raw, dataset=mnist_domain),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                seed=seed + 302,
            ),
            "backdoor": _make_loader(
                NormalizeWrapper(test_mnist_raw, dataset=mnist_domain),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                seed=seed + 303,
            ),
            cross_domain_case: _make_loader(
                NormalizeWrapper(test_cross_raw, dataset=cross_domain),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                seed=seed + 304,
            ),
        }

        case_mnist_eval_loader = {
            "benign1": _make_loader(
                NormalizeWrapper(test_mnist_raw, dataset=mnist_domain),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                seed=seed + 311,
            ),
            "benign2": _make_loader(
                NormalizeWrapper(test_mnist_raw, dataset=mnist_domain),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                seed=seed + 312,
            ),
            "backdoor": _make_loader(
                NormalizeWrapper(test_mnist_raw, dataset=mnist_domain),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                seed=seed + 313,
            ),
            cross_domain_case: _make_loader(
                NormalizeWrapper(test_mnist_raw, dataset=cross_domain),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                seed=seed + 314,
            ),
        }

        bd_evalset_raw = build_backdoor_evalset(
            test_mnist_raw,
            backdoor_cfg,
            per_class=int(backdoor_cfg.get("eval_per_class", 50)),
        )
        bd_eval_loader_mnist = _make_loader(
            NormalizeAttackEvalWrapper(bd_evalset_raw, dataset=mnist_domain),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            seed=seed + 401,
        )
        bd_eval_loader_cross_domain = _make_loader(
            NormalizeAttackEvalWrapper(bd_evalset_raw, dataset=cross_domain),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            seed=seed + 402,
        )

        case_results: Dict[str, Dict[str, Any]] = {}
        case_models: Dict[str, nn.Module] = {}
        case_summary_rows: List[Dict[str, Any]] = []

        for case_idx, case_name in enumerate(["benign1", "benign2", "backdoor", cross_domain_case]):
            domain_name = case_domain[case_name]
            raw_train_ds = case_raw_train[case_name]
            logger.log(f"[Case] {case_name} role={case_role[case_name]} domain={domain_name} train_size={len(raw_train_ds)}")

            train_ds = NormalizeWrapper(raw_train_ds, dataset=domain_name)
            train_loader = _make_loader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                seed=seed + case_idx,
            )

            set_seed(seed)
            model = build_random_init_model(
                model_cfg=model_cfg,
                dataset_name=domain_name,
                device=device,
            )

            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=float(optimizer_cfg.get("lr", 0.1)),
                momentum=float(optimizer_cfg.get("momentum", 0.9)),
                weight_decay=float(optimizer_cfg.get("weight_decay", 5e-4)),
            )

            epoch_metric_rows: List[Dict[str, Any]] = []
            for ep in range(max_epochs):
                loss = train_one_epoch(model, train_loader, optimizer, device)
                logger.log(f"[Train] {case_name} ep={ep+1:03d} loss={loss:.4f}")

                clean_acc = eval_acc(model, case_test_loader[case_name], device)
                clean_acc_native = float(clean_acc)
                clean_acc_mnist = float(eval_acc(model, case_mnist_eval_loader[case_name], device))
                clean_acc_fmnist = float(clean_acc_native) if case_name == cross_domain_case else float("nan")

                if case_name == cross_domain_case:
                    asr = float(eval_asr(model, bd_eval_loader_cross_domain, device))
                else:
                    asr = float(eval_asr(model, bd_eval_loader_mnist, device))

                epoch_metric_rows.append({
                    "case": case_name,
                    "role": case_role[case_name],
                    "domain": domain_name,
                    "epoch": int(ep + 1),
                    "train_loss": float(loss),
                    "clean_acc_native": float(clean_acc_native),
                    "clean_acc_mnist": float(clean_acc_mnist),
                    "clean_acc_fmnist": float(clean_acc_fmnist),
                    "asr": float(asr),
                })
                logger.log(
                    f"[EpochEval] {case_name} "
                    f"ep={ep+1:03d} "
                    f"loss={loss:.4f} "
                    f"clean_acc_native={clean_acc_native:.2f} "
                    f"clean_acc_mnist={clean_acc_mnist:.2f} "
                    f"clean_acc_fmnist={clean_acc_fmnist:.2f} "
                    f"asr={asr:.2f}"
                )

            if not epoch_metric_rows:
                raise RuntimeError(f"No epoch metrics recorded for case={case_name}")

            write_csv(
                csv_path=_csv_path(csv_dir, f"exp6_training_metrics_{case_name}.csv"),
                fieldnames=list(epoch_metric_rows[0].keys()),
                rows=epoch_metric_rows,
            )

            final_metrics = dict(epoch_metric_rows[-1])
            clean_acc_native = float(final_metrics["clean_acc_native"])
            clean_acc_mnist = float(final_metrics["clean_acc_mnist"])
            clean_acc_fmnist = float(final_metrics["clean_acc_fmnist"])
            asr = float(final_metrics["asr"])

            result = run_pointillism_search(
                model=model,
                num_classes=num_classes,
                out_hw=case_out_hw[case_name],
                dataset=domain_name,
                device=device,
                norm_cfg_fn=_norm_cfg,
                cfg=search_cfg,
                log_fn=logger.log,
            )

            save_signature_outputs(
                result=result,
                out_dir=out_dir / "signatures",
                model_tag=case_name,
                out_hw=case_out_hw[case_name],
            )

            counts = _effective_label_counts(raw_train_ds, num_classes=num_classes)
            total_count = int(sum(counts))
            summary_row = {
                "case": case_name,
                "role": case_role[case_name],
                "domain": domain_name,
                "native_test_domain": domain_name,
                "train_size": int(total_count),
                "clean_acc": float(clean_acc_native),
                "clean_acc_native": float(clean_acc_native),
                "clean_acc_mnist": float(clean_acc_mnist),
                "clean_acc_fmnist": float(clean_acc_fmnist),
                "asr": float(asr),
                "asr_mnist_backdoor": float(asr),
            }
            for y in range(num_classes):
                summary_row[f"label_{y:02d}_count"] = int(counts[y])
                summary_row[f"label_{y:02d}_frac"] = float(counts[y]) / float(max(total_count, 1))

            case_summary_rows.append(summary_row)
            case_models[case_name] = model
            case_results[case_name] = {
                "domain": domain_name,
                "role": case_role[case_name],
                "result": result,
                "q1": result.get("q1", {}),
                "clean_acc": float(clean_acc_native),
                "clean_acc_native": float(clean_acc_native),
                "clean_acc_mnist": float(clean_acc_mnist),
                "clean_acc_fmnist": float(clean_acc_fmnist),
                "asr": float(asr),
                "label_counts": counts,
                "topl_by_label": result.get("topl_by_label", {}),
                "edge_cases": result.get("edge_cases", {}),
            }

        if case_summary_rows:
            write_csv(
                csv_path=_csv_path(csv_dir, "exp6_case_summary.csv"),
                fieldnames=list(case_summary_rows[0].keys()),
                rows=case_summary_rows,
            )

        all_case_names = ["benign1", "benign2", "backdoor", cross_domain_case]
        display_case_names = [display_name_map[name] for name in all_case_names]
        _support_by_case, shared_labels_by_pair, q1_case_rows, q1_pair_rows = build_shared_support_lookup(
            case_names=all_case_names,
            case_results=case_results,
        )
        if q1_case_rows:
            write_csv(
                csv_path=_csv_path(csv_dir, "exp6_q1_case_support_summary.csv"),
                fieldnames=list(q1_case_rows[0].keys()),
                rows=q1_case_rows,
            )
        if q1_pair_rows:
            write_csv(
                csv_path=_csv_path(csv_dir, "exp6_q1_pairwise_shared_support.csv"),
                fieldnames=list(q1_pair_rows[0].keys()),
                rows=q1_pair_rows,
            )

        if feature_consensus_variant == "transfer":
            topl_transfer_raw_rows, topl_transfer_pair_rows = compute_topl_transfer_pair_rows(
                case_names=all_case_names,
                case_results=case_results,
                case_models=case_models,
                case_out_hw=case_out_hw,
                device=device,
                batch_size=topl_transfer_batch_size,
            )
            if topl_transfer_raw_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_topk_transfer_raw.csv"),
                    fieldnames=list(topl_transfer_raw_rows[0].keys()),
                    rows=topl_transfer_raw_rows,
                )
            if topl_transfer_pair_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_topk_transfer_pair_summary.csv"),
                    fieldnames=list(topl_transfer_pair_rows[0].keys()),
                    rows=topl_transfer_pair_rows,
                )

            edge_transfer_raw_rows, edge_transfer_summary_rows = compute_edge_transfer_pair_rows(
                case_names=all_case_names,
                case_results=case_results,
                case_models=case_models,
                case_out_hw=case_out_hw,
                device=device,
                batch_size=edge_transfer_batch_size,
            )
            if edge_transfer_raw_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_edge_transfer_raw.csv"),
                    fieldnames=list(edge_transfer_raw_rows[0].keys()),
                    rows=edge_transfer_raw_rows,
                )
            if edge_transfer_summary_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_edge_transfer_pair_summary.csv"),
                    fieldnames=list(edge_transfer_summary_rows[0].keys()),
                    rows=edge_transfer_summary_rows,
                )

            sym_topk_rows, sym_topk_summary_rows = compute_symmetric_topk_consensus_rows(
                raw_rows=topl_transfer_raw_rows,
                focus_pairs=focus_pairs,
                num_classes=num_classes,
                shared_labels_by_pair=shared_labels_by_pair,
            )
            if sym_topk_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_symmetric_topk_per_class.csv"),
                    fieldnames=list(sym_topk_rows[0].keys()),
                    rows=sym_topk_rows,
                )
            if sym_topk_summary_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_symmetric_topk_summary.csv"),
                    fieldnames=list(sym_topk_summary_rows[0].keys()),
                    rows=sym_topk_summary_rows,
                )

            all_pairs_sym_topk_rows, all_pairs_sym_topk_mean_rows = compute_all_pairs_symmetric_topk_rows(
                raw_rows=topl_transfer_raw_rows,
                case_names=all_case_names,
                display_name_map=display_name_map,
                num_classes=num_classes,
                shared_labels_by_pair=shared_labels_by_pair,
            )
            if all_pairs_sym_topk_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_all_pairs_symmetric_topk_per_class.csv"),
                    fieldnames=list(all_pairs_sym_topk_rows[0].keys()),
                    rows=all_pairs_sym_topk_rows,
                )
                save_consensus_per_class_heatmaps(
                    case_names=display_case_names,
                    per_class_rows=all_pairs_sym_topk_rows,
                    metric_key="bar_T_c",
                    metric_family="T_c",
                    out_dir=out_dir / "top_l_per_class_transfer",
                    cmap="parula",
                )
            if all_pairs_sym_topk_mean_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_all_pairs_symmetric_topk_mean_over_labels.csv"),
                    fieldnames=list(all_pairs_sym_topk_mean_rows[0].keys()),
                    rows=all_pairs_sym_topk_mean_rows,
                )
                save_consensus_mean_heatmap(
                    case_names=display_case_names,
                    mean_rows=all_pairs_sym_topk_mean_rows,
                    metric_key="mean_bar_T_c",
                    metric_family="T_c",
                    out_path=out_dir / "top_l_per_class_transfer" / "mean_T_c.png",
                    cmap="parula",
                )

            directed_edge_pc_rows, sym_edge_rows, sym_edge_summary_rows = compute_symmetric_edge_consensus_rows(
                raw_rows=edge_transfer_raw_rows,
                focus_pairs=focus_pairs,
                num_classes=num_classes,
                reduction="mean",
                shared_labels_by_pair=shared_labels_by_pair,
            )
            if directed_edge_pc_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_directed_edge_preservation_per_class.csv"),
                    fieldnames=list(directed_edge_pc_rows[0].keys()),
                    rows=directed_edge_pc_rows,
                )
            if sym_edge_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_symmetric_edge_per_class.csv"),
                    fieldnames=list(sym_edge_rows[0].keys()),
                    rows=sym_edge_rows,
                )
            if sym_edge_summary_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_symmetric_edge_summary.csv"),
                    fieldnames=list(sym_edge_summary_rows[0].keys()),
                    rows=sym_edge_summary_rows,
                )

            all_pairs_directed_pc_rows, all_pairs_sym_edge_rows, all_pairs_sym_edge_mean_rows = compute_all_pairs_symmetric_edge_rows_variant(
                raw_rows=edge_transfer_raw_rows,
                case_names=all_case_names,
                display_name_map=display_name_map,
                num_classes=num_classes,
                reduction="mean",
                shared_labels_by_pair=shared_labels_by_pair,
            )
            if all_pairs_directed_pc_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_all_pairs_directed_Pc_per_class.csv"),
                    fieldnames=list(all_pairs_directed_pc_rows[0].keys()),
                    rows=all_pairs_directed_pc_rows,
                )
            if all_pairs_sym_edge_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_all_pairs_symmetric_edge_per_class.csv"),
                    fieldnames=list(all_pairs_sym_edge_rows[0].keys()),
                    rows=all_pairs_sym_edge_rows,
                )
            if all_pairs_sym_edge_mean_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_all_pairs_symmetric_edge_mean_over_labels.csv"),
                    fieldnames=list(all_pairs_sym_edge_mean_rows[0].keys()),
                    rows=all_pairs_sym_edge_mean_rows,
                )

            edge_variant_outputs: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
            family_name = "P_c"
            variant_dir = out_dir / "edge_transfer_per_class_P_c"
            directed_rows_variant, per_class_rows_variant, mean_rows_variant = compute_all_pairs_symmetric_edge_rows_variant(
                raw_rows=edge_transfer_raw_rows,
                case_names=all_case_names,
                display_name_map=display_name_map,
                num_classes=num_classes,
                reduction="mean",
                shared_labels_by_pair=shared_labels_by_pair,
            )
            edge_variant_outputs[family_name] = (
                directed_rows_variant,
                per_class_rows_variant,
                mean_rows_variant,
            )
            if directed_rows_variant:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_all_pairs_directed_P_c_per_class.csv"),
                    fieldnames=list(directed_rows_variant[0].keys()),
                    rows=directed_rows_variant,
                )
            if per_class_rows_variant:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_all_pairs_symmetric_P_c_per_class.csv"),
                    fieldnames=list(per_class_rows_variant[0].keys()),
                    rows=per_class_rows_variant,
                )
                save_consensus_per_class_heatmaps(
                    case_names=display_case_names,
                    per_class_rows=per_class_rows_variant,
                    metric_key="bar_P_c",
                    metric_family=family_name,
                    out_dir=variant_dir,
                    cmap="copper",
                    norm_mode="log",
                )
            if mean_rows_variant:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_all_pairs_symmetric_P_c_mean_over_labels.csv"),
                    fieldnames=list(mean_rows_variant[0].keys()),
                    rows=mean_rows_variant,
                )
                save_consensus_mean_heatmap(
                    case_names=display_case_names,
                    mean_rows=mean_rows_variant,
                    metric_key="mean_bar_P_c",
                    metric_family=family_name,
                    out_path=variant_dir / "mean_P_c.png",
                    cmap="copper",
                    norm_mode="log",
                )

        else:
            topk_consistency_raw_rows, topk_consistency_pair_rows = compute_topk_consistency_pair_rows(
                case_names=all_case_names,
                case_results=case_results,
                case_models=case_models,
                case_out_hw=case_out_hw,
                device=device,
                batch_size=topl_transfer_batch_size,
            )
            if topk_consistency_raw_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_topk_consistency_raw.csv"),
                    fieldnames=list(topk_consistency_raw_rows[0].keys()),
                    rows=topk_consistency_raw_rows,
                )
            if topk_consistency_pair_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_topk_consistency_pair_summary.csv"),
                    fieldnames=list(topk_consistency_pair_rows[0].keys()),
                    rows=topk_consistency_pair_rows,
                )

            edge_consistency_raw_rows, edge_consistency_summary_rows = compute_edge_consistency_pair_rows(
                case_names=all_case_names,
                case_results=case_results,
                case_models=case_models,
                case_out_hw=case_out_hw,
                device=device,
                batch_size=edge_transfer_batch_size,
            )
            if edge_consistency_raw_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_edge_consistency_raw.csv"),
                    fieldnames=list(edge_consistency_raw_rows[0].keys()),
                    rows=edge_consistency_raw_rows,
                )
            if edge_consistency_summary_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_edge_consistency_pair_summary.csv"),
                    fieldnames=list(edge_consistency_summary_rows[0].keys()),
                    rows=edge_consistency_summary_rows,
                )

            sym_topk_rows, sym_topk_summary_rows = compute_focus_pair_topk_consistency_rows(
                raw_rows=topk_consistency_raw_rows,
                focus_pairs=focus_pairs,
                num_classes=num_classes,
                shared_labels_by_pair=shared_labels_by_pair,
            )
            if sym_topk_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_topk_consistency_per_class.csv"),
                    fieldnames=list(sym_topk_rows[0].keys()),
                    rows=sym_topk_rows,
                )
            if sym_topk_summary_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_topk_consistency_summary.csv"),
                    fieldnames=list(sym_topk_summary_rows[0].keys()),
                    rows=sym_topk_summary_rows,
                )

            all_pairs_sym_topk_rows, all_pairs_sym_topk_mean_rows = compute_all_pairs_topk_consistency_rows(
                raw_rows=topk_consistency_raw_rows,
                case_names=all_case_names,
                display_name_map=display_name_map,
                num_classes=num_classes,
                shared_labels_by_pair=shared_labels_by_pair,
            )
            if all_pairs_sym_topk_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_all_pairs_topk_consistency_per_class.csv"),
                    fieldnames=list(all_pairs_sym_topk_rows[0].keys()),
                    rows=all_pairs_sym_topk_rows,
                )
                save_consensus_per_class_heatmaps(
                    case_names=display_case_names,
                    per_class_rows=all_pairs_sym_topk_rows,
                    metric_key="T_c",
                    metric_family="T_c",
                    out_dir=out_dir / "top_k_consistency_per_class",
                    cmap="parula",
                    vmax=2.0,
                )
            if all_pairs_sym_topk_mean_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_all_pairs_topk_consistency_mean_over_labels.csv"),
                    fieldnames=list(all_pairs_sym_topk_mean_rows[0].keys()),
                    rows=all_pairs_sym_topk_mean_rows,
                )
                save_consensus_mean_heatmap(
                    case_names=display_case_names,
                    mean_rows=all_pairs_sym_topk_mean_rows,
                    metric_key="mean_T_c",
                    metric_family="T_c",
                    out_path=out_dir / "top_k_consistency_per_class" / "mean_T_c.png",
                    cmap="parula",
                    vmax=2.0,
                )

            sym_edge_rows, sym_edge_summary_rows = compute_focus_pair_edge_consistency_rows(
                raw_rows=edge_consistency_raw_rows,
                focus_pairs=focus_pairs,
                num_classes=num_classes,
                shared_labels_by_pair=shared_labels_by_pair,
            )
            if sym_edge_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_edge_consistency_per_class.csv"),
                    fieldnames=list(sym_edge_rows[0].keys()),
                    rows=sym_edge_rows,
                )
            if sym_edge_summary_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_edge_consistency_summary.csv"),
                    fieldnames=list(sym_edge_summary_rows[0].keys()),
                    rows=sym_edge_summary_rows,
                )

            all_pairs_sym_edge_rows, all_pairs_sym_edge_mean_rows = compute_all_pairs_edge_consistency_rows(
                raw_rows=edge_consistency_raw_rows,
                case_names=all_case_names,
                display_name_map=display_name_map,
                num_classes=num_classes,
                shared_labels_by_pair=shared_labels_by_pair,
            )
            if all_pairs_sym_edge_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_all_pairs_edge_consistency_per_class.csv"),
                    fieldnames=list(all_pairs_sym_edge_rows[0].keys()),
                    rows=all_pairs_sym_edge_rows,
                )
                save_consensus_per_class_heatmaps(
                    case_names=display_case_names,
                    per_class_rows=all_pairs_sym_edge_rows,
                    metric_key="P_c",
                    metric_family="P_c",
                    out_dir=out_dir / "edge_consistency_per_class",
                    cmap="copper",
                    vmax=4.0,
                )
            if all_pairs_sym_edge_mean_rows:
                write_csv(
                    csv_path=_csv_path(csv_dir, "exp6_q2_all_pairs_edge_consistency_mean_over_labels.csv"),
                    fieldnames=list(all_pairs_sym_edge_mean_rows[0].keys()),
                    rows=all_pairs_sym_edge_mean_rows,
                )
                save_consensus_mean_heatmap(
                    case_names=display_case_names,
                    mean_rows=all_pairs_sym_edge_mean_rows,
                    metric_key="mean_P_c",
                    metric_family="P_c",
                    out_path=out_dir / "edge_consistency_per_class" / "mean_P_c.png",
                    cmap="copper",
                    vmax=4.0,
                )

            edge_variant_outputs = {
                "P_c": ([], all_pairs_sym_edge_rows, all_pairs_sym_edge_mean_rows),
            }

        pair_summary_lookup: Dict[str, Dict[str, Any]] = {}
        for row in sym_topk_summary_rows:
            pair_summary_lookup[str(row["comparison_group"])] = dict(row)
        for row in sym_edge_summary_rows:
            key = str(row["comparison_group"])
            pair_summary_lookup.setdefault(key, {"comparison_group": key})
            pair_summary_lookup[key].update(row)

        pair_summary_rows = [pair_summary_lookup[str(spec["group"])] for spec in focus_pairs if str(spec["group"]) in pair_summary_lookup]
        if pair_summary_rows:
            fieldnames = []
            seen = set()
            for row in pair_summary_rows:
                for key in row.keys():
                    if key not in seen:
                        fieldnames.append(key)
                        seen.add(key)
            write_csv(
                csv_path=_csv_path(csv_dir, "exp6_q2_focus_pair_summary.csv"),
                fieldnames=fieldnames,
                rows=pair_summary_rows,
            )

        # ------------------------------------------------------------
        # compact class-level focus plots for MNIST-only pair comparisons
        # ------------------------------------------------------------
        focus_pair_dir = out_dir / "focus_pair_class_level_plots"
        ensure_dir(focus_pair_dir)

        mnist_focus_pairs = _mnist_only_focus_pairs()

        if feature_consensus_variant == "transfer":
            save_q2_feature_consensus_figure(
                topl_rows=sym_topk_rows,
                edge_rows=sym_edge_rows,
                focus_pairs=focus_pairs,
                num_classes=num_classes,
                out_path=focus_pair_dir / "exp6_q2_feature_consensus_6panel.png",
            )

            _directed_p, p_rows, _p_mean = edge_variant_outputs["P_c"]

            save_focus_pair_class_heatmaps(
                topl_rows=all_pairs_sym_topk_rows,
                edge_rows=p_rows,
                focus_pairs=mnist_focus_pairs,
                num_classes=num_classes,
                out_dir=focus_pair_dir,
                topl_cbar_label="",
                edge_cbar_label="",
            )

            save_focus_pair_metric_heatmap(
                rows=all_pairs_sym_topk_rows,
                focus_pairs=mnist_focus_pairs,
                num_classes=num_classes,
                metric_key="bar_T_c",
                out_path=focus_pair_dir / "mnist_only_focus_pairs_T_c_heatmap.png",
                cmap="parula",
                cbar_label="",
                norm_mode="linear",
                text_color_threshold=0.50,
            )

            save_focus_pair_metric_heatmap(
                rows=p_rows,
                focus_pairs=mnist_focus_pairs,
                num_classes=num_classes,
                metric_key="bar_P_c",
                out_path=focus_pair_dir / "mnist_only_focus_pairs_P_c_heatmap.png",
                cmap="copper",
                cbar_label="",
                norm_mode="log",
            )
        else:
            save_q2_feature_consensus_figure(
                topl_rows=sym_topk_rows,
                edge_rows=sym_edge_rows,
                focus_pairs=focus_pairs,
                num_classes=num_classes,
                out_path=focus_pair_dir / "exp6_q2_feature_consensus_6panel.png",
                topl_metric_key="T_c",
                edge_metric_key="P_c",
                topl_panel_title="Top-K Consistency ($T_c$)",
                edge_panel_title="Edge Consistency ($P_c$)",
                topl_ylabel="$T_c$",
                edge_ylabel="$P_c$",
                topl_ymax=2.0,
                edge_ymax=4.0,
                figure_title="Q2 Feature Consensus via Response Discrepancy",
            )

            _unused_directed, p_rows, _unused_mean = edge_variant_outputs["P_c"]

            save_focus_pair_class_heatmaps(
                topl_rows=all_pairs_sym_topk_rows,
                edge_rows=p_rows,
                focus_pairs=mnist_focus_pairs,
                num_classes=num_classes,
                out_dir=focus_pair_dir,
                topl_metric_key="T_c",
                edge_metric_key="P_c",
                topl_title=r"$T_c$",
                edge_title=r"$P_c$",
                topl_cbar_label=r"$T_c$",
                edge_cbar_label=r"$P_c$",
                topl_norm_mode="linear",
                edge_norm_mode="linear",
                topl_vmax=2.0,
                edge_vmax=4.0,
            )

            save_focus_pair_metric_heatmap(
                rows=all_pairs_sym_topk_rows,
                focus_pairs=mnist_focus_pairs,
                num_classes=num_classes,
                metric_key="T_c",
                out_path=focus_pair_dir / "mnist_only_focus_pairs_T_c_heatmap.png",
                cmap="parula",
                cbar_label=r"$T_c$",
                norm_mode="linear",
                text_color_threshold=1.0,
                vmax=2.0,
            )

            save_focus_pair_metric_heatmap(
                rows=p_rows,
                focus_pairs=mnist_focus_pairs,
                num_classes=num_classes,
                metric_key="P_c",
                out_path=focus_pair_dir / "mnist_only_focus_pairs_P_c_heatmap.png",
                cmap="copper",
                cbar_label=r"$P_c$",
                norm_mode="linear",
                text_color_threshold=2.0,
                vmax=4.0,
            )

        logger.log(f"[Done] Saved outputs to {out_dir}")
        logger.log(f"[Done] CSV outputs collected in {csv_dir}")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
