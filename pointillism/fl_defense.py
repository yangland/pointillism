from __future__ import annotations

import csv
import itertools
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pointillism.pointillism_norm import normalize_like_loader
from pointillism.pointillism_probe_search import (
    _neighbor_masks_for_local_refine,
    build_grids_from_flatidx,
    build_grids_from_flatidx_with_fill,
    render_gaussian_probe_records,
    render_two_group_probe_records,
    sample_two_group_color_centers,
    sample_two_group_color_centers_batch_cpu,
    sample_random_masks_for_d,
    sample_random_masks_for_d_batched_cpu,
)
from pointillism.pointillism_signature import build_w_list_for_grid


_STATE_BY_OUTDIR: Dict[str, "PointillismFLState"] = {}


@dataclass
class PointillismFLState:
    topk_bank: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    edge_bank: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    color_bank: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    random_pool_fresh: List[Dict[str, Any]] = field(default_factory=list)
    rounds_seen: int = 0


def _pair_key(a: int, b: int) -> str:
    x = int(min(a, b))
    y = int(max(a, b))
    return f"{x}_{y}"


def _parse_pair_key(key: str) -> Tuple[int, int]:
    a, b = str(key).split("_")
    return int(a), int(b)


def _probe_signature(spec: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        str(spec.get("probe_mode", "binary")),
        int(spec.get("grid_hw", -1)),
        int(spec.get("grid_h", spec.get("grid_hw", -1))),
        int(spec.get("grid_w", spec.get("grid_hw", -1))),
        tuple(int(x) for x in spec.get("flat_idx", [])),
        str(spec.get("polarity", "white")).lower(),
        int(spec.get("color_id", 0)),
        int(spec.get("render_seed", 0)),
    )


def _strip_probe_runtime_fields(spec: Dict[str, Any]) -> Dict[str, Any]:
    keep = {
        "grid_hw": int(spec.get("grid_hw", -1)),
        "grid_h": int(spec.get("grid_h", spec.get("grid_hw", -1))),
        "grid_w": int(spec.get("grid_w", spec.get("grid_hw", -1))),
        "flat_idx": [int(x) for x in spec.get("flat_idx", [])],
        "w": int(spec.get("w", len(spec.get("flat_idx", [])))),
        "pct": float(spec.get("pct", -1.0)),
        "polarity": str(spec.get("polarity", "white")).lower(),
        "source": str(spec.get("source", "memory")),
        "color_id": int(spec.get("color_id", 0)),
        "probe_mode": str(spec.get("probe_mode", "binary")),
        "render_seed": int(spec.get("render_seed", 0)),
    }
    if str(keep["probe_mode"]).startswith(("two_group", "gaussian")):
        keep["num_channels"] = int(spec.get("num_channels", len(spec.get("mu_fg", [1.0]))))
        # Legacy field retained so older saved specs can round-trip cleanly.
        keep["cell_size"] = int(spec.get("cell_size", 4))
        keep["sigma_fg"] = float(spec.get("sigma_fg", 0.1))
        keep["sigma_bg"] = float(spec.get("sigma_bg", 0.1))
        keep["mu_fg"] = [float(x) for x in spec.get("mu_fg", [1.0, 1.0, 1.0])]
        keep["mu_bg"] = [float(x) for x in spec.get("mu_bg", [0.0, 0.0, 0.0])]
        keep["value_range"] = [float(x) for x in spec.get("value_range", [0.0, 1.0])]
    return keep


def _two_group_color_signature(spec: Dict[str, Any]) -> Tuple[Any, ...]:
    num_channels = int(spec.get("num_channels", len(spec.get("mu_fg", []))))
    mu_fg = tuple(round(float(x), 6) for x in list(spec.get("mu_fg", []))[:num_channels])
    mu_bg = tuple(round(float(x), 6) for x in list(spec.get("mu_bg", []))[:num_channels])
    return (
        int(num_channels),
        mu_fg,
        mu_bg,
    )


def _two_group_color_exemplar(
    *,
    spec: Dict[str, Any],
    label: int,
    source: str,
    pair: Optional[str] = None,
) -> Dict[str, Any]:
    num_channels = int(spec.get("num_channels", len(spec.get("mu_fg", []))))
    out = {
        "label": int(label),
        "source": str(source),
        "pair": None if pair is None else str(pair),
        "num_channels": int(num_channels),
        "mu_fg": [float(x) for x in list(spec.get("mu_fg", []))[:num_channels]],
        "mu_bg": [float(x) for x in list(spec.get("mu_bg", []))[:num_channels]],
    }
    if "sigma_fg" in spec:
        out["sigma_fg"] = float(spec.get("sigma_fg", 0.1))
    if "sigma_bg" in spec:
        out["sigma_bg"] = float(spec.get("sigma_bg", 0.1))
    if "value_range" in spec:
        out["value_range"] = [float(x) for x in spec.get("value_range", [0.0, 1.0])]
    return out


def _build_color_memory_bank(
    *,
    topk_bank: Dict[int, List[Dict[str, Any]]],
    edge_bank: Dict[str, List[Dict[str, Any]]],
    num_classes: int,
) -> Dict[int, List[Dict[str, Any]]]:
    bank: Dict[int, List[Dict[str, Any]]] = {int(label): [] for label in range(int(num_classes))}
    seen: Dict[int, Set[Tuple[Any, ...]]] = {int(label): set() for label in range(int(num_classes))}

    def _add(label: int, spec: Dict[str, Any], source: str, pair: Optional[str] = None) -> None:
        if not str(spec.get("probe_mode", "binary")).startswith(("two_group", "gaussian")):
            return
        key = _two_group_color_signature(spec)
        if key in seen[int(label)]:
            return
        seen[int(label)].add(key)
        bank[int(label)].append(
            _two_group_color_exemplar(spec=spec, label=int(label), source=source, pair=pair)
        )

    for label in range(int(num_classes)):
        for spec in topk_bank.get(int(label), []):
            _add(int(label), spec, source="topk")

    for pair_name, specs in edge_bank.items():
        a, b = _parse_pair_key(str(pair_name))
        for spec in specs:
            _add(int(a), spec, source="edge", pair=str(pair_name))
            _add(int(b), spec, source="edge", pair=str(pair_name))

    return bank


def _flatten_color_memory_bank(
    *,
    color_bank: Dict[int, List[Dict[str, Any]]],
    num_channels: int,
) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    seen: Set[Tuple[Any, ...]] = set()
    for label in sorted(color_bank.keys()):
        for entry in color_bank.get(int(label), []):
            if int(entry.get("num_channels", 0)) != int(num_channels):
                continue
            key = (
                int(entry.get("num_channels", 0)),
                tuple(round(float(x), 6) for x in entry.get("mu_fg", [])),
                tuple(round(float(x), 6) for x in entry.get("mu_bg", [])),
            )
            if key in seen:
                continue
            seen.add(key)
            pool.append(dict(entry))
    return pool


def _get_state(out_dir: Optional[str]) -> PointillismFLState:
    key = str(out_dir or "__default__")
    if key not in _STATE_BY_OUTDIR:
        _STATE_BY_OUTDIR[key] = PointillismFLState()
    return _STATE_BY_OUTDIR[key]


def _dataset_name(cfg: Dict[str, Any]) -> str:
    return str((cfg.get("task", {}) or {}).get("dataset", "mnist")).lower()


def _num_classes(cfg: Dict[str, Any], model: torch.nn.Module) -> int:
    cfg_val = (cfg.get("model", {}) or {}).get("num_classes", None)
    if cfg_val is not None:
        return int(cfg_val)
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Linear):
            return int(module.out_features)
    raise ValueError("Could not infer num_classes for pointillism_fl")


def _in_channels(cfg: Dict[str, Any]) -> int:
    return int((cfg.get("task", {}) or {}).get("channels", 1))


def _out_hw(cfg: Dict[str, Any]) -> int:
    task_cfg = cfg.get("task", {}) or {}
    img_size = task_cfg.get("img_size", [28, 28])
    return int(img_size[0])

def _out_hws(cfg: Dict[str, Any]) -> Tuple[int, int]:
    task_cfg = cfg.get("task", {}) or {}
    img_size = task_cfg.get("img_size", [28, 28])
    return int(img_size[0]), int(img_size[1])


def _grid_hws(cfg_point: Dict[str, Any], out_hw: int) -> List[int]:
    if "grid_freq" in cfg_point or "grid_time" in cfg_point:
        return [int(cfg_point.get("grid_freq", cfg_point.get("grid_time", out_hw)))]
    vals = cfg_point.get("grid_hws", None)
    if vals is None:
        vals = [cfg_point.get("grid_hw", max(4, min(8, int(out_hw) // 4 or 4)))]
    out = sorted({int(v) for v in vals if int(v) > 1})
    if not out:
        raise ValueError("pointillism_fl requires at least one grid_hw")
    return out


def _grid_shape(cfg_point: Dict[str, Any], grid_hw: int) -> Tuple[int, int]:
    if "grid_freq" in cfg_point or "grid_time" in cfg_point:
        return int(cfg_point.get("grid_freq", grid_hw)), int(cfg_point.get("grid_time", grid_hw))
    return int(grid_hw), int(grid_hw)


def _w_list_for_cells(*, n_cells: int, w_cfg: Dict[str, Any]) -> List[int]:
    pcts = [float(x) for x in w_cfg.get("pcts", [0.08, 0.16, 0.32, 0.5])]
    rounding = str(w_cfg.get("rounding", "nearest")).lower()
    vals = []
    for pct in pcts:
        raw = float(pct) * int(n_cells)
        w = math.ceil(raw) if rounding == "ceil" else (math.floor(raw) if rounding == "floor" else round(raw))
        if 0 < int(w) < int(n_cells):
            vals.append(int(w))
    return sorted(set(vals))


def _sample_masks_for_cells(*, n_cells: int, w_list: Sequence[int], budget: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(int(seed))
    out: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, ...]] = set()
    if not w_list:
        return out
    per_w = max(1, int(math.ceil(float(budget) / len(w_list))))
    for w in w_list:
        attempts = 0
        count_w = 0
        while count_w < per_w and len(out) < int(budget):
            attempts += 1
            if attempts > per_w * 20:
                break
            idx = tuple(sorted(rng.sample(range(int(n_cells)), int(w))))
            if idx in seen:
                continue
            seen.add(idx)
            out.append({"flat_idx": list(idx), "w": int(w), "polarity": "white"})
            count_w += 1
    rng.shuffle(out)
    return out[:int(budget)]


def _safe_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _top_indices_desc(scores_1d: np.ndarray, top_n: int) -> np.ndarray:
    """
    Return indices of the largest `top_n` values in descending score order.

    This avoids sorting the full array when only a top slice is needed.
    """
    scores_1d = np.asarray(scores_1d)
    n = int(scores_1d.shape[0])
    top_n = int(top_n)
    if top_n <= 0 or n <= 0:
        return np.empty((0,), dtype=np.int64)
    if top_n >= n:
        return np.argsort(-scores_1d, kind="stable").astype(np.int64, copy=False)

    top_idx = np.argpartition(-scores_1d, kth=top_n - 1)[:top_n]
    order = np.argsort(-scores_1d[top_idx], kind="stable")
    return top_idx[order].astype(np.int64, copy=False)


def _median_or_zero(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return 0.0
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, float(value))))


def _topk_disagreement_to_agreement(value: float) -> float:
    # Top-K discrepancy sums two absolute probability differences, so it lies in [0, 2].
    return _clamp01(1.0 - float(value) / 2.0)


def _edge_disagreement_to_agreement(value: float) -> float:
    # Edge discrepancy sums two absolute margin differences, so it lies in [0, 4].
    return _clamp01(1.0 - float(value) / 4.0)


def _top2_matches_pair(pred2_row: np.ndarray, a: int, b: int) -> bool:
    vals = {int(pred2_row[0]), int(pred2_row[1])}
    return vals == {int(a), int(b)}


def _combine_bar_a(*, bar_t: float, bar_p: float, cfg_point: Optional[Dict[str, Any]] = None) -> float:
    cfg_point = dict(cfg_point or {})
    mode = str(cfg_point.get("bar_a_mode", "none")).strip().lower()
    if mode in {"none", "", "both"}:
        return float(1.0 - (1.0 - float(bar_t)) * (1.0 - float(bar_p)))
    if mode in {"tc", "t", "topk", "top_k"}:
        return float(bar_t)
    if mode in {"pc", "p", "edge"}:
        return float(bar_p)
    raise ValueError(
        f"Unsupported pointillism_fl.bar_a_mode={mode!r}. "
        "Expected one of: none, both, tc, pc."
    )


def _selection_table_mode(cfg_point: Optional[Dict[str, Any]] = None) -> str:
    cfg_point = dict(cfg_point or {})
    mode = str(cfg_point.get("selection_table", "bar_A")).strip().lower()
    aliases = {
        "": "bar_a",
        "default": "bar_a",
        "full": "bar_a",
        "a": "bar_a",
        "bar_a": "bar_a",
        "r": "bar_r",
        "bar_r": "bar_r",
        "reliability": "bar_r",
        "t": "bar_t",
        "bar_t": "bar_t",
        "tc": "bar_t",
        "topk": "bar_t",
        "top_k": "bar_t",
        "p": "bar_p",
        "bar_p": "bar_p",
        "pc": "bar_p",
        "edge": "bar_p",
    }
    if mode in aliases:
        return aliases[mode]
    raise ValueError(
        f"Unsupported pointillism_fl.selection_table={mode!r}. "
        "Expected one of: bar_A, bar_R, bar_T, bar_P."
    )


def _selection_score_from_tables(
    *,
    selection_table: str,
    bar_t: float,
    bar_p: float,
    bar_r: float,
    bar_a: float,
) -> float:
    table = str(selection_table).lower()
    if table == "bar_r":
        return float(bar_r)
    if table == "bar_t":
        return float(bar_t)
    if table == "bar_p":
        return float(bar_p)
    return float(bar_a)


def _entropy_of_distribution(q_vals: Sequence[float]) -> float:
    vals = np.asarray([max(0.0, float(x)) for x in q_vals], dtype=np.float64)
    total = float(vals.sum())
    if total <= 0.0:
        return 0.0
    probs = vals / float(total)
    probs = probs[probs > 0.0]
    if probs.size <= 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))


def _model_reliability_from_by_grid(
    by_grid: Dict[Any, Any],
    *,
    num_classes: int,
) -> Tuple[float, List[Dict[str, Any]]]:
    num_classes_i = max(1, int(num_classes))
    log_c = math.log(float(max(2, num_classes_i)))
    rows: List[Dict[str, Any]] = []

    for d_raw, rec_raw in sorted(dict(by_grid or {}).items(), key=lambda item: int(item[0])):
        rec = dict(rec_raw or {})
        q_vals = list(rec.get("q", []))[:num_classes_i]
        h_vals = list(rec.get("entropy_mean", []))[:num_classes_i]
        if len(q_vals) < num_classes_i:
            q_vals.extend([0.0 for _ in range(num_classes_i - len(q_vals))])
        if len(h_vals) < num_classes_i:
            h_vals.extend([float(log_c) for _ in range(num_classes_i - len(h_vals))])

        q_entropy = _entropy_of_distribution(q_vals)
        diversity = _clamp01(float(q_entropy) / float(log_c)) if log_c > 0.0 else 0.0
        mean_entropy_norm = float(sum(float(h) / float(log_c) for h in h_vals) / float(num_classes_i)) if log_c > 0.0 else 1.0
        confidence = _clamp01(1.0 - float(mean_entropy_norm))
        reliability = float(diversity)
        rows.append({
            "grid_hw": int(d_raw),
            "reliability_mode": "diversity",
            "diversity": float(diversity),
            "confidence": float(confidence),
            "model_reliability_d": float(reliability),
            "q_entropy": float(q_entropy),
            "mean_entropy_norm": float(mean_entropy_norm),
            "dominant_label": int(np.argmax(np.asarray(q_vals, dtype=np.float64))) if q_vals else -1,
            "dominant_q": float(max(q_vals)) if q_vals else 0.0,
        })

    if not rows:
        return 0.0, []
    return float(sum(float(row["model_reliability_d"]) for row in rows) / float(len(rows))), rows


def _model_reliability_value(q1: Dict[str, Any], *, num_classes: int) -> float:
    if "model_reliability" in q1:
        return _clamp01(float(q1.get("model_reliability", 0.0)))
    if q1.get("evidence_by_grid"):
        rel, _rows = _model_reliability_from_by_grid(q1.get("evidence_by_grid", {}) or {}, num_classes=int(num_classes))
        return _clamp01(float(rel))
    return 1.0


def _class_evidence_details(q1: Dict[str, Any], *, num_classes: int) -> List[Dict[str, Any]]:
    rel = [0.0 for _ in range(max(0, int(num_classes)))]
    by_grid = q1.get("evidence_by_grid", {}) or {}
    rows: List[Dict[str, Any]] = []
    for label in range(int(num_classes)):
        best_grid = None
        best_q = 0.0
        best_u = 0.0
        best_h = 0.0
        best_e = float(rel[int(label)]) if int(label) < len(rel) else 0.0
        best_count = 0
        for d_raw, rec_raw in by_grid.items():
            rec = dict(rec_raw or {})
            e_vals = list(rec.get("E", []))
            if int(label) >= len(e_vals):
                continue
            e_val = float(e_vals[int(label)])
            if best_grid is None or float(e_val) > float(best_e) or math.isclose(float(e_val), float(best_e)):
                q_vals = list(rec.get("q", []))
                u_vals = list(rec.get("u", []))
                h_vals = list(rec.get("entropy_mean", []))
                count_vals = list(rec.get("counts", []))
                best_grid = int(d_raw)
                best_e = float(e_val)
                best_q = float(q_vals[int(label)]) if int(label) < len(q_vals) else 0.0
                best_u = float(u_vals[int(label)]) if int(label) < len(u_vals) else 0.0
                best_h = float(h_vals[int(label)]) if int(label) < len(h_vals) else 0.0
                best_count = int(count_vals[int(label)]) if int(label) < len(count_vals) else 0
        rows.append({
            "label": int(label),
            "class_evidence": float(_clamp01(best_e)),
            "best_grid_hw": "" if best_grid is None else int(best_grid),
            "best_q": float(best_q),
            "best_entropy_mean": float(best_h),
            "best_u": float(best_u),
            "best_E": float(_clamp01(best_e)),
            "best_count": int(best_count),
        })
    return rows


def _pairwise_label_agreement_mode(cfg_point: Optional[Dict[str, Any]] = None) -> str:
    cfg_point = dict(cfg_point or {})
    mode = str(cfg_point.get("pairwise_label_agreement", "topk")).strip().lower()
    if mode in {"majority_supported", "majority-supported", "majority", "support_aware", "support-aware"}:
        return "majority_supported"
    if mode in {"class_avg", "classavg", "avg", "mean", "legacy", "old"}:
        return "class_avg"
    if mode in {"lowk", "low_k", "bottomk", "bottom_k"}:
        return "lowk"
    if mode in {"topk", "top_k", "label_topk", "structure_aware"}:
        return "topk"
    raise ValueError(
        f"Unsupported pointillism_fl.pairwise_label_agreement={mode!r}. "
        "Expected one of: topk, lowk, class_avg, majority_supported."
    )


def _pairwise_label_k_config(cfg_point: Optional[Dict[str, Any]] = None) -> float:
    cfg_point = dict(cfg_point or {})
    if "pairwise_label_k" in cfg_point:
        raw = cfg_point.get("pairwise_label_k")
    elif "pairwise_label_top_k" in cfg_point:
        raw = cfg_point.get("pairwise_label_top_k")
    elif _pairwise_label_agreement_mode(cfg_point) in {"class_avg", "majority_supported"}:
        raw = 1.0
    else:
        raw = 0.5
    return float(raw)


def _resolve_pairwise_label_k(*, k_cfg: float, num_labels: int) -> int:
    num_labels = max(0, int(num_labels))
    if num_labels <= 0:
        return 0
    k_cfg = float(k_cfg)
    if k_cfg <= 0.0:
        return int(num_labels)
    if k_cfg <= 1.0:
        return max(1, min(int(num_labels), int(math.ceil(float(k_cfg) * float(num_labels)))))
    return max(1, min(int(num_labels), int(math.ceil(k_cfg))))


def _reduce_label_agreement_rows(
    label_rows: List[Tuple[int, float, float, float, float, float]],
    *,
    mode: str,
    k_cfg: float,
    num_classes: int,
    global_selected_labels: Optional[Set[int]] = None,
    global_label_weights: Optional[Dict[int, float]] = None,
) -> Tuple[float, float, float, float, float, List[Tuple[int, float, float, float, float, float]], int]:
    if not label_rows:
        return 0.0, 0.0, 0.0, 0.0, 0.0, [], 0

    rows = list(label_rows)
    resolved_k = _resolve_pairwise_label_k(k_cfg=k_cfg, num_labels=len(rows))
    use_mode = str(mode)

    if use_mode in {"topk", "lowk", "class_avg"}:
        selected_label_set = set(int(x) for x in (global_selected_labels or set()))
        selected = [row for row in rows if int(row[0]) in selected_label_set]
        selected = sorted(selected, key=lambda row: int(row[0]))
        resolved_k = len(selected)
        t_sum = float(sum(float(row[1]) for row in selected))
        p_sum = float(sum(float(row[2]) for row in selected))
        a_sum = float(sum(float(row[3]) for row in selected))
        r_val = _safe_mean([float(row[4]) for row in selected])
        a_raw_sum = float(sum(float(row[5]) for row in selected))
    elif use_mode == "majority_supported":
        selected = sorted(rows, key=lambda row: int(row[0]))
        resolved_k = len(selected)
        weights = {int(label): _clamp01(weight) for label, weight in dict(global_label_weights or {}).items()}
        t_sum = float(sum(float(weights.get(int(row[0]), 0.0)) * float(row[1]) for row in selected))
        p_sum = float(sum(float(weights.get(int(row[0]), 0.0)) * float(row[2]) for row in selected))
        a_sum = float(sum(float(weights.get(int(row[0]), 0.0)) * float(row[3]) for row in selected))
        weight_sum = float(sum(float(weights.get(int(row[0]), 0.0)) for row in selected))
        r_val = (
            float(sum(float(weights.get(int(row[0]), 0.0)) * float(row[4]) for row in selected)) / float(weight_sum)
            if weight_sum > 0.0
            else 0.0
        )
        a_raw_sum = float(sum(float(weights.get(int(row[0]), 0.0)) * float(row[5]) for row in selected))
    else:
        raise ValueError(f"Unsupported pairwise label agreement reducer: {mode!r}")

    denom = float(max(1, int(num_classes)))
    return (
        float(t_sum / denom),
        float(p_sum / denom),
        float(a_sum / denom),
        float(r_val),
        float(a_raw_sum / denom),
        selected,
        int(resolved_k),
    )


def _select_global_agreement_labels(
    *,
    label_values: List[List[float]],
    mode: str,
    k_cfg: float,
    num_classes: int,
) -> Tuple[List[int], List[Tuple[int, float]]]:
    global_rows: List[Tuple[int, float]] = []
    for label in range(int(num_classes)):
        vals = [float(v) for v in label_values[int(label)] if np.isfinite(float(v))]
        if vals:
            global_rows.append((int(label), float(sum(vals) / float(len(vals)))))

    if not global_rows:
        return [], []

    k_global = _resolve_pairwise_label_k(k_cfg=k_cfg, num_labels=int(num_classes))
    k_global = max(1, min(int(k_global), len(global_rows)))
    if str(mode) in {"topk", "class_avg"}:
        selected_rows = sorted(global_rows, key=lambda row: (-float(row[1]), int(row[0])))[:k_global]
    elif str(mode) == "lowk":
        selected_rows = sorted(global_rows, key=lambda row: (float(row[1]), int(row[0])))[:k_global]
    else:
        raise ValueError(f"Unsupported global label agreement mode: {mode!r}")

    selected_labels = sorted(int(label) for label, _g in selected_rows)
    selected_rows_by_label = {int(label): float(g_val) for label, g_val in selected_rows}
    selected_global_rows = [(int(label), float(selected_rows_by_label[int(label)])) for label in selected_labels]
    return selected_labels, selected_global_rows


def _majority_supported_label_weights(
    *,
    label_values: List[List[float]],
    num_clients: int,
    num_classes: int,
) -> Tuple[List[int], List[Tuple[int, float]]]:
    k = max(1, int(num_clients))
    support_rows: List[Tuple[int, float]] = []
    for label in range(int(num_classes)):
        pair_vals = [float(v) for v in label_values[int(label)] if np.isfinite(float(v))]
        if not pair_vals:
            continue
        h_c = float((2.0 * sum(pair_vals)) / float(max(1, k - 1)))
        support_rows.append((int(label), float(h_c)))

    if not support_rows:
        return [], []

    max_support = max(float(h_c) for _label, h_c in support_rows)
    if max_support <= 0.0:
        weight_rows = [(int(label), 0.0) for label, _h_c in support_rows]
    else:
        weight_rows = [
            (int(label), _clamp01(float(h_c) / float(max_support)))
            for label, h_c in support_rows
        ]

    selected_labels = [int(label) for label, _weight in weight_rows]
    return selected_labels, weight_rows


def _label_coverage_counts(
    *,
    subset: Sequence[int],
    client_infos: List[Dict[str, Any]],
    num_classes: int,
) -> List[int]:
    counts = [0 for _ in range(int(num_classes))]
    for idx in subset:
        support = set(int(x) for x in client_infos[int(idx)]["q1"].get("effective_support_labels", []))
        for label in support:
            if 0 <= int(label) < int(num_classes):
                counts[int(label)] += 1
    return counts


def _label_coverage_score(counts: Sequence[int]) -> float:
    return float(sum(math.log1p(float(x)) for x in counts))


def _cfg_point(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(cfg.get("pointillism_fl", {}) or {})


def _point_probe_mode(cfg: Dict[str, Any], cfg_point: Optional[Dict[str, Any]] = None) -> str:
    cfg_point = dict(cfg_point or _cfg_point(cfg))
    mode = str(cfg_point.get("probe_mode", "binary")).strip().lower()
    if mode in {"two_group", "gaussian"}:
        return mode
    if mode not in {"binary", "two_group_rgb", "two_group_gray", "gaussian_rgb", "gaussian_gray"}:
        raise ValueError(
            f"Unsupported pointillism_fl.probe_mode={mode!r}. "
            "Expected one of: binary, two_group, gaussian, two_group_rgb, two_group_gray, gaussian_rgb, gaussian_gray."
        )
    return mode


def _point_color_mode(cfg: Dict[str, Any], cfg_point: Optional[Dict[str, Any]] = None) -> str:
    cfg_point = dict(cfg_point or _cfg_point(cfg))
    mode = str(cfg_point.get("color_mode", "mono")).strip().lower()
    if mode not in {"mono", "palette"}:
        raise ValueError(
            f"Unsupported pointillism_fl.color_mode={mode!r}. "
            "Expected one of: mono, palette."
        )
    return mode


def _point_palette_color_ids(cfg: Dict[str, Any], cfg_point: Optional[Dict[str, Any]] = None) -> List[int]:
    cfg_point = dict(cfg_point or _cfg_point(cfg))
    mode = _point_color_mode(cfg, cfg_point=cfg_point)
    in_ch = _in_channels(cfg)
    if int(in_ch) != 3 or mode == "mono":
        return [0]
    # 0=white, 1=red, 2=green, 3=blue
    palette = cfg_point.get("palette_color_ids", [0, 1, 2, 3])
    out: List[int] = []
    for val in palette:
        cid = int(val)
        if cid not in {0, 1, 2, 3}:
            raise ValueError(
                "pointillism_fl.palette_color_ids must only contain color ids "
                "from {0,1,2,3} for white/red/green/blue."
            )
        if cid not in out:
            out.append(cid)
    return out or [0]


def _point_two_group_cfg(cfg: Dict[str, Any], cfg_point: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg_point = dict(cfg_point or _cfg_point(cfg))
    color_cfg = dict(cfg_point.get("color_group", {}) or {})
    return {
        "variants_per_mask": int(color_cfg.get("variants_per_mask", 1)),
        # Legacy compatibility only. Two-group rendering now follows grid_hw
        # so each logical grid cell is a single pure-color block.
        "cell_size": int(color_cfg.get("cell_size", 4)),
        "sigma_fg": float(color_cfg.get("sigma_fg", 0.10)),
        "sigma_bg": float(color_cfg.get("sigma_bg", 0.10)),
        "d_min": float(color_cfg.get("d_min", 0.30)),
        # Two-group probes now always use the full normalized color space.
        "value_range": [0.0, 1.0],
        "center_range": [0.0, 1.0],
    }


def _point_gaussian_cfg(cfg: Dict[str, Any], cfg_point: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg_point = dict(cfg_point or _cfg_point(cfg))
    gaussian_cfg = dict(cfg_point.get("gaussian", cfg_point.get("gaussian_group", {})) or {})
    return {
        "sigma": float(gaussian_cfg.get("sigma", 0.35)),
        "mean": 0.5,
        "value_range": [0.0, 1.0],
    }


def _resolved_two_group_mode(cfg: Dict[str, Any], cfg_point: Optional[Dict[str, Any]] = None) -> str:
    mode = _point_probe_mode(cfg, cfg_point=cfg_point)
    in_ch = _in_channels(cfg)
    if mode == "two_group":
        return "two_group_gray" if int(in_ch) == 1 else "two_group_rgb"
    if mode == "gaussian":
        return "gaussian_gray" if int(in_ch) == 1 else "gaussian_rgb"
    return mode


def _stable_mask_seed(flat_idx: Sequence[int]) -> int:
    acc = 0
    for pos, val in enumerate(flat_idx):
        acc = (acc * 1000003 + (int(pos) + 1) * 9176 + int(val) + 1) % 2147483647
    return int(acc)


def _point_log(recorder, msg: str) -> None:
    if recorder is not None:
        recorder.maybe_print(str(msg))


def _render_probe_specs(
    *,
    specs: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    dataset = _dataset_name(cfg)
    out_h, out_w = _out_hws(cfg)
    out_hw = int(out_h)
    in_ch = _in_channels(cfg)

    if not specs:
        return torch.empty((0, in_ch, out_h, out_w), device=device)

    grouped: Dict[Tuple[Any, ...], List[Tuple[int, Dict[str, Any]]]] = {}
    for idx, spec in enumerate(specs):
        key = (
            str(spec.get("probe_mode", "binary")),
            int(spec["grid_hw"]),
            str(spec.get("polarity", "white")).lower(),
            int(spec.get("color_id", 0)),
        )
        grouped.setdefault(key, []).append((int(idx), spec))

    imgs = torch.zeros((len(specs), in_ch, out_h, out_w), dtype=torch.float32, device=device)
    for (probe_mode, grid_hw, polarity, color_id), group in grouped.items():
        order = [int(pos) for pos, _ in group]
        recs = [spec for _, spec in group]
        if str(probe_mode).startswith("two_group"):
            up = render_two_group_probe_records(
                recs,
                out_hw=int(out_hw),
                device=device,
            )
        elif str(probe_mode).startswith("gaussian"):
            up = render_gaussian_probe_records(
                recs,
                out_hw=int(out_hw),
                device=device,
            )
        else:
            flatidx_list = [list(spec.get("flat_idx", [])) for spec in recs]
            if str(polarity) == "black":
                grids = build_grids_from_flatidx_with_fill(
                    flatidx_list=flatidx_list,
                    grid_hw=int(grid_hw),
                    device=device,
                    background_fill=1.0,
                    active_fill=0.0,
                )
            else:
                grids = build_grids_from_flatidx(flatidx_list=flatidx_list, grid_hw=int(grid_hw), device=device)
            up = F.interpolate(grids, size=(int(out_hw), int(out_hw)), mode="nearest")
            if int(in_ch) == 3:
                rgb = torch.zeros((up.shape[0], 3, up.shape[2], up.shape[3]), dtype=up.dtype, device=device)
                cid = int(color_id)
                if cid == 0:
                    rgb[:, 0:1] = up
                    rgb[:, 1:2] = up
                    rgb[:, 2:3] = up
                elif cid == 1:
                    rgb[:, 0:1] = up
                elif cid == 2:
                    rgb[:, 1:2] = up
                elif cid == 3:
                    rgb[:, 2:3] = up
                else:
                    raise ValueError(
                        f"Unsupported pointillism_fl probe color_id={cid}. "
                        "Expected one of {0,1,2,3}."
                    )
                up = rgb
            elif int(in_ch) > 1:
                up = up.repeat(1, int(in_ch), 1, 1)
        if tuple(up.shape[-2:]) != (int(out_h), int(out_w)):
            up = F.interpolate(up, size=(int(out_h), int(out_w)), mode="nearest")
        imgs[torch.as_tensor(order, dtype=torch.long, device=device)] = up

    return normalize_like_loader(imgs, dataset=dataset)


@torch.inference_mode()
def score_probe_specs(
    *,
    models: List[torch.nn.Module],
    specs: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    if not models or not specs:
        k = len(models)
        p = len(specs)
        c = 0 if not models else _num_classes(cfg, models[0])
        return {
            "probs": np.zeros((k, p, c), dtype=np.float32),
            "pred1": np.zeros((k, p), dtype=np.int64),
            "pred2": np.zeros((k, p, 2), dtype=np.int64),
            "entropy": np.zeros((k, p), dtype=np.float32),
            "margin": np.zeros((k, p), dtype=np.float32),
        }

    imgs = _render_probe_specs(specs=specs, cfg=cfg, device=device)
    num_classes = _num_classes(cfg, models[0])

    probs_all = np.zeros((len(models), len(specs), num_classes), dtype=np.float32)
    pred1_all = np.zeros((len(models), len(specs)), dtype=np.int64)
    pred2_all = np.zeros((len(models), len(specs), 2), dtype=np.int64)
    entropy_all = np.zeros((len(models), len(specs)), dtype=np.float32)
    margin_all = np.zeros((len(models), len(specs)), dtype=np.float32)

    bs = max(1, int(batch_size))
    use_cuda_autocast = str(device.type).lower() == "cuda"
    for mi, model in enumerate(models):
        model.eval()
        start = 0
        while start < len(specs):
            end = min(len(specs), start + bs)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda_autocast):
                logits = model(imgs[start:end])
            probs_t = torch.softmax(logits.float(), dim=1)
            top2_vals, top2_idx = torch.topk(probs_t, k=min(2, probs_t.shape[1]), dim=1)
            ent = -(probs_t * torch.log(probs_t.clamp_min(1e-12))).sum(dim=1)

            probs_np = probs_t.detach().cpu().numpy().astype(np.float32, copy=False)
            top2_idx_np = top2_idx.detach().cpu().numpy().astype(np.int64, copy=False)
            top2_vals_np = top2_vals.detach().cpu().numpy().astype(np.float32, copy=False)

            probs_all[mi, start:end, :] = probs_np
            pred1_all[mi, start:end] = top2_idx_np[:, 0]
            pred2_all[mi, start:end, :] = top2_idx_np[:, :2]
            entropy_all[mi, start:end] = ent.detach().cpu().numpy().astype(np.float32, copy=False)

            if top2_vals_np.shape[1] >= 2:
                margin_all[mi, start:end] = top2_vals_np[:, 0] - top2_vals_np[:, 1]
            else:
                margin_all[mi, start:end] = top2_vals_np[:, 0]
            start = end

    return {
        "probs": probs_all,
        "pred1": pred1_all,
        "pred2": pred2_all,
        "entropy": entropy_all,
        "margin": margin_all,
    }


def build_round_probe_pool(
    *,
    cfg: Dict[str, Any],
    cfg_point: Dict[str, Any],
    state: PointillismFLState,
    rnd: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out_hw = _out_hw(cfg)
    grid_hws = _grid_hws(cfg_point, out_hw=out_hw)
    random_budget = int(cfg_point.get("random_budget_per_grid", 96))
    seed0 = int(cfg.get("seed", 0)) + 1000003 * int(rnd)

    use_memory = bool(cfg_point.get("use_memory_bank", True))
    color_inertia = _clamp01(float(cfg_point.get("color_inertia", 0.0)))
    reuse_random_pool_frac = _clamp01(float(cfg_point.get("reuse_random_pool_frac", 0.0)))
    construction_backend = str(
        cfg_point.get("pool_construction_backend", "legacy")
    ).strip().lower()
    if construction_backend not in {"legacy", "cpu_vectorized"}:
        raise ValueError(
            "pointillism_fl.pool_construction_backend must be one of "
            f"['cpu_vectorized', 'legacy'], got {construction_backend!r}"
        )
    use_vectorized_cpu = construction_backend == "cpu_vectorized"
    mask_batch_size = max(
        1, int(cfg_point.get("pool_construction_mask_batch_size", 2048))
    )
    probe_mode = _resolved_two_group_mode(cfg, cfg_point=cfg_point)
    color_ids = _point_palette_color_ids(cfg, cfg_point=cfg_point)
    two_group_cfg = _point_two_group_cfg(cfg, cfg_point=cfg_point)
    scalar_palette = [float(x) for x in cfg_point.get("palette_values", [])]
    gaussian_cfg = _point_gaussian_cfg(cfg, cfg_point=cfg_point)

    specs: List[Dict[str, Any]] = []
    sig_seen: Set[Tuple[Any, ...]] = set()
    next_random_pool: List[Dict[str, Any]] = []
    track_random_pool = float(reuse_random_pool_frac) > 0.0

    def _random_reuse_candidates(*, grid_hw: int, probe_mode_key: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for spec in list(state.random_pool_fresh):
            if int(spec.get("grid_hw", -1)) != int(grid_hw):
                continue
            if str(spec.get("probe_mode", "binary")) != str(probe_mode_key):
                continue
            out.append(dict(spec))
        return out

    def _add(spec: Dict[str, Any]) -> None:
        sig = _probe_signature(spec)
        if sig in sig_seen:
            return
        sig_seen.add(sig)
        specs.append(spec)
        if track_random_pool and str(spec.get("source", "")) == "random":
            next_random_pool.append(_strip_probe_runtime_fields(spec))

    for grid_hw in grid_hws:
        grid_h, grid_w = _grid_shape(cfg_point, int(grid_hw))
        n_cells = int(grid_h) * int(grid_w)
        reuse_candidates = _random_reuse_candidates(grid_hw=int(grid_hw), probe_mode_key=str(probe_mode))
        reuse_n = min(int(len(reuse_candidates)), int(round(float(random_budget) * float(reuse_random_pool_frac))))
        if reuse_n > 0:
            reuse_rng = random.Random(int(seed0) + int(grid_hw) * 65537 + 17)
            for spec in reuse_rng.sample(reuse_candidates, int(reuse_n)):
                _add({**dict(spec), "source": "random_reuse"})

        if str(probe_mode).startswith("gaussian"):
            num_channels = 1 if str(probe_mode) == "gaussian_gray" else 3
            fresh_budget = max(0, int(random_budget) - int(reuse_n))
            for sample_idx in range(fresh_budget):
                render_seed = (
                    int(seed0)
                    + int(grid_hw) * 10007
                    + int(sample_idx) * 1000003
                ) % 2147483647
                _add({
                    "grid_hw": int(grid_hw),
                    "flat_idx": [],
                    "w": 0,
                    "pct": -1.0,
                    "polarity": "white",
                    "source": "random",
                    "probe_mode": str(probe_mode),
                    "num_channels": int(num_channels),
                    "render_seed": int(render_seed),
                    "sigma": float(gaussian_cfg["sigma"]),
                    "mean": float(gaussian_cfg["mean"]),
                })
            continue

        w_cfg = dict(cfg_point.get("w_schedule", {}) or {})
        if int(grid_h) != int(grid_w):
            w_list = _w_list_for_cells(n_cells=int(n_cells), w_cfg=w_cfg)
        else:
            w_list = build_w_list_for_grid(grid_hw=int(grid_hw), w_cfg=w_cfg)
        if not w_list:
            continue
        fresh_budget = max(0, int(random_budget) - int(reuse_n))
        use_complement = bool(
            (cfg_point.get("w_schedule", {}) or {}).get("use_complement", False)
        )
        sampler_kwargs = {
            "grid_hw": int(grid_hw),
            "w_list": [int(x) for x in w_list],
            "n_trials_per_w": max(
                1,
                int(
                    math.ceil(
                        float(fresh_budget) / float(max(1, len(w_list)))
                    )
                ),
            ),
            "seed": int(seed0 + int(grid_hw) * 10007),
            "max_records": int(fresh_budget),
        }
        if use_vectorized_cpu and not use_complement:
            recs = sample_random_masks_for_d_batched_cpu(
                **sampler_kwargs,
                batch_size=int(mask_batch_size),
            )
        else:
            recs = sample_random_masks_for_d(
                **sampler_kwargs,
                use_complement=bool(use_complement),
            )
        color_memory_pool: List[Dict[str, Any]] = []
        if str(probe_mode).startswith("two_group") and float(color_inertia) > 0.0:
            num_channels = 1 if str(probe_mode) == "two_group_gray" else 3
            color_memory_pool = _flatten_color_memory_bank(
                color_bank=state.color_bank,
                num_channels=int(num_channels),
            )
        if (
            use_vectorized_cpu
            and str(probe_mode).startswith("two_group")
            and not color_memory_pool
        ):
            variants_per_mask = max(1, int(two_group_cfg["variants_per_mask"]))
            num_channels = 1 if str(probe_mode) == "two_group_gray" else 3
            d_max = float(math.sqrt(float(num_channels)))
            pending_specs: List[Dict[str, Any]] = []
            pending_seeds: List[int] = []
            for rec in recs:
                base_spec = {
                    "grid_hw": int(rec["grid_hw"]),
                    "flat_idx": [int(x) for x in rec["flat_idx"]],
                    "w": int(rec["w"]),
                    "pct": -1.0,
                    "polarity": str(rec.get("polarity", "white")).lower(),
                    "source": "random",
                }
                mask_seed_raw = rec.get("mask_seed")
                mask_seed = (
                    _stable_mask_seed(base_spec["flat_idx"])
                    if mask_seed_raw is None
                    else int(mask_seed_raw)
                )
                for variant_idx in range(variants_per_mask):
                    render_seed = (
                        int(seed0)
                        + int(grid_hw) * 10007
                        + int(mask_seed) * 97
                        + int(variant_idx) * 1000003
                    ) % 2147483647
                    pending_specs.append(base_spec)
                    pending_seeds.append(int(render_seed))

            mu_fg_batch, mu_bg_batch = sample_two_group_color_centers_batch_cpu(
                seeds=pending_seeds,
                d_min=float(two_group_cfg["d_min"]),
                d_max=float(d_max),
                num_channels=int(num_channels),
            )
            for row_idx, (base_spec, render_seed) in enumerate(
                zip(pending_specs, pending_seeds)
            ):
                _add({
                    **base_spec,
                    "probe_mode": str(probe_mode),
                    "num_channels": int(num_channels),
                    "render_seed": int(render_seed),
                    "cell_size": int(two_group_cfg["cell_size"]),
                    "sigma_fg": float(two_group_cfg["sigma_fg"]),
                    "sigma_bg": float(two_group_cfg["sigma_bg"]),
                    "mu_fg": mu_fg_batch[int(row_idx)].tolist(),
                    "mu_bg": mu_bg_batch[int(row_idx)].tolist(),
                    "value_range": [float(x) for x in two_group_cfg["value_range"]],
                    "color_inertia": float(color_inertia),
                    "color_source": "random",
                })
            continue

        for rec in recs:
            base_spec = {
                "grid_hw": int(rec["grid_hw"]),
                "flat_idx": [int(x) for x in rec["flat_idx"]],
                "w": int(rec["w"]),
                "pct": -1.0,
                "polarity": str(rec.get("polarity", "white")).lower(),
                "source": "random",
            }
            if str(probe_mode).startswith("two_group"):
                variants_per_mask = max(1, int(two_group_cfg["variants_per_mask"]))
                mask_seed_raw = rec.get("mask_seed")
                mask_seed = (
                    _stable_mask_seed(base_spec["flat_idx"])
                    if mask_seed_raw is None
                    else int(mask_seed_raw)
                )
                num_channels = 1 if str(probe_mode) == "two_group_gray" else 3
                d_max = float(math.sqrt(float(num_channels)))
                for variant_idx in range(variants_per_mask):
                    render_seed = (
                        int(seed0)
                        + int(grid_hw) * 10007
                        + int(mask_seed) * 97
                        + int(variant_idx) * 1000003
                    ) % 2147483647
                    use_color_memory = False
                    color_rng = None
                    if color_memory_pool and float(color_inertia) > 0.0:
                        color_rng = random.Random(int(render_seed))
                        use_color_memory = color_rng.random() < float(color_inertia)
                    if use_color_memory:
                        assert color_rng is not None
                        color_rec = color_memory_pool[int(color_rng.randrange(len(color_memory_pool)))]
                        mu_fg = [float(x) for x in color_rec.get("mu_fg", [])[: int(num_channels)]]
                        mu_bg = [float(x) for x in color_rec.get("mu_bg", [])[: int(num_channels)]]
                    else:
                        if scalar_palette and int(num_channels) == 1:
                            palette_rng = random.Random(int(render_seed))
                            mu_fg = [float(scalar_palette[palette_rng.randrange(len(scalar_palette))])]
                            remaining = [v for v in scalar_palette if float(v) != float(mu_fg[0])]
                            mu_bg = [float(remaining[palette_rng.randrange(len(remaining))] if remaining else mu_fg[0])]
                        else:
                            mu_fg, mu_bg = sample_two_group_color_centers(
                                seed=int(render_seed),
                                d_min=float(two_group_cfg["d_min"]),
                                d_max=float(d_max),
                                num_channels=int(num_channels),
                                center_range=(0.0, 1.0),
                            )
                    _add({
                        **base_spec,
                        "probe_mode": str(probe_mode),
                        "num_channels": int(num_channels),
                        "render_seed": int(render_seed),
                        "cell_size": int(two_group_cfg["cell_size"]),
                        "sigma_fg": float(two_group_cfg["sigma_fg"]),
                        "sigma_bg": float(two_group_cfg["sigma_bg"]),
                        "mu_fg": [float(x) for x in mu_fg],
                        "mu_bg": [float(x) for x in mu_bg],
                        "value_range": ([float(min(scalar_palette)), float(max(scalar_palette))] if scalar_palette else [float(x) for x in two_group_cfg["value_range"]]),
                        "color_inertia": float(color_inertia),
                        "color_source": "memory" if use_color_memory else "random",
                    })
            else:
                for color_id in color_ids:
                    _add({
                        **base_spec,
                        "probe_mode": "binary",
                        "color_id": int(color_id),
                    })

    if use_memory:
        for label in sorted(state.topk_bank.keys()):
            for spec in list(state.topk_bank.get(int(label), [])):
                _add({**_strip_probe_runtime_fields(spec), "source": "bank_topk"})
        for pair_name in sorted(state.edge_bank.keys()):
            for spec in list(state.edge_bank.get(str(pair_name), [])):
                _add({**_strip_probe_runtime_fields(spec), "source": "bank_edge"})

    for pid, spec in enumerate(specs):
        spec["probe_id"] = int(pid)

    state.random_pool_fresh = list(next_random_pool)

    meta = {
        "construction_backend": str(construction_backend),
        "num_random": int(sum(1 for s in specs if str(s.get("source")) == "random")),
        "num_random_reuse": int(sum(1 for s in specs if str(s.get("source")) == "random_reuse")),
        "num_bank_topk": int(sum(1 for s in specs if str(s.get("source")) == "bank_topk")),
        "num_bank_edge": int(sum(1 for s in specs if str(s.get("source")) == "bank_edge")),
        "num_total": int(len(specs)),
    }
    return specs, meta


def _min_max_w_for_grid(
    *,
    cfg_point: Dict[str, Any],
    grid_hw: int,
) -> Tuple[int, int]:
    w_list = build_w_list_for_grid(grid_hw=int(grid_hw), w_cfg=dict(cfg_point.get("w_schedule", {}) or {}))
    if not w_list:
        n_cells = int(grid_hw) * int(grid_hw)
        return 1, max(1, n_cells - 1)
    return int(min(w_list)), int(max(w_list))


def collect_refinement_specs(
    *,
    base_specs: List[Dict[str, Any]],
    base_scores: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    cfg_point: Dict[str, Any],
    rnd: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    refine_cfg = dict(cfg_point.get("refine", {}) or {})
    if not bool(refine_cfg.get("enabled", True)):
        return [], {"enabled": False}
    if str(_point_probe_mode(cfg, cfg_point=cfg_point)).startswith("gaussian"):
        return [], {"enabled": False, "reason": "gaussian_probe_mode_no_refine"}

    probs = base_scores["probs"]
    pred1 = base_scores["pred1"]
    num_clients = int(probs.shape[0])
    num_classes = int(probs.shape[2]) if probs.ndim == 3 else 0

    top_m = int(refine_cfg.get("candidate_top_m", 16))
    top_k = int(cfg_point.get("top_k", 8))
    max_labels_per_client = int(refine_cfg.get("max_labels_per_client", 2))
    seeds_per_label = int(refine_cfg.get("seeds_per_label", 2))
    neighbors_per_seed = int(refine_cfg.get("neighbors_per_seed", 6))
    flip_cells = int(refine_cfg.get("flip_cells", 1))
    swap_cells = int(refine_cfg.get("swap_cells", 2))
    max_edit_steps = int(refine_cfg.get("max_edit_steps", 1))

    all_specs: List[Dict[str, Any]] = []
    all_sigs: Set[Tuple[Any, ...]] = set()
    refined_labels = 0

    rng_seed = int(cfg.get("seed", 0)) + 7919 * int(rnd)

    for client_idx in range(num_clients):
        deficits: List[Tuple[int, int]] = []
        for label in range(num_classes):
            cand = [int(idx) for idx in _top_indices_desc(probs[client_idx, :, label], max(1, top_m))]
            hit_count = sum(1 for idx in cand if int(pred1[client_idx, idx]) == int(label))
            if int(hit_count) < int(top_k):
                deficits.append((int(top_k - hit_count), int(label)))

        deficits.sort(key=lambda x: (-int(x[0]), int(x[1])))
        deficits = deficits[: max(0, int(max_labels_per_client))]

        for _, label in deficits:
            refined_labels += 1
            seed_indices = [
                int(idx)
                for idx in _top_indices_desc(probs[client_idx, :, label], max(1, top_m))[: max(1, seeds_per_label)]
            ]

            for local_seed_idx, pool_idx in enumerate(seed_indices):
                seed_spec = base_specs[int(pool_idx)]
                grid_hw = int(seed_spec["grid_hw"])
                grid_h = int(seed_spec.get("grid_h", grid_hw))
                grid_w = int(seed_spec.get("grid_w", grid_hw))
                n_cells = int(grid_h) * int(grid_w)
                if int(grid_h) != int(grid_w):
                    ref_ws = _w_list_for_cells(n_cells=int(n_cells), w_cfg=dict(cfg_point.get("w_schedule", {}) or {}))
                    min_w, max_w = (min(ref_ws), max(ref_ws)) if ref_ws else (1, n_cells - 1)
                else:
                    min_w, max_w = _min_max_w_for_grid(cfg_point=cfg_point, grid_hw=int(grid_hw))

                masks_curr = [[int(x) for x in seed_spec.get("flat_idx", [])]]
                for edit_step in range(max(1, max_edit_steps)):
                    next_masks: List[List[int]] = []
                    for mask_idx, mask in enumerate(masks_curr):
                        rng = random.Random(
                            int(rng_seed)
                            + int(client_idx) * 100003
                            + int(label) * 1009
                            + int(local_seed_idx) * 17
                            + int(edit_step) * 10000019
                            + int(mask_idx)
                        )
                        neigh = _neighbor_masks_for_local_refine(
                            flatidx=[int(x) for x in mask],
                            n_cells=int(n_cells),
                            rng=rng,
                            min_w=int(min_w),
                            max_w=int(max_w),
                            max_add_per_seed=max(0, int(flip_cells)),
                            max_remove_per_seed=max(0, int(flip_cells)),
                            max_swap_per_seed=max(0, int(swap_cells)),
                        )
                        next_masks.extend(neigh[: max(0, int(neighbors_per_seed))])
                    masks_curr = next_masks[: max(0, int(neighbors_per_seed))]

                for flatidx in masks_curr:
                    spec = {
                        "grid_hw": int(grid_hw),
                        "grid_h": int(grid_h),
                        "grid_w": int(grid_w),
                        "flat_idx": [int(x) for x in flatidx],
                        "w": int(len(flatidx)),
                        "pct": -1.0,
                        "polarity": str(seed_spec.get("polarity", "white")).lower(),
                        "source": "refine",
                        "probe_mode": str(seed_spec.get("probe_mode", "binary")),
                        "color_id": int(seed_spec.get("color_id", 0)),
                        "render_seed": int(seed_spec.get("render_seed", 0)),
                    }
                    if str(spec["probe_mode"]).startswith(("two_group", "gaussian")):
                        spec["num_channels"] = int(seed_spec.get("num_channels", 3))
                        spec["cell_size"] = int(seed_spec.get("cell_size", 4))
                        spec["sigma_fg"] = float(seed_spec.get("sigma_fg", 0.1))
                        spec["sigma_bg"] = float(seed_spec.get("sigma_bg", 0.1))
                        spec["mu_fg"] = [float(x) for x in seed_spec.get("mu_fg", [1.0, 1.0, 1.0])]
                        spec["mu_bg"] = [float(x) for x in seed_spec.get("mu_bg", [0.0, 0.0, 0.0])]
                        spec["value_range"] = [float(x) for x in seed_spec.get("value_range", [0.0, 1.0])]
                    sig = _probe_signature(spec)
                    if sig in all_sigs:
                        continue
                    all_sigs.add(sig)
                    all_specs.append(spec)

    for pid, spec in enumerate(all_specs):
        spec["probe_id"] = int(pid)

    return all_specs, {
        "enabled": True,
        "num_specs": int(len(all_specs)),
        "num_refined_labels": int(refined_labels),
    }


def merge_probe_pools(
    *,
    specs_a: List[Dict[str, Any]],
    scores_a: Dict[str, np.ndarray],
    specs_b: List[Dict[str, Any]],
    scores_b: Dict[str, np.ndarray],
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    if not specs_b:
        return specs_a, scores_a

    merged_specs = list(specs_a) + list(specs_b)
    merged_scores = {
        "probs": np.concatenate([scores_a["probs"], scores_b["probs"]], axis=1),
        "pred1": np.concatenate([scores_a["pred1"], scores_b["pred1"]], axis=1),
        "pred2": np.concatenate([scores_a["pred2"], scores_b["pred2"]], axis=1),
        "entropy": np.concatenate([scores_a["entropy"], scores_b["entropy"]], axis=1),
        "margin": np.concatenate([scores_a["margin"], scores_b["margin"]], axis=1),
    }
    return merged_specs, merged_scores


def compute_q1_summary_for_client(
    *,
    client_idx: int,
    specs: List[Dict[str, Any]],
    scores: Dict[str, np.ndarray],
    num_classes: int,
    tau_r: Optional[float],
) -> Dict[str, Any]:
    probs = scores["probs"][client_idx]
    pred1 = scores["pred1"][client_idx]
    entropy = scores["entropy"][client_idx]
    log_c = math.log(float(max(2, num_classes)))

    by_grid: Dict[int, Dict[str, Any]] = {}
    evidence_max = [0.0 for _ in range(int(num_classes))]

    grid_hws = sorted({int(spec["grid_hw"]) for spec in specs})
    for d in grid_hws:
        idxs = [idx for idx, spec in enumerate(specs) if int(spec["grid_hw"]) == int(d)]
        if not idxs:
            continue

        q = [0.0 for _ in range(num_classes)]
        u = [0.0 for _ in range(num_classes)]
        E = [0.0 for _ in range(num_classes)]
        H = [float(log_c) for _ in range(num_classes)]
        counts = [0 for _ in range(num_classes)]

        pred_d = pred1[idxs]
        ent_d = entropy[idxs]

        for label in range(num_classes):
            mask = (pred_d == int(label))
            q[label] = float(mask.mean()) if mask.size > 0 else 0.0
            count = int(mask.sum())
            counts[label] = int(count)
            if count > 0:
                H[label] = float(ent_d[mask].mean())
                u[label] = max(0.0, 1.0 - (float(H[label]) / float(log_c)))
            else:
                H[label] = float(log_c)
                u[label] = 0.0
            E[label] = float(q[label] * u[label])
            evidence_max[label] = max(float(evidence_max[label]), float(E[label]))

        by_grid[int(d)] = {
            "q": [float(x) for x in q],
            "u": [float(x) for x in u],
            "E": [float(x) for x in E],
            "entropy_mean": [float(x) for x in H],
            "counts": [int(x) for x in counts],
            "sum_E_d": float(sum(E)),
        }

    model_reliability, model_reliability_by_grid = _model_reliability_from_by_grid(
        by_grid,
        num_classes=int(num_classes),
    )
    r_f = float(model_reliability)
    support = [int(label) for label, val in enumerate(evidence_max) if float(val) > 0.0]
    is_disc = None if tau_r is None else bool(r_f >= float(tau_r))
    effective = list(support) if tau_r is None or bool(is_disc) else []
    return {
        "R_f": float(r_f),
        "support_labels": [int(x) for x in support],
        "effective_support_labels": [int(x) for x in effective],
        "tau_R": (None if tau_r is None else float(tau_r)),
        "is_discriminative": is_disc,
        "evidence_by_grid": by_grid,
        "model_reliability": float(model_reliability),
        "model_reliability_by_grid": model_reliability_by_grid,
        "class_evidence_max": [float(x) for x in evidence_max],
        # Legacy field retained for older analysis scripts; it is no longer used as the A-table weight.
        "evidence_by_label_max": [float(x) for x in evidence_max],
    }


def build_client_signature(
    *,
    client_idx: int,
    specs: List[Dict[str, Any]],
    scores: Dict[str, np.ndarray],
    num_classes: int,
    cfg_point: Dict[str, Any],
) -> Dict[str, Any]:
    probs = scores["probs"][client_idx]
    pred1 = scores["pred1"][client_idx]
    pred2 = scores["pred2"][client_idx]
    margin = scores["margin"][client_idx]

    top_m = int(cfg_point.get("candidate_top_m", 16))
    top_k = int(cfg_point.get("top_k", 8))
    edge_top_k = int(cfg_point.get("edge_top_k", 4))

    topk_by_label: Dict[int, List[int]] = {}
    for label in range(num_classes):
        order = np.argsort(-probs[:, label])
        cand = [int(idx) for idx in order[: max(1, top_m)]]
        hits = [
            int(idx)
            for idx in cand
            if int(pred1[idx]) == int(label) and float(probs[int(idx), int(label)]) > 0.0
        ]
        topk_by_label[int(label)] = hits[: max(0, int(top_k))]

    edge_by_pair: Dict[str, List[int]] = {}
    for idx in range(len(specs)):
        if pred2.shape[1] < 2:
            continue
        p = int(pred2[idx, 0])
        q = int(pred2[idx, 1])
        if int(p) == int(q):
            continue
        if float(probs[int(idx), int(p)]) <= 0.0 or float(probs[int(idx), int(q)]) <= 0.0:
            continue
        edge_by_pair.setdefault(_pair_key(p, q), []).append(int(idx))

    edge_ranked: Dict[str, List[int]] = {}
    for key, idxs in edge_by_pair.items():
        idxs_sorted = sorted(
            [int(x) for x in idxs],
            key=lambda x: (float(margin[x]), int(x)),
        )
        edge_ranked[str(key)] = idxs_sorted[: max(0, int(edge_top_k))]

    return {
        "topk_by_label": topk_by_label,
        "edge_by_pair": edge_ranked,
    }


def compute_pairwise_consensus(
    *,
    client_infos: List[Dict[str, Any]],
    scores: Dict[str, np.ndarray],
    num_classes: int,
    cfg_point: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    k = len(client_infos)
    A = np.zeros((k, k), dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    label_agreement_mode = _pairwise_label_agreement_mode(cfg_point)
    label_k_cfg = _pairwise_label_k_config(cfg_point)
    selection_table = _selection_table_mode(cfg_point)

    def _build_pair_label_rows(i: int, j: int, shared: Sequence[int]) -> List[Tuple[int, float, float, float, float, float]]:
        sig_i = client_infos[int(i)]["signature"]
        sig_j = client_infos[int(j)]["signature"]
        rel_i = _model_reliability_value(client_infos[int(i)].get("q1", {}), num_classes=int(num_classes))
        rel_j = _model_reliability_value(client_infos[int(j)].get("q1", {}), num_classes=int(num_classes))
        r_pair = math.sqrt(max(0.0, float(rel_i)) * max(0.0, float(rel_j)))
        label_rows_local: List[Tuple[int, float, float, float, float, float]] = []

        for label in shared:
            idxs_i = list(sig_i["topk_by_label"].get(int(label), []))
            idxs_j = list(sig_j["topk_by_label"].get(int(label), []))
            t_i_to_j = float(scores["probs"][j, idxs_i, int(label)].mean()) if idxs_i else 0.0
            t_j_to_i = float(scores["probs"][i, idxs_j, int(label)].mean()) if idxs_j else 0.0
            label_t = 0.5 * (float(t_i_to_j) + float(t_j_to_i))

            p_i_vals: List[float] = []
            p_j_vals: List[float] = []
            for q in shared:
                if int(q) == int(label):
                    continue
                pair_name = _pair_key(int(label), int(q))

                edge_i = list(sig_i["edge_by_pair"].get(pair_name, []))
                if edge_i:
                    probs_p_i = scores["probs"][j, edge_i, int(label)]
                    probs_q_i = scores["probs"][j, edge_i, int(q)]
                    acc_i = float(np.mean(np.minimum(probs_p_i, probs_q_i)))
                    p_i_vals.append(float(acc_i))

                edge_j = list(sig_j["edge_by_pair"].get(pair_name, []))
                if edge_j:
                    probs_p_j = scores["probs"][i, edge_j, int(label)]
                    probs_q_j = scores["probs"][i, edge_j, int(q)]
                    acc_j = float(np.mean(np.minimum(probs_p_j, probs_q_j)))
                    p_j_vals.append(float(acc_j))

            # Old class-level edge aggregation:
            # p_i = _median_or_zero(p_i_vals)
            # p_j = _median_or_zero(p_j_vals)
            p_i = _safe_mean(p_i_vals)
            p_j = _safe_mean(p_j_vals)
            label_p = 0.5 * (float(p_i) + float(p_j))
            label_a_raw = _combine_bar_a(bar_t=label_t, bar_p=label_p, cfg_point=cfg_point)
            label_a = float(r_pair) * float(label_a_raw)
            label_rows_local.append((
                int(label),
                float(label_t),
                float(label_p),
                float(label_a),
                float(r_pair),
                float(label_a_raw),
            ))

        return label_rows_local

    global_label_values: List[List[float]] = [[] for _ in range(int(num_classes))]
    for i in range(k):
        eff_i = set(int(x) for x in client_infos[i]["q1"].get("effective_support_labels", []))
        for j in range(i + 1, k):
            eff_j = set(int(x) for x in client_infos[j]["q1"].get("effective_support_labels", []))
            shared = sorted(eff_i & eff_j)
            for label, _label_t, _label_p, label_a, _r_pair, _label_a_raw in _build_pair_label_rows(i, j, shared):
                if 0 <= int(label) < int(num_classes):
                    global_label_values[int(label)].append(float(label_a))

    if label_agreement_mode == "majority_supported":
        global_selected_labels, global_selected_rows = _majority_supported_label_weights(
            label_values=global_label_values,
            num_clients=int(k),
            num_classes=int(num_classes),
        )
    else:
        global_selected_labels, global_selected_rows = _select_global_agreement_labels(
            label_values=global_label_values,
            mode=label_agreement_mode,
            k_cfg=label_k_cfg,
            num_classes=int(num_classes),
        )
    global_selected_label_set = set(int(label) for label in global_selected_labels)
    global_label_weights = {int(label): float(val) for label, val in global_selected_rows}
    global_selected_labels_csv = ",".join(str(int(label)) for label in global_selected_labels)
    global_selected_values_csv = ",".join(f"{float(g_val):.6g}" for _label, g_val in global_selected_rows)

    for i in range(k):
        sig_i = client_infos[i]["signature"]
        q1_i = client_infos[i]["q1"]
        rel_i = _model_reliability_value(q1_i, num_classes=int(num_classes))
        eff_i = sorted(set(int(x) for x in q1_i.get("effective_support_labels", [])))
        self_label_rows: List[Tuple[int, float, float, float, float, float]] = []
        for label in eff_i:
            idxs_i = list(sig_i["topk_by_label"].get(int(label), []))
            self_t = float(scores["probs"][i, idxs_i, int(label)].mean()) if idxs_i else 0.0
            self_p = 1.0
            self_a_raw = _combine_bar_a(bar_t=self_t, bar_p=self_p, cfg_point=cfg_point)
            self_r = float(rel_i)
            self_a = float(self_r) * float(self_a_raw)
            self_label_rows.append((
                int(label),
                float(self_t),
                float(self_p),
                float(self_a),
                float(self_r),
                float(self_a_raw),
            ))

        self_bar_t, self_bar_p, self_bar_a, self_bar_r, self_bar_a_raw, self_selected_rows, self_label_k = _reduce_label_agreement_rows(
            self_label_rows,
            mode=label_agreement_mode,
            k_cfg=label_k_cfg,
            num_classes=int(num_classes),
            global_selected_labels=global_selected_label_set,
            global_label_weights=global_label_weights,
        )
        self_selection_score = _selection_score_from_tables(
            selection_table=selection_table,
            bar_t=float(self_bar_t),
            bar_p=float(self_bar_p),
            bar_r=float(self_bar_r),
            bar_a=float(self_bar_a),
        )
        A[i, i] = float(self_selection_score)
        rows.append({
            "cid_a": int(client_infos[i]["cid"]),
            "cid_b": int(client_infos[i]["cid"]),
            "selection_table": str(selection_table),
            "selection_score": float(self_selection_score),
            "shared_support_size": int(len(eff_i)),
            "shared_support_labels": ",".join(str(x) for x in eff_i),
            "pairwise_label_agreement": str(label_agreement_mode),
            "pairwise_label_k": float(label_k_cfg),
            "pairwise_label_k_resolved": int(self_label_k),
            "global_selected_labels": str(global_selected_labels_csv),
            "global_selected_values": str(global_selected_values_csv),
            "selected_agreement_labels": ",".join(str(row[0]) for row in self_selected_rows),
            "selected_agreement_values": ",".join(f"{float(row[3]):.6g}" for row in self_selected_rows),
            "selected_agreement_raw_values": ",".join(f"{float(row[5]):.6g}" for row in self_selected_rows),
            "selected_reliability_weights": ",".join(f"{float(row[4]):.6g}" for row in self_selected_rows),
            "selected_reliability_zero_count": int(sum(1 for row in self_selected_rows if float(row[4]) <= 1e-12)),
            "bar_T": float(self_bar_t),
            "bar_P": float(self_bar_p),
            "bar_R": float(self_bar_r),
            "bar_A_raw": float(self_bar_a_raw),
            "bar_A": float(self_bar_a),
        })

    for i in range(k):
        sig_i = client_infos[i]["signature"]
        q1_i = client_infos[i]["q1"]
        eff_i = set(int(x) for x in q1_i.get("effective_support_labels", []))

        for j in range(i + 1, k):
            sig_j = client_infos[j]["signature"]
            q1_j = client_infos[j]["q1"]
            eff_j = set(int(x) for x in q1_j.get("effective_support_labels", []))
            rel_i = _model_reliability_value(q1_i, num_classes=int(num_classes))
            rel_j = _model_reliability_value(q1_j, num_classes=int(num_classes))
            pair_r = math.sqrt(max(0.0, float(rel_i)) * max(0.0, float(rel_j)))

            shared = sorted(eff_i & eff_j)
            if not shared:
                selection_score = _selection_score_from_tables(
                    selection_table=selection_table,
                    bar_t=0.0,
                    bar_p=0.0,
                    bar_r=float(pair_r),
                    bar_a=0.0,
                )
                A[i, j] = float(selection_score)
                A[j, i] = float(selection_score)
                row = {
                    "cid_a": int(client_infos[i]["cid"]),
                    "cid_b": int(client_infos[j]["cid"]),
                    "selection_table": str(selection_table),
                    "selection_score": float(selection_score),
                    "shared_support_size": 0,
                    "shared_support_labels": "",
                    "pairwise_label_agreement": str(label_agreement_mode),
                    "pairwise_label_k": float(label_k_cfg),
                    "pairwise_label_k_resolved": 0,
                    "global_selected_labels": str(global_selected_labels_csv),
                    "global_selected_values": str(global_selected_values_csv),
                    "selected_agreement_labels": "",
                    "selected_agreement_values": "",
                    "selected_agreement_raw_values": "",
                    "selected_reliability_weights": "",
                    "selected_reliability_zero_count": 0,
                    "bar_T": 0.0,
                    "bar_P": 0.0,
                    "bar_R": float(pair_r),
                    "bar_A_raw": 0.0,
                    "bar_A": 0.0,
                }
                rows.append(row)
                continue

            label_rows = _build_pair_label_rows(i, j, shared)

            bar_t, bar_p, bar_a, bar_r, bar_a_raw, selected_rows, label_k = _reduce_label_agreement_rows(
                label_rows,
                mode=label_agreement_mode,
                k_cfg=label_k_cfg,
                num_classes=int(num_classes),
                global_selected_labels=global_selected_label_set,
                global_label_weights=global_label_weights,
            )

            selection_score = _selection_score_from_tables(
                selection_table=selection_table,
                bar_t=float(bar_t),
                bar_p=float(bar_p),
                bar_r=float(bar_r),
                bar_a=float(bar_a),
            )
            A[i, j] = float(selection_score)
            A[j, i] = float(selection_score)

            rows.append({
                "cid_a": int(client_infos[i]["cid"]),
                "cid_b": int(client_infos[j]["cid"]),
                "selection_table": str(selection_table),
                "selection_score": float(selection_score),
                "shared_support_size": int(len(shared)),
                "shared_support_labels": ",".join(str(x) for x in shared),
                "pairwise_label_agreement": str(label_agreement_mode),
                "pairwise_label_k": float(label_k_cfg),
                "pairwise_label_k_resolved": int(label_k),
                "global_selected_labels": str(global_selected_labels_csv),
                "global_selected_values": str(global_selected_values_csv),
                "selected_agreement_labels": ",".join(str(row[0]) for row in selected_rows),
                "selected_agreement_values": ",".join(f"{float(row[3]):.6g}" for row in selected_rows),
                "selected_agreement_raw_values": ",".join(f"{float(row[5]):.6g}" for row in selected_rows),
                "selected_reliability_weights": ",".join(f"{float(row[4]):.6g}" for row in selected_rows),
                "selected_reliability_zero_count": int(sum(1 for row in selected_rows if float(row[4]) <= 1e-12)),
                "bar_T": float(bar_t),
                "bar_P": float(bar_p),
                "bar_R": float(bar_r),
                "bar_A_raw": float(bar_a_raw),
                "bar_A": float(bar_a),
            })

    return A, rows


def select_client_subset(
    *,
    client_infos: List[Dict[str, Any]],
    A: np.ndarray,
    num_classes: int,
    cfg_point: Dict[str, Any],
    rng_seed: Optional[int] = None,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    k = int(len(client_infos))
    if k <= 1:
        return list(range(k)), [], {
            "mode": "singleton",
            "subset_rows": [],
            "selected_score": 0.0,
        }

    rng = random.Random(0 if rng_seed is None else int(rng_seed))
    shuffled_order = list(range(k))
    rng.shuffle(shuffled_order)
    order_rank = {int(idx): pos for pos, idx in enumerate(shuffled_order)}

    min_keep = int(cfg_point.get("min_keep", 0))
    if min_keep <= 0:
        min_keep = max(1, int(math.ceil(float(k) * float(cfg_point.get("min_keep_frac", 0.5)))))
    min_keep = max(1, min(int(min_keep), int(k)))

    selection_mode = str(cfg_point.get("selection_mode", "Pareto")).strip().lower()
    selection_mode = selection_mode.replace("-", "_")
    if selection_mode == "greedyconsensus":
        selection_mode = "greedy_consensus"
    consensus_objective = str(cfg_point.get("consensus_objective", "mean")).strip().lower()
    if consensus_objective in {"average", "avg"}:
        consensus_objective = "mean"
    if consensus_objective in {"low-quantile", "quantile", "lower_quantile"}:
        consensus_objective = "low_quantile"
    if consensus_objective not in {"mean", "median", "low_quantile"}:
        raise ValueError(
            f"Unknown pointillism_fl.consensus_objective='{cfg_point.get('consensus_objective')}'. "
            "Expected 'mean', 'median', or 'low_quantile'."
        )
    consensus_quantile_alpha = float(cfg_point.get("consensus_quantile_alpha", 0.25))
    if consensus_objective == "low_quantile" and not (0.0 < float(consensus_quantile_alpha) <= 0.5):
        raise ValueError(
            f"pointillism_fl.consensus_quantile_alpha must be in (0, 0.5], got {consensus_quantile_alpha}"
        )

    def _consensus_score(pair_vals: List[float]) -> float:
        if not pair_vals:
            return 0.0
        if consensus_objective == "median":
            return _median_or_zero(pair_vals)
        if consensus_objective == "low_quantile":
            return float(np.quantile(np.asarray(pair_vals, dtype=np.float32), float(consensus_quantile_alpha)))
        return _safe_mean(pair_vals)

    candidate_subsets: List[List[int]]
    mode_name: str
    if selection_mode == "pareto":
        current = list(shuffled_order)
        candidate_subsets = [list(current)]

        while len(current) > int(min_keep):
            g_vals: List[Tuple[float, int]] = []
            for idx in current:
                others = [j for j in current if int(j) != int(idx)]
                g_i = _safe_mean([float(A[int(idx), int(j)]) for j in others]) if others else 0.0
                g_vals.append((float(g_i), int(idx)))
            g_vals.sort(key=lambda x: (float(x[0]), int(order_rank[int(x[1])])))
            remove_idx = int(g_vals[0][1])
            current = [idx for idx in current if int(idx) != int(remove_idx)]
            candidate_subsets.append(list(current))
        mode_name = "pareto_peeling"
    elif selection_mode == "consensus":
        target_keep = max(int(min_keep), int(k // 2 + 1))
        target_keep = max(1, min(int(target_keep), int(k)))
        candidate_subsets = [list(combo) for combo in itertools.combinations(shuffled_order, int(target_keep))]
        mode_name = "consensus_median"
    elif selection_mode == "greedy_consensus":
        target_keep = max(int(min_keep), int(k // 2 + 1))
        target_keep = max(1, min(int(target_keep), int(k)))
        current = list(shuffled_order)

        # For the mean objective, incrementally maintained row sums make the
        # complete peeling path O(k^2). Median and low-quantile objectives use
        # the same deterministic rule with direct row reductions and remain
        # polynomial rather than enumerating all majority subsets.
        row_sums = {
            int(idx): float(
                sum(float(A[int(idx), int(other)]) for other in current if int(other) != int(idx))
            )
            for idx in current
        }
        while len(current) > int(target_keep):
            client_scores: List[Tuple[float, int]] = []
            for idx in current:
                if consensus_objective == "mean":
                    num_others = int(len(current) - 1)
                    score_i = (
                        float(row_sums[int(idx)]) / float(num_others)
                        if num_others > 0
                        else 0.0
                    )
                else:
                    others = [other for other in current if int(other) != int(idx)]
                    score_i = _consensus_score(
                        [float(A[int(idx), int(other)]) for other in others]
                    )
                client_scores.append((float(score_i), int(idx)))

            _, remove_idx_raw = min(
                client_scores,
                key=lambda item: (
                    float(item[0]),
                    int(order_rank[int(item[1])]),
                ),
            )
            remove_idx = int(remove_idx_raw)
            current = [idx for idx in current if int(idx) != int(remove_idx)]
            if consensus_objective == "mean":
                for idx in current:
                    row_sums[int(idx)] -= float(A[int(idx), int(remove_idx)])
            row_sums.pop(int(remove_idx), None)

        candidate_subsets = [list(current)]
        mode_name = "greedy_consensus_peeling"
    else:
        raise ValueError(
            f"Unknown pointillism_fl.selection_mode='{cfg_point.get('selection_mode')}'. "
            "Expected 'Consensus', 'GreedyConsensus', or 'Pareto'."
        )

    subset_rows: List[Dict[str, Any]] = []
    cov_vals: List[float] = []
    con_vals: List[float] = []

    for subset in candidate_subsets:
        counts = _label_coverage_counts(
            subset=subset,
            client_infos=client_infos,
            num_classes=int(num_classes),
        )
        j_cov = _label_coverage_score(counts)

        pair_vals: List[float] = []
        for pos_a in range(len(subset)):
            for pos_b in range(pos_a + 1, len(subset)):
                pair_vals.append(float(A[int(subset[pos_a]), int(subset[pos_b])]))
        j_con_avg = _safe_mean(pair_vals) if pair_vals else 0.0
        j_con = _consensus_score(pair_vals)
        j_med = _median_or_zero(pair_vals)

        cov_vals.append(float(j_cov))
        con_vals.append(float(j_con))
        subset_rows.append({
            "subset": [int(x) for x in subset],
            "size": int(len(subset)),
            "J_cov": float(j_cov),
            "J_con": float(j_con),
            "J_con_avg": float(j_con_avg),
            "J_con_low_quantile": float(np.quantile(np.asarray(pair_vals, dtype=np.float32), float(consensus_quantile_alpha))) if pair_vals else 0.0,
            "J_con_quantile_alpha": float(consensus_quantile_alpha),
            "J_con_objective": str(consensus_objective),
            "J_med": float(j_med),
        })

    cov_max = max([float(x) for x in cov_vals] + [1.0])
    lambda_cov = float(cfg_point.get("lambda_cov", 0.35))
    lambda_con = float(cfg_point.get("lambda_con", 0.65))

    best_idx = 0
    best_score = float("-inf")
    for idx, row in enumerate(subset_rows):
        cov_norm = float(row["J_cov"]) / float(cov_max) if cov_max > 0.0 else 0.0
        if selection_mode == "pareto":
            score = float(lambda_cov) * float(cov_norm) + float(lambda_con) * float(row["J_con"])
        else:
            score = float(row["J_con"])
        row["score"] = float(score)
        row["cov_norm"] = float(cov_norm)
        if (
            float(score) > float(best_score)
            or (math.isclose(float(score), float(best_score)) and float(row["J_con"]) > float(subset_rows[best_idx]["J_con"]))
            or (math.isclose(float(score), float(best_score)) and int(row["size"]) > int(subset_rows[best_idx]["size"]))
        ):
            best_idx = int(idx)
            best_score = float(score)

    keep_idx = sorted(int(x) for x in subset_rows[int(best_idx)]["subset"])
    reject_idx = sorted(idx for idx in range(k) if int(idx) not in set(keep_idx))
    return keep_idx, reject_idx, {
        "mode": str(mode_name),
        "subset_rows": subset_rows,
        "selected_score": float(best_score),
        "selected_subset": [int(x) for x in keep_idx],
        "selection_order": [int(x) for x in shuffled_order],
        "consensus_objective": str(consensus_objective),
        "consensus_quantile_alpha": float(consensus_quantile_alpha),
    }


def update_memory_bank(
    *,
    state: PointillismFLState,
    client_infos: List[Dict[str, Any]],
    specs: List[Dict[str, Any]],
    scores: Dict[str, np.ndarray],
    keep_idx: List[int],
    num_classes: int,
    cfg_point: Dict[str, Any],
) -> Dict[str, Any]:
    top_k = max(0, int(cfg_point.get("top_k", 8)))
    edge_top_k = max(0, int(cfg_point.get("edge_top_k", 4)))
    client_bank_idx = list(range(len(client_infos)))

    topk_bank: Dict[int, List[Dict[str, Any]]] = {}
    edge_bank: Dict[str, List[Dict[str, Any]]] = {}
    topk_fill_rows: List[Dict[str, Any]] = []
    edge_fill_rows: List[Dict[str, Any]] = []

    for label in range(num_classes):
        grouped: Dict[Tuple[int, Tuple[int, ...], str, int], Dict[str, Any]] = {}
        for idx in client_bank_idx:
            for probe_idx in client_infos[int(idx)]["signature"]["topk_by_label"].get(int(label), []):
                spec = specs[int(probe_idx)]
                sig = _probe_signature(spec)
                rec = grouped.setdefault(sig, {"spec": _strip_probe_runtime_fields(spec), "scores": [], "count": 0})
                rec["scores"].append(float(scores["probs"][int(idx), int(probe_idx), int(label)]))
                rec["count"] += 1
        rows = list(grouped.values())
        rows.sort(
            key=lambda r: (-int(r["count"]), -_safe_mean(r["scores"]), int(r["spec"]["w"]), tuple(r["spec"]["flat_idx"]))
        )
        selected_rows = rows[: int(top_k)] if int(top_k) > 0 else []
        max_client_assets = int(len(client_bank_idx) * int(top_k))
        topk_bank[int(label)] = [dict(r["spec"]) for r in selected_rows]
        topk_fill_rows.append({
            "label": int(label),
            "available_unique": int(len(rows)),
            "filled": int(len(selected_rows)),
            "capacity": int(top_k),
            "max_client_assets": int(max_client_assets),
            "shortfall": max(0, int(top_k) - int(len(selected_rows))),
        })

    all_pairs = [_pair_key(a, b) for a in range(num_classes) for b in range(a + 1, num_classes)]
    for pair_name in all_pairs:
        a, b = _parse_pair_key(pair_name)
        grouped: Dict[Tuple[int, Tuple[int, ...], str, int], Dict[str, Any]] = {}
        for idx in client_bank_idx:
            for probe_idx in client_infos[int(idx)]["signature"]["edge_by_pair"].get(pair_name, []):
                spec = specs[int(probe_idx)]
                sig = _probe_signature(spec)
                rec = grouped.setdefault(sig, {"spec": _strip_probe_runtime_fields(spec), "margins": [], "count": 0})
                rec["margins"].append(float(scores["margin"][int(idx), int(probe_idx)]))
                rec["count"] += 1
        rows = list(grouped.values())
        rows.sort(
            key=lambda r: (-int(r["count"]), _safe_mean(r["margins"]), int(r["spec"]["w"]), tuple(r["spec"]["flat_idx"]))
        )
        selected_rows = rows[: int(edge_top_k)] if int(edge_top_k) > 0 else []
        max_client_assets = int(len(client_bank_idx) * int(edge_top_k))
        edge_bank[str(pair_name)] = [dict(r["spec"]) for r in selected_rows]
        edge_fill_rows.append({
            "pair": str(pair_name),
            "label_a": int(a),
            "label_b": int(b),
            "available_unique": int(len(rows)),
            "filled": int(len(selected_rows)),
            "capacity": int(edge_top_k),
            "max_client_assets": int(max_client_assets),
            "shortfall": max(0, int(edge_top_k) - int(len(selected_rows))),
        })

    state.topk_bank = topk_bank
    state.edge_bank = edge_bank
    state.color_bank = _build_color_memory_bank(
        topk_bank=topk_bank,
        edge_bank=edge_bank,
        num_classes=int(num_classes),
    )
    state.rounds_seen += 1

    topk_filled_total = int(sum(int(row["filled"]) for row in topk_fill_rows))
    edge_filled_total = int(sum(int(row["filled"]) for row in edge_fill_rows))
    topk_capacity_total = int(sum(int(row["capacity"]) for row in topk_fill_rows))
    edge_capacity_total = int(sum(int(row["capacity"]) for row in edge_fill_rows))
    topk_max_client_assets_total = int(num_classes * len(client_bank_idx) * int(top_k))
    edge_max_client_assets_total = int(len(all_pairs) * len(client_bank_idx) * int(edge_top_k))
    color_bank_total = int(sum(len(v) for v in state.color_bank.values()))
    return {
        "bank_topk_labels": int(sum(1 for _, v in topk_bank.items() if v)),
        "bank_edge_pairs": int(sum(1 for _, v in edge_bank.items() if v)),
        "bank_color_labels": int(sum(1 for _, v in state.color_bank.items() if v)),
        "bank_color_total": int(color_bank_total),
        "topk_keep_per_label": int(top_k),
        "edge_keep_per_pair": int(edge_top_k),
        "topk_filled_total": int(topk_filled_total),
        "edge_filled_total": int(edge_filled_total),
        "topk_capacity_total": int(topk_capacity_total),
        "edge_capacity_total": int(edge_capacity_total),
        "topk_max_client_assets_total": int(topk_max_client_assets_total),
        "edge_max_client_assets_total": int(edge_max_client_assets_total),
        "topk_shortfall_total": int(sum(int(row["shortfall"]) for row in topk_fill_rows)),
        "edge_shortfall_total": int(sum(int(row["shortfall"]) for row in edge_fill_rows)),
        "topk_fill_rows": topk_fill_rows,
        "edge_fill_rows": edge_fill_rows,
    }


def save_state_snapshot(
    *,
    out_dir: Optional[str],
    rnd: int,
    state: PointillismFLState,
    round_meta: Dict[str, Any],
) -> None:
    if out_dir is None:
        return
    os.makedirs(str(out_dir), exist_ok=True)
    path = os.path.join(str(out_dir), "pointillism_fl_state.json")
    payload = {
        "round": int(rnd),
        "rounds_seen": int(state.rounds_seen),
        "topk_bank": {str(k): v for k, v in state.topk_bank.items()},
        "edge_bank": {str(k): v for k, v in state.edge_bank.items()},
        "color_bank": {str(k): v for k, v in state.color_bank.items()},
        "random_pool_fresh": list(state.random_pool_fresh),
        "round_meta": dict(round_meta),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _append_dict_rows_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _heatmap_scale_and_label(mat: np.ndarray, metric_key: str) -> Tuple[float, str]:
    vals = np.asarray(mat[np.isfinite(mat)], dtype=np.float64)
    vals = vals[vals >= 0.0]
    if vals.size <= 0:
        return 1.0, "fixed [0,1]"

    max_val = float(np.max(vals))
    if str(metric_key) in {"selection_score", "bar_A", "bar_R"} and 0.0 < max_val < 0.25:
        vmax = max(float(max_val), 1e-12)
        return vmax, f"auto [0,{vmax:.2g}]"
    return 1.0, "fixed [0,1]"


def _format_heatmap_value(value: float) -> str:
    if not np.isfinite(float(value)):
        return "nan"
    value_f = float(value)
    if value_f == 0.0:
        return "0"
    if abs(value_f) < 0.01:
        return f"{value_f:.1e}"
    return f"{value_f:.2f}"


def _save_fl_pairwise_heatmap(
    *,
    out_dir: str,
    rnd: int,
    client_infos: List[Dict[str, Any]],
    pair_rows: List[Dict[str, Any]],
    metric_key: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(out_dir, "plots_fl_consensus", str(metric_key))
    os.makedirs(plot_dir, exist_ok=True)
    client_ids = [int(info["cid"]) for info in client_infos]
    client_labels = [
        f"{int(info['cid'])}*" if int(info.get("is_malicious", 0)) == 1 else str(int(info["cid"]))
        for info in client_infos
    ]
    idx_by_cid = {int(cid): pos for pos, cid in enumerate(client_ids)}
    n = len(client_ids)
    mat = np.full((n, n), np.nan, dtype=np.float32)

    for row in pair_rows:
        a = int(row["cid_a"])
        b = int(row["cid_b"])
        if a not in idx_by_cid or b not in idx_by_cid:
            continue
        i = idx_by_cid[a]
        j = idx_by_cid[b]
        val = float(row.get(metric_key, 0.0))
        mat[i, j] = val
        mat[j, i] = val

    fig = plt.figure(figsize=(7.2, 6.0), dpi=180)
    ax = fig.add_subplot(1, 1, 1)
    vmax, scale_label = _heatmap_scale_and_label(mat, str(metric_key))
    im = ax.imshow(mat, vmin=0.0, vmax=float(vmax), cmap="viridis", aspect="equal")
    ax.set_title(f"{metric_key} heatmap | round {int(rnd)} | {scale_label}")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(client_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(client_labels, fontsize=8)
    ax.set_xlabel("Selected clients")
    ax.set_ylabel("Selected clients")

    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            text = _format_heatmap_value(float(val))
            frac = float(val) / float(vmax) if np.isfinite(val) and float(vmax) > 0.0 else 0.0
            color = "white" if np.isfinite(val) and float(frac) < 0.65 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(f"{metric_key} ({scale_label})", rotation=90)
    fig.tight_layout()
    fig.savefig(
        os.path.join(plot_dir, f"{metric_key}_round{int(rnd):03d}.png"),
        bbox_inches="tight",
    )
    plt.close(fig)


def log_round_outputs(
    *,
    out_dir: Optional[str],
    rnd: int,
    pool_meta: Dict[str, Any],
    refine_meta: Dict[str, Any],
    client_infos: List[Dict[str, Any]],
    pair_rows: List[Dict[str, Any]],
    select_info: Dict[str, Any],
    bank_meta: Optional[Dict[str, Any]] = None,
    cfg_point: Optional[Dict[str, Any]] = None,
) -> None:
    if out_dir is None:
        return

    num_classes = 0
    if client_infos:
        sample_q1 = client_infos[0]["q1"]
        by_grid = sample_q1.get("evidence_by_grid", {}) or {}
        if by_grid:
            first_grid = next(iter(by_grid.values()))
            num_classes = int(len(first_grid.get("E", [])))
        else:
            num_classes = int(len(sample_q1.get("class_evidence_max", sample_q1.get("evidence_by_label_max", [])) or []))
    selected_subset = [int(x) for x in select_info.get("selected_subset", [])]
    keep_set = set(selected_subset)
    reject_subset = [int(idx) for idx in range(len(client_infos)) if int(idx) not in keep_set]

    selected_counts = _label_coverage_counts(
        subset=selected_subset,
        client_infos=client_infos,
        num_classes=int(num_classes),
    )
    rejected_counts = _label_coverage_counts(
        subset=reject_subset,
        client_infos=client_infos,
        num_classes=int(num_classes),
    )
    selected_j_cov = _label_coverage_score(selected_counts)
    rejected_j_cov = _label_coverage_score(rejected_counts)

    round_rows = [{
        "round": int(rnd),
        "num_pool": int(pool_meta.get("num_total", 0)),
        "num_random": int(pool_meta.get("num_random", 0)),
        "num_bank_topk": int(pool_meta.get("num_bank_topk", 0)),
        "num_bank_edge": int(pool_meta.get("num_bank_edge", 0)),
        "num_refine": int(refine_meta.get("num_specs", 0)),
        "bank_topk_filled_total": int((bank_meta or {}).get("topk_filled_total", 0)),
        "bank_topk_capacity_total": int((bank_meta or {}).get("topk_capacity_total", 0)),
        "bank_topk_shortfall_total": int((bank_meta or {}).get("topk_shortfall_total", 0)),
        "bank_edge_filled_total": int((bank_meta or {}).get("edge_filled_total", 0)),
        "bank_edge_capacity_total": int((bank_meta or {}).get("edge_capacity_total", 0)),
        "bank_edge_shortfall_total": int((bank_meta or {}).get("edge_shortfall_total", 0)),
        "bank_color_labels": int((bank_meta or {}).get("bank_color_labels", 0)),
        "bank_color_total": int((bank_meta or {}).get("bank_color_total", 0)),
        "keep_ids": ",".join(str(x) for x in select_info.get("keep_ids", [])),
        "reject_ids": ",".join(str(x) for x in select_info.get("reject_ids", [])),
        "selected_score": float(select_info.get("selected_score", 0.0)),
        "selected_J_cov": float(selected_j_cov),
        "rejected_J_cov": float(rejected_j_cov),
    }]
    _append_dict_rows_csv(
        os.path.join(str(out_dir), "pointillism_fl_rounds.csv"),
        list(round_rows[0].keys()),
        round_rows,
    )

    topk_fill_rows = [
        {
            "round": int(rnd),
            **row,
        }
        for row in list((bank_meta or {}).get("topk_fill_rows", []))
    ]
    if topk_fill_rows:
        _append_dict_rows_csv(
            os.path.join(str(out_dir), "pointillism_fl_bank_topk_fill.csv"),
            list(topk_fill_rows[0].keys()),
            topk_fill_rows,
        )

    edge_fill_rows = [
        {
            "round": int(rnd),
            **row,
        }
        for row in list((bank_meta or {}).get("edge_fill_rows", []))
    ]
    if edge_fill_rows:
        _append_dict_rows_csv(
            os.path.join(str(out_dir), "pointillism_fl_bank_edge_fill.csv"),
            list(edge_fill_rows[0].keys()),
            edge_fill_rows,
        )

    client_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    for idx, info in enumerate(client_infos):
        q1 = info["q1"]
        support = set(int(x) for x in q1.get("effective_support_labels", []))
        is_malicious = info.get("is_malicious", "")
        client_rows.append({
            "round": int(rnd),
            "cid": int(info["cid"]),
            "keep": int(idx in keep_set),
            "is_malicious": is_malicious,
            "R_f": float(q1.get("R_f", 0.0)),
            "support_size": int(len(support)),
            "support_labels": ",".join(str(x) for x in sorted(support)),
        })
        coverage_row = {
            "round": int(rnd),
            "cid": int(info["cid"]),
            "keep": int(idx in keep_set),
            "is_malicious": is_malicious,
            "R_f": float(q1.get("R_f", 0.0)),
            "support_size": int(len(support)),
            "support_labels": ",".join(str(x) for x in sorted(support)),
        }
        for label in range(int(num_classes)):
            coverage_row[f"label_{int(label)}"] = int(int(label) in support)
        coverage_rows.append(coverage_row)
    if client_rows:
        _append_dict_rows_csv(
            os.path.join(str(out_dir), "pointillism_fl_clients.csv"),
            list(client_rows[0].keys()),
            client_rows,
        )
    if coverage_rows:
        _append_dict_rows_csv(
            os.path.join(str(out_dir), "pointillism_fl_client_label_coverage.csv"),
            list(coverage_rows[0].keys()),
            coverage_rows,
        )

    model_reliability_rows: List[Dict[str, Any]] = []
    model_reliability_summary_rows: List[Dict[str, Any]] = []
    class_evidence_rows: List[Dict[str, Any]] = []
    for idx, info in enumerate(client_infos):
        q1 = info["q1"]
        support = set(int(x) for x in q1.get("support_labels", []))
        effective = set(int(x) for x in q1.get("effective_support_labels", []))
        is_malicious = info.get("is_malicious", "")

        model_r = _model_reliability_value(q1, num_classes=int(num_classes))
        grid_details = list(q1.get("model_reliability_by_grid", []) or [])
        if not grid_details and q1.get("evidence_by_grid"):
            _model_r, grid_details = _model_reliability_from_by_grid(
                q1.get("evidence_by_grid", {}) or {},
                num_classes=int(num_classes),
            )
        for rec in grid_details:
            model_reliability_rows.append({
                "round": int(rnd),
                "cid": int(info["cid"]),
                "keep": int(idx in keep_set),
                "is_malicious": is_malicious,
                "grid_hw": int(rec.get("grid_hw", -1)),
                "reliability_mode": str(rec.get("reliability_mode", "diversity")),
                "model_reliability_d": float(rec.get("model_reliability_d", 0.0)),
                "diversity": float(rec.get("diversity", 0.0)),
                "confidence": float(rec.get("confidence", 0.0)),
                "q_entropy": float(rec.get("q_entropy", 0.0)),
                "mean_entropy_norm": float(rec.get("mean_entropy_norm", 1.0)),
                "dominant_label": int(rec.get("dominant_label", -1)),
                "dominant_q": float(rec.get("dominant_q", 0.0)),
            })

        rel_by_grid = [float(rec.get("model_reliability_d", 0.0)) for rec in grid_details]
        diversity_by_grid = [float(rec.get("diversity", 0.0)) for rec in grid_details]
        confidence_by_grid = [float(rec.get("confidence", 0.0)) for rec in grid_details]
        dominant_q_by_grid = [float(rec.get("dominant_q", 0.0)) for rec in grid_details]
        model_reliability_summary_rows.append({
            "round": int(rnd),
            "cid": int(info["cid"]),
            "keep": int(idx in keep_set),
            "is_malicious": is_malicious,
            "reliability_mode": "diversity",
            "model_reliability": float(model_r),
            "grid_reliability_min": float(min(rel_by_grid)) if rel_by_grid else 0.0,
            "grid_reliability_max": float(max(rel_by_grid)) if rel_by_grid else 0.0,
            "diversity_mean": _safe_mean(diversity_by_grid),
            "confidence_mean": _safe_mean(confidence_by_grid),
            "dominant_q_mean": _safe_mean(dominant_q_by_grid),
            "dominant_q_max": float(max(dominant_q_by_grid)) if dominant_q_by_grid else 0.0,
            "dominant_labels": ",".join(str(int(rec.get("dominant_label", -1))) for rec in grid_details),
            "support_size": int(len(support)),
            "effective_support_size": int(len(effective)),
        })

        for rec in _class_evidence_details(q1, num_classes=int(num_classes)):
            label = int(rec["label"])
            class_evidence_rows.append({
                "round": int(rnd),
                "cid": int(info["cid"]),
                "keep": int(idx in keep_set),
                "is_malicious": is_malicious,
                "label": int(label),
                "class_evidence": float(rec["class_evidence"]),
                "support": int(label in support),
                "effective_support": int(label in effective),
                "best_grid_hw": rec["best_grid_hw"],
                "best_q": float(rec["best_q"]),
                "best_entropy_mean": float(rec["best_entropy_mean"]),
                "best_u": float(rec["best_u"]),
                "best_E": float(rec["best_E"]),
                "best_count": int(rec["best_count"]),
            })
    if model_reliability_rows:
        _append_dict_rows_csv(
            os.path.join(str(out_dir), "pointillism_fl_reliability_model.csv"),
            list(model_reliability_rows[0].keys()),
            model_reliability_rows,
        )
    if model_reliability_summary_rows:
        _append_dict_rows_csv(
            os.path.join(str(out_dir), "pointillism_fl_reliability_model_summary.csv"),
            list(model_reliability_summary_rows[0].keys()),
            model_reliability_summary_rows,
        )
        grouped_rows: List[Dict[str, Any]] = []
        for group_name, group_rows in [
            ("all", model_reliability_summary_rows),
            ("benign", [row for row in model_reliability_summary_rows if str(row.get("is_malicious", "")) == "0"]),
            ("malicious", [row for row in model_reliability_summary_rows if str(row.get("is_malicious", "")) == "1"]),
        ]:
            if not group_rows:
                continue
            grouped_rows.append({
                "round": int(rnd),
                "group": str(group_name),
                "num_clients": int(len(group_rows)),
                "num_kept": int(sum(int(row.get("keep", 0)) for row in group_rows)),
                "model_reliability": _safe_mean([float(row["model_reliability"]) for row in group_rows]),
                "grid_reliability_min_mean": _safe_mean([float(row["grid_reliability_min"]) for row in group_rows]),
                "grid_reliability_max_mean": _safe_mean([float(row["grid_reliability_max"]) for row in group_rows]),
                "diversity_mean": _safe_mean([float(row["diversity_mean"]) for row in group_rows]),
                "confidence_mean": _safe_mean([float(row["confidence_mean"]) for row in group_rows]),
                "dominant_q_mean": _safe_mean([float(row["dominant_q_mean"]) for row in group_rows]),
                "dominant_q_max_mean": _safe_mean([float(row["dominant_q_max"]) for row in group_rows]),
                "support_size_mean": _safe_mean([float(row["support_size"]) for row in group_rows]),
                "effective_support_size_mean": _safe_mean([float(row["effective_support_size"]) for row in group_rows]),
            })
        if grouped_rows:
            _append_dict_rows_csv(
                os.path.join(str(out_dir), "pointillism_fl_reliability_model_by_malicious.csv"),
                list(grouped_rows[0].keys()),
                grouped_rows,
            )
    if class_evidence_rows:
        _append_dict_rows_csv(
            os.path.join(str(out_dir), "pointillism_fl_class_evidence.csv"),
            list(class_evidence_rows[0].keys()),
            class_evidence_rows,
        )

    pair_rows_rnd = [{"round": int(rnd), **row} for row in pair_rows]
    if pair_rows_rnd:
        _append_dict_rows_csv(
            os.path.join(str(out_dir), "pointillism_fl_pairwise.csv"),
            list(pair_rows_rnd[0].keys()),
            pair_rows_rnd,
        )
        pair_reliability_debug_rows = [
            {
                "round": int(rnd),
                "cid_a": int(row["cid_a"]),
                "cid_b": int(row["cid_b"]),
                "selection_table": str(row.get("selection_table", "bar_a")),
                "selection_score": float(row.get("selection_score", row.get("bar_A", 0.0))),
                "shared_support_size": int(row.get("shared_support_size", 0)),
                "selected_agreement_labels": str(row.get("selected_agreement_labels", "")),
                "selected_reliability_weights": str(row.get("selected_reliability_weights", "")),
                "selected_reliability_zero_count": int(row.get("selected_reliability_zero_count", 0)),
                "bar_R": float(row.get("bar_R", 0.0)),
                "bar_A_raw": float(row.get("bar_A_raw", 0.0)),
                "bar_A": float(row.get("bar_A", 0.0)),
            }
            for row in pair_rows
        ]
        if pair_reliability_debug_rows:
            _append_dict_rows_csv(
                os.path.join(str(out_dir), "pointillism_fl_pairwise_reliability_debug.csv"),
                list(pair_reliability_debug_rows[0].keys()),
                pair_reliability_debug_rows,
            )
        cfg_point_l = dict(cfg_point or {})
        save_heatmaps = bool(cfg_point_l.get("save_pairwise_heatmaps", True))
        if bool(cfg_point_l.get("debug", cfg_point_l.get("debug_mode", False))):
            save_heatmaps = True
        if save_heatmaps:
            for metric_key in ["selection_score", "bar_A", "bar_A_raw", "bar_T", "bar_P", "bar_R"]:
                if metric_key not in pair_rows_rnd[0]:
                    continue
                _save_fl_pairwise_heatmap(
                    out_dir=str(out_dir),
                    rnd=int(rnd),
                    client_infos=client_infos,
                    pair_rows=pair_rows,
                    metric_key=str(metric_key),
                )

    coverage_summary_rows = [{
        "round": int(rnd),
        "selected_ids": ",".join(str(int(client_infos[idx]["cid"])) for idx in selected_subset),
        "rejected_ids": ",".join(str(int(client_infos[idx]["cid"])) for idx in reject_subset),
        "selected_size": int(len(selected_subset)),
        "rejected_size": int(len(reject_subset)),
        "selected_J_cov": float(selected_j_cov),
        "rejected_J_cov": float(rejected_j_cov),
        "selected_counts": ",".join(str(int(x)) for x in selected_counts),
        "rejected_counts": ",".join(str(int(x)) for x in rejected_counts),
    }]
    for label in range(int(num_classes)):
        coverage_summary_rows[0][f"selected_label_{int(label)}"] = int(selected_counts[int(label)])
        coverage_summary_rows[0][f"rejected_label_{int(label)}"] = int(rejected_counts[int(label)])
    _append_dict_rows_csv(
        os.path.join(str(out_dir), "pointillism_fl_label_coverage.csv"),
        list(coverage_summary_rows[0].keys()),
        coverage_summary_rows,
    )

    subset_rows = [{"round": int(rnd), **row} for row in select_info.get("subset_rows", [])]
    if subset_rows:
        _append_dict_rows_csv(
            os.path.join(str(out_dir), "pointillism_fl_subsets.csv"),
            list(subset_rows[0].keys()),
            subset_rows,
        )
