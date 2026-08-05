from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE

from pointillism.pointillism_probe_search import (
    render_gaussian_probe_records,
    render_two_group_probe_records,
    score_mask_records_all_labels,
)
from utils.norm import _norm_cfg


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def save_json(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def save_topl_summary_csv(
    *,
    topl_by_label: Dict[int, Dict[str, Any]],
    csv_path: Path,
) -> None:
    rows: List[Dict[str, Any]] = []

    for label, info in sorted(topl_by_label.items(), key=lambda kv: int(kv[0])):
        topl = info.get("topl", [])
        selection_mode = str(info.get("selection_mode", "robustness_rank"))
        selection_score_key = str(info.get("selection_score_key", "robust_target_mean"))
        for rank, rec in enumerate(topl):
            selection_score = float(
                rec.get(
                    "selection_score",
                    rec.get(selection_score_key, rec.get("robust_target_mean", rec.get("target_score", 0.0))),
                )
            )
            rows.append(
                {
                    "label": int(label),
                    "rank": int(rank),
                    "selection_mode": selection_mode,
                    "selection_score_key": selection_score_key,
                    "selection_score": selection_score,
                    "grid_hw": int(rec.get("grid_hw", -1)),
                    "w": int(rec.get("w", -1)),
                    "pred_label": int(rec.get("pred_label", -1)),
                    "pred_score": float(rec.get("pred_score", 0.0)),
                    "target_score": float(rec.get("target_score", 0.0)),
                    "robust_target_mean": float(rec.get("robust_target_mean", 0.0)),
                    "robust_target_std": float(rec.get("robust_target_std", 0.0)),
                    "robust_count": int(rec.get("robust_count", 0)),
                    "flat_idx": " ".join(str(int(x)) for x in rec.get("flat_idx", [])),
                }
            )

    fieldnames = [
        "label",
        "rank",
        "selection_mode",
        "selection_score_key",
        "selection_score",
        "grid_hw",
        "w",
        "pred_label",
        "pred_score",
        "target_score",
        "robust_target_mean",
        "robust_target_std",
        "robust_count",
        "flat_idx",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_missing_label_fallback_rows(
    *,
    topl_by_label: Dict[int, Dict[str, Any]],
    grid_hws: List[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ordered_grid_hws = [int(x) for x in sorted(int(d) for d in grid_hws)]
    for label, info in sorted(topl_by_label.items(), key=lambda kv: int(kv[0])):
        selection_mode = str(info.get("selection_mode", ""))
        if "missing_label" not in selection_mode:
            continue
        selection_info = dict(info.get("selection_info", {}))
        row: Dict[str, Any] = {
            "label": int(label),
            "selection_mode": selection_mode,
            "selection_score_key": str(info.get("selection_score_key", "target_score")),
            "fallback_mode": str(selection_info.get("fallback_mode", "seed_only")),
            "direct_hit_count": int(selection_info.get("direct_hit_count", 0)),
            "best_d": int(selection_info.get("best_d", -1)),
            "seed_count_total": int(selection_info.get("seed_count_total", 0)),
            "best_seed_score": float(selection_info.get("best_seed_score", 0.0)),
            "mean_seed_score_top5": float(selection_info.get("mean_seed_score_top5", 0.0)),
            "refined_count": int(selection_info.get("refined_count", 0)),
            "best_refined_score": selection_info.get("best_refined_score", None),
            "best_improvement": selection_info.get("best_improvement", None),
            "final_selected_count": int(selection_info.get("final_selected_count", len(info.get("topl", [])))),
        }
        best_by_d = dict(selection_info.get("per_d_best_target_score", {}))
        mean_top5_by_d = dict(selection_info.get("per_d_mean_top5_target_score", {}))
        quota_by_d = dict(selection_info.get("per_d_seed_quota", {}))
        count_by_d = dict(selection_info.get("per_d_seed_counts", {}))
        for d in ordered_grid_hws:
            row[f"best_target_score_d{int(d)}"] = float(best_by_d.get(int(d), best_by_d.get(str(int(d)), 0.0)))
            row[f"mean_top5_target_score_d{int(d)}"] = float(mean_top5_by_d.get(int(d), mean_top5_by_d.get(str(int(d)), 0.0)))
            row[f"seed_quota_d{int(d)}"] = int(quota_by_d.get(int(d), quota_by_d.get(str(int(d)), 0)))
            row[f"seed_count_d{int(d)}"] = int(count_by_d.get(int(d), count_by_d.get(str(int(d)), 0)))
        rows.append(row)
    return rows


def save_missing_label_fallback_summary(
    *,
    topl_by_label: Dict[int, Dict[str, Any]],
    grid_hws: List[int],
    out_json: Path,
    out_csv: Path,
) -> None:
    rows = extract_missing_label_fallback_rows(
        topl_by_label=topl_by_label,
        grid_hws=grid_hws,
    )
    save_json({"rows": rows}, out_json)
    if not rows:
        fieldnames = ["label", "selection_mode", "fallback_mode"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    preferred = [
        "label",
        "selection_mode",
        "selection_score_key",
        "fallback_mode",
        "direct_hit_count",
        "best_d",
        "seed_count_total",
        "best_seed_score",
        "mean_seed_score_top5",
        "refined_count",
        "best_refined_score",
        "best_improvement",
        "final_selected_count",
    ]
    dynamic = [key for key in rows[0].keys() if key not in preferred]
    fieldnames = preferred + sorted(dynamic)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# ---------------------------------------------------------------------
# sonar compare helpers
# ---------------------------------------------------------------------

def _lookup_sonar_info_by_d(sonar_map: Dict[Any, Dict[str, Any]], d: int) -> Dict[str, Any]:
    if int(d) in sonar_map:
        return dict(sonar_map[int(d)])
    d_str = str(int(d))
    if d_str in sonar_map:
        return dict(sonar_map[d_str])
    return {}


def sonar_ratio_matrix(
    sonar_map: Dict[int, Dict[str, Any]],
    *,
    grid_hws: List[int],
    num_classes: int,
) -> np.ndarray:
    rows: List[List[float]] = []
    for d in grid_hws:
        info = _lookup_sonar_info_by_d(sonar_map, int(d))
        ratios = info.get("class_ratios", [0.0] * int(num_classes))
        row = [float(x) for x in ratios[: int(num_classes)]]
        if len(row) < int(num_classes):
            row += [0.0] * (int(num_classes) - len(row))
        rows.append(row)
    if len(rows) == 0:
        return np.zeros((0, int(num_classes)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _normalize_prob(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).copy()
    x = np.maximum(x, 0.0)
    s = float(x.sum())
    if s <= 1e-12:
        if x.size == 0:
            return x
        return np.full_like(x, 1.0 / float(x.size))
    return x / s


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _normalize_prob(np.asarray(p, dtype=np.float64).reshape(-1))
    q = _normalize_prob(np.asarray(q, dtype=np.float64).reshape(-1))
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        aa = np.clip(a, eps, None)
        bb = np.clip(b, eps, None)
        return float(np.sum(aa * np.log(aa / bb)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def plot_pairwise_heatmap(
    *,
    values: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    out_path: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 0.75 * len(col_labels)), max(4, 0.55 * len(row_labels))))
    im = ax.imshow(values, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{float(values[i, j]):.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sonar_ratio_delta_heatmap(
    *,
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    tag_a: str,
    tag_b: str,
    out_path: Path,
) -> None:
    delta = np.asarray(mat_b, dtype=np.float64) - np.asarray(mat_a, dtype=np.float64)
    vmax = float(np.max(np.abs(delta))) if delta.size > 0 else 0.0
    vmax = max(vmax, 0.05)

    fig, ax = plt.subplots(figsize=(max(7, 0.95 * len(col_labels)), max(4.6, 1.05 * len(row_labels))))
    im = ax.imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"Sonar Ratio Delta ({tag_b} - {tag_a})")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            dval = float(delta[i, j])
            aval = float(mat_a[i, j])
            bval = float(mat_b[i, j])
            txt = f"{dval:+.2f}\n{aval:.2f}|{bval:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{tag_b} ratio - {tag_a} ratio")
    ax.set_xlabel("Class label")
    ax.set_ylabel("Grid resolution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pairwise_metrics_and_plots(
    *,
    result_a: Dict[str, Any],
    result_b: Dict[str, Any],
    grid_hws: List[int],
    num_classes: int,
    out_dir: Path,
    tag_a: str,
    tag_b: str,
) -> None:
    ensure_dir(out_dir)

    mat_a = sonar_ratio_matrix(
        result_a.get("sonar_map", {}),
        grid_hws=grid_hws,
        num_classes=num_classes,
    )
    mat_b = sonar_ratio_matrix(
        result_b.get("sonar_map", {}),
        grid_hws=grid_hws,
        num_classes=num_classes,
    )

    if mat_a.shape != mat_b.shape:
        save_json(
            {
                "warning": "shape mismatch",
                "shape_a": list(mat_a.shape),
                "shape_b": list(mat_b.shape),
            },
            out_dir / "pairwise_warning.json",
        )
        return

    per_d_cos = []
    per_d_js = []
    for i, d in enumerate(grid_hws):
        row_a = mat_a[i]
        row_b = mat_b[i]
        per_d_cos.append(
            {
                "grid_hw": int(d),
                "cosine": float(safe_cosine(row_a, row_b)),
            }
        )
        per_d_js.append(
            {
                "grid_hw": int(d),
                "js": float(js_divergence(row_a, row_b)),
            }
        )

    delta_matrix = mat_b - mat_a
    flat_cos = float(safe_cosine(mat_a.reshape(-1), mat_b.reshape(-1)))
    flat_js = float(js_divergence(mat_a.reshape(-1), mat_b.reshape(-1)))
    mean_abs_delta = float(np.mean(np.abs(delta_matrix))) if delta_matrix.size > 0 else 0.0
    max_abs_delta = float(np.max(np.abs(delta_matrix))) if delta_matrix.size > 0 else 0.0

    save_json(
        {
            "tag_a": str(tag_a),
            "tag_b": str(tag_b),
            "grid_hws": [int(x) for x in grid_hws],
            "num_classes": int(num_classes),
            "flat_cosine": flat_cos,
            "flat_js": flat_js,
            "per_d_cosine": per_d_cos,
            "per_d_js": per_d_js,
            "mean_abs_delta": mean_abs_delta,
            "max_abs_delta": max_abs_delta,
        },
        out_dir / "pairwise_metrics.json",
    )

    row_labels = [f"d={int(d)}" for d in grid_hws]
    col_labels = [f"c{int(i)}" for i in range(int(num_classes))]

    plot_sonar_ratio_delta_heatmap(
        mat_a=mat_a,
        mat_b=mat_b,
        row_labels=row_labels,
        col_labels=col_labels,
        tag_a=str(tag_a),
        tag_b=str(tag_b),
        out_path=out_dir / "sonar_ratio_delta_heatmap.png",
    )


def _score_records_grouped_by_grid(
    *,
    model,
    records: List[Dict[str, Any]],
    out_hw: int,
    dataset: str,
    device: torch.device,
    batch_size: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(int(rec.get("grid_hw", 1)), []).append(dict(rec))

    out: List[Dict[str, Any]] = []
    for grid_hw in sorted(grouped.keys()):
        out.extend(
            score_mask_records_all_labels(
                model=model,
                records=grouped[int(grid_hw)],
                out_hw=int(out_hw),
                dataset=str(dataset),
                device=device,
                norm_cfg_fn=_norm_cfg,
                batch_size=int(batch_size),
            )
        )
    return out


def compute_topl_transfer_target_softmax(
    *,
    source_tag: str,
    source_topl_by_label: Dict[int, Dict[str, Any]],
    eval_tag: str,
    eval_model,
    out_hw: int,
    dataset: str,
    device: torch.device,
    batch_size: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for label, info in sorted(source_topl_by_label.items(), key=lambda kv: int(kv[0])):
        topl = list(info.get("topl", []))
        scored = _score_records_grouped_by_grid(
            model=eval_model,
            records=topl,
            out_hw=int(out_hw),
            dataset=str(dataset),
            device=device,
            batch_size=int(batch_size),
        )
        count = int(len(scored))
        if count <= 0:
            avg_target_softmax = 0.0
        else:
            target_probs = [float(rec.get("all_probs", [0.0] * (int(label) + 1))[int(label)]) for rec in scored]
            avg_target_softmax = float(sum(target_probs) / float(count))

        rows.append(
            {
                "source_model": str(source_tag),
                "eval_model": str(eval_tag),
                "label": int(label),
                "num_examples": count,
                "avg_target_softmax": float(avg_target_softmax),
            }
        )

    return rows


def save_topl_transfer_target_softmax_csv(
    *,
    rows: List[Dict[str, Any]],
    csv_path: Path,
) -> None:
    fieldnames = [
        "source_model",
        "eval_model",
        "label",
        "num_examples",
        "avg_target_softmax",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_topl_transfer_target_softmax(
    *,
    rows: List[Dict[str, Any]],
    source_tag: str,
    eval_tags: List[str],
    out_path: Path,
) -> None:
    rows_src = [r for r in rows if str(r.get("source_model")) == str(source_tag)]
    if not rows_src:
        return

    labels = sorted({int(r["label"]) for r in rows_src})
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(labels)), 4.8))
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    mean_by_eval: Dict[str, float] = {}

    for idx, eval_tag in enumerate(eval_tags):
        vals = []
        for lab in labels:
            match = next((r for r in rows_src if int(r["label"]) == int(lab) and str(r["eval_model"]) == str(eval_tag)), None)
            vals.append(float(match["avg_target_softmax"]) if match is not None else 0.0)
        mean_by_eval[str(eval_tag)] = float(sum(vals) / float(len(vals))) if vals else 0.0
        offset = (idx - (len(eval_tags) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            label=f"{str(eval_tag)} (avg={mean_by_eval[str(eval_tag)]:.3f})",
            color=palette[idx % len(palette)],
            alpha=0.9,
        )

    ax.set_title(f"Top-L Avg Target Softmax | source={source_tag}")
    ax.set_xlabel("Class label")
    ax.set_ylabel("Average target-label softmax")
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{int(l)}" for l in labels])
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_topl_consistency_analysis(
    *,
    result_by_tag: Dict[str, Dict[str, Any]],
    model_by_tag: Dict[str, torch.nn.Module],
    out_hw: int,
    dataset: str,
    device: torch.device,
    out_dir: Path,
    batch_size: int = 4096,
) -> None:
    ensure_dir(out_dir)
    tags = [str(x) for x in sorted(result_by_tag.keys())]
    rows: List[Dict[str, Any]] = []

    for source_tag in tags:
        source_topl_by_label = result_by_tag[str(source_tag)].get("topl_by_label", {})
        for eval_tag in tags:
            rows.extend(
                compute_topl_transfer_target_softmax(
                    source_tag=str(source_tag),
                    source_topl_by_label=source_topl_by_label,
                    eval_tag=str(eval_tag),
                    eval_model=model_by_tag[str(eval_tag)],
                    out_hw=int(out_hw),
                    dataset=str(dataset),
                    device=device,
                    batch_size=int(batch_size),
                )
            )

    save_topl_transfer_target_softmax_csv(
        rows=rows,
        csv_path=out_dir / "topl_transfer_target_softmax.csv",
    )
    for source_tag in tags:
        plot_topl_transfer_target_softmax(
            rows=rows,
            source_tag=str(source_tag),
            eval_tags=tags,
            out_path=out_dir / f"topl_transfer_target_softmax_source_{str(source_tag)}.png",
        )


# ---------------------------------------------------------------------
# sonar/top-l viz helpers
# ---------------------------------------------------------------------

def _resolve_tsne_perplexity(n_points: int, requested: float = 40.0) -> float:
    if int(n_points) < 3:
        raise ValueError(f"t-SNE requires at least 3 points, got {int(n_points)}")
    return float(min(float(requested), max(2.0, float(int(n_points) - 1))))


def _categorical_colors(n: int) -> List[Any]:
    if int(n) <= 0:
        return ["#1f77b4"]
    palette = []
    for cmap_name in ["tab10", "Dark2", "Set1", "Set2", "tab20"]:
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, "colors"):
            palette.extend(list(cmap.colors))
    if int(n) <= len(palette):
        return palette[: int(n)]
    extra = plt.get_cmap("hsv", int(n))
    while len(palette) < int(n):
        palette.append(extra(len(palette)))
    return palette[: int(n)]


def _cluster_anchor_point(points: np.ndarray) -> np.ndarray:
    if int(points.shape[0]) == 0:
        return np.zeros((2,), dtype=np.float64)
    center = np.median(points, axis=0)
    d2 = np.sum((points - center[None, :]) ** 2, axis=1)
    return np.asarray(points[int(np.argmin(d2))], dtype=np.float64)


def _resolve_label_positions(anchors: np.ndarray, x_span: float, y_span: float) -> np.ndarray:
    pos = np.asarray(anchors, dtype=np.float64).copy()
    min_dx = 0.05 * float(max(x_span, 1e-6))
    min_dy = 0.05 * float(max(y_span, 1e-6))
    max_shift_x = 0.08 * float(max(x_span, 1e-6))
    max_shift_y = 0.08 * float(max(y_span, 1e-6))
    for _ in range(80):
        moved = False
        for i in range(int(pos.shape[0])):
            for j in range(i + 1, int(pos.shape[0])):
                dx = float(pos[j, 0] - pos[i, 0])
                dy = float(pos[j, 1] - pos[i, 1])
                if abs(dx) >= min_dx or abs(dy) >= min_dy:
                    continue
                push_x = 0.5 * (min_dx - abs(dx) + 1e-6)
                push_y = 0.5 * (min_dy - abs(dy) + 1e-6)
                sx = 1.0 if dx >= 0.0 else -1.0
                sy = 1.0 if dy >= 0.0 else -1.0
                trial_i = pos[i] + np.asarray([-sx * push_x, -sy * push_y], dtype=np.float64)
                trial_j = pos[j] + np.asarray([sx * push_x, sy * push_y], dtype=np.float64)
                for idx, trial in [(i, trial_i), (j, trial_j)]:
                    shift = trial - anchors[idx]
                    shift[0] = float(np.clip(shift[0], -max_shift_x, max_shift_x))
                    shift[1] = float(np.clip(shift[1], -max_shift_y, max_shift_y))
                    pos[idx] = anchors[idx] + shift
                moved = True
        if not moved:
            break
    return pos


def _stratified_sample_indices(labels: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=np.int64)
    n = int(labels_arr.shape[0])
    if n <= int(max_points):
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    uniq, counts = np.unique(labels_arr, return_counts=True)
    quotas = np.floor(counts.astype(np.float64) / float(n) * float(max_points)).astype(np.int64)
    quotas = np.maximum(quotas, 1)
    quotas = np.minimum(quotas, counts)

    while int(quotas.sum()) > int(max_points):
        idx = int(np.argmax(quotas))
        if quotas[idx] > 1:
            quotas[idx] -= 1
        else:
            break

    while int(quotas.sum()) < int(max_points):
        slack = counts - quotas
        idx = int(np.argmax(slack))
        if slack[idx] <= 0:
            break
        quotas[idx] += 1

    selected: List[int] = []
    for label, quota in zip(uniq.tolist(), quotas.tolist()):
        idx = np.flatnonzero(labels_arr == int(label))
        if int(quota) >= int(idx.shape[0]):
            selected.extend(int(x) for x in idx.tolist())
        else:
            picked = rng.choice(idx, size=int(quota), replace=False)
            selected.extend(int(x) for x in picked.tolist())
    selected = sorted(selected)
    if len(selected) > int(max_points):
        selected = selected[: int(max_points)]
    return np.asarray(selected, dtype=np.int64)


def _plot_prob_tsne(
    *,
    emb: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_png: Path,
) -> None:
    uniq = sorted(int(x) for x in np.unique(labels))
    max_label = max(uniq) if uniq else 0
    palette = _categorical_colors(max(1, int(max_label) + 1))
    color_map = {int(label): palette[int(label)] for label in uniq}
    x_span = max(float(np.max(emb[:, 0]) - np.min(emb[:, 0])), 1e-6)
    y_span = max(float(np.max(emb[:, 1]) - np.min(emb[:, 1])), 1e-6)
    anchors = []
    cluster_indices = []
    for label in uniq:
        idx = np.flatnonzero(labels == int(label))
        cluster_indices.append(idx)
        anchors.append(_cluster_anchor_point(emb[idx]))
    anchor_arr = np.asarray(anchors, dtype=np.float64)
    label_pos = _resolve_label_positions(anchor_arr, x_span=float(x_span), y_span=float(y_span))

    fig, ax = plt.subplots(figsize=(9.5, 7.8))
    for i, label in enumerate(uniq):
        idx = cluster_indices[i]
        ax.scatter(
            emb[idx, 0],
            emb[idx, 1],
            s=8,
            alpha=0.62,
            c=[color_map[int(label)]],
            edgecolors="white",
            linewidths=0.15,
            label=f"{int(label)}",
        )
        cx = float(anchor_arr[i, 0])
        cy = float(anchor_arr[i, 1])
        tx = float(label_pos[i, 0])
        ty = float(label_pos[i, 1])
        ax.annotate(
            str(int(label)),
            xy=(cx, cy),
            xytext=(tx, ty),
            textcoords="data",
            fontsize=11.5,
            weight="bold",
            ha="center",
            va="center",
            color="black",
            bbox={"boxstyle": "square,pad=0.22", "facecolor": "white", "edgecolor": "#333333", "alpha": 0.92},
            arrowprops={
                "arrowstyle": "-",
                "color": "#333333",
                "lw": 0.95,
                "shrinkA": 4,
                "shrinkB": 4,
                "alpha": 0.9,
            },
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.18)
    ax.legend(
        title="Pred Label",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        ncol=2,
        fontsize=9,
        title_fontsize=10,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#444444",
        markerscale=2.2,
        labelspacing=0.55,
        borderpad=0.6,
        handletextpad=0.45,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _records_for_grid_hw(result: Dict[str, Any], grid_hw: int) -> List[Dict[str, Any]]:
    by_d = result.get("candidate_pool_by_label_by_d", {})
    records: List[Dict[str, Any]] = []
    d_payload = by_d.get(int(grid_hw), {})
    for recs in d_payload.values():
        records.extend(list(recs))
    return records


def save_stage1_prob_tsne_plots(
    *,
    result: Dict[str, Any],
    out_dir: Path,
    model_tag: str,
    max_points: int = 2000,
    seed: int = 123,
) -> None:
    ensure_dir(out_dir)
    grid_hws = [int(x) for x in result.get("grid_hws", [])]

    for d in grid_hws:
        records = _records_for_grid_hw(result, int(d))
        probs = []
        labels = []
        for rec in records:
            prob_row = rec.get("all_probs", None)
            pred_label = rec.get("pred_label", None)
            if prob_row is None or pred_label is None:
                continue
            probs.append([float(x) for x in prob_row])
            labels.append(int(pred_label))

        if len(probs) < 3:
            continue

        probs_arr = np.asarray(probs, dtype=np.float32)
        labels_arr = np.asarray(labels, dtype=np.int64)
        keep = _stratified_sample_indices(labels_arr, max_points=int(max_points), seed=int(seed) + int(d))
        probs_arr = probs_arr[keep]
        labels_arr = labels_arr[keep]
        n_points = int(probs_arr.shape[0])

        metric_specs = [
            ("cosine", probs_arr, "cosine"),
            ("js", probs_arr.astype(np.float64, copy=False), "precomputed"),
        ]

        for metric_name, metric_payload, metric_arg in metric_specs:
            if metric_name == "js":
                x_in = squareform(pdist(metric_payload, metric="jensenshannon"))
            else:
                x_in = metric_payload

            tsne = TSNE(
                n_components=2,
                metric=metric_arg,
                init="random",
                learning_rate="auto",
                random_state=int(seed),
                perplexity=_resolve_tsne_perplexity(n_points, requested=40.0),
            )
            emb = tsne.fit_transform(x_in)

            png_path = out_dir / f"{model_tag}_d{int(d)}_stage1_probs_{metric_name}_tsne.png"
            csv_path = out_dir / f"{model_tag}_d{int(d)}_stage1_probs_{metric_name}_tsne_points.csv"
            _plot_prob_tsne(
                emb=emb,
                labels=labels_arr,
                title=f"{model_tag} d={int(d)} Stage-1 probs t-SNE ({metric_name.upper()}) n={n_points}",
                out_png=png_path,
            )
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["model_tag", "grid_hw", "pred_label", "x", "y", "metric", "n_points"])
                writer.writeheader()
                for idx in range(n_points):
                    writer.writerow(
                        {
                            "model_tag": str(model_tag),
                            "grid_hw": int(d),
                            "pred_label": int(labels_arr[idx]),
                            "x": float(emb[idx, 0]),
                            "y": float(emb[idx, 1]),
                            "metric": str(metric_name),
                            "n_points": int(n_points),
                        }
                    )


def save_sonar_histogram_panel(
    *,
    result: Dict[str, Any],
    out_path: Path,
    model_tag: str,
) -> None:
    sonar_map = result.get("sonar_map", {})
    grid_hws = [int(x) for x in result.get("grid_hws", [])]
    num_classes = len(result.get("coverage_status", {}).get("counts", []))
    if num_classes <= 0 or not grid_hws:
        return

    fig, axes = plt.subplots(
        len(grid_hws),
        1,
        figsize=(max(10, 0.72 * num_classes), max(2.8 * len(grid_hws), 4.8)),
        sharex=True,
    )
    if len(grid_hws) == 1:
        axes = [axes]

    x = np.arange(num_classes)
    for ax, d in zip(axes, grid_hws):
        info = sonar_map.get(int(d), {})
        counts = [int(x) for x in info.get("class_counts", [0] * num_classes)]
        ratios = [float(x) for x in info.get("class_ratios", [0.0] * num_classes)]
        max_count = max(counts + [1])
        text_pad = max(1.0, 0.05 * float(max_count))
        ax.bar(x, counts, color="#4C78A8", alpha=0.88, edgecolor="white", linewidth=0.6)
        ax.set_ylabel(f"d={int(d)}")
        ax.set_title(f"d={int(d)} | total={int(sum(counts))}")
        ax.set_ylim(0.0, float(max_count) + float(text_pad) * (1.8 if num_classes <= 15 else 0.9))
        ax.margins(x=0.01)
        ax.grid(True, axis="y", alpha=0.18)
        if num_classes <= 15:
            for idx, (count, ratio) in enumerate(zip(counts, ratios)):
                ax.text(
                    idx,
                    float(count) + float(text_pad) * 0.35,
                    f"{int(count)}\n{float(ratio):.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"c{int(i)}" for i in range(num_classes)], rotation=45, ha="right")
    fig.suptitle(f"{model_tag} Sonar Class Counts by Resolution", y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_mask_record_to_image(rec: Dict[str, Any], out_hw: int) -> np.ndarray:
    if str(rec.get("probe_mode", "binary")).startswith("two_group"):
        img = render_two_group_probe_records(
            [rec],
            out_hw=int(out_hw),
            device=torch.device("cpu"),
        )[0].cpu().numpy()
        if img.shape[0] == 1:
            return np.asarray(img[0], dtype=np.float32)
        return np.asarray(np.transpose(img, (1, 2, 0)), dtype=np.float32)
    if str(rec.get("probe_mode", "binary")).startswith("gaussian"):
        img = render_gaussian_probe_records(
            [rec],
            out_hw=int(out_hw),
            device=torch.device("cpu"),
        )[0].cpu().numpy()
        if img.shape[0] == 1:
            return np.asarray(img[0], dtype=np.float32)
        return np.asarray(np.transpose(img, (1, 2, 0)), dtype=np.float32)

    grid_hw = int(rec.get("grid_hw", 1))
    background_fill = float(rec.get("background_fill", 0.0))
    active_fill = float(rec.get("active_fill", 1.0))
    grid = torch.full((1, 1, int(grid_hw), int(grid_hw)), float(background_fill), dtype=torch.float32)
    flat = grid[0, 0].view(-1)
    flat[[int(x) for x in rec.get("flat_idx", [])]] = float(active_fill)
    img = F.interpolate(grid, size=(int(out_hw), int(out_hw)), mode="nearest")[0, 0].cpu().numpy()
    return np.asarray(img, dtype=np.float32)

def save_entropy_summary_csv(
    *,
    entropy_summary: Dict[str, Any],
    csv_path: Path,
) -> None:
    rows: List[Dict[str, Any]] = []
    by_label_by_d = dict(entropy_summary.get("by_pred_label_by_d", {}))

    for d_key in sorted(by_label_by_d.keys(), key=lambda x: int(x)):
        d = int(d_key)
        per_d = dict(by_label_by_d[d_key])
        for label_key in sorted(per_d.keys(), key=lambda x: int(x)):
            label = int(label_key)
            rec = dict(per_d[label_key])
            rows.append(
                {
                    "grid_hw": int(d),
                    "label": int(label),
                    "count": int(rec.get("count", 0)),
                    "entropy_mean": float(rec.get("entropy_mean", 0.0)),
                    "entropy_median": float(rec.get("entropy_median", 0.0)),
                    "entropy_std": float(rec.get("entropy_std", 0.0)),
                    "entropy_q10": float(rec.get("entropy_q10", 0.0)),
                    "entropy_q90": float(rec.get("entropy_q90", 0.0)),
                    "margin_mean": float(rec.get("margin_mean", 0.0)),
                    "margin_median": float(rec.get("margin_median", 0.0)),
                }
            )

    fieldnames = [
        "grid_hw",
        "label",
        "count",
        "entropy_mean",
        "entropy_median",
        "entropy_std",
        "entropy_q10",
        "entropy_q90",
        "margin_mean",
        "margin_median",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_edge_case_summary_csv(
    *,
    edge_cases: Dict[str, Any],
    csv_path: Path,
) -> None:
    rows: List[Dict[str, Any]] = []
    summary_by_pair = dict(edge_cases.get("summary_by_pair", {}))

    for pair_key in sorted(summary_by_pair.keys()):
        rec = dict(summary_by_pair[pair_key])
        pair = rec.get("pair", [-1, -1])
        grid_hist = rec.get("grid_hist", {})
        rows.append(
            {
                "pair_key": str(pair_key),
                "class_a": int(pair[0]),
                "class_b": int(pair[1]),
                "count_total": int(rec.get("count_total", 0)),
                "count_kept": int(rec.get("count_kept", 0)),
                "margin_mean": float(rec.get("margin_mean", 0.0)),
                "margin_median": float(rec.get("margin_median", 0.0)),
                "entropy_mean": float(rec.get("entropy_mean", 0.0)),
                "entropy_median": float(rec.get("entropy_median", 0.0)),
                "grid_hist": json.dumps(grid_hist, ensure_ascii=False),
            }
        )

    fieldnames = [
        "pair_key",
        "class_a",
        "class_b",
        "count_total",
        "count_kept",
        "margin_mean",
        "margin_median",
        "entropy_mean",
        "entropy_median",
        "grid_hist",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_topl_examples(
    *,
    topl_by_label: Dict[int, Dict[str, Any]],
    out_dir: Path,
    out_hw: int,
    num_examples: int = 20,
) -> None:
    ensure_dir(out_dir)

    for label, info in sorted(topl_by_label.items(), key=lambda kv: int(kv[0])):
        label_dir = out_dir / f"label_{int(label):02d}"
        ensure_dir(label_dir)
        topl = list(info.get("topl", []))[: int(num_examples)]
        if not topl:
            continue

        selection_mode = str(info.get("selection_mode", "robustness_rank"))
        selection_score_key = str(info.get("selection_score_key", "robust_target_mean"))
        score_tag = "tgt" if selection_score_key == "target_score" else "rob"

        cols = 5
        rows = int(math.ceil(float(len(topl)) / float(cols)))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9))
        axes_arr = np.atleast_1d(axes).reshape(rows, cols)

        for rank, rec in enumerate(topl):
            img = render_mask_record_to_image(rec, out_hw=int(out_hw))
            polarity = str(rec.get("polarity", "white"))
            score_value = float(
                rec.get(
                    "selection_score",
                    rec.get(selection_score_key, rec.get("robust_target_mean", rec.get("target_score", 0.0))),
                )
            )
            grid_hw = int(rec.get("grid_hw", -1))
            file_name = f"rank_{int(rank):02d}_d{int(grid_hw)}_pol_{polarity}_{score_tag}_{score_value:.4f}.png"
            file_path = label_dir / file_name

            fig_single, ax_single = plt.subplots(figsize=(2.8, 2.8))
            ax_single.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax_single.set_title(f"r={int(rank)} d={int(grid_hw)} {polarity}\n{score_tag}={score_value:.3f}")
            ax_single.axis("off")
            fig_single.tight_layout()
            fig_single.savefig(file_path, dpi=180, bbox_inches="tight")
            plt.close(fig_single)

            ax = axes_arr[rank // cols, rank % cols]
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(f"r={int(rank)} d={int(grid_hw)}\n{polarity} | {score_value:.3f}", fontsize=9)
            ax.axis("off")

        for idx in range(len(topl), rows * cols):
            axes_arr[idx // cols, idx % cols].axis("off")

        fig.suptitle(f"label={int(label):02d} Top-{len(topl)} Examples ({selection_mode})", y=0.995)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])
        fig.savefig(out_dir / f"top_L_label{int(label):02d}_grid.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

def save_entropy_visualizations(
    *,
    result: Dict[str, Any],
    out_dir: Path,
    model_tag: str,
) -> None:
    """
    Save entropy visualizations:
      - entropy_heatmap.png
      - margin_heatmap.png
      - entropy_histograms_by_grid.png
    Uses result["entropy_summary"].
    """
    entropy_summary = dict(result.get("entropy_summary", {}))
    by_label_by_d = dict(entropy_summary.get("by_pred_label_by_d", {}))
    if not by_label_by_d:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    grid_hws = sorted(int(k) for k in by_label_by_d.keys())
    num_classes = len(result.get("coverage_status", {}).get("counts", []))
    if num_classes <= 0 or len(grid_hws) == 0:
        return

    entropy_mat = np.zeros((len(grid_hws), int(num_classes)), dtype=np.float32)
    margin_mat = np.zeros((len(grid_hws), int(num_classes)), dtype=np.float32)
    count_mat = np.zeros((len(grid_hws), int(num_classes)), dtype=np.int32)

    for r, d in enumerate(grid_hws):
        rec_d = dict(by_label_by_d.get(int(d), by_label_by_d.get(str(int(d)), {})))
        for c in range(int(num_classes)):
            rec = dict(rec_d.get(int(c), rec_d.get(str(int(c)), {})))
            entropy_mat[r, c] = float(rec.get("entropy_mean", 0.0))
            margin_mat[r, c] = float(rec.get("margin_mean", 0.0))
            count_mat[r, c] = int(rec.get("count", 0))

    row_labels = [f"d={int(d)}" for d in grid_hws]
    col_labels = [f"c{int(i)}" for i in range(int(num_classes))]

    # ---- entropy heatmap ----
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * num_classes), max(4.5, 1.0 * len(grid_hws))))
    im = ax.imshow(entropy_mat, aspect="auto")
    ax.set_title(f"{model_tag} Entropy Mean by Grid and Predicted Class")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(entropy_mat.shape[0]):
        for j in range(entropy_mat.shape[1]):
            txt = f"{float(entropy_mat[i, j]):.2f}\nN={int(count_mat[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean entropy")
    fig.tight_layout()
    fig.savefig(out_dir / "entropy_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- margin heatmap ----
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * num_classes), max(4.5, 1.0 * len(grid_hws))))
    im = ax.imshow(margin_mat, aspect="auto")
    ax.set_title(f"{model_tag} Top-2 Margin Mean by Grid and Predicted Class")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(margin_mat.shape[0]):
        for j in range(margin_mat.shape[1]):
            txt = f"{float(margin_mat[i, j]):.2f}\nN={int(count_mat[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean top-2 margin")
    fig.tight_layout()
    fig.savefig(out_dir / "margin_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- entropy histograms by grid ----
    by_d = dict(entropy_summary.get("by_d", {}))
    if by_d:
        nrows = len(grid_hws)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            figsize=(9, max(3.0 * nrows, 4.5)),
            sharex=True,
        )
        if nrows == 1:
            axes = [axes]

        # We only have summary stats, not raw values, so this panel is an interval summary bar-style plot.
        # If later you want true histograms, we can save raw entropies too.
        for ax, d in zip(axes, grid_hws):
            rec = dict(by_d.get(int(d), by_d.get(str(int(d)), {})))
            q10 = float(rec.get("entropy_q10", 0.0))
            q90 = float(rec.get("entropy_q90", 0.0))
            mean = float(rec.get("entropy_mean", 0.0))
            med = float(rec.get("entropy_median", 0.0))
            count = int(rec.get("count", 0))

            ax.hlines(y=0, xmin=q10, xmax=q90, linewidth=6)
            ax.plot([mean], [0], marker="o", markersize=8, label="mean")
            ax.plot([med], [0], marker="s", markersize=7, label="median")
            ax.set_yticks([])
            ax.set_ylabel(f"d={int(d)}", rotation=0, labelpad=25)
            ax.set_title(f"d={int(d)} | N={int(count)} | q10={q10:.2f} mean={mean:.2f} med={med:.2f} q90={q90:.2f}")
            ax.grid(True, axis="x", alpha=0.2)

        axes[-1].set_xlabel("Entropy")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"{model_tag} Entropy Summary by Grid", y=0.995)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
        fig.savefig(out_dir / "entropy_histograms_by_grid.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def _infer_num_classes_from_result(result: Dict[str, Any]) -> int:
    counts = list(result.get("coverage_status", {}).get("counts", []))
    if counts:
        return int(len(counts))

    sonar_map = dict(result.get("sonar_map", {}))
    for info in sonar_map.values():
        ratios = list(info.get("class_ratios", []))
        if ratios:
            return int(len(ratios))
        class_counts = list(info.get("class_counts", []))
        if class_counts:
            return int(len(class_counts))

    by_label_by_d = dict(result.get("entropy_summary", {}).get("by_pred_label_by_d", {}))
    max_label = -1
    for rec_d in by_label_by_d.values():
        for label in rec_d.keys():
            max_label = max(max_label, int(label))
    return int(max_label + 1) if max_label >= 0 else 0


def save_global_response_statistics_sg(
    *,
    result: Dict[str, Any],
    out_dir: Path,
    model_tag: str,
) -> None:
    """
    Save paper-ready global response statistics S_g visualizations:
      - entropy_by_grid_and_class.png
      - class_label_distribution_by_grid.png
    """
    entropy_summary = dict(result.get("entropy_summary", {}))
    by_label_by_d = dict(entropy_summary.get("by_pred_label_by_d", {}))
    sonar_map = dict(result.get("sonar_map", {}))
    num_classes = _infer_num_classes_from_result(result)

    if num_classes <= 0:
        return

    grid_hws = sorted(
        {
            *[int(k) for k in by_label_by_d.keys()],
            *[int(k) for k in sonar_map.keys()],
        }
    )
    if not grid_hws:
        return

    ensure_dir(out_dir)

    entropy_mat = np.zeros((len(grid_hws), int(num_classes)), dtype=np.float32)
    count_mat = np.zeros((len(grid_hws), int(num_classes)), dtype=np.int32)
    ratio_mat = np.zeros((len(grid_hws), int(num_classes)), dtype=np.float32)
    row_totals = np.zeros(len(grid_hws), dtype=np.int32)

    for row_idx, d in enumerate(grid_hws):
        rec_d = dict(by_label_by_d.get(int(d), by_label_by_d.get(str(int(d)), {})))
        sonar_d = dict(sonar_map.get(int(d), sonar_map.get(str(int(d)), {})))
        class_ratios = [float(x) for x in sonar_d.get("class_ratios", [0.0] * int(num_classes))]
        if len(class_ratios) < int(num_classes):
            class_ratios = class_ratios + [0.0] * (int(num_classes) - len(class_ratios))
        ratio_mat[row_idx] = np.asarray(class_ratios[: int(num_classes)], dtype=np.float32)
        row_totals[row_idx] = int(sonar_d.get("num_trials", int(np.sum(sonar_d.get("class_counts", [])))))

        for label in range(int(num_classes)):
            rec = dict(rec_d.get(int(label), rec_d.get(str(int(label)), {})))
            entropy_mat[row_idx, label] = float(rec.get("entropy_mean", 0.0))
            count_mat[row_idx, label] = int(rec.get("count", 0))

    row_labels = [f"$\\mathcal{{P}}_{{d={int(d)}}}$" for d in grid_hws]
    # Use a moderately compact canvas so the panel stays paper-friendly
    # without making the heatmap look cramped.
    fig_w = max(6.4, 0.48 * num_classes + 1.6)
    fig_h = max(2.10, 0.46 * len(grid_hws) + 0.72)

    cbar_fraction = 0.032
    cbar_pad = 0.018
    left_margin = 0.12
    right_margin = 0.89
    bottom_margin = 0.22
    top_margin = 0.95

    # ---- entropy heatmap ----
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(entropy_mat, aspect="auto", cmap="viridis")
    ax.set_xlabel("Predicted class label")
    ax.set_ylabel("Probe subset")
    ax.set_xticks(np.arange(int(num_classes)))
    ax.set_xticklabels([str(i) for i in range(int(num_classes))])
    ax.set_yticks(np.arange(len(grid_hws)))
    ax.set_yticklabels(row_labels, linespacing=1.0)

    max_entropy = float(np.max(entropy_mat)) if entropy_mat.size else 0.0
    text_threshold = 0.58 * max_entropy if max_entropy > 0.0 else 0.0
    for i in range(entropy_mat.shape[0]):
        for j in range(entropy_mat.shape[1]):
            val = float(entropy_mat[i, j])
            has_mass = int(count_mat[i, j]) > 0
            text = f"{val:.2f}" if has_mass else "-"
            color = "black" if has_mass and val >= text_threshold else "white"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    fig.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin)
    cbar = fig.colorbar(im, ax=ax, fraction=cbar_fraction, pad=cbar_pad)
    if max_entropy > 0.0 and max_entropy <= 1.25:
        tick_max = 1.0 if max_entropy <= 1.10 else 0.2 * math.ceil(max_entropy / 0.2)
        cbar.set_ticks(np.arange(0.0, tick_max + 1e-9, 0.2))
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar.set_label("Mean entropy")
    fig.savefig(out_dir / f"{model_tag}_entropy_by_grid_and_class.png", dpi=220)
    plt.close(fig)

    # ---- class-label distribution heatmap ----
    ratio_pct_mat = 100.0 * ratio_mat
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(ratio_pct_mat, aspect="auto", cmap="magma")
    ax.set_xlabel("Predicted class label")
    ax.set_ylabel("Probe subset")
    ax.set_xticks(np.arange(int(num_classes)))
    ax.set_xticklabels([str(i) for i in range(int(num_classes))])
    ax.set_yticks(np.arange(len(grid_hws)))
    ax.set_yticklabels(row_labels, linespacing=1.0)

    max_pct = float(np.max(ratio_pct_mat)) if ratio_pct_mat.size else 0.0
    text_threshold = 0.55 * max_pct if max_pct > 0.0 else 0.0
    for i in range(ratio_pct_mat.shape[0]):
        for j in range(ratio_pct_mat.shape[1]):
            val = float(ratio_pct_mat[i, j])
            color = "black" if val >= text_threshold and val > 0.0 else "white"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8, color=color)

    fig.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin)
    cbar = fig.colorbar(im, ax=ax, fraction=cbar_fraction, pad=cbar_pad)
    cbar.set_label("Class dist. (%)")
    fig.savefig(out_dir / f"{model_tag}_class_label_distribution_by_grid.png", dpi=220)
    plt.close(fig)
        
def save_edge_case_examples(
    *,
    result: Dict[str, Any],
    out_dir: Path,
    out_hw: int,
    max_pairs: int = 10,
    num_examples_per_pair: int = 20,
) -> None:
    """
    Save edge-case probe examples.

    Folder structure:
      edge_case_examples/
        pair_00_01/
          pair_00_01_grid.png
          rank_00_....png
          ...
    """
    edge_cases = dict(result.get("edge_cases", {}))
    summary_by_pair = dict(edge_cases.get("summary_by_pair", {}))
    top_records_by_pair = dict(edge_cases.get("top_records_by_pair", {}))
    if not summary_by_pair or not top_records_by_pair:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # rank pairs by smaller margin_mean first, then larger entropy_mean
    ranked_pairs = []
    for pair_key, rec in summary_by_pair.items():
        ranked_pairs.append(
            (
                float(rec.get("margin_mean", 1e9)),
                -float(rec.get("entropy_mean", 0.0)),
                str(pair_key),
            )
        )
    ranked_pairs.sort()

    selected_pair_keys = [pair_key for _, _, pair_key in ranked_pairs[: max(1, int(max_pairs))]]

    for pair_key in selected_pair_keys:
        recs = list(top_records_by_pair.get(pair_key, []))[: max(1, int(num_examples_per_pair))]
        if not recs:
            continue

        pair_dir = out_dir / f"pair_{str(pair_key)}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        cols = 5
        rows = int(math.ceil(float(len(recs)) / float(cols)))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.9, rows * 3.1))
        axes_arr = np.atleast_1d(axes).reshape(rows, cols)

        for rank, rec in enumerate(recs):
            img = render_mask_record_to_image(rec, out_hw=int(out_hw))
            grid_hw = int(rec.get("grid_hw", -1))
            margin = float(rec.get("margin", 0.0))
            entropy = float(rec.get("entropy", 0.0))
            top1_prob = float(rec.get("top1_prob", rec.get("pred_score", 0.0)))
            top2_prob = float(rec.get("top2_prob", 0.0))
            top1_label = int(rec.get("top1_label", rec.get("pred_label", -1)))
            top2_label = int(rec.get("top2_label", -1))
            polarity = str(rec.get("polarity", "white"))

            file_name = (
                f"rank_{int(rank):02d}_d{int(grid_hw)}_"
                f"m_{margin:.4f}_h_{entropy:.4f}_"
                f"p1_{int(top1_label)}_{top1_prob:.3f}_"
                f"p2_{int(top2_label)}_{top2_prob:.3f}_"
                f"{polarity}.png"
            )
            file_path = pair_dir / file_name

            fig_single, ax_single = plt.subplots(figsize=(2.8, 2.8))
            ax_single.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax_single.set_title(
                f"r={int(rank)} d={int(grid_hw)}\n"
                f"m={margin:.3f} H={entropy:.3f}\n"
                f"{int(top1_label)}:{top1_prob:.2f} | {int(top2_label)}:{top2_prob:.2f}",
                fontsize=9,
            )
            ax_single.axis("off")
            fig_single.tight_layout()
            fig_single.savefig(file_path, dpi=180, bbox_inches="tight")
            plt.close(fig_single)

            ax = axes_arr[rank // cols, rank % cols]
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(
                f"r={int(rank)} d={int(grid_hw)}\n"
                f"m={margin:.3f} H={entropy:.3f}\n"
                f"{int(top1_label)}:{top1_prob:.2f} | {int(top2_label)}:{top2_prob:.2f}",
                fontsize=8.5,
            )
            ax.axis("off")

        for idx in range(len(recs), rows * cols):
            axes_arr[idx // cols, idx % cols].axis("off")

        summary = dict(summary_by_pair.get(pair_key, {}))
        pair = summary.get("pair", [-1, -1])
        fig.suptitle(
            f"Edge Cases pair=({int(pair[0])},{int(pair[1])}) "
            f"| mean_margin={float(summary.get('margin_mean', 0.0)):.4f} "
            f"| mean_entropy={float(summary.get('entropy_mean', 0.0)):.4f}",
            y=0.995,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
        fig.savefig(pair_dir / f"pair_{str(pair_key)}_grid.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
