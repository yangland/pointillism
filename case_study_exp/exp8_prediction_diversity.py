#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml

from data.loaders import LoaderCfg, _seed_worker, build_test_loaders, build_train_loader
from data.registry import get_dataset_spec
from pointillism.pointillism_signature import run_pointillism_search
from utils.norm import _norm_cfg

from case_study_exp.common import (
    RunLogger,
    build_random_init_model,
    ensure_dir,
    now_tag,
    resolve_device,
    save_json,
    save_signature_outputs,
    set_seed,
    write_csv,
)


def _default_cases(num_classes: int) -> List[Dict[str, Any]]:
    even = [int(x) for x in range(int(num_classes)) if int(x) % 2 == 0]
    odd = [int(x) for x in range(int(num_classes)) if int(x) % 2 == 1]
    miss_two = [even[0], odd[0]] if even and odd else [0]
    return [
        {"tag": "iid_trained", "group": "iid_trained", "train": True, "keep_labels": "all"},
        {"tag": "untrained", "group": "untrained", "train": False, "keep_labels": "all"},
        {"tag": "miss_even", "group": "missing_even", "train": True, "missing_labels": even},
        {"tag": "miss_odd", "group": "missing_odd", "train": True, "missing_labels": odd},
        {"tag": "miss_two_labels", "group": "missing_two_labels", "train": True, "missing_labels": miss_two},
    ]


def _labels_from_dataset(dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        base_labels = _labels_from_dataset(dataset.dataset)
        return np.asarray(base_labels, dtype=np.int64)[np.asarray(dataset.indices, dtype=np.int64)]
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        if torch.is_tensor(targets):
            return targets.detach().cpu().numpy().astype(np.int64)
        return np.asarray(targets, dtype=np.int64)
    labels: List[int] = []
    for idx in range(len(dataset)):
        _x, y = dataset[idx]
        labels.append(int(y))
    return np.asarray(labels, dtype=np.int64)


def _subset_by_labels(dataset, keep_labels: List[int]) -> Subset:
    keep = {int(x) for x in keep_labels}
    labels = _labels_from_dataset(dataset)
    indices = [int(i) for i, y in enumerate(labels.tolist()) if int(y) in keep]
    return Subset(dataset, indices)


def _label_counts(dataset, *, num_classes: int) -> List[int]:
    labels = _labels_from_dataset(dataset)
    return [int(np.sum(labels == int(c))) for c in range(int(num_classes))]


def _make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int,
) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers) and int(num_workers) > 0,
        worker_init_fn=_seed_worker,
        generator=g,
    )


def _case_keep_labels(case_cfg: Dict[str, Any], *, num_classes: int) -> List[int]:
    keep_raw = case_cfg.get("keep_labels", None)
    if keep_raw is None or str(keep_raw).lower() == "all":
        keep = list(range(int(num_classes)))
    else:
        keep = [int(x) for x in list(keep_raw)]

    missing = case_cfg.get("missing_labels", None)
    if missing is not None:
        missing_set = {int(x) for x in list(missing)}
        keep = [int(x) for x in range(int(num_classes)) if int(x) not in missing_set]

    return sorted({int(x) for x in keep if 0 <= int(x) < int(num_classes)})


def _missing_labels(keep_labels: List[int], *, num_classes: int) -> List[int]:
    keep = {int(x) for x in keep_labels}
    return [int(x) for x in range(int(num_classes)) if int(x) not in keep]


def _to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device=device, non_blocking=True)


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_items = 0
    for x, y in loader:
        x = _to_device(x, device)
        y = _to_device(y, device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        bs = int(x.shape[0])
        total_loss += float(loss.detach().cpu()) * bs
        total_items += bs
    return total_loss / float(max(1, total_items))


@torch.no_grad()
def eval_acc(*, model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = _to_device(x, device)
        y = _to_device(y, device)
        pred = torch.argmax(model(x), dim=1)
        correct += int((pred == y).sum().detach().cpu())
        total += int(y.numel())
    return 100.0 * float(correct) / float(max(1, total))


def _make_optimizer(cfg: Dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    opt_cfg = dict(cfg.get("optimizer", {}) or {})
    return torch.optim.SGD(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 0.01)),
        momentum=float(opt_cfg.get("momentum", 0.9)),
        weight_decay=float(opt_cfg.get("weight_decay", 5e-4)),
    )


def _mean_entropy_global(result: Dict[str, Any]) -> float:
    by_d = dict(result.get("entropy_summary", {}).get("by_d", {}) or {})
    vals = [float(dict(v).get("entropy_mean", 0.0)) for v in by_d.values()]
    if not vals:
        return float("nan")
    return float(sum(vals) / float(len(vals)))


def _q1_grid_rows(
    *,
    model_tag: str,
    group: str,
    result: Dict[str, Any],
    num_classes: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    q1 = dict(result.get("q1", {}) or {})
    by_grid = dict(q1.get("evidence_by_grid", {}) or {})
    rel_by_grid = list(q1.get("model_reliability_by_grid", []) or [])
    rel_by_d = {int(row.get("grid_hw", -1)): dict(row) for row in rel_by_grid}

    q_rows: List[Dict[str, Any]] = []
    rel_rows: List[Dict[str, Any]] = []
    for d_raw in sorted(by_grid.keys(), key=lambda x: int(x)):
        d = int(d_raw)
        rec = dict(by_grid[d_raw] or {})
        q_vals = list(rec.get("q", []))
        u_vals = list(rec.get("u", []))
        e_vals = list(rec.get("E", []))
        h_vals = list(rec.get("entropy_mean", []))
        counts = list(rec.get("counts", []))
        for label in range(int(num_classes)):
            q_rows.append(
                {
                    "model_tag": str(model_tag),
                    "group": str(group),
                    "grid_hw": int(d),
                    "label": int(label),
                    "q": float(q_vals[label]) if label < len(q_vals) else 0.0,
                    "count": int(counts[label]) if label < len(counts) else 0,
                    "entropy_mean": float(h_vals[label]) if label < len(h_vals) else 0.0,
                    "u": float(u_vals[label]) if label < len(u_vals) else 0.0,
                    "E": float(e_vals[label]) if label < len(e_vals) else 0.0,
                }
            )

        rel = dict(rel_by_d.get(int(d), {}))
        rel_rows.append(
            {
                "model_tag": str(model_tag),
                "group": str(group),
                "grid_hw": int(d),
                "reliability_mode": str(rel.get("reliability_mode", "diversity")),
                "model_reliability_d": float(rel.get("model_reliability_d", 0.0)),
                "diversity": float(rel.get("diversity", rel.get("model_reliability_d", 0.0))),
                "q_entropy": float(rel.get("q_entropy", 0.0)),
                "dominant_label": int(rel.get("dominant_label", -1)),
                "dominant_q": float(rel.get("dominant_q", 0.0)),
                "mean_entropy_norm": float(rel.get("mean_entropy_norm", 0.0)),
            }
        )
    return q_rows, rel_rows


def _plot_label_counts(rows: List[Dict[str, Any]], *, out_path: Path, num_classes: int) -> None:
    if not rows:
        return
    tags = [str(row["model_tag"]) for row in rows]
    mat = np.asarray([[int(row.get(f"train_count_c{c}", 0)) for c in range(int(num_classes))] for row in rows])
    fig, ax = plt.subplots(figsize=(max(8.0, 0.7 * num_classes + 2.0), max(3.5, 0.55 * len(tags) + 1.2)))
    im = ax.imshow(mat, aspect="auto", cmap="Blues")
    ax.set_xlabel("Training label")
    ax.set_ylabel("Model")
    ax.set_xticks(np.arange(int(num_classes)))
    ax.set_xticklabels([str(c) for c in range(int(num_classes))])
    ax.set_yticks(np.arange(len(tags)))
    ax.set_yticklabels(tags)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_q_or_entropy_grid(
    rows: List[Dict[str, Any]],
    *,
    out_path: Path,
    value_key: str,
    label: str,
    num_classes: int,
) -> None:
    if not rows:
        return
    row_keys = sorted({(str(r["model_tag"]), int(r["grid_hw"])) for r in rows}, key=lambda x: (x[0], x[1]))
    mat = np.zeros((len(row_keys), int(num_classes)), dtype=np.float32)
    lookup = {(str(r["model_tag"]), int(r["grid_hw"]), int(r["label"])): float(r.get(value_key, 0.0)) for r in rows}
    for ridx, (tag, d) in enumerate(row_keys):
        for c in range(int(num_classes)):
            mat[ridx, c] = float(lookup.get((tag, d, c), 0.0))

    fig, ax = plt.subplots(figsize=(max(8.0, 0.65 * num_classes + 2.0), max(4.0, 0.38 * len(row_keys) + 1.5)))
    im = ax.imshow(mat, aspect="auto", cmap="magma" if value_key == "q" else "viridis")
    ax.set_xlabel("Predicted class label")
    ax.set_ylabel("Model / grid")
    ax.set_xticks(np.arange(int(num_classes)))
    ax.set_xticklabels([str(c) for c in range(int(num_classes))])
    ax.set_yticks(np.arange(len(row_keys)))
    ax.set_yticklabels([f"{tag} d={d}" for tag, d in row_keys], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            text = f"{100.0 * float(mat[i, j]):.1f}%" if value_key == "q" else f"{float(mat[i, j]):.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label(str(label))
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_reliability(rows: List[Dict[str, Any]], *, out_path: Path) -> None:
    if not rows:
        return
    tags = sorted({str(row["model_tag"]) for row in rows})
    grid_hws = sorted({int(row["grid_hw"]) for row in rows})
    lookup = {(str(row["model_tag"]), int(row["grid_hw"])): float(row["model_reliability_d"]) for row in rows}

    x = np.arange(len(grid_hws))
    width = 0.78 / float(max(1, len(tags)))
    fig, ax = plt.subplots(figsize=(max(8.0, 1.1 * len(grid_hws) + 3.0), 4.5))
    for idx, tag in enumerate(tags):
        vals = [float(lookup.get((tag, d), 0.0)) for d in grid_hws]
        ax.bar(x + (idx - (len(tags) - 1) / 2.0) * width, vals, width=width, label=tag)
    ax.set_xlabel("Grid size d")
    ax.set_ylabel("r(fi) by grid")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in grid_hws])
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_reliability_average(rows: List[Dict[str, Any]], *, out_path: Path) -> None:
    if not rows:
        return
    paper_order = [
        ("iid_trained", "Trained"),
        ("miss_even", "Trained (missing-even)"),
        ("untrained", "Untrained"),
    ]
    present = {str(row["model_tag"]) for row in rows}
    tag_label_pairs = [(tag, label) for tag, label in paper_order if tag in present]
    if tag_label_pairs:
        tags = [tag for tag, _label in tag_label_pairs]
        display_labels = [label for _tag, label in tag_label_pairs]
    else:
        tags = sorted(present)
        display_labels = list(tags)
    means: List[float] = []
    stds: List[float] = []
    vals_by_tag: Dict[str, List[float]] = {}
    for tag in tags:
        vals = [
            float(row["model_reliability_d"])
            for row in rows
            if str(row["model_tag"]) == str(tag) and np.isfinite(float(row["model_reliability_d"]))
        ]
        vals_by_tag[str(tag)] = vals
        means.append(float(np.mean(vals)) if vals else 0.0)
        stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)

    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(7.0545, 2.4636))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="#4c78a8", edgecolor="#263746", linewidth=0.8)

    for idx, tag in enumerate(tags):
        vals = vals_by_tag.get(str(tag), [])
        if vals:
            jitter = np.linspace(-0.10, 0.10, len(vals)) if len(vals) > 1 else np.asarray([0.0])
            ax.scatter(
                np.full(len(vals), float(x[idx])) + jitter,
                vals,
                s=26,
                color="#f58518",
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        ax.text(
            bars[idx].get_x() + bars[idx].get_width() / 2.0,
            min(1.03, float(means[idx]) + float(stds[idx]) + 0.025),
            f"{float(means[idx]):.3f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
        )

    ax.set_ylabel(r"$r(f_i)$ prediction diversity", fontsize=11.5)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=0, ha="center", fontsize=11.0)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.set_ylim(0.0, 1.08)
    ax.grid(True, axis="y", alpha=0.25)
    fig.subplots_adjust(left=0.13, right=0.83, bottom=0.24, top=0.88)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", required=True, help="Path to exp8 yaml config")
    args = parser.parse_args()

    with open(args.yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = dict(cfg.get("experiment", {}) or {})
    data_cfg = dict(cfg.get("data", {}) or {})
    model_cfg = dict(cfg.get("model", {}) or {})
    train_cfg = dict(cfg.get("train", {}) or {})
    search_cfg = dict(cfg.get("search", cfg.get("pointillism_fl", {})) or {})

    exp_name = str(exp_cfg.get("name", "exp8_prediction_diversity"))
    log_root = Path(str(exp_cfg.get("log_root", "log_standalone/signature_series")))
    seed = int(exp_cfg.get("seed", 123))
    device = resolve_device(str(exp_cfg.get("device", "cuda")))
    dataset_name = str(data_cfg["dataset"])
    ds_spec = get_dataset_spec(dataset_name)
    num_classes = int(model_cfg.get("num_classes", ds_spec.num_classes))
    out_hw = int(ds_spec.default_size)

    out_dir = log_root / f"{exp_name}_{now_tag()}"
    ensure_dir(out_dir)
    logger = RunLogger(out_dir / "run.log")

    try:
        logger.log(f"[Experiment] {exp_name}")
        logger.log(f"[Output] {out_dir}")
        logger.log(f"[Dataset] {dataset_name}")
        logger.log(f"[Device] {device}")
        logger.log(f"[Seed] {seed}")
        save_json(cfg, out_dir / "config_snapshot.json")

        set_seed(seed)
        loader_cfg = LoaderCfg(
            dataset=dataset_name,
            data_root=str(data_cfg.get("data_root", "./data")),
            batch_size=int(data_cfg.get("batch_size", 256)),
            num_workers=int(data_cfg.get("num_workers", 4)),
            pin_memory=bool(data_cfg.get("pin_memory", True)),
            persistent_workers=bool(data_cfg.get("persistent_workers", True)),
            seed=int(seed),
            data_fraction=float(data_cfg.get("data_fraction", 1.0)),
        )
        base_train_ds = build_train_loader(loader_cfg).dataset
        test_loader = build_test_loaders(loader_cfg)["clean"]

        case_cfgs = list(cfg.get("cases", []) or _default_cases(num_classes))
        max_epochs = int(train_cfg.get("max_epochs", 20))

        model_infos: List[Dict[str, Any]] = []
        train_rows: List[Dict[str, Any]] = []

        logger.log("[Phase] build/train models")
        for case_idx, raw_case in enumerate(case_cfgs):
            case = dict(raw_case or {})
            tag = str(case.get("tag", f"case_{case_idx:02d}"))
            group = str(case.get("group", tag))
            train_enabled = bool(case.get("train", True))
            model_seed = int(case.get("seed", seed + case_idx))
            keep_labels = _case_keep_labels(case, num_classes=int(num_classes))
            missing = _missing_labels(keep_labels, num_classes=int(num_classes))

            set_seed(model_seed)
            model = build_random_init_model(
                model_cfg=model_cfg,
                dataset_name=dataset_name,
                device=device,
            )

            if train_enabled:
                train_ds = _subset_by_labels(base_train_ds, keep_labels)
            else:
                train_ds = base_train_ds

            train_counts = _label_counts(train_ds, num_classes=int(num_classes))
            train_loader = _make_loader(
                train_ds,
                batch_size=int(data_cfg.get("batch_size", 256)),
                shuffle=True,
                num_workers=int(data_cfg.get("num_workers", 4)),
                pin_memory=bool(data_cfg.get("pin_memory", True)),
                persistent_workers=bool(data_cfg.get("persistent_workers", True)),
                seed=int(model_seed),
            )

            logger.log(
                f"[Model] {tag} group={group} train={int(train_enabled)} "
                f"keep={keep_labels} missing={missing} n={len(train_ds)}"
            )

            final_loss = float("nan")
            if train_enabled:
                optimizer = _make_optimizer(cfg, model)
                for epoch in range(1, max_epochs + 1):
                    final_loss = train_one_epoch(
                        model=model,
                        loader=train_loader,
                        optimizer=optimizer,
                        device=device,
                    )
                    logger.log(f"[Train] {tag} epoch={epoch:04d} loss={final_loss:.6f}")
            else:
                logger.log(f"[Train] {tag} skipped; random initialization is the condition")

            test_acc = eval_acc(model=model, loader=test_loader, device=device)
            ckpt_path = out_dir / "checkpoints" / f"{tag}.pth"
            ensure_dir(ckpt_path.parent)
            torch.save(model.state_dict(), ckpt_path)
            logger.log(f"[Save] {tag}: {ckpt_path}")

            train_row = {
                "model_tag": tag,
                "group": group,
                "train_enabled": int(train_enabled),
                "seed": int(model_seed),
                "keep_labels": " ".join(str(x) for x in keep_labels),
                "missing_labels": " ".join(str(x) for x in missing),
                "train_size": int(len(train_ds)),
                "final_loss": float(final_loss),
                "test_acc": float(test_acc),
                "ckpt_path": str(ckpt_path),
            }
            for c in range(int(num_classes)):
                train_row[f"train_count_c{c}"] = int(train_counts[c])
            train_rows.append(train_row)
            model_infos.append(
                {
                    "tag": tag,
                    "group": group,
                    "model": model,
                    "train_row": train_row,
                }
            )

        write_csv(
            csv_path=out_dir / "exp8_training_label_distribution.csv",
            fieldnames=list(train_rows[0].keys()),
            rows=train_rows,
        )
        _plot_label_counts(train_rows, out_path=out_dir / "training_label_distribution.png", num_classes=int(num_classes))

        summary_rows: List[Dict[str, Any]] = []
        q_rows_all: List[Dict[str, Any]] = []
        rel_rows_all: List[Dict[str, Any]] = []

        logger.log("[Phase] one-round probe search")
        for info in model_infos:
            tag = str(info["tag"])
            group = str(info["group"])
            model = info["model"]
            model.eval()

            logger.log(f"[Probe] {tag}")
            result = run_pointillism_search(
                model=model,
                num_classes=int(num_classes),
                out_hw=int(out_hw),
                dataset=dataset_name,
                device=device,
                norm_cfg_fn=_norm_cfg,
                cfg=search_cfg,
                log_fn=logger.log,
            )
            save_signature_outputs(
                result=result,
                out_dir=out_dir / "models",
                model_tag=tag,
                out_hw=int(out_hw),
            )

            q_rows, rel_rows = _q1_grid_rows(
                model_tag=tag,
                group=group,
                result=result,
                num_classes=int(num_classes),
            )
            q_rows_all.extend(q_rows)
            rel_rows_all.extend(rel_rows)

            q1 = dict(result.get("q1", {}) or {})
            cov = dict(result.get("coverage_status", {}) or {})
            train_row = dict(info["train_row"])
            summary_rows.append(
                {
                    "model_tag": tag,
                    "group": group,
                    "train_enabled": int(train_row["train_enabled"]),
                    "seed": int(train_row["seed"]),
                    "keep_labels": str(train_row["keep_labels"]),
                    "missing_labels": str(train_row["missing_labels"]),
                    "train_size": int(train_row["train_size"]),
                    "test_acc": float(train_row["test_acc"]),
                    "model_reliability": float(q1.get("model_reliability", q1.get("R_f", 0.0))),
                    "R_f": float(q1.get("R_f", 0.0)),
                    "support_labels": " ".join(str(int(x)) for x in q1.get("support_labels", [])),
                    "effective_support_labels": " ".join(str(int(x)) for x in q1.get("effective_support_labels", [])),
                    "covered_labels": int(len(cov.get("covered_labels", []))),
                    "undercovered_labels": " ".join(str(int(x)) for x in cov.get("undercovered_labels", [])),
                    "mean_entropy_global": float(_mean_entropy_global(result)),
                    "num_grid_hws": int(len(result.get("grid_hws", []))),
                    "stop_reason": str(result.get("stop_reason", "")),
                    "elapsed_sec": float(result.get("elapsed_sec", 0.0)),
                }
            )
            logger.log(
                f"[Probe] {tag} r(fi)={float(summary_rows[-1]['model_reliability']):.4f} "
                f"covered={summary_rows[-1]['covered_labels']}/{num_classes}"
            )

        write_csv(
            csv_path=out_dir / "exp8_model_summary.csv",
            fieldnames=list(summary_rows[0].keys()),
            rows=summary_rows,
        )
        write_csv(
            csv_path=out_dir / "exp8_q_by_grid.csv",
            fieldnames=["model_tag", "group", "grid_hw", "label", "q", "count", "entropy_mean", "u", "E"],
            rows=q_rows_all,
        )
        write_csv(
            csv_path=out_dir / "exp8_prediction_diversity_by_grid.csv",
            fieldnames=[
                "model_tag",
                "group",
                "grid_hw",
                "reliability_mode",
                "model_reliability_d",
                "diversity",
                "q_entropy",
                "dominant_label",
                "dominant_q",
                "mean_entropy_norm",
            ],
            rows=rel_rows_all,
        )

        _plot_q_or_entropy_grid(
            q_rows_all,
            out_path=out_dir / "q_distribution_by_model_grid.png",
            value_key="q",
            label="q(d)",
            num_classes=int(num_classes),
        )
        _plot_q_or_entropy_grid(
            q_rows_all,
            out_path=out_dir / "classwise_entropy_by_model_grid.png",
            value_key="entropy_mean",
            label="Mean softmax entropy",
            num_classes=int(num_classes),
        )
        _plot_reliability(rel_rows_all, out_path=out_dir / "prediction_diversity_by_grid.png")
        _plot_reliability_average(rel_rows_all, out_path=out_dir / "prediction_diversity_average_across_grid.png")

        logger.log("[Done]")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
