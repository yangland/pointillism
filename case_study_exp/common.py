#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import numpy as np
import torch

from data.registry import get_dataset_spec
from models import build_model
from pointillism.pointillism_signature_viz import (
    save_edge_case_examples,
    save_edge_case_summary_csv,
    save_entropy_summary_csv,
    save_entropy_visualizations,
    save_global_response_statistics_sg,
    save_json,
    save_missing_label_fallback_summary,
    save_sonar_histogram_panel,
    save_stage1_prob_tsne_plots,
    save_topl_examples,
    save_topl_summary_csv,
)


def set_seed(seed: int) -> None:
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_device(device_str: str) -> torch.device:
    requested = str(device_str).strip()
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if requested == "cuda":
            return torch.device("cuda")
        try:
            idx = int(requested.split(":", 1)[1])
        except (IndexError, ValueError):
            return torch.device("cuda")
        if idx >= torch.cuda.device_count():
            return torch.device("cpu")
    return torch.device(requested)


class RunLogger:
    def __init__(self, log_path: Path):
        self.f = open(log_path, "w", buffering=1)

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def _load_state_dict_any(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        return ckpt
    raise ValueError("Unsupported checkpoint format")


def build_model_from_cfg(
    *,
    model_cfg: Dict[str, Any],
    dataset_name: str,
    device: torch.device,
) -> torch.nn.Module:
    ds_spec = get_dataset_spec(str(dataset_name))
    model = build_model(
        arch=str(model_cfg["arch"]),
        num_classes=int(model_cfg["num_classes"]),
        in_channels=int(model_cfg.get("in_channels", ds_spec.in_channels)),
    ).to(device)
    model.eval()
    return model


def load_model_from_ckpt(
    *,
    ckpt_path: str,
    model_cfg: Dict[str, Any],
    dataset_name: str,
    device: torch.device,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    model = build_model_from_cfg(
        model_cfg=model_cfg,
        dataset_name=dataset_name,
        device=device,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _load_state_dict_any(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=True)
    model.eval()
    return model, {"missing": list(missing), "unexpected": list(unexpected)}


def build_random_init_model(
    *,
    model_cfg: Dict[str, Any],
    dataset_name: str,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model_from_cfg(
        model_cfg=model_cfg,
        dataset_name=dataset_name,
        device=device,
    )
    model.eval()
    return model


def save_signature_outputs(
    *,
    result: Dict[str, Any],
    out_dir: Path,
    model_tag: str,
    out_hw: int,
) -> None:
    model_dir = out_dir if out_dir.name == str(model_tag) else out_dir / model_tag
    ensure_dir(model_dir)

    save_json(result.get("sonar_map", {}), model_dir / "sonar_map.json")
    save_json(result.get("coverage_status", {}), model_dir / "coverage_status.json")
    save_json(result.get("search_trace", []), model_dir / "search_trace.json")
    save_json(
        {
            "stop_reason": result.get("stop_reason", ""),
            "elapsed_sec": result.get("elapsed_sec", 0.0),
        },
        model_dir / "run_summary.json",
    )

    torch.save(result.get("topl_by_label", {}), model_dir / "topl_by_label.pt")

    save_topl_summary_csv(
        topl_by_label=result.get("topl_by_label", {}),
        csv_path=model_dir / "topl_summary.csv",
    )
    save_missing_label_fallback_summary(
        topl_by_label=result.get("topl_by_label", {}),
        grid_hws=result.get("grid_hws", []),
        out_json=model_dir / "missing_label_fallback_summary.json",
        out_csv=model_dir / "missing_label_fallback_summary.csv",
    )

    save_json(result.get("entropy_summary", {}), model_dir / "entropy_summary.json")
    save_json(result.get("edge_cases", {}), model_dir / "edge_cases.json")
    save_json(result.get("q1", {}), model_dir / "q1_summary.json")

    save_entropy_summary_csv(
        entropy_summary=result.get("entropy_summary", {}),
        csv_path=model_dir / "entropy_summary.csv",
    )
    save_edge_case_summary_csv(
        edge_cases=result.get("edge_cases", {}),
        csv_path=model_dir / "edge_case_summary.csv",
    )

    save_sonar_histogram_panel(
        result=result,
        out_path=model_dir / "sonar_class_count_histograms.png",
        model_tag=model_tag,
    )
    save_stage1_prob_tsne_plots(
        result=result,
        out_dir=model_dir / "sonar_tsne",
        model_tag=model_tag,
    )
    save_topl_examples(
        topl_by_label=result.get("topl_by_label", {}),
        out_dir=model_dir / "topl_examples",
        out_hw=int(out_hw),
        num_examples=20,
    )
    save_entropy_visualizations(
        result=result,
        out_dir=model_dir / "entropy_viz",
        model_tag=model_tag,
    )
    save_global_response_statistics_sg(
        result=result,
        out_dir=model_dir / "global_response_statistics_Sg",
        model_tag=model_tag,
    )
    save_edge_case_examples(
        result=result,
        out_dir=model_dir / "edge_case_examples",
        out_hw=int(out_hw),
        max_pairs=10,
        num_examples_per_pair=20,
    )


def write_csv(
    *,
    csv_path: Path,
    fieldnames: List[str],
    rows: List[Dict[str, Any]],
) -> None:
    ensure_dir(csv_path.parent)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
