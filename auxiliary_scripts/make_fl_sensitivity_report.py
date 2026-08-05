#!/usr/bin/env python3
"""Build sensitivity ACC/ASR CSV reports from FL series folders.

The script reads one or more ``S_<dataset>_<timestamp>_<attack>`` folders. Each
folder must contain ``master_results.csv`` as produced by ``main_fl.py`` and
should contain the copied YAML config for metadata.

Example:

    python auxiliary_scripts/make_fl_sensitivity_report.py \
        --log-root log_fl \
        --glob 'S_fmnist_*_badnet' \
        --out-prefix log_fl/fmnist_sensitivity \
        --config-root configs/fl/fmnist_sensitivity

This writes:

    log_fl/fmnist_sensitivity_records.csv
    log_fl/fmnist_sensitivity_summary.csv
    log_fl/fmnist_sensitivity_acc_organized.csv
    log_fl/fmnist_sensitivity_asr_organized.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import stdev
from typing import Any, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - only used when PyYAML is unavailable.
    yaml = None


DISPLAY_DEFENSES = {
    "fedavg": "FedAvg",
    "normbound": "NormBound",
    "rfa": "RFA",
    "krum": "Krum",
    "flame": "FLAME",
    "pointillism_fl": "Pointillism",
}
SUMMARY_DEFENSES = ["krum", "flame"]
RE_SEED = re.compile(r"_seed(\d+)(?:$|[^0-9])")
RE_REP = re.compile(r"_rep(\d+)(?:_|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sensitivity records/summary/ACC/ASR CSV files."
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="*",
        help="Series directories containing master_results.csv.",
    )
    parser.add_argument(
        "--links-file",
        type=Path,
        help="Optional text file with one series directory per line.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("log_fl"),
        help="Root used with --glob. Default: log_fl",
    )
    parser.add_argument(
        "--glob",
        action="append",
        dest="globs",
        help="Glob under --log-root, e.g. 'S_fmnist_*_badnet'. Can repeat.",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        required=True,
        help="Output prefix, e.g. log_fl/fmnist_sensitivity.",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        help=(
            "Optional canonical config directory. If set, config_path is written "
            "as config-root/<copied-yaml-name> instead of the series copy."
        ),
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=4,
        help="Decimal digits for rounded report values. Default: 4",
    )
    parser.add_argument(
        "--require-defenses",
        default="krum,flame",
        help=(
            "Comma-separated defenses each selected series must contain. "
            "Default: krum,flame. Use '' to keep incomplete series."
        ),
    )
    return parser.parse_args()


def read_series_dirs(args: argparse.Namespace) -> list[Path]:
    raw_paths: list[Path] = [Path(p) for p in args.experiment_dirs]

    if args.links_file:
        for raw in args.links_file.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                raw_paths.append(Path(line))

    if args.globs:
        for pattern in args.globs:
            raw_paths.extend(sorted(args.log_root.glob(pattern)))

    # Preserve first occurrence while removing duplicates.
    seen: set[Path] = set()
    series_dirs: list[Path] = []
    for path in raw_paths:
        norm = path
        if norm not in seen:
            seen.add(norm)
            series_dirs.append(norm)

    if not series_dirs:
        raise SystemExit("No experiment directories were provided.")

    missing = [p for p in series_dirs if not (p / "master_results.csv").exists()]
    if missing:
        joined = "\n".join(str(p) for p in missing[:10])
        extra = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
        raise SystemExit(f"Missing master_results.csv in:\n{joined}{extra}")

    return series_dirs


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    with path.open() as fh:
        loaded = yaml.safe_load(fh) or {}
    return loaded if isinstance(loaded, dict) else {}


def find_series_config(series_dir: Path) -> Path | None:
    non_frozen = sorted(
        p
        for p in series_dir.glob("*.yaml")
        if not p.name.endswith("_frozen.yaml") and p.name != "cfg_effective.yaml"
    )
    if non_frozen:
        return non_frozen[0]
    frozen = sorted(series_dir.glob("*_frozen.yaml"))
    return frozen[0] if frozen else None


def get_nested(data: dict[str, Any], path: Iterable[str]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def parse_float(text: Any) -> float | None:
    if text is None or text == "":
        return None
    value = float(text)
    if math.isnan(value):
        return None
    return value


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    value_f = float(value)
    return str(round(value_f, digits))


def fmt_fixed(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def sample_std(values: list[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    return stdev(values)


def extract_seed(out_dir: str) -> str:
    match = RE_SEED.search(str(out_dir))
    return match.group(1) if match else ""


def extract_rep(out_dir: str, defense: str, counts: dict[str, int]) -> str:
    if defense != "pointillism_fl":
        return ""
    match = RE_REP.search(str(out_dir))
    if match:
        return match.group(1)
    counts[defense] = counts.get(defense, 0) + 1
    return str(counts[defense])


def display_defense(defense: str) -> str:
    return DISPLAY_DEFENSES.get(defense, defense)


def resolve_config_path(series_cfg: Path | None, config_root: Path | None) -> str:
    if series_cfg is None:
        return ""
    if config_root is not None:
        return str((config_root / series_cfg.name).resolve())
    return str(series_cfg.resolve())


def series_sort_key(item: tuple[dict[str, Any], list[dict[str, Any]]]) -> tuple[float, float, str]:
    meta, _ = item
    mali = meta.get("mali_rate")
    alpha = meta.get("dirichlet_alpha")
    return (
        float("inf") if mali is None else float(mali),
        float("inf") if alpha is None else float(alpha),
        str(meta.get("source_log", "")),
    )


def record_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    defense = str(row.get("defense", ""))
    rank = {"flame": 0, "krum": 1, "pointillism_fl": 2}.get(defense, 99)
    rep = row.get("rep")
    rep_i = 0 if rep in ("", None) else int(rep)
    return (rank, rep_i, defense)


def build_records(
    series_dirs: list[Path],
    config_root: Path | None,
    require_defenses: set[str],
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], list[dict[str, Any]]]]]:
    all_series: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    all_records: list[dict[str, Any]] = []

    for series_dir in series_dirs:
        master_path = series_dir / "master_results.csv"
        master_rows = load_csv(master_path)
        series_cfg = find_series_config(series_dir)
        cfg = load_yaml(series_cfg) if series_cfg is not None else {}

        task = str(get_nested(cfg, ("task", "dataset")) or (master_rows[0].get("task", "") if master_rows else ""))
        attack = str(get_nested(cfg, ("attack", "atk_name")) or (master_rows[0].get("atk_name", "") if master_rows else ""))
        model = str(get_nested(cfg, ("model", "model_name")) or "")
        rounds = get_nested(cfg, ("train", "rounds"))
        lr = get_nested(cfg, ("train", "lr"))
        mali = get_nested(cfg, ("clients_setting", "mali_rate"))
        alpha = get_nested(cfg, ("clients_setting", "non_iid", "dirichlet_alpha"))
        setting = f"mali={fmt(mali, 4)}, alpha={fmt(alpha, 4)}"
        config_path = resolve_config_path(series_cfg, config_root)

        meta = {
            "setting": setting,
            "mali_rate": parse_float(mali),
            "dirichlet_alpha": parse_float(alpha),
            "task": task,
            "attack": attack,
            "model": model,
            "rounds": rounds,
            "lr": lr,
            "config_path": config_path,
            "source_log": str(series_dir),
            "source_master": str(master_path),
        }

        rep_counts: dict[str, int] = {}
        series_records: list[dict[str, Any]] = []
        for master_row in master_rows:
            defense = str(master_row.get("defense", "")).strip().lower()
            out_dir = str(master_row.get("out_dir", ""))
            record = {
                **meta,
                "defense": defense,
                "display_defense": display_defense(defense),
                "rep": extract_rep(out_dir, defense, rep_counts),
                "seed": extract_seed(out_dir),
                "acc": parse_float(master_row.get("acc")),
                "asr": parse_float(master_row.get("asr")),
                "out_dir": out_dir,
            }
            series_records.append(record)

        seen_defenses = {str(row.get("defense", "")).lower() for row in series_records}
        if require_defenses and not require_defenses.issubset(seen_defenses):
            continue

        series_records.sort(key=record_sort_key)
        all_series.append((meta, series_records))
        all_records.extend(series_records)

    all_series.sort(key=series_sort_key)
    all_records = [record for _, rows in all_series for record in rows]
    return all_records, all_series


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]], digits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, float):
                    out[key] = fmt(value, digits)
                else:
                    out[key] = value
            writer.writerow(out)


def values_for(rows: list[dict[str, Any]], defense: str, metric: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        if row.get("defense") == defense and row.get(metric) is not None:
            vals.append(float(row[metric]))
    return vals


def first_for(rows: list[dict[str, Any]], defense: str, metric: str) -> float | None:
    vals = values_for(rows, defense, metric)
    return vals[0] if vals else None


def build_summary_rows(
    all_series: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    digits: int,
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for meta, rows in all_series:
        point_acc = values_for(rows, "pointillism_fl", "acc")
        point_asr = values_for(rows, "pointillism_fl", "asr")
        summary_rows.append(
            {
                "setting": meta.get("setting", ""),
                "mali_rate": meta.get("mali_rate"),
                "dirichlet_alpha": meta.get("dirichlet_alpha"),
                "Krum_acc": first_for(rows, "krum", "acc"),
                "Krum_asr": first_for(rows, "krum", "asr"),
                "FLAME_acc": first_for(rows, "flame", "acc"),
                "FLAME_asr": first_for(rows, "flame", "asr"),
                "Pointillism_acc_mean": (sum(point_acc) / len(point_acc)) if point_acc else None,
                "Pointillism_acc_std": sample_std(point_acc),
                "Pointillism_asr_mean": (sum(point_asr) / len(point_asr)) if point_asr else None,
                "Pointillism_asr_std": sample_std(point_asr),
                "Pointillism_acc_values": ";".join(fmt_fixed(v, digits) for v in point_acc),
                "Pointillism_asr_values": ";".join(fmt_fixed(v, digits) for v in point_asr),
                "source_log": meta.get("source_log", ""),
            }
        )
    return summary_rows


def build_metric_rows(
    all_series: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    metric: str,
) -> list[dict[str, Any]]:
    organized_rows: list[dict[str, Any]] = []
    max_pointillism = 0
    per_series_values: list[tuple[dict[str, Any], list[float], dict[str, float | None]]] = []

    for meta, rows in all_series:
        point_values = values_for(rows, "pointillism_fl", metric)
        max_pointillism = max(max_pointillism, len(point_values))
        per_series_values.append(
            (
                meta,
                point_values,
                {
                    "Krum": first_for(rows, "krum", metric),
                    "FLAME": first_for(rows, "flame", metric),
                },
            )
        )

    for meta, point_values, base_values in per_series_values:
        row: dict[str, Any] = {
            "setting": meta.get("setting", ""),
            "mali_rate": meta.get("mali_rate"),
            "dirichlet_alpha": meta.get("dirichlet_alpha"),
            "Krum": base_values.get("Krum"),
            "FLAME": base_values.get("FLAME"),
        }
        for idx in range(max_pointillism):
            row[f"Pointillism{idx + 1}"] = point_values[idx] if idx < len(point_values) else None
        row["Pointillism_mean"] = (sum(point_values) / len(point_values)) if point_values else None
        row["Pointillism_std"] = sample_std(point_values)
        row["source_log"] = meta.get("source_log", "")
        organized_rows.append(row)

    return organized_rows


def main() -> None:
    args = parse_args()
    series_dirs = read_series_dirs(args)
    require_defenses = {
        item.strip().lower()
        for item in str(args.require_defenses).split(",")
        if item.strip()
    }
    all_records, all_series = build_records(series_dirs, args.config_root, require_defenses)

    record_fields = [
        "setting",
        "mali_rate",
        "dirichlet_alpha",
        "task",
        "attack",
        "defense",
        "display_defense",
        "rep",
        "seed",
        "acc",
        "asr",
        "model",
        "rounds",
        "lr",
        "config_path",
        "source_log",
        "source_master",
        "out_dir",
    ]
    summary_fields = [
        "setting",
        "mali_rate",
        "dirichlet_alpha",
        "Krum_acc",
        "Krum_asr",
        "FLAME_acc",
        "FLAME_asr",
        "Pointillism_acc_mean",
        "Pointillism_acc_std",
        "Pointillism_asr_mean",
        "Pointillism_asr_std",
        "Pointillism_acc_values",
        "Pointillism_asr_values",
        "source_log",
    ]

    acc_rows = build_metric_rows(all_series, "acc")
    asr_rows = build_metric_rows(all_series, "asr")
    metric_fields = list(acc_rows[0].keys()) if acc_rows else [
        "setting",
        "mali_rate",
        "dirichlet_alpha",
        "Krum",
        "FLAME",
        "Pointillism_mean",
        "Pointillism_std",
        "source_log",
    ]

    out_prefix = args.out_prefix
    write_csv(out_prefix.with_name(out_prefix.name + "_records.csv"), record_fields, all_records, args.digits)
    write_csv(out_prefix.with_name(out_prefix.name + "_summary.csv"), summary_fields, build_summary_rows(all_series, args.digits), args.digits)
    write_csv(out_prefix.with_name(out_prefix.name + "_acc_organized.csv"), metric_fields, acc_rows, args.digits)
    write_csv(out_prefix.with_name(out_prefix.name + "_asr_organized.csv"), metric_fields, asr_rows, args.digits)

    print(f"Wrote {out_prefix}_records.csv")
    print(f"Wrote {out_prefix}_summary.csv")
    print(f"Wrote {out_prefix}_acc_organized.csv")
    print(f"Wrote {out_prefix}_asr_organized.csv")


if __name__ == "__main__":
    main()
