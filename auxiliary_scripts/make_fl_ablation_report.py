#!/usr/bin/env python3
"""Build ablation-study CSV reports from FL series folders.

The script reads one or more ``S_<dataset>_<timestamp>_<attack>`` folders. Each
folder must contain ``master_results.csv`` as produced by ``main_fl.py`` and
should contain the copied ablation YAML config.

Example:

    python auxiliary_scripts/make_fl_ablation_report.py \
        log_fl/S_fmnist_20260428_155509_badnet \
        log_fl/S_fmnist_20260428_155522_badnet \
        --out-prefix log_fl/fmnist_ablation_mali040_alpha05 \
        --config-root configs/fl/fmnist_ablation

This writes:

    log_fl/fmnist_ablation_mali040_alpha05_records.csv
    log_fl/fmnist_ablation_mali040_alpha05_summary.csv
    log_fl/fmnist_ablation_mali040_alpha05_acc_organized.csv
    log_fl/fmnist_ablation_mali040_alpha05_asr_organized.csv
    log_fl/fmnist_ablation_mali040_alpha05_wide.csv
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


ATTACK_ORDER = ["badnet", "label_perturbation"]
VARIANT_ORDER = ["full_default", "p_only", "r_only", "t_only"]
VARIANT_LABELS = {
    "full_default": "Full/default (P+R+T)",
    "p_only": "P only",
    "r_only": "R only",
    "t_only": "T only",
}
SELECTION_TABLES = {
    "full_default": "full_default",
    "p_only": "bar_P",
    "r_only": "bar_R",
    "t_only": "bar_T",
}
RE_SEED = re.compile(r"_seed(\d+)(?:$|[^0-9])")
RE_REP = re.compile(r"_rep(\d+)(?:_|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ablation records/summary/organized/wide CSV files."
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
        help="Glob under --log-root. Can repeat.",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        required=True,
        help="Output prefix, e.g. log_fl/fmnist_ablation_mali040_alpha05.",
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

    seen: set[Path] = set()
    series_dirs: list[Path] = []
    for path in raw_paths:
        if path not in seen:
            seen.add(path)
            series_dirs.append(path)

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


def load_yaml(path: Path | None) -> dict[str, Any]:
    if yaml is None or path is None or not path.exists():
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
    return str(round(float(value), digits))


def sample_std(values: list[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    return stdev(values)


def mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def extract_seed(out_dir: str) -> str:
    match = RE_SEED.search(str(out_dir))
    return match.group(1) if match else ""


def extract_rep(out_dir: str, fallback_idx: int) -> str:
    match = RE_REP.search(str(out_dir))
    if match:
        return match.group(1)
    return str(fallback_idx)


def resolve_config_path(series_cfg: Path | None, config_root: Path | None) -> str:
    if series_cfg is None:
        return ""
    if config_root is not None:
        return str((config_root / series_cfg.name).resolve())
    return str(series_cfg.resolve())


def infer_variant(config_path: Path | None, cfg: dict[str, Any]) -> str:
    name = config_path.name if config_path is not None else ""
    for variant in VARIANT_ORDER:
        if variant in name:
            return variant

    selection_table = str(get_nested(cfg, ("pointillism_fl", "selection_table")) or "")
    for variant, table in SELECTION_TABLES.items():
        if selection_table == table:
            return variant
    return selection_table or "full_default"


def attack_type(attack: str) -> str:
    if attack == "label_perturbation":
        return "untargeted"
    return "backdoor"


def sort_key_for_record(row: dict[str, Any]) -> tuple[int, int, str, int, str]:
    attack = str(row.get("attack", ""))
    variant = str(row.get("variant", ""))
    attack_rank = ATTACK_ORDER.index(attack) if attack in ATTACK_ORDER else len(ATTACK_ORDER)
    variant_rank = VARIANT_ORDER.index(variant) if variant in VARIANT_ORDER else len(VARIANT_ORDER)
    rep = row.get("rep")
    rep_i = 0 if rep in ("", None) else int(rep)
    return (attack_rank, variant_rank, str(row.get("source_log", "")), rep_i, str(row.get("out_dir", "")))


def summary_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    attack = str(row.get("attack", ""))
    variant = str(row.get("variant", ""))
    attack_rank = ATTACK_ORDER.index(attack) if attack in ATTACK_ORDER else len(ATTACK_ORDER)
    variant_rank = VARIANT_ORDER.index(variant) if variant in VARIANT_ORDER else len(VARIANT_ORDER)
    return (attack_rank, variant_rank, str(row.get("source_logs", "")))


def build_records(series_dirs: list[Path], config_root: Path | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for series_dir in series_dirs:
        master_path = series_dir / "master_results.csv"
        master_rows = load_csv(master_path)
        series_cfg = find_series_config(series_dir)
        cfg = load_yaml(series_cfg)

        task = str(get_nested(cfg, ("task", "dataset")) or (master_rows[0].get("task", "") if master_rows else ""))
        attack = str(get_nested(cfg, ("attack", "atk_name")) or (master_rows[0].get("atk_name", "") if master_rows else ""))
        variant = infer_variant(series_cfg, cfg)
        selection_table = str(get_nested(cfg, ("pointillism_fl", "selection_table")) or SELECTION_TABLES.get(variant, variant))
        model = str(get_nested(cfg, ("model", "model_name")) or "")
        rounds = get_nested(cfg, ("train", "rounds"))
        lr = get_nested(cfg, ("train", "lr"))
        mali = get_nested(cfg, ("clients_setting", "mali_rate"))
        alpha = get_nested(cfg, ("clients_setting", "non_iid", "dirichlet_alpha"))
        config_path = resolve_config_path(series_cfg, config_root)

        point_rows = [row for row in master_rows if str(row.get("defense", "")).strip().lower() == "pointillism_fl"]
        for idx, master_row in enumerate(point_rows, start=1):
            out_dir = str(master_row.get("out_dir", ""))
            records.append(
                {
                    "attack": attack,
                    "attack_type": attack_type(attack),
                    "variant": variant,
                    "variant_label": VARIANT_LABELS.get(variant, variant),
                    "selection_table": selection_table,
                    "task": task,
                    "defense": "pointillism_fl",
                    "rep": extract_rep(out_dir, idx),
                    "seed": extract_seed(out_dir),
                    "acc": parse_float(master_row.get("acc")),
                    "asr": parse_float(master_row.get("asr")),
                    "mali_rate": parse_float(mali),
                    "dirichlet_alpha": parse_float(alpha),
                    "model": model,
                    "rounds": rounds,
                    "lr": lr,
                    "config_path": config_path,
                    "source_log": str(series_dir),
                    "source_master": str(master_path),
                    "out_dir": out_dir,
                }
            )

    records.sort(key=sort_key_for_record)
    return records


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


def group_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("attack", "")), str(row.get("variant", "")))


def metric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    return [float(row[metric]) for row in rows if row.get(metric) is not None]


def build_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(group_key(record), []).append(record)

    rows: list[dict[str, Any]] = []
    for (attack, variant), group in grouped.items():
        acc_values = metric_values(group, "acc")
        asr_values = metric_values(group, "asr")
        source_logs = sorted({str(row.get("source_log", "")) for row in group})
        first = group[0]
        rows.append(
            {
                "attack": attack,
                "attack_type": first.get("attack_type", ""),
                "variant": variant,
                "variant_label": first.get("variant_label", ""),
                "selection_table": first.get("selection_table", ""),
                "n": len(group),
                "acc_mean": mean(acc_values),
                "acc_std": sample_std(acc_values),
                "acc_min": min(acc_values) if acc_values else None,
                "acc_max": max(acc_values) if acc_values else None,
                "asr_mean": mean(asr_values),
                "asr_std": sample_std(asr_values),
                "asr_min": min(asr_values) if asr_values else None,
                "asr_max": max(asr_values) if asr_values else None,
                "mali_rate": first.get("mali_rate"),
                "dirichlet_alpha": first.get("dirichlet_alpha"),
                "source_logs": ";".join(source_logs),
            }
        )

    rows.sort(key=summary_sort_key)
    return rows


def get_summary_value(summary_rows: list[dict[str, Any]], attack: str, variant: str, metric: str) -> Any:
    for row in summary_rows:
        if row.get("attack") == attack and row.get("variant") == variant:
            return row.get(metric)
    return None


def attacks_in_order(summary_rows: list[dict[str, Any]]) -> list[str]:
    attacks = {str(row.get("attack", "")) for row in summary_rows}
    ordered = [attack for attack in ATTACK_ORDER if attack in attacks]
    ordered.extend(sorted(attacks - set(ordered)))
    return ordered


def build_organized(summary_rows: list[dict[str, Any]], metric_prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attack in attacks_in_order(summary_rows):
        for metric in [f"{metric_prefix}_mean", f"{metric_prefix}_std"]:
            row: dict[str, Any] = {"attack": attack, "metric": metric}
            for variant in VARIANT_ORDER:
                row[variant] = get_summary_value(summary_rows, attack, variant, metric)
            rows.append(row)
    return rows


def build_wide(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = ["acc_mean", "acc_std", "asr_mean", "asr_std", "n"]
    for attack in attacks_in_order(summary_rows):
        for metric in metrics:
            row: dict[str, Any] = {"attack": attack, "metric": metric}
            for variant in VARIANT_ORDER:
                row[variant] = get_summary_value(summary_rows, attack, variant, metric)
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    series_dirs = read_series_dirs(args)
    records = build_records(series_dirs, args.config_root)
    summary_rows = build_summary(records)
    acc_rows = build_organized(summary_rows, "acc")
    asr_rows = build_organized(summary_rows, "asr")
    wide_rows = build_wide(summary_rows)

    record_fields = [
        "attack",
        "attack_type",
        "variant",
        "variant_label",
        "selection_table",
        "task",
        "defense",
        "rep",
        "seed",
        "acc",
        "asr",
        "mali_rate",
        "dirichlet_alpha",
        "model",
        "rounds",
        "lr",
        "config_path",
        "source_log",
        "source_master",
        "out_dir",
    ]
    summary_fields = [
        "attack",
        "attack_type",
        "variant",
        "variant_label",
        "selection_table",
        "n",
        "acc_mean",
        "acc_std",
        "acc_min",
        "acc_max",
        "asr_mean",
        "asr_std",
        "asr_min",
        "asr_max",
        "mali_rate",
        "dirichlet_alpha",
        "source_logs",
    ]
    organized_fields = ["attack", "metric", *VARIANT_ORDER]

    out_prefix = args.out_prefix
    write_csv(out_prefix.with_name(out_prefix.name + "_records.csv"), record_fields, records, args.digits)
    write_csv(out_prefix.with_name(out_prefix.name + "_summary.csv"), summary_fields, summary_rows, args.digits)
    write_csv(out_prefix.with_name(out_prefix.name + "_acc_organized.csv"), organized_fields, acc_rows, args.digits)
    write_csv(out_prefix.with_name(out_prefix.name + "_asr_organized.csv"), organized_fields, asr_rows, args.digits)
    write_csv(out_prefix.with_name(out_prefix.name + "_wide.csv"), organized_fields, wide_rows, args.digits)

    print(f"Wrote {out_prefix}_records.csv")
    print(f"Wrote {out_prefix}_summary.csv")
    print(f"Wrote {out_prefix}_acc_organized.csv")
    print(f"Wrote {out_prefix}_asr_organized.csv")
    print(f"Wrote {out_prefix}_wide.csv")


if __name__ == "__main__":
    main()
