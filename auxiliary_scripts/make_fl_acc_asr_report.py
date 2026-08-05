#!/usr/bin/env python3
"""Build spreadsheet-style ACC/ASR reports from FL series folders.

The script expects one or more experiment-series directories, each containing a
``master_results.csv`` with columns:

    task, atk_name, defense, acc, asr, out_dir

Example:

    python auxiliary_scripts/make_fl_acc_asr_report.py \
        log_fl/S_cifar10_20260425_230812_badnet \
        log_fl/S_cifar10_20260425_230826_dba \
        --title baby01 \
        --out log_fl/cifar10_acc_asr_report.csv

You can also pass a text file with one experiment path per line:

    python auxiliary_scripts/make_fl_acc_asr_report.py \
        --links-file my_experiments.txt \
        --out log_fl/report.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - only used when PyYAML is unavailable.
    yaml = None


DEFENSE_ORDER = ["fedavg", "normbound", "rfa", "krum", "flame"]
DEFENSE_LABELS = {
    "fedavg": "FedAvg",
    "normbound": "NormBound",
    "rfa": "RFA",
    "krum": "Krum",
    "flame": "FLAME",
}
ATTACK_LABELS = {
    "a3fl": "A3FL",
    "badnet": "BadNet",
    "dba": "DBA",
    "label_flipping": "LabelFlipp",
    "label_perturbation": "LabelPerturb",
    "neurotoxin": "Neurotoxin",
    "reverse_grad_sign": "ReverseGrad",
    "scale": "Scale",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an ACC/ASR report from FL experiment links."
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="*",
        help="Experiment-series directories containing master_results.csv.",
    )
    parser.add_argument(
        "--links-file",
        type=Path,
        help="Optional text file with one experiment-series directory per line.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("fl_acc_asr_report.csv"),
        help="Output CSV path. Default: fl_acc_asr_report.csv",
    )
    parser.add_argument(
        "--title",
        default="report",
        help="First cell of the first row, e.g. baby01. Default: report",
    )
    parser.add_argument(
        "--pointillism-count",
        type=int,
        default=4,
        help="Minimum number of pointillism columns to include. Default: 4",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Decimal digits for numeric cells. Default: 2",
    )
    parser.add_argument(
        "--no-meta",
        action="store_true",
        help="Do not add the dataset/alpha/mali/lr/model metadata row.",
    )
    return parser.parse_args()


def read_links(args: argparse.Namespace) -> list[Path]:
    links: list[str] = []
    links.extend(args.experiment_dirs)
    if args.links_file:
        for raw in args.links_file.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                links.append(line)

    if not links:
        raise SystemExit("No experiment directories were provided.")

    return [Path(link) for link in links]


def read_master_csv(series_dir: Path) -> list[dict[str, str]]:
    csv_path = series_dir / "master_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing master_results.csv: {csv_path}")

    with csv_path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def load_series_config(series_dir: Path) -> dict[str, Any]:
    if yaml is None:
        return {}

    frozen = sorted(series_dir.glob("*_frozen.yaml"))
    candidates = frozen or sorted(series_dir.glob("*.yaml"))
    if not candidates:
        return {}

    with candidates[0].open() as fh:
        loaded = yaml.safe_load(fh) or {}
    return loaded if isinstance(loaded, dict) else {}


def get_nested(data: dict[str, Any], path: Iterable[str]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def format_model_name(name: Any) -> str:
    if name is None:
        return ""
    text = str(name)
    known = {"resnet20": "ResNet20", "resnet18": "ResNet18", "vgg11": "VGG11"}
    return known.get(text.lower(), text)


def infer_metadata(series_dirs: list[Path], all_rows: list[list[dict[str, str]]]) -> list[str]:
    first_rows = next((rows for rows in all_rows if rows), [])
    dataset = first_rows[0].get("task", "") if first_rows else ""

    cfg = {}
    for series_dir in series_dirs:
        cfg = load_series_config(series_dir)
        if cfg:
            break

    if not dataset:
        dataset = str(get_nested(cfg, ("task", "dataset")) or "")

    alpha = get_nested(cfg, ("clients_setting", "non_iid", "dirichlet_alpha"))
    mali = get_nested(cfg, ("clients_setting", "mali_rate"))
    lr = get_nested(cfg, ("train", "lr"))
    model = format_model_name(get_nested(cfg, ("model", "model_name")))

    return [
        dataset,
        f"alpha={alpha}" if alpha is not None else "",
        f"mali={mali}" if mali is not None else "",
        f"lr={lr}" if lr is not None else "",
        model,
    ]


def attack_label(raw_name: str, series_dir: Path) -> str:
    if raw_name:
        return ATTACK_LABELS.get(raw_name, raw_name)

    # Fallback for empty CSVs: S_<dataset>_<timestamp>_<attack>
    parts = series_dir.name.split("_", 3)
    if len(parts) == 4:
        return ATTACK_LABELS.get(parts[3], parts[3])
    return series_dir.name


def parse_float(text: str | None) -> float | None:
    if text is None or text == "":
        return None
    value = float(text)
    if math.isnan(value):
        return None
    return value


def format_number(value: float | None, digits: int) -> str:
    if value is None:
        return ""
    rounded = round(value, digits)
    text = f"{rounded:.{digits}f}"
    return text.rstrip("0").rstrip(".")


def build_attack_records(
    series_dirs: list[Path], all_rows: list[list[dict[str, str]]]
) -> tuple[OrderedDict[str, dict[str, dict[str, float | None]]], int]:
    attacks: OrderedDict[str, dict[str, dict[str, float | None]]] = OrderedDict()
    max_pointillism = 0

    for series_dir, rows in zip(series_dirs, all_rows):
        raw_attack = rows[0].get("atk_name", "") if rows else ""
        label = attack_label(raw_attack, series_dir)
        acc: dict[str, float | None] = {}
        asr: dict[str, float | None] = {}
        pointillism_idx = 0

        for row in rows:
            defense = (row.get("defense") or "").strip().lower()
            if defense == "pointillism_fl":
                pointillism_idx += 1
                col = f"pointillism{pointillism_idx}"
            else:
                col = defense

            acc[col] = parse_float(row.get("acc"))
            asr[col] = parse_float(row.get("asr"))

        max_pointillism = max(max_pointillism, pointillism_idx)
        attacks[label] = {"acc": acc, "asr": asr}

    return attacks, max_pointillism


def build_metric_rows(
    attacks: OrderedDict[str, dict[str, dict[str, float | None]]],
    metric: str,
    columns: list[str],
    digits: int,
) -> list[list[str]]:
    rows = [["Attack", *[DEFENSE_LABELS.get(col, col) for col in columns]]]
    for attack, values_by_metric in attacks.items():
        values = values_by_metric[metric]
        rows.append(
            [attack, *[format_number(values.get(col), digits) for col in columns]]
        )
    return rows


def write_report(
    out_path: Path,
    title: str,
    series_dirs: list[Path],
    metadata: list[str] | None,
    acc_rows: list[list[str]],
    asr_rows: list[list[str]],
) -> None:
    width = max(len(acc_rows[0]), len(asr_rows[0]), 2)

    def padded(row: list[str]) -> list[str]:
        return row + [""] * (width - len(row))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(padded([title, str(series_dirs[0])]))
        if metadata:
            writer.writerow(padded(metadata))
        writer.writerow([])
        writer.writerow(padded(["ACC"]))
        writer.writerows(padded(row) for row in acc_rows)
        writer.writerow([])
        writer.writerow(padded(["ASR"]))
        writer.writerows(padded(row) for row in asr_rows)


def main() -> None:
    args = parse_args()
    series_dirs = read_links(args)
    all_rows = [read_master_csv(series_dir) for series_dir in series_dirs]

    attacks, observed_pointillism_count = build_attack_records(series_dirs, all_rows)
    pointillism_count = max(args.pointillism_count, observed_pointillism_count)
    columns = [
        *DEFENSE_ORDER,
        *[f"pointillism{i}" for i in range(1, pointillism_count + 1)],
    ]

    metadata = None if args.no_meta else infer_metadata(series_dirs, all_rows)
    acc_rows = build_metric_rows(attacks, "acc", columns, args.digits)
    asr_rows = build_metric_rows(attacks, "asr", columns, args.digits)
    write_report(args.out, args.title, series_dirs, metadata, acc_rows, asr_rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
