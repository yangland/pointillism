from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import torch
import numpy as np

from pointillism.pointillism_signature_viz import (
    compute_topl_transfer_target_softmax,
    js_divergence,
    safe_cosine,
    sonar_ratio_matrix,
    _score_records_grouped_by_grid,
)


# ------------------------------------------------------------
# generic helpers
# ------------------------------------------------------------

def safe_mean(values: Iterable[float], default: float = float("nan")) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return float(default)
    return float(sum(vals) / float(len(vals)))


def pair_type(domain_a: str, domain_b: str) -> str:
    da = str(domain_a).lower()
    db = str(domain_b).lower()
    if da == db == "mnist":
        return "within_mnist"
    if da == db == "fmnist":
        return "within_fmnist"
    return "cross_domain"


def iter_class_pairs(num_classes: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for i in range(int(num_classes)):
        for j in range(i + 1, int(num_classes)):
            out.append((int(i), int(j)))
    return out


def pair_key(i: int, j: int) -> str:
    a = int(min(i, j))
    b = int(max(i, j))
    return f"{a}_{b}"


def parse_pair_key(key: Any) -> Optional[Tuple[int, int]]:
    if isinstance(key, tuple) and len(key) == 2:
        try:
            a = int(key[0])
            b = int(key[1])
            if a == b:
                return None
            return (min(a, b), max(a, b))
        except Exception:
            return None

    s = str(key).strip()
    if not s:
        return None

    for sep in ["_", "-", ",", "|", " "]:
        parts = [p for p in s.split(sep) if p != ""]
        if len(parts) == 2:
            try:
                a = int(parts[0])
                b = int(parts[1])
                if a == b:
                    return None
                return (min(a, b), max(a, b))
            except Exception:
                pass
    return None


def js_on_prob_vectors(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)

    p = p / (np.sum(p) + 1e-12)
    q = q / (np.sum(q) + 1e-12)
    return float(js_divergence(p, q))


def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sum(np.abs(a - b)))


def common_grid_hws(result_a: Dict[str, Any], result_b: Dict[str, Any]) -> List[int]:
    a = {int(x) for x in result_a.get("grid_hws", [])}
    b = {int(x) for x in result_b.get("grid_hws", [])}
    return sorted(a.intersection(b))


# ------------------------------------------------------------
# territory / sonar helpers
# ------------------------------------------------------------

def compute_label_presence_from_sonar(
    sonar_map: Dict[str, Any],
    num_classes: int,
) -> np.ndarray:
    acc = np.zeros(int(num_classes), dtype=np.float64)
    count = 0

    for _d, info in sonar_map.items():
        ratios = np.asarray(info.get("class_ratios", []), dtype=np.float64)
        if ratios.shape[0] != int(num_classes):
            continue
        acc += ratios
        count += 1

    if count > 0:
        acc /= float(count)
    return acc


def compute_territory_from_result(
    result: Dict[str, Any],
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sonar_presence = compute_label_presence_from_sonar(
        result.get("sonar_map", {}),
        num_classes=int(num_classes),
    )
    territory = sonar_presence / (np.sum(sonar_presence) + 1e-8)
    return sonar_presence, territory


def compute_pairwise_signature_metrics(
    *,
    case_a: str,
    case_b: str,
    domain_a: str,
    domain_b: str,
    result_a: Dict[str, Any],
    result_b: Dict[str, Any],
    territory_a: np.ndarray,
    territory_b: np.ndarray,
    num_classes: int,
) -> Dict[str, Any]:
    grid_hws = common_grid_hws(result_a, result_b)

    if grid_hws:
        mat_a = sonar_ratio_matrix(
            result_a.get("sonar_map", {}),
            grid_hws=grid_hws,
            num_classes=int(num_classes),
        )
        mat_b = sonar_ratio_matrix(
            result_b.get("sonar_map", {}),
            grid_hws=grid_hws,
            num_classes=int(num_classes),
        )
        delta = mat_b - mat_a
        sonar_flat_cosine = float(safe_cosine(mat_a.reshape(-1), mat_b.reshape(-1)))
        sonar_flat_js = float(js_divergence(mat_a.reshape(-1), mat_b.reshape(-1)))
        sonar_mean_abs_delta = float(np.mean(np.abs(delta)))
        sonar_max_abs_delta = float(np.max(np.abs(delta)))
    else:
        sonar_flat_cosine = float("nan")
        sonar_flat_js = float("nan")
        sonar_mean_abs_delta = float("nan")
        sonar_max_abs_delta = float("nan")

    return {
        "case_a": str(case_a),
        "case_b": str(case_b),
        "domain_a": str(domain_a),
        "domain_b": str(domain_b),
        "pair_type": pair_type(domain_a, domain_b),
        "territory_l1": float(np.sum(np.abs(territory_a - territory_b))),
        "territory_js": float(js_on_prob_vectors(territory_a, territory_b)),
        "territory_cosine": float(safe_cosine(territory_a, territory_b)),
        "sonar_flat_cosine": sonar_flat_cosine,
        "sonar_flat_js": sonar_flat_js,
        "sonar_mean_abs_delta": sonar_mean_abs_delta,
        "sonar_max_abs_delta": sonar_max_abs_delta,
    }


def compute_experiment_pairwise_signature_rows(
    *,
    case_names: List[str],
    case_results: Dict[str, Dict[str, Any]],
    num_classes: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for case_a, case_b in combinations(case_names, 2):
        result_a = case_results[case_a]["result"]
        result_b = case_results[case_b]["result"]
        territory_a = np.asarray(case_results[case_a]["territory"], dtype=np.float64)
        territory_b = np.asarray(case_results[case_b]["territory"], dtype=np.float64)

        row = compute_pairwise_signature_metrics(
            case_a=case_a,
            case_b=case_b,
            domain_a=str(case_results[case_a]["domain"]),
            domain_b=str(case_results[case_b]["domain"]),
            result_a=result_a,
            result_b=result_b,
            territory_a=territory_a,
            territory_b=territory_b,
            num_classes=int(num_classes),
        )
        rows.append(row)

    return rows


# ------------------------------------------------------------
# Top-L transfer
# ------------------------------------------------------------

def compute_topl_transfer_pair_rows(
    *,
    case_names: List[str],
    case_results: Dict[str, Dict[str, Any]],
    case_models: Dict[str, Any],
    case_out_hw: Dict[str, int],
    device,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    for source_name in case_names:
        source_topl = case_results[source_name]["topl_by_label"]
        source_domain = str(case_results[source_name]["domain"])

        for eval_name in case_names:
            eval_domain = str(case_results[eval_name]["domain"])
            eval_model = case_models[eval_name]

            rows = compute_topl_transfer_target_softmax(
                source_tag=source_name,
                source_topl_by_label=source_topl,
                eval_tag=eval_name,
                eval_model=eval_model,
                out_hw=int(case_out_hw[eval_name]),
                dataset=eval_domain,
                device=device,
                batch_size=int(batch_size),
            )

            for row in rows:
                raw_rows.append({
                    "source_model": str(source_name),
                    "source_domain": str(source_domain),
                    "eval_model": str(eval_name),
                    "eval_domain": str(eval_domain),
                    "pair_type": pair_type(source_domain, eval_domain),
                    **row,
                })

            vals = [float(row["avg_target_softmax"]) for row in rows]
            pair_rows.append({
                "source_model": str(source_name),
                "source_domain": str(source_domain),
                "eval_model": str(eval_name),
                "eval_domain": str(eval_domain),
                "pair_type": pair_type(source_domain, eval_domain),
                "num_labels": int(len(vals)),
                "mean_target_softmax": safe_mean(vals),
            })

    return raw_rows, pair_rows


def _score_records_on_two_models(
    *,
    records: List[Dict[str, Any]],
    model_a,
    dataset_a: str,
    out_hw_a: int,
    model_b,
    dataset_b: str,
    out_hw_b: int,
    device,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    scored_a = _score_records_grouped_by_grid(
        model=model_a,
        records=records,
        out_hw=int(out_hw_a),
        dataset=str(dataset_a),
        device=device,
        batch_size=int(batch_size),
    )
    scored_b = _score_records_grouped_by_grid(
        model=model_b,
        records=records,
        out_hw=int(out_hw_b),
        dataset=str(dataset_b),
        device=device,
        batch_size=int(batch_size),
    )
    return scored_a, scored_b


def _mean_absdiff_target_softmax_from_scored(
    *,
    scored_a: List[Dict[str, Any]],
    scored_b: List[Dict[str, Any]],
    label: int,
) -> Dict[str, float]:
    n = min(len(scored_a), len(scored_b))
    if n <= 0:
        return {
            "num_examples": 0,
            "avg_absdiff_target_softmax": 0.0,
        }

    diffs: List[float] = []
    target_idx = int(label)
    for idx in range(n):
        probs_a = np.asarray(scored_a[idx].get("all_probs", []), dtype=np.float64).reshape(-1)
        probs_b = np.asarray(scored_b[idx].get("all_probs", []), dtype=np.float64).reshape(-1)
        if probs_a.size <= target_idx or probs_b.size <= target_idx:
            continue
        diffs.append(abs(float(probs_a[target_idx]) - float(probs_b[target_idx])))

    if not diffs:
        return {
            "num_examples": 0,
            "avg_absdiff_target_softmax": 0.0,
        }

    return {
        "num_examples": int(len(diffs)),
        "avg_absdiff_target_softmax": float(np.mean(np.asarray(diffs, dtype=np.float64))),
    }


def compute_topk_consistency_pair_rows(
    *,
    case_names: List[str],
    case_results: Dict[str, Dict[str, Any]],
    case_models: Dict[str, Any],
    case_out_hw: Dict[str, int],
    device,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    ordered_names = [str(x) for x in case_names]
    for idx_a, case_a in enumerate(ordered_names):
        domain_a = str(case_results[case_a]["domain"])
        model_a = case_models[case_a]
        out_hw_a = int(case_out_hw[case_a])

        for idx_b in range(idx_a + 1, len(ordered_names)):
            case_b = ordered_names[idx_b]
            domain_b = str(case_results[case_b]["domain"])
            model_b = case_models[case_b]
            out_hw_b = int(case_out_hw[case_b])

            rows_this_pair: List[Dict[str, Any]] = []
            for probe_source in (case_a, case_b):
                probe_domain = str(case_results[probe_source]["domain"])
                source_topl = case_results[probe_source]["topl_by_label"]

                for label, info in sorted(source_topl.items(), key=lambda kv: int(kv[0])):
                    topl = list(info.get("topl", []))
                    scored_a, scored_b = _score_records_on_two_models(
                        records=topl,
                        model_a=model_a,
                        dataset_a=domain_a,
                        out_hw_a=out_hw_a,
                        model_b=model_b,
                        dataset_b=domain_b,
                        out_hw_b=out_hw_b,
                        device=device,
                        batch_size=int(batch_size),
                    )
                    stats = _mean_absdiff_target_softmax_from_scored(
                        scored_a=scored_a,
                        scored_b=scored_b,
                        label=int(label),
                    )

                    row = {
                        "case_a": str(case_a),
                        "domain_a": str(domain_a),
                        "case_b": str(case_b),
                        "domain_b": str(domain_b),
                        "pair_type": pair_type(domain_a, domain_b),
                        "probe_source_model": str(probe_source),
                        "probe_source_domain": str(probe_domain),
                        "label": int(label),
                        "num_examples": int(stats["num_examples"]),
                        "avg_absdiff_target_softmax": float(stats["avg_absdiff_target_softmax"]),
                    }
                    raw_rows.append(row)
                    rows_this_pair.append(row)

            vals = [float(row["avg_absdiff_target_softmax"]) for row in rows_this_pair]
            pair_rows.append({
                "case_a": str(case_a),
                "domain_a": str(domain_a),
                "case_b": str(case_b),
                "domain_b": str(domain_b),
                "pair_type": pair_type(domain_a, domain_b),
                "num_rows": int(len(rows_this_pair)),
                "mean_absdiff_target_softmax": safe_mean(vals),
            })

    return raw_rows, pair_rows


def _build_topk_consistency_lookup(
    raw_rows: List[Dict[str, Any]],
) -> Dict[Tuple[str, str, str, int], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    for row in raw_rows:
        lookup[(
            str(row["case_a"]),
            str(row["case_b"]),
            str(row["probe_source_model"]),
            int(row["label"]),
        )] = row
    return lookup


def compute_focus_pair_topk_consistency_rows(
    *,
    raw_rows: List[Dict[str, Any]],
    focus_pairs: List[Dict[str, str]],
    num_classes: int,
    shared_labels_by_pair: Optional[Dict[Tuple[str, str], List[int]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    lookup = _build_topk_consistency_lookup(raw_rows)
    out_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for spec in focus_pairs:
        case_a = str(spec["case_a"])
        case_b = str(spec["case_b"])
        group = str(spec["group"])
        display_name = str(spec["display_name"])
        key_a, key_b = sorted([case_a, case_b])

        vals_t: List[float] = []
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

            if case_a == case_b:
                out_rows.append({
                    "comparison_group": group,
                    "display_name": display_name,
                    "case_a": case_a,
                    "case_b": case_b,
                    "class_label": int(c),
                    "t_on_a_probes": 0.0,
                    "t_on_b_probes": 0.0,
                    "T_c": 0.0,
                    "num_examples_a_probes": 0,
                    "num_examples_b_probes": 0,
                })
                vals_t.append(0.0)
                num_available += 1
                continue

            row_a = lookup.get((key_a, key_b, case_a, int(c)))
            row_b = lookup.get((key_a, key_b, case_b, int(c)))
            if row_a is None or row_b is None:
                continue

            t_on_a = float(row_a["avg_absdiff_target_softmax"])
            t_on_b = float(row_b["avg_absdiff_target_softmax"])
            t_c = t_on_a + t_on_b
            vals_t.append(t_c)
            num_available += 1

            out_rows.append({
                "comparison_group": group,
                "display_name": display_name,
                "case_a": case_a,
                "case_b": case_b,
                "domain_a": str(row_a.get("domain_a", "")),
                "domain_b": str(row_a.get("domain_b", "")),
                "class_label": int(c),
                "t_on_a_probes": float(t_on_a),
                "t_on_b_probes": float(t_on_b),
                "T_c": float(t_c),
                "num_examples_a_probes": int(row_a.get("num_examples", 0)),
                "num_examples_b_probes": int(row_b.get("num_examples", 0)),
            })

        summary_rows.append({
            "comparison_group": group,
            "display_name": display_name,
            "case_a": case_a,
            "case_b": case_b,
            "num_classes_available": int(num_available),
            "mean_T_c": float(safe_mean(vals_t)),
        })

    return out_rows, summary_rows


def compute_all_pairs_topk_consistency_rows(
    *,
    raw_rows: List[Dict[str, Any]],
    case_names: List[str],
    display_name_map: Dict[str, str],
    num_classes: int,
    shared_labels_by_pair: Optional[Dict[Tuple[str, str], List[int]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    lookup = _build_topk_consistency_lookup(raw_rows)
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
            key_a, key_b = sorted([case_a, case_b])

            for c in range(int(num_classes)):
                if int(c) not in shared_labels:
                    continue

                if case_a == case_b:
                    t_c = 0.0
                else:
                    row_a = lookup.get((key_a, key_b, case_a, int(c)))
                    row_b = lookup.get((key_a, key_b, case_b, int(c)))
                    if row_a is None or row_b is None:
                        continue
                    t_c = float(row_a["avg_absdiff_target_softmax"]) + float(row_b["avg_absdiff_target_softmax"])

                vals.append(t_c)
                num_available += 1
                per_class_rows.append({
                    "case_a": display_name_map.get(case_a, case_a),
                    "case_b": display_name_map.get(case_b, case_b),
                    "raw_case_a": case_a,
                    "raw_case_b": case_b,
                    "class_label": int(c),
                    "T_c": float(t_c),
                })

            mean_rows.append({
                "case_a": display_name_map.get(case_a, case_a),
                "case_b": display_name_map.get(case_b, case_b),
                "raw_case_a": case_a,
                "raw_case_b": case_b,
                "num_classes_available": int(num_available),
                "mean_T_c": float(safe_mean(vals)),
            })

    return per_class_rows, mean_rows


# ------------------------------------------------------------
# edge-case parsing helpers
# ------------------------------------------------------------

def _extract_prob_vector_from_record(rec: Dict[str, Any]) -> Optional[np.ndarray]:
    candidate_keys = [
        "all_probs",
        "probs",
        "softmax",
        "softmax_probs",
        "avg_probs",
        "mean_probs",
        "prob_vector",
    ]
    for key in candidate_keys:
        if key in rec:
            try:
                arr = np.asarray(rec[key], dtype=np.float64).reshape(-1)
                if arr.size > 0:
                    return arr
            except Exception:
                pass
    return None


def _extract_entropy_from_record(rec: Dict[str, Any], probs: Optional[np.ndarray]) -> float:
    for key in ["entropy", "avg_entropy", "mean_entropy"]:
        if key in rec:
            try:
                return float(rec[key])
            except Exception:
                pass

    if probs is not None and probs.size > 0:
        p = np.clip(probs, 1e-12, 1.0)
        p = p / (np.sum(p) + 1e-12)
        return float(-np.sum(p * np.log(p)))
    return float("nan")


def _extract_pair_scores_from_record(
    rec: Dict[str, Any],
    i: int,
    j: int,
    probs: Optional[np.ndarray],
) -> Tuple[float, float, float, float]:
    if probs is not None and probs.size > max(i, j):
        pi = float(probs[i])
        pj = float(probs[j])
        pair_mass = float(pi + pj)
        pair_gap = float(abs(pi - pj))
        return pi, pj, pair_mass, pair_gap

    pi = float(rec.get(f"p_{i}", rec.get("p_i", 0.0)))
    pj = float(rec.get(f"p_{j}", rec.get("p_j", 0.0)))

    pair_mass = rec.get("pair_mass", None)
    if pair_mass is None:
        pair_mass = pi + pj

    pair_gap = rec.get("pair_gap", None)
    if pair_gap is None:
        pair_gap = abs(pi - pj)

    return float(pi), float(pj), float(pair_mass), float(pair_gap)


def _coerce_edge_record_list(
    pair_payload: Any,
) -> List[Dict[str, Any]]:
    if isinstance(pair_payload, list):
        return [x for x in pair_payload if isinstance(x, dict)]

    if isinstance(pair_payload, dict):
        for key in [
            "records",
            "examples",
            "top_records",
            "top_k",
            "items",
            "samples",
            "candidates",
        ]:
            value = pair_payload.get(key, None)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

        if all(not isinstance(v, (list, dict)) for v in pair_payload.values()):
            return [pair_payload]

    return []


# ------------------------------------------------------------
# edge-case summarization
# ------------------------------------------------------------

def summarize_edge_cases(
    *,
    edge_cases: Dict[str, Any],
    num_classes: int,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    expected_pairs = set(iter_class_pairs(int(num_classes)))

    raw_edge_cases = edge_cases or {}
    summary_by_pair = raw_edge_cases.get("summary_by_pair", {})
    top_records_by_pair = raw_edge_cases.get("top_records_by_pair", {})

    for raw_key, payload in summary_by_pair.items():
        parsed = parse_pair_key(raw_key)
        if parsed is None:
            if isinstance(payload, dict):
                pair = payload.get("pair", None)
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    try:
                        parsed = (min(int(pair[0]), int(pair[1])), max(int(pair[0]), int(pair[1])))
                    except Exception:
                        parsed = None

        if parsed is None:
            continue

        i, j = parsed
        if (i, j) not in expected_pairs:
            continue

        key = pair_key(i, j)

        count_kept = int(payload.get("count_kept", 0))
        count_total = int(payload.get("count_total", count_kept))
        margin_mean = float(payload.get("margin_mean", float("nan")))
        entropy_mean = float(payload.get("entropy_mean", float("nan")))

        grid_hist = payload.get("grid_hist", {})
        if isinstance(grid_hist, dict):
            grid_hist_clean = {
                int(k): int(v) for k, v in grid_hist.items()
            }
        else:
            grid_hist_clean = {}

        hist_total = float(sum(grid_hist_clean.values()))
        frac_d4 = float(grid_hist_clean.get(4, 0)) / hist_total if hist_total > 0 else 0.0
        frac_d8 = float(grid_hist_clean.get(8, 0)) / hist_total if hist_total > 0 else 0.0
        frac_d16 = float(grid_hist_clean.get(16, 0)) / hist_total if hist_total > 0 else 0.0

        # keep placeholder fields for compatibility with the rest of the code
        # these are not directly available from summary_by_pair
        out[key] = {
            "pair_i": int(i),
            "pair_j": int(j),
            "pair_name": key,
            "count": int(count_kept),
            "count_total": int(count_total),
            "mean_p_i": float("nan"),
            "mean_p_j": float("nan"),
            "mean_pair_mass": float("nan"),
            "mean_pair_gap": float(margin_mean),
            "mean_entropy": float(entropy_mean),
            "mean_offpair_mass": float("nan"),
            "grid_hist": grid_hist_clean,
            "frac_d4": float(frac_d4),
            "frac_d8": float(frac_d8),
            "frac_d16": float(frac_d16),
            "has_top_records": bool(key in top_records_by_pair),
        }

    # fill missing pairs so later comparison stays aligned
    for i, j in iter_class_pairs(int(num_classes)):
        key = pair_key(i, j)
        if key not in out:
            out[key] = {
                "pair_i": int(i),
                "pair_j": int(j),
                "pair_name": key,
                "count": 0,
                "count_total": 0,
                "mean_p_i": float("nan"),
                "mean_p_j": float("nan"),
                "mean_pair_mass": float("nan"),
                "mean_pair_gap": float("nan"),
                "mean_entropy": float("nan"),
                "mean_offpair_mass": float("nan"),
                "grid_hist": {},
                "frac_d4": 0.0,
                "frac_d8": 0.0,
                "frac_d16": 0.0,
                "has_top_records": False,
            }

    return out


def edge_summary_to_rows(
    *,
    case_name: str,
    domain: str,
    edge_summary: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in sorted(edge_summary.keys(), key=lambda x: tuple(int(v) for v in x.split("_"))):
        info = dict(edge_summary[key])
        rows.append({
            "case": str(case_name),
            "domain": str(domain),
            **info,
        })
    return rows


def edge_feature_vector(edge_info: Dict[str, Any]) -> np.ndarray:
    count = float(edge_info.get("count", 0))
    count_norm = min(count, 64.0) / 64.0

    vals = [
        count_norm,
        edge_info.get("mean_pair_gap", float("nan")),
        edge_info.get("mean_entropy", float("nan")),
        edge_info.get("frac_d4", 0.0),
        edge_info.get("frac_d8", 0.0),
        edge_info.get("frac_d16", 0.0),
    ]
    arr = np.asarray(vals, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def compare_edge_summaries(
    *,
    case_a: str,
    case_b: str,
    domain_a: str,
    domain_b: str,
    edge_summary_a: Dict[str, Dict[str, Any]],
    edge_summary_b: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    keys = sorted(set(edge_summary_a.keys()).intersection(edge_summary_b.keys()))

    for key in keys:
        a = edge_summary_a[key]
        b = edge_summary_b[key]

        vec_a = edge_feature_vector(a)
        vec_b = edge_feature_vector(b)

        cosine_sim = float(safe_cosine(vec_a, vec_b))
        cosine_dist = float(1.0 - cosine_sim)

        rows.append({
            "case_a": str(case_a),
            "case_b": str(case_b),
            "domain_a": str(domain_a),
            "domain_b": str(domain_b),
            "pair_type": pair_type(domain_a, domain_b),
            "edge_pair": str(key),
            "pair_i": int(a["pair_i"]),
            "pair_j": int(a["pair_j"]),
            "count_a": int(a.get("count", 0)),
            "count_b": int(b.get("count", 0)),
            "edge_cosine": cosine_dist,
            "edge_l1": float(l1_distance(vec_a, vec_b)),
            "edge_js": float(js_on_prob_vectors(vec_a, vec_b)),
        })

    return rows


def aggregate_edge_pair_rows(
    *,
    case_a: str,
    case_b: str,
    domain_a: str,
    domain_b: str,
    pair_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    cosine_vals = [float(r["edge_cosine"]) for r in pair_rows]
    l1_vals = [float(r["edge_l1"]) for r in pair_rows]
    js_vals = [float(r["edge_js"]) for r in pair_rows]

    return {
        "case_a": str(case_a),
        "case_b": str(case_b),
        "domain_a": str(domain_a),
        "domain_b": str(domain_b),
        "pair_type": pair_type(domain_a, domain_b),
        "edge_mean_cosine": safe_mean(cosine_vals),
        "edge_mean_l1": safe_mean(l1_vals),
        "edge_mean_js": safe_mean(js_vals),
        "edge_worst_l1": float(max(l1_vals)) if l1_vals else float("nan"),
        "edge_worst_js": float(max(js_vals)) if js_vals else float("nan"),
        "num_pairs": int(len(pair_rows)),
    }


def compute_experiment_edge_rows(
    *,
    case_names: List[str],
    case_results: Dict[str, Dict[str, Any]],
    num_classes: int,
    debug: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    per_case_rows: List[Dict[str, Any]] = []
    pairwise_by_pair_rows: List[Dict[str, Any]] = []
    pairwise_summary_rows: List[Dict[str, Any]] = []
    edge_summary_by_case: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for case_name in case_names:
        domain = str(case_results[case_name]["domain"])
        result = case_results[case_name]["result"]

        raw_edge_cases = result.get("edge_cases", {})

        if debug:
            print(f"[DEBUG edge raw] case={case_name} type={type(raw_edge_cases)}")
            if isinstance(raw_edge_cases, dict):
                raw_keys = list(raw_edge_cases.keys())
                print(f"[DEBUG edge raw] case={case_name} num_keys={len(raw_keys)} sample_keys={raw_keys[:10]}")

        summary = summarize_edge_cases(
            edge_cases=raw_edge_cases,
            num_classes=int(num_classes),
        )

        if debug:
            print(f"[DEBUG edge summary] case={case_name} num_pairs={len(summary)}")

        edge_summary_by_case[case_name] = summary
        per_case_rows.extend(
            edge_summary_to_rows(
                case_name=case_name,
                domain=domain,
                edge_summary=summary,
            )
        )

    for case_a, case_b in combinations(case_names, 2):
        rows = compare_edge_summaries(
            case_a=case_a,
            case_b=case_b,
            domain_a=str(case_results[case_a]["domain"]),
            domain_b=str(case_results[case_b]["domain"]),
            edge_summary_a=edge_summary_by_case[case_a],
            edge_summary_b=edge_summary_by_case[case_b],
        )

        if debug:
            print(f"[DEBUG edge compare] {case_a} vs {case_b} num_rows={len(rows)}")

        pairwise_by_pair_rows.extend(rows)
        pairwise_summary_rows.append(
            aggregate_edge_pair_rows(
                case_a=case_a,
                case_b=case_b,
                domain_a=str(case_results[case_a]["domain"]),
                domain_b=str(case_results[case_b]["domain"]),
                pair_rows=rows,
            )
        )

    return per_case_rows, pairwise_by_pair_rows, pairwise_summary_rows


# ------------------------------------------------------------
# grouped summaries
# ------------------------------------------------------------

def aggregate_rows_by_group(
    *,
    rows: List[Dict[str, Any]],
    group_key: str,
    metric_keys: List[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        g = str(row[group_key])
        grouped.setdefault(g, []).append(row)

    out: List[Dict[str, Any]] = []
    for group_name, group_rows in grouped.items():
        agg_row: Dict[str, Any] = {"group": str(group_name)}
        for key in metric_keys:
            vals = [float(r[key]) for r in group_rows]
            agg_row[f"{key}_mean"] = safe_mean(vals)
        out.append(agg_row)

    out.sort(key=lambda r: r["group"])
    return out

# ------------------------------------------------------------
# Top-L per-class analysis
# ------------------------------------------------------------

def _topl_label_vector(topl_entry: Any) -> np.ndarray:
    """
    Convert one label entry in topl_by_label into a comparable numeric vector.

    This is intentionally tolerant to different topl entry layouts.
    """
    if topl_entry is None:
        return np.zeros(1, dtype=np.float64)

    if isinstance(topl_entry, np.ndarray):
        arr = np.asarray(topl_entry, dtype=np.float64).reshape(-1)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if isinstance(topl_entry, (list, tuple)):
        try:
            arr = np.asarray(topl_entry, dtype=np.float64).reshape(-1)
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass

    if not isinstance(topl_entry, dict):
        return np.zeros(1, dtype=np.float64)

    # common numeric summary fields first
    numeric_fields = [
        "best_score",
        "score",
        "mean_score",
        "avg_score",
        "target_softmax",
        "avg_target_softmax",
        "mean_target_softmax",
        "entropy",
        "mean_entropy",
        "avg_entropy",
        "count",
        "num_examples",
        "num_records",
    ]

    vals: List[float] = []
    for k in numeric_fields:
        if k in topl_entry:
            try:
                vals.append(float(topl_entry[k]))
            except Exception:
                pass

    # optional probability-like vectors
    for k in ["avg_probs", "mean_probs", "probs", "softmax", "prob_vector"]:
        if k in topl_entry:
            try:
                arr = np.asarray(topl_entry[k], dtype=np.float64).reshape(-1)
                vals.extend(arr.tolist())
                break
            except Exception:
                pass

    if len(vals) == 0:
        # final fallback: extract scalar numeric fields in sorted order
        for k in sorted(topl_entry.keys()):
            v = topl_entry[k]
            if isinstance(v, (int, float, np.integer, np.floating)):
                vals.append(float(v))

    if len(vals) == 0:
        return np.zeros(1, dtype=np.float64)

    arr = np.asarray(vals, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def summarize_topl_by_class(
    *,
    topl_by_label: Dict[Any, Any],
    num_classes: int,
) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}

    raw = topl_by_label or {}
    for y in range(int(num_classes)):
        payload = raw.get(y, raw.get(str(y), None))
        vec = _topl_label_vector(payload)
        out[int(y)] = {
            "label": int(y),
            "vector": vec,
            "dim": int(vec.size),
            "has_entry": payload is not None,
        }

    return out


def topl_summary_to_rows(
    *,
    case_name: str,
    domain: str,
    topl_summary: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for y in range(len(topl_summary)):
        info = topl_summary[int(y)]
        rows.append({
            "case": str(case_name),
            "domain": str(domain),
            "label": int(y),
            "dim": int(info["dim"]),
            "has_entry": bool(info["has_entry"]),
        })
    return rows


def _align_vectors(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    m = max(int(aa.size), int(bb.size))
    if aa.size < m:
        aa = np.pad(aa, (0, m - aa.size))
    if bb.size < m:
        bb = np.pad(bb, (0, m - bb.size))
    return aa, bb


def compare_topl_summaries(
    *,
    case_a: str,
    case_b: str,
    domain_a: str,
    domain_b: str,
    topl_summary_a: Dict[int, Dict[str, Any]],
    topl_summary_b: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    common_labels = sorted(set(topl_summary_a.keys()).intersection(topl_summary_b.keys()))
    for y in common_labels:
        vec_a, vec_b = _align_vectors(
            topl_summary_a[int(y)]["vector"],
            topl_summary_b[int(y)]["vector"],
        )

        cosine_sim = float(safe_cosine(vec_a, vec_b))
        cosine_dist = float(1.0 - cosine_sim)

        rows.append({
            "case_a": str(case_a),
            "case_b": str(case_b),
            "domain_a": str(domain_a),
            "domain_b": str(domain_b),
            "pair_type": pair_type(domain_a, domain_b),
            "label": int(y),
            "topl_cosine_similarity": cosine_sim,
            "topl_cosine_distance": cosine_dist,
            "topl_l1": float(l1_distance(vec_a, vec_b)),
            "topl_js": float(js_on_prob_vectors(vec_a, vec_b)),
            "dim": int(max(vec_a.size, vec_b.size)),
        })

    return rows


def aggregate_topl_class_rows(
    *,
    case_a: str,
    case_b: str,
    domain_a: str,
    domain_b: str,
    class_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    cosine_dist_vals = [float(r["topl_cosine_distance"]) for r in class_rows]
    l1_vals = [float(r["topl_l1"]) for r in class_rows]
    js_vals = [float(r["topl_js"]) for r in class_rows]

    return {
        "case_a": str(case_a),
        "case_b": str(case_b),
        "domain_a": str(domain_a),
        "domain_b": str(domain_b),
        "pair_type": pair_type(domain_a, domain_b),
        "topl_mean_cosine_distance": safe_mean(cosine_dist_vals),
        "topl_mean_l1": safe_mean(l1_vals),
        "topl_mean_js": safe_mean(js_vals),
        "topl_worst_cosine_distance": float(max(cosine_dist_vals)) if cosine_dist_vals else float("nan"),
        "topl_worst_l1": float(max(l1_vals)) if l1_vals else float("nan"),
        "topl_worst_js": float(max(js_vals)) if js_vals else float("nan"),
        "num_labels": int(len(class_rows)),
    }


def compute_experiment_topl_per_class_rows(
    *,
    case_names: List[str],
    case_results: Dict[str, Dict[str, Any]],
    num_classes: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[int, Dict[str, Any]]]]:
    per_case_rows: List[Dict[str, Any]] = []
    pairwise_by_class_rows: List[Dict[str, Any]] = []
    pairwise_summary_rows: List[Dict[str, Any]] = []
    topl_summary_by_case: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for case_name in case_names:
        domain = str(case_results[case_name]["domain"])
        result = case_results[case_name]["result"]
        topl_summary = summarize_topl_by_class(
            topl_by_label=result.get("topl_by_label", {}),
            num_classes=int(num_classes),
        )
        topl_summary_by_case[case_name] = topl_summary
        per_case_rows.extend(
            topl_summary_to_rows(
                case_name=case_name,
                domain=domain,
                topl_summary=topl_summary,
            )
        )

    for case_a, case_b in combinations(case_names, 2):
        rows = compare_topl_summaries(
            case_a=case_a,
            case_b=case_b,
            domain_a=str(case_results[case_a]["domain"]),
            domain_b=str(case_results[case_b]["domain"]),
            topl_summary_a=topl_summary_by_case[case_a],
            topl_summary_b=topl_summary_by_case[case_b],
        )
        pairwise_by_class_rows.extend(rows)
        pairwise_summary_rows.append(
            aggregate_topl_class_rows(
                case_a=case_a,
                case_b=case_b,
                domain_a=str(case_results[case_a]["domain"]),
                domain_b=str(case_results[case_b]["domain"]),
                class_rows=rows,
            )
        )

    return per_case_rows, pairwise_by_class_rows, pairwise_summary_rows, topl_summary_by_case


def filter_target_centered_edge_rows(
    *,
    edge_rows: List[Dict[str, Any]],
    target_label: int,
) -> List[Dict[str, Any]]:
    tgt = int(target_label)
    return [
        row for row in edge_rows
        if int(row["pair_i"]) == tgt or int(row["pair_j"]) == tgt
    ]


def filter_target_label_topl_rows(
    *,
    topl_rows: List[Dict[str, Any]],
    target_label: int,
) -> List[Dict[str, Any]]:
    tgt = int(target_label)
    return [row for row in topl_rows if int(row["label"]) == tgt]


def rank_topl_classes_for_case_pair(
    *,
    pairwise_by_class_rows: List[Dict[str, Any]],
    case_a: str,
    case_b: str,
    metric_key: str = "topl_js",
    descending: bool = True,
) -> List[Dict[str, Any]]:
    rows = [
        r for r in pairwise_by_class_rows
        if str(r["case_a"]) == str(case_a) and str(r["case_b"]) == str(case_b)
    ]
    rows = sorted(rows, key=lambda r: float(r[metric_key]), reverse=bool(descending))
    return rows


def rank_edge_pairs_for_case_pair(
    *,
    pairwise_by_pair_rows: List[Dict[str, Any]],
    case_a: str,
    case_b: str,
    metric_key: str = "edge_js",
    descending: bool = True,
) -> List[Dict[str, Any]]:
    rows = [
        r for r in pairwise_by_pair_rows
        if str(r["case_a"]) == str(case_a) and str(r["case_b"]) == str(case_b)
    ]
    rows = sorted(rows, key=lambda r: float(r[metric_key]), reverse=bool(descending))
    return rows

# ------------------------------------------------------------
# Top-L transfer per-label analysis
# ------------------------------------------------------------

def group_topl_transfer_rows_by_label(
    *,
    raw_rows: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in raw_rows:
        y = int(row["label"])
        grouped.setdefault(y, []).append(row)
    return grouped


def summarize_topl_transfer_by_label(
    *,
    raw_rows: List[Dict[str, Any]],
    case_names: List[str],
) -> List[Dict[str, Any]]:
    """
    Keep one row per (source_model, eval_model, label), with explicit metadata.
    This is mostly a normalized pass-through for downstream CSV / plotting.
    """
    case_set = set(str(x) for x in case_names)
    rows: List[Dict[str, Any]] = []

    for row in raw_rows:
        src = str(row["source_model"])
        tgt = str(row["eval_model"])
        if src not in case_set or tgt not in case_set:
            continue

        rows.append({
            "source_model": src,
            "source_domain": str(row.get("source_domain", "")),
            "eval_model": tgt,
            "eval_domain": str(row.get("eval_domain", "")),
            "pair_type": str(row.get("pair_type", "")),
            "label": int(row["label"]),
            "num_examples": int(row.get("num_examples", 0)),
            "avg_target_softmax": float(row.get("avg_target_softmax", 0.0)),
        })

    return rows


def rank_topl_transfer_labels_for_case_pair(
    *,
    raw_rows: List[Dict[str, Any]],
    source_model: str,
    eval_model: str,
    descending: bool = True,
) -> List[Dict[str, Any]]:
    rows = [
        r for r in raw_rows
        if str(r["source_model"]) == str(source_model)
        and str(r["eval_model"]) == str(eval_model)
    ]
    rows = sorted(rows, key=lambda r: float(r["avg_target_softmax"]), reverse=bool(descending))
    return rows

def compute_topl_transfer_mean_over_labels(
    *,
    raw_rows: List[Dict[str, Any]],
    case_names: List[str],
) -> List[Dict[str, Any]]:
    """
    Aggregate per-label Top-L transfer into mean over labels.
    """

    case_set = set(str(x) for x in case_names)

    # (src, tgt) -> list of values
    agg: Dict[Tuple[str, str], List[float]] = {}

    for r in raw_rows:
        src = str(r["source_model"])
        tgt = str(r["eval_model"])

        if src not in case_set or tgt not in case_set:
            continue

        key = (src, tgt)
        agg.setdefault(key, []).append(float(r["avg_target_softmax"]))

    rows: List[Dict[str, Any]] = []

    for (src, tgt), vals in agg.items():
        if len(vals) == 0:
            mean_val = 0.0
        else:
            mean_val = float(sum(vals) / len(vals))

        rows.append({
            "source_model": src,
            "eval_model": tgt,
            "num_labels": int(len(vals)),
            "mean_target_softmax": mean_val,
        })

    return rows

# ------------------------------------------------------------
# edge transfer analysis
# ------------------------------------------------------------

def _build_tensor_batch_from_records(
    *,
    records: List[Dict[str, Any]],
    out_hw: int,
) -> Optional[torch.Tensor]:
    """
    Build a tensor batch from stored edge/top-l records.

    This is intentionally tolerant to several record layouts.
    """
    xs: List[torch.Tensor] = []

    for rec in records:
        candidate = None
        for key in ["x", "image", "probe", "tensor"]:
            if key in rec:
                candidate = rec[key]
                break

        if candidate is None:
            # common layout from pointillism search records
            for key in ["input", "img", "rendered"]:
                if key in rec:
                    candidate = rec[key]
                    break

        if candidate is None:
            continue

        try:
            x = torch.as_tensor(candidate, dtype=torch.float32)
        except Exception:
            continue

        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            pass
        else:
            continue

        xs.append(x)

    if len(xs) == 0:
        return None

    return torch.stack(xs, dim=0)


@torch.no_grad()
def _forward_probs_batched(
    *,
    model,
    x_batch: torch.Tensor,
    device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outs: List[np.ndarray] = []

    n = int(x_batch.shape[0])
    bs = int(batch_size)

    for start in range(0, n, bs):
        xb = x_batch[start:start + bs].to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        outs.append(probs)

    if len(outs) == 0:
        return np.zeros((0, 1), dtype=np.float64)

    return np.concatenate(outs, axis=0).astype(np.float64)


def _entropy_from_probs_np(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    p = p / (np.sum(p, axis=1, keepdims=True) + 1e-12)
    return -np.sum(p * np.log(p), axis=1)


def _pair_top2_acc_np(
    probs: np.ndarray,
    *,
    pair_i: int,
    pair_j: int,
) -> float:
    if probs.shape[0] == 0:
        return 0.0

    top2 = np.argsort(probs, axis=1)[:, -2:]
    good = 0
    target_set = {int(pair_i), int(pair_j)}

    for row in top2:
        if set(int(x) for x in row.tolist()) == target_set:
            good += 1

    return float(good) / float(probs.shape[0])


def _pair_metrics_from_probs_np(
    probs: np.ndarray,
    *,
    pair_i: int,
    pair_j: int,
) -> Dict[str, float]:
    if probs.shape[0] == 0:
        return {
            "num_examples": 0,
            "avg_pair_mass": 0.0,
            "avg_pair_gap": 0.0,
            "avg_pair_ambiguity": 0.0,
            "avg_offpair_mass": 0.0,
            "avg_entropy": 0.0,
            "pair_top2_acc": 0.0,
        }

    pi = probs[:, int(pair_i)]
    pj = probs[:, int(pair_j)]
    pair_mass = pi + pj
    pair_gap = np.abs(pi - pj)
    pair_ambiguity = np.minimum(pi, pj)
    offpair_mass = 1.0 - pair_mass
    ent = _entropy_from_probs_np(probs)
    top2_acc = _pair_top2_acc_np(probs, pair_i=int(pair_i), pair_j=int(pair_j))

    return {
        "num_examples": int(probs.shape[0]),
        "avg_pair_mass": float(np.mean(pair_mass)),
        "avg_pair_gap": float(np.mean(pair_gap)),
        "avg_pair_ambiguity": float(np.mean(pair_ambiguity)),
        "avg_offpair_mass": float(np.mean(offpair_mass)),
        "avg_entropy": float(np.mean(ent)),
        "pair_top2_acc": float(top2_acc),
    }


def compute_edge_transfer_metrics_for_records(
    *,
    edge_pair: Tuple[int, int],
    records: List[Dict[str, Any]],
    eval_model,
    dataset: str,
    device,
    batch_size: int,
    out_hw: int,
) -> Dict[str, float]:
    """
    Replay source edge records on one eval model and summarize edge behavior.
    Reuses the same record-scoring path as Top-L transfer.
    """
    pair_i, pair_j = int(edge_pair[0]), int(edge_pair[1])

    scored = _score_records_grouped_by_grid(
        model=eval_model,
        records=records,
        out_hw=int(out_hw),
        dataset=str(dataset),
        device=device,
        batch_size=int(batch_size),
    )

    if len(scored) == 0:
        return {
            "num_examples": 0,
            "avg_pair_mass": 0.0,
            "avg_pair_gap": 0.0,
            "avg_pair_ambiguity": 0.0,
            "avg_offpair_mass": 0.0,
            "avg_entropy": 0.0,
            "pair_top2_acc": 0.0,
        }

    probs_list = []
    top2_good = 0
    target_set = {int(pair_i), int(pair_j)}

    for rec in scored:
        probs = np.asarray(rec.get("all_probs", []), dtype=np.float64)
        if probs.ndim != 1 or probs.size == 0:
            continue
        probs_list.append(probs)

        top2 = np.argsort(probs)[-2:]
        if set(int(x) for x in top2.tolist()) == target_set:
            top2_good += 1

    if len(probs_list) == 0:
        return {
            "num_examples": 0,
            "avg_pair_mass": 0.0,
            "avg_pair_gap": 0.0,
            "avg_pair_ambiguity": 0.0,
            "avg_offpair_mass": 0.0,
            "avg_entropy": 0.0,
            "pair_top2_acc": 0.0,
        }

    probs_mat = np.stack(probs_list, axis=0)
    pi = probs_mat[:, int(pair_i)]
    pj = probs_mat[:, int(pair_j)]
    pair_mass = pi + pj
    pair_gap = np.abs(pi - pj)
    pair_ambiguity = np.minimum(pi, pj)
    offpair_mass = 1.0 - pair_mass
    ent = _entropy_from_probs_np(probs_mat)

    return {
        "num_examples": int(probs_mat.shape[0]),
        "avg_pair_mass": float(np.mean(pair_mass)),
        "avg_pair_gap": float(np.mean(pair_gap)),
        "avg_pair_ambiguity": float(np.mean(pair_ambiguity)),
        "avg_offpair_mass": float(np.mean(offpair_mass)),
        "avg_entropy": float(np.mean(ent)),
        "pair_top2_acc": float(top2_good) / float(probs_mat.shape[0]),
    }


def _extract_top_records_by_pair(
    *,
    edge_cases: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    raw = edge_cases or {}
    payload = raw.get("top_records_by_pair", {})

    out: Dict[str, List[Dict[str, Any]]] = {}
    if not isinstance(payload, dict):
        return out

    for raw_key, value in payload.items():
        parsed = parse_pair_key(raw_key)
        if parsed is None:
            continue
        key = pair_key(parsed[0], parsed[1])

        if isinstance(value, list):
            out[key] = [x for x in value if isinstance(x, dict)]
        elif isinstance(value, dict):
            out[key] = _coerce_edge_record_list(value)
        else:
            out[key] = []

    return out


def compute_edge_transfer_pair_rows(
    *,
    case_names: List[str],
    case_results: Dict[str, Dict[str, Any]],
    case_models: Dict[str, Any],
    case_out_hw: Dict[str, int],
    device,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Raw rows:
      one row per (source_model, eval_model, edge_pair)

    Summary rows:
      one row per (source_model, eval_model), aggregated over all edge pairs
    """
    raw_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for source_name in case_names:
        source_domain = str(case_results[source_name]["domain"])
        source_model = case_models[source_name]
        source_result = case_results[source_name]["result"]
        source_top_records_by_pair = _extract_top_records_by_pair(
            edge_cases=source_result.get("edge_cases", {}),
        )

        for eval_name in case_names:
            eval_domain = str(case_results[eval_name]["domain"])
            eval_model = case_models[eval_name]
            eval_out_hw = int(case_out_hw[eval_name])

            rows_this_pairing: List[Dict[str, Any]] = []

            for edge_pair_name in sorted(source_top_records_by_pair.keys(), key=lambda x: tuple(int(v) for v in x.split("_"))):
                records = source_top_records_by_pair[edge_pair_name]
                if len(records) == 0:
                    continue

                parsed = parse_pair_key(edge_pair_name)
                if parsed is None:
                    continue
                pair_i, pair_j = int(parsed[0]), int(parsed[1])

                source_metrics = compute_edge_transfer_metrics_for_records(
                    edge_pair=(pair_i, pair_j),
                    records=records,
                    eval_model=source_model,
                    dataset=str(source_domain),
                    device=device,
                    batch_size=int(batch_size),
                    out_hw=int(case_out_hw[source_name]),
                )

                eval_metrics = compute_edge_transfer_metrics_for_records(
                    edge_pair=(pair_i, pair_j),
                    records=records,
                    eval_model=eval_model,
                    dataset=str(eval_domain),
                    device=device,
                    batch_size=int(batch_size),
                    out_hw=eval_out_hw,
                )

                avg_ref_pair_mass_absdiff = abs(
                    float(source_metrics["avg_pair_mass"]) - float(eval_metrics["avg_pair_mass"])
                )
                avg_ref_pair_gap_absdiff = abs(
                    float(source_metrics["avg_pair_gap"]) - float(eval_metrics["avg_pair_gap"])
                )
                avg_ref_offpair_absdiff = abs(
                    float(source_metrics["avg_offpair_mass"]) - float(eval_metrics["avg_offpair_mass"])
                )
                avg_ref_entropy_absdiff = abs(
                    float(source_metrics["avg_entropy"]) - float(eval_metrics["avg_entropy"])
                )

                edge_ref_drift = (
                    avg_ref_pair_mass_absdiff
                    + avg_ref_pair_gap_absdiff
                    + avg_ref_offpair_absdiff
                    + avg_ref_entropy_absdiff
                )

                row = {
                    "source_model": str(source_name),
                    "source_domain": str(source_domain),
                    "eval_model": str(eval_name),
                    "eval_domain": str(eval_domain),
                    "pair_type": pair_type(source_domain, eval_domain),

                    "edge_pair": str(edge_pair_name),
                    "pair_i": int(pair_i),
                    "pair_j": int(pair_j),

                    "num_examples": int(eval_metrics["num_examples"]),

                    "avg_pair_mass": float(eval_metrics["avg_pair_mass"]),
                    "avg_pair_gap": float(eval_metrics["avg_pair_gap"]),
                    "avg_pair_ambiguity": float(eval_metrics["avg_pair_ambiguity"]),
                    "avg_offpair_mass": float(eval_metrics["avg_offpair_mass"]),
                    "avg_entropy": float(eval_metrics["avg_entropy"]),
                    "pair_top2_acc": float(eval_metrics["pair_top2_acc"]),

                    "avg_ref_pair_mass_absdiff": float(avg_ref_pair_mass_absdiff),
                    "avg_ref_pair_gap_absdiff": float(avg_ref_pair_gap_absdiff),
                    "avg_ref_offpair_absdiff": float(avg_ref_offpair_absdiff),
                    "avg_ref_entropy_absdiff": float(avg_ref_entropy_absdiff),
                    "edge_ref_drift": float(edge_ref_drift),
                }
                raw_rows.append(row)
                rows_this_pairing.append(row)

            if len(rows_this_pairing) == 0:
                continue

            mean_pair_mass = safe_mean(float(r["avg_pair_mass"]) for r in rows_this_pairing)
            mean_pair_gap = safe_mean(float(r["avg_pair_gap"]) for r in rows_this_pairing)
            mean_pair_ambiguity = safe_mean(float(r["avg_pair_ambiguity"]) for r in rows_this_pairing)
            mean_offpair_mass = safe_mean(float(r["avg_offpair_mass"]) for r in rows_this_pairing)
            mean_entropy = safe_mean(float(r["avg_entropy"]) for r in rows_this_pairing)
            mean_pair_top2_acc = safe_mean(float(r["pair_top2_acc"]) for r in rows_this_pairing)
            mean_edge_ref_drift = safe_mean(float(r["edge_ref_drift"]) for r in rows_this_pairing)

            summary_rows.append({
                "source_model": str(source_name),
                "source_domain": str(source_domain),
                "eval_model": str(eval_name),
                "eval_domain": str(eval_domain),
                "pair_type": pair_type(source_domain, eval_domain),

                "num_pairs": int(len(rows_this_pairing)),

                "mean_pair_mass": float(mean_pair_mass),
                "mean_pair_gap": float(mean_pair_gap),
                "mean_pair_ambiguity": float(mean_pair_ambiguity),
                "mean_offpair_mass": float(mean_offpair_mass),
                "mean_entropy": float(mean_entropy),
                "mean_pair_top2_acc": float(mean_pair_top2_acc),
                "mean_edge_ref_drift": float(mean_edge_ref_drift),

                "worst_pair_gap": float(max(float(r["avg_pair_gap"]) for r in rows_this_pairing)),
                "worst_offpair_mass": float(max(float(r["avg_offpair_mass"]) for r in rows_this_pairing)),
                "worst_edge_ref_drift": float(max(float(r["edge_ref_drift"]) for r in rows_this_pairing)),
            })

    return raw_rows, summary_rows


def _mean_absdiff_pair_margin_from_scored(
    *,
    scored_a: List[Dict[str, Any]],
    scored_b: List[Dict[str, Any]],
    pair_i: int,
    pair_j: int,
) -> Dict[str, float]:
    n = min(len(scored_a), len(scored_b))
    if n <= 0:
        return {
            "num_examples": 0,
            "avg_margin_absdiff": 0.0,
        }

    diffs: List[float] = []
    idx_i = int(pair_i)
    idx_j = int(pair_j)
    for idx in range(n):
        probs_a = np.asarray(scored_a[idx].get("all_probs", []), dtype=np.float64).reshape(-1)
        probs_b = np.asarray(scored_b[idx].get("all_probs", []), dtype=np.float64).reshape(-1)
        if probs_a.size <= max(idx_i, idx_j) or probs_b.size <= max(idx_i, idx_j):
            continue
        margin_a = float(probs_a[idx_i]) - float(probs_a[idx_j])
        margin_b = float(probs_b[idx_i]) - float(probs_b[idx_j])
        diffs.append(abs(margin_a - margin_b))

    if not diffs:
        return {
            "num_examples": 0,
            "avg_margin_absdiff": 0.0,
        }

    return {
        "num_examples": int(len(diffs)),
        "avg_margin_absdiff": float(np.mean(np.asarray(diffs, dtype=np.float64))),
    }


def compute_edge_consistency_pair_rows(
    *,
    case_names: List[str],
    case_results: Dict[str, Dict[str, Any]],
    case_models: Dict[str, Any],
    case_out_hw: Dict[str, int],
    device,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    ordered_names = [str(x) for x in case_names]
    for idx_a, case_a in enumerate(ordered_names):
        domain_a = str(case_results[case_a]["domain"])
        model_a = case_models[case_a]
        out_hw_a = int(case_out_hw[case_a])

        for idx_b in range(idx_a + 1, len(ordered_names)):
            case_b = ordered_names[idx_b]
            domain_b = str(case_results[case_b]["domain"])
            model_b = case_models[case_b]
            out_hw_b = int(case_out_hw[case_b])

            rows_this_pair: List[Dict[str, Any]] = []
            for probe_source in (case_a, case_b):
                probe_domain = str(case_results[probe_source]["domain"])
                source_result = case_results[probe_source]["result"]
                top_records_by_pair = _extract_top_records_by_pair(
                    edge_cases=source_result.get("edge_cases", {}),
                )

                for edge_pair_name in sorted(top_records_by_pair.keys(), key=lambda x: tuple(int(v) for v in x.split("_"))):
                    records = top_records_by_pair[edge_pair_name]
                    if len(records) == 0:
                        continue

                    parsed = parse_pair_key(edge_pair_name)
                    if parsed is None:
                        continue
                    pair_i, pair_j = int(parsed[0]), int(parsed[1])

                    scored_a, scored_b = _score_records_on_two_models(
                        records=records,
                        model_a=model_a,
                        dataset_a=domain_a,
                        out_hw_a=out_hw_a,
                        model_b=model_b,
                        dataset_b=domain_b,
                        out_hw_b=out_hw_b,
                        device=device,
                        batch_size=int(batch_size),
                    )
                    stats = _mean_absdiff_pair_margin_from_scored(
                        scored_a=scored_a,
                        scored_b=scored_b,
                        pair_i=int(pair_i),
                        pair_j=int(pair_j),
                    )

                    row = {
                        "case_a": str(case_a),
                        "domain_a": str(domain_a),
                        "case_b": str(case_b),
                        "domain_b": str(domain_b),
                        "pair_type": pair_type(domain_a, domain_b),
                        "probe_source_model": str(probe_source),
                        "probe_source_domain": str(probe_domain),
                        "edge_pair": str(edge_pair_name),
                        "pair_i": int(pair_i),
                        "pair_j": int(pair_j),
                        "num_examples": int(stats["num_examples"]),
                        "avg_margin_absdiff": float(stats["avg_margin_absdiff"]),
                    }
                    raw_rows.append(row)
                    rows_this_pair.append(row)

            vals = [float(row["avg_margin_absdiff"]) for row in rows_this_pair]
            summary_rows.append({
                "case_a": str(case_a),
                "domain_a": str(domain_a),
                "case_b": str(case_b),
                "domain_b": str(domain_b),
                "pair_type": pair_type(domain_a, domain_b),
                "num_rows": int(len(rows_this_pair)),
                "mean_margin_absdiff": safe_mean(vals),
            })

    return raw_rows, summary_rows


def _build_edge_consistency_lookup(
    raw_rows: List[Dict[str, Any]],
) -> Dict[Tuple[str, str, str, int, int], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str, str, int, int], Dict[str, Any]] = {}
    for row in raw_rows:
        lookup[(
            str(row["case_a"]),
            str(row["case_b"]),
            str(row["probe_source_model"]),
            int(row["pair_i"]),
            int(row["pair_j"]),
        )] = row
    return lookup


def _class_edge_values_from_consistency_lookup(
    *,
    lookup: Dict[Tuple[str, str, str, int, int], Dict[str, Any]],
    case_a: str,
    case_b: str,
    class_label: int,
    shared_labels: Sequence[int],
) -> List[float]:
    vals: List[float] = []
    if case_a == case_b:
        return [0.0 for _ in shared_labels if int(_) != int(class_label)]

    key_a, key_b = sorted([case_a, case_b])
    c = int(class_label)
    for q in shared_labels:
        q_int = int(q)
        if q_int == c:
            continue
        pair_i = int(min(c, q_int))
        pair_j = int(max(c, q_int))
        row_a = lookup.get((key_a, key_b, case_a, pair_i, pair_j))
        row_b = lookup.get((key_a, key_b, case_b, pair_i, pair_j))
        if row_a is None or row_b is None:
            continue
        vals.append(float(row_a["avg_margin_absdiff"]) + float(row_b["avg_margin_absdiff"]))
    return vals


def compute_focus_pair_edge_consistency_rows(
    *,
    raw_rows: List[Dict[str, Any]],
    focus_pairs: List[Dict[str, str]],
    num_classes: int,
    shared_labels_by_pair: Optional[Dict[Tuple[str, str], List[int]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    lookup = _build_edge_consistency_lookup(raw_rows)
    out_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for spec in focus_pairs:
        case_a = str(spec["case_a"])
        case_b = str(spec["case_b"])
        group = str(spec["group"])
        display_name = str(spec["display_name"])

        vals_p: List[float] = []
        num_available = 0
        shared_labels = list(
            _shared_labels_for_pair(
                shared_labels_by_pair=shared_labels_by_pair,
                case_a=case_a,
                case_b=case_b,
                num_classes=int(num_classes),
            )
        )

        for c in range(int(num_classes)):
            if int(c) not in set(shared_labels):
                continue

            pair_vals = _class_edge_values_from_consistency_lookup(
                lookup=lookup,
                case_a=case_a,
                case_b=case_b,
                class_label=int(c),
                shared_labels=shared_labels,
            )
            if not pair_vals:
                continue

            p_c = float(np.median(np.asarray(pair_vals, dtype=np.float64)))
            vals_p.append(p_c)
            num_available += 1
            out_rows.append({
                "comparison_group": group,
                "display_name": display_name,
                "case_a": case_a,
                "case_b": case_b,
                "class_label": int(c),
                "num_edge_pairs": int(len(pair_vals)),
                "P_c": float(p_c),
            })

        summary_rows.append({
            "comparison_group": group,
            "display_name": display_name,
            "case_a": case_a,
            "case_b": case_b,
            "num_classes_available": int(num_available),
            "mean_P_c": float(safe_mean(vals_p)),
        })

    return out_rows, summary_rows


def compute_all_pairs_edge_consistency_rows(
    *,
    raw_rows: List[Dict[str, Any]],
    case_names: List[str],
    display_name_map: Dict[str, str],
    num_classes: int,
    shared_labels_by_pair: Optional[Dict[Tuple[str, str], List[int]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    lookup = _build_edge_consistency_lookup(raw_rows)
    per_class_rows: List[Dict[str, Any]] = []
    mean_rows: List[Dict[str, Any]] = []

    ordered_names = [str(x) for x in case_names]
    for idx_a, case_a in enumerate(ordered_names):
        for idx_b in range(idx_a, len(ordered_names)):
            case_b = ordered_names[idx_b]
            vals: List[float] = []
            num_available = 0
            shared_labels = list(
                _shared_labels_for_pair(
                    shared_labels_by_pair=shared_labels_by_pair,
                    case_a=case_a,
                    case_b=case_b,
                    num_classes=int(num_classes),
                )
            )

            for c in range(int(num_classes)):
                if int(c) not in set(shared_labels):
                    continue
                pair_vals = _class_edge_values_from_consistency_lookup(
                    lookup=lookup,
                    case_a=case_a,
                    case_b=case_b,
                    class_label=int(c),
                    shared_labels=shared_labels,
                )
                if not pair_vals:
                    continue

                p_c = float(np.median(np.asarray(pair_vals, dtype=np.float64)))
                vals.append(p_c)
                num_available += 1
                per_class_rows.append({
                    "case_a": display_name_map.get(case_a, case_a),
                    "case_b": display_name_map.get(case_b, case_b),
                    "raw_case_a": case_a,
                    "raw_case_b": case_b,
                    "class_label": int(c),
                    "P_c": float(p_c),
                })

            mean_rows.append({
                "case_a": display_name_map.get(case_a, case_a),
                "case_b": display_name_map.get(case_b, case_b),
                "raw_case_a": case_a,
                "raw_case_b": case_b,
                "num_classes_available": int(num_available),
                "mean_P_c": float(safe_mean(vals)),
            })

    return per_class_rows, mean_rows


def filter_target_centered_edge_transfer_rows(
    *,
    rows: List[Dict[str, Any]],
    target_label: int,
) -> List[Dict[str, Any]]:
    tgt = int(target_label)
    return [
        r for r in rows
        if int(r["pair_i"]) == tgt or int(r["pair_j"]) == tgt
    ]


def rank_edge_transfer_pairs_for_case_pair(
    *,
    raw_rows: List[Dict[str, Any]],
    source_model: str,
    eval_model: str,
    metric_key: str = "edge_ref_drift",
    descending: bool = True,
) -> List[Dict[str, Any]]:
    rows = [
        r for r in raw_rows
        if str(r["source_model"]) == str(source_model)
        and str(r["eval_model"]) == str(eval_model)
    ]
    rows = sorted(rows, key=lambda r: float(r[metric_key]), reverse=bool(descending))
    return rows

# ------------------------------------------------------------
# edge transfer per-class analysis
# ------------------------------------------------------------

def compute_edge_transfer_per_class_rows(
    *,
    raw_rows: List[Dict[str, Any]],
    num_classes: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convert pair-level edge transfer rows into class-level rows.

    Each edge pair (i, j) contributes to both class i and class j.

    Returns
    -------
    per_class_rows:
        one row per (source_model, eval_model, class_label)

    mean_over_classes_rows:
        one row per (source_model, eval_model), aggregated over class labels
    """
    grouped: Dict[Tuple[str, str, str, str, str, int], List[Dict[str, Any]]] = {}

    for row in raw_rows:
        src = str(row["source_model"])
        src_domain = str(row.get("source_domain", ""))
        tgt = str(row["eval_model"])
        tgt_domain = str(row.get("eval_domain", ""))
        ptype = str(row.get("pair_type", ""))

        pair_i = int(row["pair_i"])
        pair_j = int(row["pair_j"])

        for c in [pair_i, pair_j]:
            key = (src, src_domain, tgt, tgt_domain, ptype, int(c))
            grouped.setdefault(key, []).append(row)

    per_class_rows: List[Dict[str, Any]] = []

    for (src, src_domain, tgt, tgt_domain, ptype, c), rows_c in grouped.items():
        if len(rows_c) == 0:
            continue

        mean_pair_mass = safe_mean(float(r["avg_pair_mass"]) for r in rows_c)
        mean_pair_gap = safe_mean(float(r["avg_pair_gap"]) for r in rows_c)
        mean_pair_ambiguity = safe_mean(
            float(r.get("avg_pair_ambiguity", r["pair_top2_acc"])) for r in rows_c
        )
        mean_pair_top2_acc = safe_mean(float(r["pair_top2_acc"]) for r in rows_c)
        mean_edge_ref_drift = safe_mean(float(r["edge_ref_drift"]) for r in rows_c)

        worst_pair_gap = max(float(r["avg_pair_gap"]) for r in rows_c)
        worst_edge_ref_drift = max(float(r["edge_ref_drift"]) for r in rows_c)

        per_class_rows.append({
            "source_model": str(src),
            "source_domain": str(src_domain),
            "eval_model": str(tgt),
            "eval_domain": str(tgt_domain),
            "pair_type": str(ptype),

            "class_label": int(c),
            "num_edge_pairs": int(len(rows_c)),

            "class_mean_pair_mass": float(mean_pair_mass),
            "class_mean_pair_gap": float(mean_pair_gap),
            "class_mean_pair_ambiguity": float(mean_pair_ambiguity),
            "class_mean_pair_top2_acc": float(mean_pair_top2_acc),
            "class_mean_edge_ref_drift": float(mean_edge_ref_drift),

            "class_worst_pair_gap": float(worst_pair_gap),
            "class_worst_edge_ref_drift": float(worst_edge_ref_drift),
        })

    mean_over_classes_rows: List[Dict[str, Any]] = []
    grouped_casepair: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = {}

    for row in per_class_rows:
        key = (
            str(row["source_model"]),
            str(row["source_domain"]),
            str(row["eval_model"]),
            str(row["eval_domain"]),
            str(row["pair_type"]),
        )
        grouped_casepair.setdefault(key, []).append(row)

    for (src, src_domain, tgt, tgt_domain, ptype), rows_ab in grouped_casepair.items():
        if len(rows_ab) == 0:
            continue

        mean_over_classes_rows.append({
            "source_model": str(src),
            "source_domain": str(src_domain),
            "eval_model": str(tgt),
            "eval_domain": str(tgt_domain),
            "pair_type": str(ptype),

            "num_classes": int(len(rows_ab)),

            "mean_over_classes_pair_mass": float(
                safe_mean(float(r["class_mean_pair_mass"]) for r in rows_ab)
            ),
            "mean_over_classes_pair_top2_acc": float(
                safe_mean(float(r["class_mean_pair_top2_acc"]) for r in rows_ab)
            ),
            "mean_over_classes_pair_ambiguity": float(
                safe_mean(float(r["class_mean_pair_ambiguity"]) for r in rows_ab)
            ),
            "mean_over_classes_edge_ref_drift": float(
                safe_mean(float(r["class_mean_edge_ref_drift"]) for r in rows_ab)
            ),
            "worst_over_classes_edge_ref_drift": float(
                max(float(r["class_worst_edge_ref_drift"]) for r in rows_ab)
            ),
        })

    return per_class_rows, mean_over_classes_rows


def rank_edge_transfer_classes_for_case_pair(
    *,
    per_class_rows: List[Dict[str, Any]],
    source_model: str,
    eval_model: str,
    metric_key: str = "class_mean_edge_ref_drift",
    descending: bool = True,
) -> List[Dict[str, Any]]:
    rows = [
        r for r in per_class_rows
        if str(r["source_model"]) == str(source_model)
        and str(r["eval_model"]) == str(eval_model)
    ]
    rows = sorted(rows, key=lambda r: float(r[metric_key]), reverse=bool(descending))
    return rows
