from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import math
import random
import time

import torch

from data.registry import get_dataset_spec

from .pointillism_probe_search import (
    compile_w_plan,
    sample_random_masks_for_d,
    score_mask_records_all_labels,
    accumulate_candidates_by_label,
    summarize_sonar_map,
    get_class_coverage_status,
    get_candidate_pool_for_label,
    prefilter_top_candidates_for_label,
    make_translation_shifts,
    evaluate_candidate_robustness,
    select_topl_for_label,
    weak_local_refine_candidates,
    summarize_entropy_by_d,
    summarize_entropy_by_label_and_d,
    collect_edge_cases_from_records,
)

def _log(log_fn: Optional[Callable[[str], None]], msg: str) -> None:
    if log_fn is not None:
        log_fn(str(msg))


def derive_grid_hws_for_image(*, out_hw: int) -> List[int]:
    grid_hws: List[int] = []
    d = 4
    max_hw = int(out_hw)

    while d <= max_hw:
        grid_hws.append(int(d))
        d *= 2

    if not grid_hws:
        raise ValueError(f"No valid power-of-two grid_hws for out_hw={int(out_hw)}")
    return grid_hws


def build_w_list_for_grid(
    *,
    grid_hw: int,
    w_cfg: Dict[str, Any],
) -> List[int]:
    n_cells = int(grid_hw) * int(grid_hw)

    plan = compile_w_plan(
        grid_hw=int(grid_hw),
        w_schedule=w_cfg,
    )

    w_list = [int(x) for x in plan.search_ws]
    w_list = [x for x in w_list if 0 < x < n_cells]
    w_list = sorted(set(w_list))
    return w_list

def run_one_resolution_round(
    *,
    model,
    grid_hw: int,
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn,
    w_cfg: Dict[str, Any],
    round_budget: int,
    batch_size: int,
    seed: int,
    use_complement: bool = False,
    seen_base_masks: Optional[Set[Tuple[int, ...]]] = None,
    probe_mode: str = "binary",
    two_group_cfg: Optional[Dict[str, Any]] = None,
    num_channels: int = 3,
) -> List[Dict[str, Any]]:
    """
    One random probing round at one resolution d=grid_hw.
    Returns scored records with:
      - flat_idx
      - grid_hw
      - w
      - all_probs
      - pred_label
      - pred_score
    """
    w_list = build_w_list_for_grid(
        grid_hw=int(grid_hw),
        w_cfg=w_cfg,
    )
    if not w_list:
        return []

    round_budget_i = max(0, int(round_budget))
    if round_budget_i == 0:
        return []

    trial_record_cost = 2 if bool(use_complement) else 1
    n_trials_per_w = max(
        1,
        math.ceil(float(round_budget_i) / float(max(1, len(w_list) * trial_record_cost))),
    )

    raw_records = sample_random_masks_for_d(
        grid_hw=int(grid_hw),
        w_list=w_list,
        n_trials_per_w=int(n_trials_per_w),
        seed=int(seed),
        use_complement=bool(use_complement),
        max_records=int(round_budget_i),
        seen_base_masks=seen_base_masks,
        probe_mode=str(probe_mode),
        two_group_cfg=dict(two_group_cfg or {}),
        num_channels=int(num_channels),
    )

    scored_records = score_mask_records_all_labels(
        model=model,
        records=raw_records,
        out_hw=int(out_hw),
        dataset=str(dataset),
        device=device,
        norm_cfg_fn=norm_cfg_fn,
        batch_size=int(batch_size),
    )
    return scored_records

def _total_unique_base_masks_for_grid(
    *,
    grid_hw: int,
    w_cfg: Dict[str, Any],
) -> int:
    w_list = build_w_list_for_grid(
        grid_hw=int(grid_hw),
        w_cfg=w_cfg,
    )
    n_cells = int(grid_hw) * int(grid_hw)
    total = 0
    for w in w_list:
        w_i = int(w)
        if 0 < w_i < n_cells:
            total += int(math.comb(int(n_cells), int(w_i)))
    return int(total)


def _max_target_softmax_by_label_by_d(
    *,
    grid_hws: List[int],
    all_scored_records_by_d: Dict[int, List[Dict[str, Any]]],
    num_classes: int,
) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {
        int(d): [0.0 for _ in range(int(num_classes))]
        for d in grid_hws
    }

    for d in grid_hws:
        best_scores = out[int(d)]
        for rec in all_scored_records_by_d.get(int(d), []):
            probs = rec.get("all_probs", None)
            if not isinstance(probs, list):
                continue
            limit = min(int(num_classes), len(probs))
            for label in range(limit):
                score = float(probs[label])
                if score > best_scores[label]:
                    best_scores[label] = float(score)
    return out


def _route_undercovered_labels_by_grid(
    *,
    grid_hws: List[int],
    undercovered_labels: List[int],
    all_scored_records_by_d: Dict[int, List[Dict[str, Any]]],
    num_classes: int,
    blocked_grid_hws: Set[int],
) -> Tuple[Dict[int, List[int]], Dict[int, Dict[str, Any]], Dict[int, List[float]]]:
    score_by_d = _max_target_softmax_by_label_by_d(
        grid_hws=grid_hws,
        all_scored_records_by_d=all_scored_records_by_d,
        num_classes=int(num_classes),
    )

    grouped: Dict[int, List[int]] = {}
    label_route: Dict[int, Dict[str, Any]] = {}

    for label in undercovered_labels:
        ranked_d = [
            int(d)
            for d in sorted(
                [int(x) for x in grid_hws],
                key=lambda d: (
                    int(int(d) in blocked_grid_hws),
                    -float(score_by_d[int(d)][int(label)]),
                    int(d),
                ),
            )
            if int(d) not in blocked_grid_hws
        ]
        if not ranked_d:
            continue

        best_d = int(ranked_d[0])
        best_score = float(score_by_d[int(best_d)][int(label)])
        grouped.setdefault(int(best_d), []).append(int(label))
        label_route[int(label)] = {
            "best_d": int(best_d),
            "best_score": float(best_score),
        }

    return grouped, label_route, score_by_d


def choose_next_grid_hw(
    *,
    grid_hws: List[int],
    coverage_status: Dict[str, Any],
    round_idx: int,
    visited_grid_hws: Set[int],
    all_scored_records_by_d: Dict[int, List[Dict[str, Any]]],
    num_classes: int,
    exhausted_grid_hws: Set[int],
    revisit_round_counts_by_d: Dict[int, int],
    revisit_round_limit_per_d: int,
) -> Tuple[int, Dict[str, Any]]:
    """
    Search policy:
    1. Visit every grid resolution once in ascending order.
    2. Afterwards, route each uncovered class to the d where it achieved the
       strongest observed target softmax, using only non-exhausted d values.
    3. Revisit the d that currently serves the largest uncovered-label group.
    """
    grid_hws = [int(x) for x in grid_hws]
    if not grid_hws:
        raise ValueError("grid_hws is empty")

    for d in grid_hws:
        if int(d) not in visited_grid_hws:
            return int(d), {
                "mode": "initial_pass",
                "target_labels": [],
                "group_size": 0,
                "group_score_sum": 0.0,
                "exhausted_grid_hws": sorted(int(x) for x in exhausted_grid_hws),
            }

    undercovered = [int(x) for x in coverage_status.get("undercovered_labels", [])]
    if not undercovered:
        for d in grid_hws:
            if int(d) not in exhausted_grid_hws:
                return int(d), {
                    "mode": "fallback_cycle",
                    "target_labels": [],
                    "group_size": 0,
                    "group_score_sum": 0.0,
                    "exhausted_grid_hws": sorted(int(x) for x in exhausted_grid_hws),
                }
        return int(grid_hws[int(round_idx) % len(grid_hws)]), {
            "mode": "fallback_cycle",
            "target_labels": [],
            "group_size": 0,
            "group_score_sum": 0.0,
            "exhausted_grid_hws": sorted(int(x) for x in exhausted_grid_hws),
        }

    capped_grid_hws = {
        int(d)
        for d in grid_hws
        if int(revisit_round_counts_by_d.get(int(d), 0)) >= int(revisit_round_limit_per_d)
    }
    grouped, label_route, score_by_d = _route_undercovered_labels_by_grid(
        grid_hws=grid_hws,
        undercovered_labels=undercovered,
        all_scored_records_by_d=all_scored_records_by_d,
        num_classes=int(num_classes),
        blocked_grid_hws=set(int(x) for x in exhausted_grid_hws).union(capped_grid_hws),
    )

    cycle_reset = False
    if not grouped:
        for d_reset in grid_hws:
            if int(d_reset) not in exhausted_grid_hws:
                revisit_round_counts_by_d[int(d_reset)] = 0
        grouped, label_route, score_by_d = _route_undercovered_labels_by_grid(
            grid_hws=grid_hws,
            undercovered_labels=undercovered,
            all_scored_records_by_d=all_scored_records_by_d,
            num_classes=int(num_classes),
            blocked_grid_hws=set(int(x) for x in exhausted_grid_hws),
        )
        cycle_reset = True

    ranked_groups = []
    for d, labels in grouped.items():
        score_sum = float(sum(float(label_route[int(label)]["best_score"]) for label in labels))
        ranked_groups.append((int(len(labels)), float(score_sum), int(d), list(labels)))

    ranked_groups.sort(key=lambda item: (-int(item[0]), -float(item[1]), int(item[2])))
    for group_size, score_sum, d, labels in ranked_groups:
        return int(d), {
            "mode": "targeted_revisit",
            "target_labels": [int(x) for x in labels],
            "group_size": int(group_size),
            "group_score_sum": float(score_sum),
            "label_route": {int(k): dict(v) for k, v in label_route.items()},
            "score_by_d": {int(k): [float(x) for x in v] for k, v in score_by_d.items()},
            "exhausted_grid_hws": sorted(int(x) for x in exhausted_grid_hws),
            "cycle_reset": bool(cycle_reset),
        }

    for d in grid_hws:
        if int(d) not in exhausted_grid_hws:
            return int(d), {
                "mode": "fallback_non_exhausted",
                "target_labels": [],
                "group_size": 0,
                "group_score_sum": 0.0,
                "exhausted_grid_hws": sorted(int(x) for x in exhausted_grid_hws),
            }

    return int(grid_hws[int(round_idx) % len(grid_hws)]), {
        "mode": "fallback_cycle",
        "target_labels": [],
        "group_size": 0,
        "group_score_sum": 0.0,
        "exhausted_grid_hws": sorted(int(x) for x in exhausted_grid_hws),
    }


def _rank_records_for_label(
    *,
    candidate_pool: List[Dict[str, Any]],
    label: int,
) -> List[Dict[str, Any]]:
    if not candidate_pool:
        return []
    return prefilter_top_candidates_for_label(
        candidate_pool=list(candidate_pool),
        label=int(label),
        prefilter_k=int(len(candidate_pool)),
    )


def _annotate_selected_records(
    *,
    records: List[Dict[str, Any]],
    selection_mode: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in records:
        new_rec = dict(rec)
        score = float(new_rec.get("target_score", 0.0))
        new_rec["selection_score"] = score
        new_rec["selection_score_key"] = "target_score"
        new_rec["selection_mode"] = str(selection_mode)
        out.append(new_rec)
    return out


def _filter_records_predicted_as_label(
    *,
    records: List[Dict[str, Any]],
    label: int,
) -> List[Dict[str, Any]]:
    return [
        dict(rec)
        for rec in records
        if int(rec.get("pred_label", -1)) == int(label)
    ]


def _pack_records_by_d_for_label(
    *,
    records: List[Dict[str, Any]],
    label: int,
) -> Dict[int, Dict[int, List[Dict[str, Any]]]]:
    out: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
    for rec in records:
        d = int(rec.get("grid_hw", -1))
        out.setdefault(int(d), {}).setdefault(int(label), []).append(dict(rec))
    return out


def compute_q1_signature_summary(
    *,
    sonar_map: Dict[int, Dict[str, Any]],
    entropy_summary: Dict[str, Any],
    num_classes: int,
    tau_r: Optional[float] = None,
) -> Dict[str, Any]:
    num_classes_i = int(num_classes)
    if num_classes_i <= 0:
        return {
            "R_f": 0.0,
            "support_labels": [],
            "effective_support_labels": [],
            "tau_R": (float(tau_r) if tau_r is not None else None),
            "is_discriminative": (False if tau_r is not None else None),
            "evidence_by_grid": {},
            "model_reliability": 0.0,
            "model_reliability_by_grid": [],
            "class_evidence_max": [],
            "evidence_by_label_max": [],
        }

    log_c = math.log(float(num_classes_i)) if num_classes_i > 1 else 1.0
    entropy_by_d = dict(entropy_summary.get("by_pred_label_by_d", {}))
    grid_hws = sorted(
        {
            int(k)
            for k in list(sonar_map.keys()) + list(entropy_by_d.keys())
        }
    )

    evidence_by_label_max = [0.0 for _ in range(num_classes_i)]
    evidence_by_grid: Dict[int, Dict[str, Any]] = {}

    for d in grid_hws:
        sonar_rec = sonar_map.get(int(d), sonar_map.get(str(d), {}))
        ratios = list(sonar_rec.get("class_ratios", []))
        if len(ratios) < num_classes_i:
            ratios = ratios + [0.0] * (num_classes_i - len(ratios))

        ent_rec_by_label = entropy_by_d.get(int(d), entropy_by_d.get(str(d), {}))
        q_vals: List[float] = []
        u_vals: List[float] = []
        evidence_vals: List[float] = []
        entropy_vals: List[float] = []
        count_vals: List[int] = []

        sum_evidence = 0.0
        for label in range(num_classes_i):
            entropy_rec = {}
            if isinstance(ent_rec_by_label, dict):
                entropy_rec = ent_rec_by_label.get(int(label), ent_rec_by_label.get(str(label), {}))
            count = int(entropy_rec.get("count", 0))
            entropy_mean = float(entropy_rec.get("entropy_mean", log_c)) if count > 0 else float(log_c)
            q = float(ratios[label])
            u = max(0.0, 1.0 - (entropy_mean / float(log_c))) if log_c > 0.0 else 0.0
            if count <= 0 or q <= 0.0:
                u = 0.0
            evidence = float(q * u)

            evidence_by_label_max[label] = max(evidence_by_label_max[label], evidence)
            sum_evidence += evidence

            q_vals.append(float(q))
            u_vals.append(float(u))
            evidence_vals.append(float(evidence))
            entropy_vals.append(float(entropy_mean))
            count_vals.append(int(count))

        evidence_by_grid[int(d)] = {
            "q": [float(x) for x in q_vals],
            "u": [float(x) for x in u_vals],
            "E": [float(x) for x in evidence_vals],
            "entropy_mean": [float(x) for x in entropy_vals],
            "counts": [int(x) for x in count_vals],
            "sum_E_d": float(sum_evidence),
        }

    support_labels = [int(label) for label, e_max in enumerate(evidence_by_label_max) if float(e_max) > 0.0]
    model_reliability, model_reliability_by_grid = _model_reliability_from_evidence_by_grid(
        evidence_by_grid,
        num_classes=int(num_classes_i),
    )
    r_f = float(model_reliability)
    is_discriminative = None if tau_r is None else bool(r_f >= float(tau_r))
    effective_support_labels = (
        list(support_labels)
        if tau_r is None or bool(is_discriminative)
        else []
    )

    return {
        "R_f": float(r_f),
        "support_labels": [int(x) for x in support_labels],
        "effective_support_labels": [int(x) for x in effective_support_labels],
        "tau_R": (float(tau_r) if tau_r is not None else None),
        "is_discriminative": is_discriminative,
        "evidence_by_grid": evidence_by_grid,
        "model_reliability": float(model_reliability),
        "model_reliability_by_grid": model_reliability_by_grid,
        "class_evidence_max": [float(x) for x in evidence_by_label_max],
        "evidence_by_label_max": [float(x) for x in evidence_by_label_max],
    }


def _entropy_of_distribution(q_vals: List[float]) -> float:
    vals = [max(0.0, float(x)) for x in q_vals]
    total = float(sum(vals))
    if total <= 0.0:
        return 0.0
    ent = 0.0
    for val in vals:
        if float(val) <= 0.0:
            continue
        p = float(val) / float(total)
        ent -= float(p) * math.log(float(p))
    return float(ent)


def _model_reliability_from_evidence_by_grid(
    by_grid: Dict[int, Dict[str, Any]],
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
        diversity = max(0.0, min(1.0, float(q_entropy) / float(log_c))) if log_c > 0.0 else 0.0
        mean_entropy_norm = (
            float(sum(float(h) / float(log_c) for h in h_vals) / float(num_classes_i))
            if log_c > 0.0
            else 1.0
        )
        confidence = max(0.0, min(1.0, 1.0 - float(mean_entropy_norm)))
        rows.append(
            {
                "grid_hw": int(d_raw),
                "reliability_mode": "diversity",
                "diversity": float(diversity),
                "confidence": float(confidence),
                "model_reliability_d": float(diversity),
                "q_entropy": float(q_entropy),
                "mean_entropy_norm": float(mean_entropy_norm),
                "dominant_label": int(max(range(len(q_vals)), key=lambda idx: float(q_vals[idx]))) if q_vals else -1,
                "dominant_q": float(max(q_vals)) if q_vals else 0.0,
            }
        )

    if not rows:
        return 0.0, []
    return float(sum(float(row["model_reliability_d"]) for row in rows) / float(len(rows))), rows


def _select_topl_prioritize_low_res(
    *,
    label: int,
    records_by_label_by_d: Dict[int, Dict[int, List[Dict[str, Any]]]],
    top_l: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ordered_grid_hws = sorted(int(d) for d in records_by_label_by_d.keys())
    selected: List[Dict[str, Any]] = []
    per_d_pool_sizes: Dict[int, int] = {}
    per_d_selected_counts: Dict[int, int] = {}

    for d in ordered_grid_hws:
        pool_d = list(records_by_label_by_d.get(int(d), {}).get(int(label), []))
        per_d_pool_sizes[int(d)] = int(len(pool_d))

        if len(selected) >= int(top_l) or len(pool_d) == 0:
            per_d_selected_counts[int(d)] = 0
            continue

        ranked_d = _rank_records_for_label(
            candidate_pool=pool_d,
            label=int(label),
        )
        remaining = max(0, int(top_l) - len(selected))
        take = list(ranked_d[:remaining])
        per_d_selected_counts[int(d)] = int(len(take))
        selected.extend(
            _annotate_selected_records(
                records=take,
                selection_mode="prioritize_low_res",
            )
        )

    return selected, {
        "mode": "prioritize_low_res",
        "ordered_grid_hws": [int(x) for x in ordered_grid_hws],
        "per_d_pool_sizes": {int(k): int(v) for k, v in per_d_pool_sizes.items()},
        "per_d_selected_counts": {int(k): int(v) for k, v in per_d_selected_counts.items()},
    }


def _allocate_proportional_counts(
    *,
    per_d_pool_sizes: Dict[int, int],
    top_l: int,
) -> Dict[int, int]:
    available = {int(d): max(0, int(v)) for d, v in per_d_pool_sizes.items() if int(v) > 0}
    if not available or int(top_l) <= 0:
        return {int(d): 0 for d in per_d_pool_sizes}

    total_available = int(sum(available.values()))
    target_total = min(int(top_l), total_available)
    alloc = {int(d): 0 for d in per_d_pool_sizes}
    if target_total <= 0:
        return alloc

    remainders: List[Tuple[float, int]] = []
    assigned = 0
    for d in sorted(available.keys()):
        raw = float(target_total) * float(available[int(d)]) / float(total_available)
        base = min(int(available[int(d)]), int(math.floor(raw)))
        alloc[int(d)] = int(base)
        assigned += int(base)
        remainders.append((float(raw - float(base)), int(d)))

    remaining = max(0, int(target_total) - int(assigned))
    remainders.sort(key=lambda item: (-float(item[0]), int(item[1])))
    while remaining > 0:
        moved = False
        for _, d in remainders:
            if alloc[int(d)] < available[int(d)]:
                alloc[int(d)] += 1
                remaining -= 1
                moved = True
                if remaining <= 0:
                    break
        if not moved:
            break
    return alloc


def _select_topl_proportional_top(
    *,
    label: int,
    records_by_label_by_d: Dict[int, Dict[int, List[Dict[str, Any]]]],
    top_l: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ordered_grid_hws = sorted(int(d) for d in records_by_label_by_d.keys())
    per_d_pool_sizes = {
        int(d): int(len(records_by_label_by_d.get(int(d), {}).get(int(label), [])))
        for d in ordered_grid_hws
    }
    quota_by_d = _allocate_proportional_counts(
        per_d_pool_sizes=per_d_pool_sizes,
        top_l=int(top_l),
    )

    selected: List[Dict[str, Any]] = []
    per_d_selected_counts: Dict[int, int] = {}
    for d in ordered_grid_hws:
        pool_d = list(records_by_label_by_d.get(int(d), {}).get(int(label), []))
        take_n = int(quota_by_d.get(int(d), 0))
        if take_n <= 0 or len(pool_d) == 0:
            per_d_selected_counts[int(d)] = 0
            continue
        ranked_d = _rank_records_for_label(
            candidate_pool=pool_d,
            label=int(label),
        )
        take = list(ranked_d[:take_n])
        per_d_selected_counts[int(d)] = int(len(take))
        selected.extend(
            _annotate_selected_records(
                records=take,
                selection_mode="proportional_top",
            )
        )

    return selected, {
        "mode": "proportional_top",
        "ordered_grid_hws": [int(x) for x in ordered_grid_hws],
        "per_d_pool_sizes": {int(k): int(v) for k, v in per_d_pool_sizes.items()},
        "per_d_selected_counts": {int(k): int(v) for k, v in per_d_selected_counts.items()},
        "per_d_quota_counts": {int(k): int(v) for k, v in quota_by_d.items()},
    }


def _select_topl_proportional_random(
    *,
    label: int,
    records_by_label_by_d: Dict[int, Dict[int, List[Dict[str, Any]]]],
    top_l: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ordered_grid_hws = sorted(int(d) for d in records_by_label_by_d.keys())
    per_d_pool_sizes = {
        int(d): int(len(records_by_label_by_d.get(int(d), {}).get(int(label), [])))
        for d in ordered_grid_hws
    }
    quota_by_d = _allocate_proportional_counts(
        per_d_pool_sizes=per_d_pool_sizes,
        top_l=int(top_l),
    )

    selected: List[Dict[str, Any]] = []
    per_d_selected_counts: Dict[int, int] = {}
    sampled_indices_by_d: Dict[int, List[int]] = {}
    for d in ordered_grid_hws:
        pool_d = list(records_by_label_by_d.get(int(d), {}).get(int(label), []))
        take_n = int(quota_by_d.get(int(d), 0))
        if take_n <= 0 or len(pool_d) == 0:
            per_d_selected_counts[int(d)] = 0
            sampled_indices_by_d[int(d)] = []
            continue

        ranked_d = _rank_records_for_label(
            candidate_pool=pool_d,
            label=int(label),
        )
        choose_n = min(int(take_n), int(len(ranked_d)))
        rng = random.Random(int(seed) + int(label) * 10007 + int(d) * 1000003)
        if choose_n >= len(ranked_d):
            chosen_idx = list(range(len(ranked_d)))
        else:
            chosen_idx = sorted(rng.sample(range(len(ranked_d)), int(choose_n)))
        take = [ranked_d[idx] for idx in chosen_idx]
        take.sort(key=lambda rec: float(rec.get("target_score", 0.0)), reverse=True)
        per_d_selected_counts[int(d)] = int(len(take))
        sampled_indices_by_d[int(d)] = [int(x) for x in chosen_idx]
        selected.extend(
            _annotate_selected_records(
                records=take,
                selection_mode="proportional_random",
            )
        )

    return selected, {
        "mode": "proportional_random",
        "ordered_grid_hws": [int(x) for x in ordered_grid_hws],
        "per_d_pool_sizes": {int(k): int(v) for k, v in per_d_pool_sizes.items()},
        "per_d_selected_counts": {int(k): int(v) for k, v in per_d_selected_counts.items()},
        "per_d_quota_counts": {int(k): int(v) for k, v in quota_by_d.items()},
        "per_d_sampled_rank_indices": {int(k): [int(x) for x in v] for k, v in sampled_indices_by_d.items()},
    }


def _allocate_balanced_counts(
    *,
    per_d_pool_sizes: Dict[int, int],
    total_limit: int,
    ordered_grid_hws: List[int],
) -> Dict[int, int]:
    alloc = {int(d): 0 for d in per_d_pool_sizes}
    available = {int(d): max(0, int(per_d_pool_sizes.get(int(d), 0))) for d in ordered_grid_hws}
    active = [int(d) for d in ordered_grid_hws if int(available.get(int(d), 0)) > 0]
    if not active or int(total_limit) <= 0:
        return alloc

    target_total = min(int(total_limit), int(sum(available[int(d)] for d in active)))
    base = int(target_total) // int(len(active))
    remainder = int(target_total) % int(len(active))

    for d in active:
        take = min(int(base), int(available[int(d)]))
        alloc[int(d)] = int(take)

    remaining = int(target_total) - int(sum(alloc.values()))
    for d in active:
        if remaining <= 0:
            break
        if alloc[int(d)] < available[int(d)]:
            extra = min(int(available[int(d)] - alloc[int(d)]), int(remainder if remainder > 0 else 1))
            if extra > 0:
                alloc[int(d)] += int(extra)
                remaining -= int(extra)
                remainder = max(0, int(remainder) - int(extra))

    while remaining > 0:
        moved = False
        for d in active:
            if alloc[int(d)] < available[int(d)]:
                alloc[int(d)] += 1
                remaining -= 1
                moved = True
                if remaining <= 0:
                    break
        if not moved:
            break
    return alloc


def _record_exact_key(rec: Dict[str, Any]) -> Tuple[int, str, Tuple[int, ...]]:
    return (
        int(rec.get("grid_hw", -1)),
        str(rec.get("polarity", "white")).lower(),
        tuple(int(x) for x in rec.get("flat_idx", [])),
    )


def _dedup_records_keep_best(
    *,
    records: List[Dict[str, Any]],
    score_key: str,
) -> List[Dict[str, Any]]:
    best: Dict[Tuple[int, str, Tuple[int, ...]], Dict[str, Any]] = {}
    for rec in records:
        key = _record_exact_key(rec)
        curr_score = float(rec.get(score_key, 0.0))
        prev = best.get(key)
        prev_score = float(prev.get(score_key, 0.0)) if prev is not None else float('-inf')
        if prev is None or curr_score > prev_score:
            best[key] = dict(rec)
    out = list(best.values())
    out.sort(key=lambda rec: float(rec.get(score_key, 0.0)), reverse=True)
    return out


def _mask_iou(rec_a: Dict[str, Any], rec_b: Dict[str, Any]) -> float:
    if int(rec_a.get("grid_hw", -1)) != int(rec_b.get("grid_hw", -1)):
        return 0.0
    if str(rec_a.get("polarity", "white")).lower() != str(rec_b.get("polarity", "white")).lower():
        return 0.0
    a = set(int(x) for x in rec_a.get("flat_idx", []))
    b = set(int(x) for x in rec_b.get("flat_idx", []))
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return float(len(a & b)) / float(len(union))


def _select_diverse_topl(
    *,
    candidates: List[Dict[str, Any]],
    top_l: int,
    score_key: str,
    max_iou_same_d: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ranked = _dedup_records_keep_best(
        records=list(candidates),
        score_key=str(score_key),
    )
    if int(top_l) <= 0:
        return [], {
            "input_count": int(len(candidates)),
            "dedup_count": int(len(ranked)),
            "selected_count": 0,
            "diversity_skipped_count": 0,
            "max_iou_same_d": float(max_iou_same_d),
        }

    selected: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for rec in ranked:
        too_similar = False
        for prev in selected:
            if _mask_iou(rec, prev) > float(max_iou_same_d):
                too_similar = True
                break
        if too_similar:
            skipped.append(rec)
            continue
        selected.append(rec)
        if len(selected) >= int(top_l):
            break

    if len(selected) < int(top_l):
        for rec in skipped:
            selected.append(rec)
            if len(selected) >= int(top_l):
                break

    return selected[: max(1, int(top_l))], {
        "input_count": int(len(candidates)),
        "dedup_count": int(len(ranked)),
        "selected_count": int(min(len(selected), max(1, int(top_l)))),
        "diversity_skipped_count": int(len(skipped)),
        "max_iou_same_d": float(max_iou_same_d),
    }


def _build_missing_label_seed_pool(
    *,
    label: int,
    grid_hws: List[int],
    all_scored_records_by_d: Dict[int, List[Dict[str, Any]]],
    seed_budget_total: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ordered_grid_hws = [int(d) for d in sorted(int(x) for x in grid_hws)]
    ranked_by_d: Dict[int, List[Dict[str, Any]]] = {
        int(d): _rank_records_for_label(
            candidate_pool=list(all_scored_records_by_d.get(int(d), [])),
            label=int(label),
        )
        for d in ordered_grid_hws
    }
    per_d_pool_sizes = {int(d): int(len(ranked_by_d[int(d)])) for d in ordered_grid_hws}
    per_d_best_target_score: Dict[int, float] = {}
    per_d_mean_top5_target_score: Dict[int, float] = {}
    for d in ordered_grid_hws:
        ranked_d = list(ranked_by_d.get(int(d), []))
        if ranked_d:
            per_d_best_target_score[int(d)] = float(ranked_d[0].get("target_score", 0.0))
            top5 = ranked_d[: min(5, len(ranked_d))]
            per_d_mean_top5_target_score[int(d)] = float(
                sum(float(rec.get("target_score", 0.0)) for rec in top5) / float(len(top5))
            )
        else:
            per_d_best_target_score[int(d)] = 0.0
            per_d_mean_top5_target_score[int(d)] = 0.0

    quota_by_d = _allocate_balanced_counts(
        per_d_pool_sizes=per_d_pool_sizes,
        total_limit=int(seed_budget_total),
        ordered_grid_hws=ordered_grid_hws,
    )

    seeds: List[Dict[str, Any]] = []
    per_d_selected_counts: Dict[int, int] = {}
    for d in ordered_grid_hws:
        take_n = int(quota_by_d.get(int(d), 0))
        ranked_d = list(ranked_by_d.get(int(d), []))
        take = list(ranked_d[:take_n])
        per_d_selected_counts[int(d)] = int(len(take))
        seeds.extend(take)

    seed_scores = [float(rec.get("target_score", 0.0)) for rec in seeds]
    top5_seed_scores = sorted(seed_scores, reverse=True)[: min(5, len(seed_scores))]
    best_seed_score = float(max(seed_scores)) if seed_scores else 0.0
    mean_seed_score_top5 = float(sum(top5_seed_scores) / float(len(top5_seed_scores))) if top5_seed_scores else 0.0
    best_d = int(max(ordered_grid_hws, key=lambda d: float(per_d_best_target_score.get(int(d), 0.0)))) if ordered_grid_hws else -1

    return seeds, {
        "mode": "missing_label_seed_pool",
        "ordered_grid_hws": [int(x) for x in ordered_grid_hws],
        "per_d_pool_sizes": {int(k): int(v) for k, v in per_d_pool_sizes.items()},
        "per_d_seed_quota": {int(k): int(v) for k, v in quota_by_d.items()},
        "per_d_seed_counts": {int(k): int(v) for k, v in per_d_selected_counts.items()},
        "per_d_best_target_score": {int(k): float(v) for k, v in per_d_best_target_score.items()},
        "per_d_mean_top5_target_score": {int(k): float(v) for k, v in per_d_mean_top5_target_score.items()},
        "best_d": int(best_d),
        "best_seed_score": float(best_seed_score),
        "mean_seed_score_top5": float(mean_seed_score_top5),
        "seed_count_total": int(len(seeds)),
    }


def _build_missing_label_fallback_topl(
    *,
    model,
    label: int,
    base_pool: List[Dict[str, Any]],
    grid_hws: List[int],
    all_scored_records_by_d: Dict[int, List[Dict[str, Any]]],
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn,
    topl_cfg: Dict[str, Any],
    refine_cfg: Dict[str, Any],
    batch_size: int,
    seed: int,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    top_l = int(topl_cfg.get("top_l", 64))
    diversity_iou_threshold = float(topl_cfg.get("diversity_iou_threshold", 0.8))
    seed_budget_total = max(int(top_l), int(refine_cfg.get("max_candidates", 32)))

    seed_pool, seed_info = _build_missing_label_seed_pool(
        label=int(label),
        grid_hws=grid_hws,
        all_scored_records_by_d=all_scored_records_by_d,
        seed_budget_total=int(seed_budget_total),
    )

    direct_hit_count = int(len(base_pool))
    refine_enabled = bool(refine_cfg.get("enabled", False))
    if log_fn is not None:
        best_by_d = seed_info.get("per_d_best_target_score", {})
        best_str = ",".join(f"{int(d)}:{float(best_by_d.get(int(d), 0.0)):.3f}" for d in seed_info.get("ordered_grid_hws", []))
        seed_counts = seed_info.get("per_d_seed_counts", {})
        seed_str = ",".join(f"{int(d)}:{int(seed_counts.get(int(d), 0))}" for d in seed_info.get("ordered_grid_hws", []))
        mode_str = "refine" if refine_enabled else "seed_only"
        _log(
            log_fn,
            f"[Pointillism][miss] y={int(label)} direct={int(direct_hit_count)} best={{{best_str}}} seeds={{{seed_str}}} mode={mode_str}",
        )

    refined: List[Dict[str, Any]] = []
    refine_info: Dict[str, Any] = {"enabled": False}

    if refine_enabled and seed_pool:
        grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
        for rec in seed_pool:
            key = (int(rec.get("grid_hw", -1)), str(rec.get("polarity", "white")).lower())
            grouped.setdefault(key, []).append(rec)

        refine_info = {"enabled": True, "groups": {}}
        for (d, polarity), group in grouped.items():
            group_refined, info = weak_local_refine_candidates(
                model=model,
                candidates=group,
                label=int(label),
                grid_hw=int(d),
                out_hw=int(out_hw),
                dataset=str(dataset),
                device=device,
                norm_cfg_fn=norm_cfg_fn,
                batch_size=int(batch_size),
                max_candidates=min(int(refine_cfg.get("max_candidates", 32)), int(len(group))),
                max_edit_steps=int(refine_cfg.get("max_edit_steps", 2)),
                seed=int(seed) + int(label) * 1000 + int(d) + (0 if str(polarity) == "white" else 1_000_000),
                log_fn=log_fn,
            )
            refined.extend(group_refined)
            refine_info["groups"][f"d={int(d)}|pol={str(polarity)}"] = info

    refined_scores = [float(rec.get("target_score", rec.get("score", 0.0))) for rec in refined]
    best_refined_score = float(max(refined_scores)) if refined_scores else None
    best_improvement = (
        float(best_refined_score) - float(seed_info.get("best_seed_score", 0.0))
        if best_refined_score is not None else None
    )
    if log_fn is not None:
        if refine_enabled:
            seed_best = float(seed_info.get("best_seed_score", 0.0))
            refined_best = float(best_refined_score) if best_refined_score is not None else seed_best
            _log(
                log_fn,
                f"[Pointillism][refine] y={int(label)} seeds={int(len(seed_pool))} refined={int(len(refined))} best={seed_best:.3f}->{refined_best:.3f}",
            )
        else:
            _log(
                log_fn,
                f"[Pointillism][refine] y={int(label)} seeds={int(len(seed_pool))} refined=0 best={float(seed_info.get('best_seed_score', 0.0)):.3f}->n/a",
            )

    candidate_pool = list(base_pool) + list(seed_pool) + list(refined)
    for rec in candidate_pool:
        rec.setdefault("selection_score", float(rec.get("target_score", 0.0)))
        rec.setdefault("selection_score_key", "target_score")
        rec.setdefault(
            "selection_mode",
            "missing_label_refine_fallback" if bool(refine_enabled) else "missing_label_seed_fallback",
        )

    topl, diversity_info = _select_diverse_topl(
        candidates=candidate_pool,
        top_l=int(top_l),
        score_key="target_score",
        max_iou_same_d=float(diversity_iou_threshold),
    )
    topl = _annotate_selected_records(
        records=topl,
        selection_mode=("missing_label_refine_fallback" if bool(refine_enabled) else "missing_label_seed_fallback"),
    )

    return {
        "label": int(label),
        "candidate_pool_size": int(len(candidate_pool)),
        "prefilter_size": int(len(seed_pool)),
        "refine_info": refine_info,
        "selection_mode": ("missing_label_refine_fallback" if bool(refine_enabled) else "missing_label_seed_fallback"),
        "selection_score_key": "target_score",
        "selection_info": {
            **seed_info,
            "direct_hit_count": int(direct_hit_count),
            "fallback_mode": ("refine" if bool(refine_enabled) else "seed_only"),
            "refined_count": int(len(refined)),
            "best_refined_score": (float(best_refined_score) if best_refined_score is not None else None),
            "best_improvement": (float(best_improvement) if best_improvement is not None else None),
            "final_selected_count": int(len(topl)),
            "diversity": diversity_info,
            "source_pool": "all_init_records_by_target_score",
        },
        "topl": topl,
    }


def build_topl_for_label(
    *,
    model,
    label: int,
    grid_hws: List[int],
    records_by_label_by_d: Dict[int, Dict[int, List[Dict[str, Any]]]],
    all_scored_records_by_d: Dict[int, List[Dict[str, Any]]],
    coverage_status: Dict[str, Any],
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn,
    topl_cfg: Dict[str, Any],
    refine_cfg: Dict[str, Any],
    batch_size: int,
    seed: int,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Build Top-L representative masks for one label from pooled random candidates.
    """
    top_l = int(topl_cfg.get("top_l", 64))
    prefilter_ratio = int(topl_cfg.get("prefilter_ratio", 4))
    shift_radius_pct = float(topl_cfg.get("shift_radius_pct", 0.25))
    diversity_iou_threshold = float(topl_cfg.get("diversity_iou_threshold", 0.8))
    refine_enabled = bool(refine_cfg.get("enabled", False))
    undercovered_labels = set(int(x) for x in coverage_status.get("undercovered_labels", []))

    pool = get_candidate_pool_for_label(
        all_scored_records_by_d=all_scored_records_by_d,
        label=int(label),
    )
    direct_hit_pool = _filter_records_predicted_as_label(
        records=pool,
        label=int(label),
    )

    if int(label) in undercovered_labels:
        return _build_missing_label_fallback_topl(
            model=model,
            label=int(label),
            base_pool=direct_hit_pool,
            grid_hws=grid_hws,
            all_scored_records_by_d=all_scored_records_by_d,
            out_hw=int(out_hw),
            dataset=str(dataset),
            device=device,
            norm_cfg_fn=norm_cfg_fn,
            topl_cfg=topl_cfg,
            refine_cfg=refine_cfg,
            batch_size=int(batch_size),
            seed=int(seed),
            log_fn=log_fn,
        )

    prefilter_k = max(top_l, prefilter_ratio * top_l)
    prefiltered = prefilter_top_candidates_for_label(
        candidate_pool=pool,
        label=int(label),
        prefilter_k=int(prefilter_k),
    )
    direct_prefiltered = _filter_records_predicted_as_label(
        records=prefiltered,
        label=int(label),
    )

    refined: List[Dict[str, Any]] = []
    refine_info: Dict[str, Any] = {"enabled": False, "reason": "not_needed"}

    if refine_enabled and len(prefiltered) > 0 and len(direct_prefiltered) < int(top_l):
        grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
        for rec in prefiltered:
            d = int(rec["grid_hw"])
            polarity = str(rec.get("polarity", "white")).lower()
            grouped.setdefault((int(d), str(polarity)), []).append(rec)

        refined = []
        refine_info = {"enabled": True, "groups": {}}

        for (d, polarity), group in grouped.items():
            group_refined, info = weak_local_refine_candidates(
                model=model,
                candidates=group,
                label=int(label),
                grid_hw=int(d),
                out_hw=int(out_hw),
                dataset=str(dataset),
                device=device,
                norm_cfg_fn=norm_cfg_fn,
                batch_size=int(batch_size),
                max_candidates=int(refine_cfg.get("max_candidates", 32)),
                max_edit_steps=int(refine_cfg.get("max_edit_steps", 2)),
                seed=int(seed) + int(d) + int(label) * 1000 + (0 if str(polarity) == "white" else 1_000_000),
                log_fn=log_fn,
            )
            refined.extend(group_refined)
            refine_info["groups"][f"d={int(d)}|pol={str(polarity)}"] = info

    candidate_hits = _filter_records_predicted_as_label(
        records=list(direct_prefiltered) + list(refined),
        label=int(label),
    )
    candidate_hits_by_d = _pack_records_by_d_for_label(
        records=candidate_hits,
        label=int(label),
    )

    if shift_radius_pct <= 0.0:
        selection_mode = str(topl_cfg.get("selection_mode", "prioritize_low_res")).strip().lower()
        if selection_mode == "prioritize_low_res":
            topl, selection_info = _select_topl_prioritize_low_res(
                label=int(label),
                records_by_label_by_d=candidate_hits_by_d,
                top_l=int(top_l),
            )
        elif selection_mode == "proportional_top":
            topl, selection_info = _select_topl_proportional_top(
                label=int(label),
                records_by_label_by_d=candidate_hits_by_d,
                top_l=int(top_l),
            )
        elif selection_mode == "proportional_random":
            topl, selection_info = _select_topl_proportional_random(
                label=int(label),
                records_by_label_by_d=candidate_hits_by_d,
                top_l=int(top_l),
                seed=int(seed),
            )
        else:
            raise ValueError(f"Unknown topl.selection_mode for no-robustness mode: {selection_mode}")
        topl, diversity_info = _select_diverse_topl(
            candidates=topl,
            top_l=int(top_l),
            score_key="target_score",
            max_iou_same_d=float(diversity_iou_threshold),
        )
        topl = _annotate_selected_records(records=topl, selection_mode=str(selection_mode))
        return {
            "label": int(label),
            "candidate_pool_size": int(len(pool)),
            "prefilter_size": int(len(prefiltered)),
            "refine_info": refine_info,
            "selection_mode": str(selection_mode),
            "selection_score_key": "target_score",
            "selection_info": {
                **selection_info,
                "prefilter_hits": int(len(direct_prefiltered)),
                "post_refine_hits": int(len(candidate_hits)),
                "diversity": diversity_info,
            },
            "topl": topl,
        }

    shift_include_origin = bool(topl_cfg.get("shift_include_origin", False))

    robust_scored: List[Dict[str, Any]] = []
    grouped_for_robust: Dict[int, List[Dict[str, Any]]] = {}
    for rec in candidate_hits:
        d = int(rec["grid_hw"])
        grouped_for_robust.setdefault(d, []).append(rec)

    for d, group in grouped_for_robust.items():
        shift_radius = max(1, int(round(float(d) * shift_radius_pct)))
        shifts = make_translation_shifts(
            grid_hw=int(d),
            radius=int(shift_radius),
            include_origin=bool(shift_include_origin),
        )

        robust_group = evaluate_candidate_robustness(
            model=model,
            candidates=group,
            label=int(label),
            out_hw=int(out_hw),
            dataset=str(dataset),
            device=device,
            norm_cfg_fn=norm_cfg_fn,
            shifts=shifts,
            batch_size=int(topl_cfg.get("robustness_batch_size", batch_size)),
        )
        robust_scored.extend(robust_group)

    topl, diversity_info = _select_diverse_topl(
        candidates=robust_scored,
        top_l=int(top_l),
        score_key="robust_target_mean",
        max_iou_same_d=float(diversity_iou_threshold),
    )
    topl = [
        {
            **dict(rec),
            "selection_score": float(rec.get("robust_target_mean", 0.0)),
            "selection_score_key": "robust_target_mean",
            "selection_mode": "robustness_rank",
        }
        for rec in topl
    ]

    return {
        "label": int(label),
        "candidate_pool_size": int(len(pool)),
        "prefilter_size": int(len(prefiltered)),
        "refine_info": refine_info,
        "selection_mode": "robustness_rank",
        "selection_score_key": "robust_target_mean",
        "selection_info": {
            "mode": "robustness_rank",
            "prefilter_hits": int(len(direct_prefiltered)),
            "post_refine_hits": int(len(candidate_hits)),
            "diversity": diversity_info,
        },
        "topl": topl,
    }
    

def build_topl_all_labels(
    *,
    model,
    num_classes: int,
    grid_hws: List[int],
    records_by_label_by_d: Dict[int, Dict[int, List[Dict[str, Any]]]],
    all_scored_records_by_d: Dict[int, List[Dict[str, Any]]],
    coverage_status: Dict[str, Any],
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn,
    topl_cfg: Dict[str, Any],
    refine_cfg: Dict[str, Any],
    batch_size: int,
    seed: int,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for label in range(int(num_classes)):
        out[int(label)] = build_topl_for_label(
            model=model,
            label=int(label),
            grid_hws=grid_hws,
            records_by_label_by_d=records_by_label_by_d,
            all_scored_records_by_d=all_scored_records_by_d,
            coverage_status=coverage_status,
            out_hw=int(out_hw),
            dataset=str(dataset),
            device=device,
            norm_cfg_fn=norm_cfg_fn,
            topl_cfg=topl_cfg,
            refine_cfg=refine_cfg,
            batch_size=int(batch_size),
            seed=int(seed),
            log_fn=log_fn,
        )
    return out

def build_pointillism_outputs(
    *,
    grid_hws: List[int],
    sonar_map: Dict[int, Dict[str, Any]],
    records_by_label: Dict[int, List[Dict[str, Any]]],
    records_by_label_by_d: Dict[int, Dict[int, List[Dict[str, Any]]]],
    coverage_status: Dict[str, Any],
    topl_by_label: Dict[int, Dict[str, Any]],
    entropy_summary: Dict[str, Any],
    edge_cases: Dict[str, Any],
    q1_summary: Dict[str, Any],
    search_trace: List[Dict[str, Any]],
    stop_reason: str,
    elapsed_sec: float,
) -> Dict[str, Any]:
    return {
        "grid_hws": [int(x) for x in grid_hws],
        "sonar_map": sonar_map,
        "coverage_status": coverage_status,
        "candidate_pool_by_label": {
            int(k): v for k, v in records_by_label.items()
        },
        "candidate_pool_by_label_by_d": {
            int(d): {int(k): v for k, v in by_lab.items()}
            for d, by_lab in records_by_label_by_d.items()
        },
        "topl_by_label": topl_by_label,
        "entropy_summary": entropy_summary,
        "edge_cases": edge_cases,
        "q1": q1_summary,
        "search_trace": search_trace,
        "stop_reason": str(stop_reason),
        "elapsed_sec": float(elapsed_sec),
    }
    
def run_pointillism_search(
    *,
    model,
    num_classes: int,
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn,
    cfg: Dict[str, Any],
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Pointillism:
      Stage 1: budgeted multi-resolution random probing
      Stage 2: Top-L representative mask extraction
    """
    t0 = time.time()

    min_candidates_per_class = int(cfg.get("min_candidates_per_class", 128))
    batch_size = int(cfg.get("batch_size", 4096))
    seed = int(cfg.get("seed", 123))
    probe_mode = str(cfg.get("probe_mode", "binary")).strip().lower()
    two_group_cfg = dict(cfg.get("two_group", cfg.get("color_group", {})) or {})
    w_cfg = dict(cfg.get("w_schedule", {}))
    use_complement = bool(cfg.get("use_complement", w_cfg.get("use_complement", False)))
    topl_cfg = dict(cfg.get("topl", {}))
    refine_cfg = dict(cfg.get("refine", {}))

    grid_hws_cfg = cfg.get("grid_hws", None)
    if grid_hws_cfg is None:
        grid_hws = derive_grid_hws_for_image(out_hw=int(out_hw))
    else:
        grid_hws = sorted({int(x) for x in list(grid_hws_cfg) if int(x) > 1})
        if not grid_hws:
            raise ValueError("search.grid_hws must contain at least one value > 1")

    if "round_budget_per_d" in cfg:
        round_budget_per_d = int(cfg.get("round_budget_per_d", 5000))
    else:
        round_budget_per_d = int(cfg.get("random_budget_per_grid", 5000))

    if "total_budget" in cfg:
        total_budget = int(cfg.get("total_budget", 100000))
    elif "random_budget_per_grid" in cfg:
        total_budget = int(round_budget_per_d) * int(len(grid_hws))
    else:
        total_budget = 100000

    try:
        dataset_channels = int(get_dataset_spec(str(dataset)).in_channels)
    except Exception:
        dataset_channels = 3
    in_channels = int(cfg.get("num_channels", cfg.get("in_channels", dataset_channels)))
    if probe_mode == "two_group":
        probe_mode = "two_group_gray" if int(in_channels) == 1 else "two_group_rgb"
    elif probe_mode == "gaussian":
        probe_mode = "gaussian_gray" if int(in_channels) == 1 else "gaussian_rgb"
    if probe_mode.startswith(("two_group", "gaussian")):
        two_group_cfg.setdefault("num_channels", int(in_channels))

    _log(log_fn, f"[Pointillism] derived grid_hws={grid_hws} from out_hw={int(out_hw)}")

    records_by_label: Dict[int, List[Dict[str, Any]]] = {
        int(i): [] for i in range(int(num_classes))
    }
    records_by_label_by_d: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
    all_scored_records_by_d: Dict[int, List[Dict[str, Any]]] = {}
    seen_base_masks_by_d: Dict[int, Set[Tuple[int, ...]]] = {
        int(d): set() for d in grid_hws
    }
    search_space_size_by_d: Dict[int, int] = {
        int(d): _total_unique_base_masks_for_grid(
            grid_hw=int(d),
            w_cfg=w_cfg,
        )
        for d in grid_hws
    }
    visited_grid_hws: Set[int] = set()
    exhausted_grid_hws: Set[int] = set()
    revisit_round_limit_per_d = max(1, int(cfg.get("revisit_round_limit_per_d", len(grid_hws))))
    revisit_round_counts_by_d: Dict[int, int] = {int(d): 0 for d in grid_hws}
    search_trace: List[Dict[str, Any]] = []

    total_trials_used = 0
    stop_reason = "budget_exhausted"
    round_idx = 0

    while total_trials_used < int(total_budget):
        remaining_budget = max(0, int(total_budget) - int(total_trials_used))
        if remaining_budget <= 0:
            stop_reason = "budget_exhausted"
            break

        coverage_status = get_class_coverage_status(
            records_by_label=records_by_label,
            num_classes=int(num_classes),
            min_candidates_per_class=int(min_candidates_per_class),
        )
        d, route_info = choose_next_grid_hw(
            grid_hws=grid_hws,
            coverage_status=coverage_status,
            round_idx=int(round_idx),
            visited_grid_hws=visited_grid_hws,
            all_scored_records_by_d=all_scored_records_by_d,
            num_classes=int(num_classes),
            exhausted_grid_hws=exhausted_grid_hws,
            revisit_round_counts_by_d=revisit_round_counts_by_d,
            revisit_round_limit_per_d=int(revisit_round_limit_per_d),
        )

        round_seed = int(seed) + int(round_idx) * 10007 + int(d)
        round_budget = min(int(round_budget_per_d), int(remaining_budget))
        visited_grid_hws.add(int(d))

        scored_records = run_one_resolution_round(
            model=model,
            grid_hw=int(d),
            out_hw=int(out_hw),
            dataset=str(dataset),
            device=device,
            norm_cfg_fn=norm_cfg_fn,
            w_cfg=w_cfg,
            round_budget=int(round_budget),
            batch_size=int(batch_size),
            seed=int(round_seed),
            use_complement=bool(use_complement),
            seen_base_masks=seen_base_masks_by_d.setdefault(int(d), set()),
            probe_mode=str(probe_mode),
            two_group_cfg=two_group_cfg,
            num_channels=int(in_channels),
        )

        if scored_records:
            all_scored_records_by_d.setdefault(int(d), []).extend(scored_records)

        accumulate_candidates_by_label(
            scored_records=scored_records,
            records_by_label=records_by_label,
            records_by_label_by_d=records_by_label_by_d,
        )

        if str(route_info.get("mode", "")) != "initial_pass":
            revisit_round_counts_by_d[int(d)] = int(revisit_round_counts_by_d.get(int(d), 0)) + 1

        total_trials_used += int(len(scored_records))
        coverage_status = get_class_coverage_status(
            records_by_label=records_by_label,
            num_classes=int(num_classes),
            min_candidates_per_class=int(min_candidates_per_class),
        )
        seen_count_for_d = int(len(seen_base_masks_by_d.get(int(d), set())))
        if seen_count_for_d >= int(search_space_size_by_d.get(int(d), 0)):
            exhausted_grid_hws.add(int(d))
        if len(scored_records) == 0:
            exhausted_grid_hws.add(int(d))

        trace_row = {
            "round_idx": int(round_idx),
            "grid_hw": int(d),
            "new_trials": int(len(scored_records)),
            "total_trials_used": int(total_trials_used),
            "coverage_counts": list(coverage_status["counts"]),
            "undercovered_labels": list(coverage_status["undercovered_labels"]),
            "route_mode": str(route_info.get("mode", "")),
            "route_target_labels": [int(x) for x in route_info.get("target_labels", [])],
            "route_group_size": int(route_info.get("group_size", 0)),
            "route_group_score_sum": float(route_info.get("group_score_sum", 0.0)),
            "seen_base_masks_for_d": int(seen_count_for_d),
            "search_space_size_for_d": int(search_space_size_by_d.get(int(d), 0)),
            "exhausted_grid_hws": sorted(int(x) for x in exhausted_grid_hws),
            "revisit_round_count_for_d": int(revisit_round_counts_by_d.get(int(d), 0)),
        }
        search_trace.append(trace_row)

        _log(
            log_fn,
            f"[Pointillism] r={round_idx} d={d} "
            f"new={len(scored_records)} total={total_trials_used} "
            f"cov={len(coverage_status['covered_labels'])}/{num_classes} "
            f"mode={str(route_info.get('mode', ''))} "
            f"tgt={list(route_info.get('target_labels', []))}",
        )

        if total_trials_used >= int(total_budget):
            stop_reason = "budget_exhausted"
            break
        if len(exhausted_grid_hws) >= len(grid_hws):
            stop_reason = "search_space_exhausted"
            break
        round_idx += 1

    if total_trials_used < int(total_budget) and stop_reason == "budget_exhausted":
        stop_reason = "search_complete"

    final_sonar_map = summarize_sonar_map(
        records_by_label_by_d=records_by_label_by_d,
        num_classes=int(num_classes),
    )
    final_coverage_status = get_class_coverage_status(
        records_by_label=records_by_label,
        num_classes=int(num_classes),
        min_candidates_per_class=int(min_candidates_per_class),
    )
    _log(
        log_fn,
        f"[Pointillism][init] covered={len(final_coverage_status['covered_labels'])}/{num_classes} missing={list(final_coverage_status['undercovered_labels'])}",
    )

    topl_by_label = build_topl_all_labels(
        model=model,
        num_classes=int(num_classes),
        grid_hws=grid_hws,
        records_by_label_by_d=records_by_label_by_d,
        all_scored_records_by_d=all_scored_records_by_d,
        coverage_status=final_coverage_status,
        out_hw=int(out_hw),
        dataset=str(dataset),
        device=device,
        norm_cfg_fn=norm_cfg_fn,
        topl_cfg=topl_cfg,
        refine_cfg=refine_cfg,
        batch_size=int(batch_size),
        seed=int(seed),
        log_fn=log_fn,
    )

    entropy_summary = {
        "by_d": summarize_entropy_by_d(
            all_scored_records_by_d=all_scored_records_by_d,
        ),
        "by_pred_label_by_d": summarize_entropy_by_label_and_d(
            all_scored_records_by_d=all_scored_records_by_d,
            num_classes=int(num_classes),
            label_key="pred_label",
        ),
    }

    edge_cases = collect_edge_cases_from_records(
        all_scored_records_by_d=all_scored_records_by_d,
        num_classes=int(num_classes),
        top_k_per_pair=int(cfg.get("edge_top_k_per_pair", 64)),
    )
    tau_r_cfg = cfg.get("tau_r", cfg.get("tau_R", None))
    q1_summary = compute_q1_signature_summary(
        sonar_map=final_sonar_map,
        entropy_summary=entropy_summary,
        num_classes=int(num_classes),
        tau_r=(None if tau_r_cfg is None else float(tau_r_cfg)),
    )

    elapsed_sec = time.time() - t0

    out = build_pointillism_outputs(
        grid_hws=grid_hws,
        sonar_map=final_sonar_map,
        records_by_label=records_by_label,
        records_by_label_by_d=records_by_label_by_d,
        coverage_status=final_coverage_status,
        topl_by_label=topl_by_label,
        entropy_summary=entropy_summary,
        edge_cases=edge_cases,
        q1_summary=q1_summary,
        search_trace=search_trace,
        stop_reason=stop_reason,
        elapsed_sec=float(elapsed_sec),
    )
    return out
