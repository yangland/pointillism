from __future__ import annotations

import math
import csv
from time import perf_counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from aggregation import fedavg as fedavg_mod
from aggregation import deepsight as deepsight_mod
from aggregation import flame as flame_mod
from aggregation import krum as krum_mod
from aggregation import normbound as normbound_mod
from aggregation import rfa as rfa_mod
from pointillism.fl_defense import (
    PointillismFLState,
    _cfg_point,
    _dataset_name,
    _get_state,
    _num_classes,
    build_client_signature,
    build_round_probe_pool,
    collect_refinement_specs,
    compute_pairwise_consensus,
    compute_q1_summary_for_client,
    log_round_outputs,
    merge_probe_pools,
    save_state_snapshot,
    score_probe_specs,
    select_client_subset,
    update_memory_bank,
)


def _round_malicious_by_cid(*, recorder, rnd: int, round_client_ids: List[int]) -> Dict[int, int]:
    selected = {int(cid) for cid in round_client_ids}
    out: Dict[int, int] = {}

    for row in list(getattr(recorder, "_sel_buf", []) or []):
        if len(row) < 3:
            continue
        if int(row[0]) == int(rnd) and int(row[1]) in selected:
            out[int(row[1])] = int(row[2])

    sel_path = getattr(recorder, "sel_path", None)
    if sel_path is not None and Path(str(sel_path)).exists():
        with open(str(sel_path), newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if int(row.get("round", -1)) != int(rnd):
                    continue
                cid = int(row.get("cid", -1))
                if cid in selected:
                    out[int(cid)] = int(row.get("is_malicious", 0))

    return out


def _dispatch_base_aggregate(
    *,
    base_agg: str,
    cfg: Dict[str, Any],
    device: torch.device,
    recorder,
    global_state,
    client_states,
    client_weights,
    model_fn,
    rnd,
    round_client_ids,
):
    key = str(base_agg).lower()
    common = dict(
        cfg=cfg,
        device=device,
        recorder=recorder,
        global_state=global_state,
        client_states=client_states,
        client_weights=client_weights,
        model_fn=model_fn,
        rnd=rnd,
        round_client_ids=round_client_ids,
    )
    if key == "fedavg":
        return fedavg_mod.aggregate(**common)
    if key == "normbound":
        return normbound_mod.aggregate(**common)
    if key == "rfa":
        return rfa_mod.aggregate(**common)
    if key == "krum":
        return krum_mod.aggregate(**common)
    if key == "flame":
        return flame_mod.aggregate(**common)
    if key == "deepsight":
        return deepsight_mod.aggregate(**common)
    raise ValueError(f"Unsupported pointillism_fl.base_agg: {base_agg}")


def _log_step_time(recorder, rnd: int, step: str, elapsed_s: float) -> None:
    if recorder is None or not bool(getattr(recorder, "log_pointillism_step_times", True)):
        return
    recorder.maybe_print(
        f"[pointillism_fl] rnd={int(rnd)} step={step} time={float(elapsed_s):.3f}s"
    )


def audit_client_states(
    *, cfg, device, client_states, model_fn, rnd, round_client_ids,
    private_probe_seed,
):
    """Score candidates on an attacker-generated pool, isolated from server state."""
    # The attacker knows the generation algorithm, but does not receive the
    # server's probes, random-pool history, or memory bank. Recreate a private
    # state for every audit; repeated audits share a pool via the private seed.
    audit_cfg = deepcopy(cfg)
    audit_cfg["seed"] = int(private_probe_seed)
    cfg_point = _cfg_point(audit_cfg)
    models = []
    try:
        for client_state in client_states:
            model = model_fn().to(device)
            model.load_state_dict(client_state, strict=True)
            model.eval()
            models.append(model)
        if not models:
            return {"scores": [], "keep_idx": [], "reject_idx": []}
        state = PointillismFLState()
        specs, _ = build_round_probe_pool(
            cfg=audit_cfg, cfg_point=cfg_point, state=state, rnd=int(rnd))
        scores = score_probe_specs(
            models=models, specs=specs, cfg=audit_cfg, device=device,
            batch_size=int(cfg_point.get("batch_size", 256)))
        refine_specs, _ = collect_refinement_specs(
            base_specs=specs, base_scores=scores, cfg=audit_cfg,
            cfg_point=cfg_point, rnd=int(rnd))
        if refine_specs:
            refine_scores = score_probe_specs(
                models=models, specs=refine_specs, cfg=audit_cfg, device=device,
                batch_size=int(cfg_point.get("batch_size", 256)))
            specs, scores = merge_probe_pools(
                specs_a=specs, scores_a=scores,
                specs_b=refine_specs, scores_b=refine_scores)
        num_classes = _num_classes(audit_cfg, models[0])
        tau_cfg = (cfg_point.get("q1", {}) or {}).get("tau_R", cfg_point.get("tau_R", None))
        tau_r = None if tau_cfg is None else float(tau_cfg)
        infos = []
        for idx, cid in enumerate(round_client_ids):
            infos.append({
                "cid": int(cid), "is_malicious": 1,
                "q1": compute_q1_summary_for_client(
                    client_idx=idx, specs=specs, scores=scores,
                    num_classes=num_classes, tau_r=tau_r),
                "signature": build_client_signature(
                    client_idx=idx, specs=specs, scores=scores,
                    num_classes=num_classes, cfg_point=cfg_point),
            })
        A, _ = compute_pairwise_consensus(
            client_infos=infos, scores=scores, num_classes=num_classes,
            cfg_point=cfg_point)
        keep, reject, select_info = select_client_subset(
            client_infos=infos, A=A, num_classes=num_classes,
            cfg_point=cfg_point,
            rng_seed=int(cfg.get("seed", 0)) + 1009 * int(rnd))
        client_scores = []
        for idx in range(len(infos)):
            peers = [float(A[idx, j]) for j in range(len(infos)) if j != idx]
            client_scores.append(float(np.mean(peers)) if peers else float(A[idx, idx]))
        return {
            "scores": client_scores, "keep_idx": keep, "reject_idx": reject,
            "A": A, "select_info": select_info,
        }
    finally:
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()


def audit_adaptive_candidate_states(
    *, cfg, device, clean_states, candidate_states, model_fn, rnd,
    round_client_ids, private_probe_seed,
):
    """Audit all interpolated candidates with one shared private probe pass."""
    if len(clean_states) != len(candidate_states):
        raise ValueError("clean_states and candidate_states must have the same length")
    if len(clean_states) != len(round_client_ids):
        raise ValueError("round_client_ids must match clean_states")

    audit_cfg = deepcopy(cfg)
    audit_cfg["seed"] = int(private_probe_seed)
    cfg_point = _cfg_point(audit_cfg)
    flat_states = list(clean_states)
    candidate_indices = []
    for client_candidates in candidate_states:
        indices = []
        for candidate_state in client_candidates:
            indices.append(len(flat_states))
            flat_states.append(candidate_state)
        candidate_indices.append(indices)

    models = []
    try:
        for client_state in flat_states:
            model = model_fn().to(device)
            model.load_state_dict(client_state, strict=True)
            model.eval()
            models.append(model)
        if not models:
            empty = {"scores": [], "keep_idx": [], "reject_idx": []}
            return {"baseline": empty, "candidates": []}

        state = PointillismFLState()
        specs, _ = build_round_probe_pool(
            cfg=audit_cfg, cfg_point=cfg_point, state=state, rnd=int(rnd))
        scores = score_probe_specs(
            models=models, specs=specs, cfg=audit_cfg, device=device,
            batch_size=int(cfg_point.get("batch_size", 256)))

        clean_count = len(clean_states)
        clean_base_scores = {
            key: np.take(value, list(range(clean_count)), axis=0)
            for key, value in scores.items()
        }
        refine_specs, _ = collect_refinement_specs(
            base_specs=specs, base_scores=clean_base_scores, cfg=audit_cfg,
            cfg_point=cfg_point, rnd=int(rnd))
        if refine_specs:
            refine_scores = score_probe_specs(
                models=models, specs=refine_specs, cfg=audit_cfg, device=device,
                batch_size=int(cfg_point.get("batch_size", 256)))
            specs, scores = merge_probe_pools(
                specs_a=specs, scores_a=scores,
                specs_b=refine_specs, scores_b=refine_scores)

        num_classes = _num_classes(audit_cfg, models[0])
        tau_cfg = (cfg_point.get("q1", {}) or {}).get(
            "tau_R", cfg_point.get("tau_R", None))
        tau_r = None if tau_cfg is None else float(tau_cfg)
        infos_all = []
        for idx in range(len(flat_states)):
            infos_all.append({
                "cid": int(idx), "is_malicious": 1,
                "q1": compute_q1_summary_for_client(
                    client_idx=idx, specs=specs, scores=scores,
                    num_classes=num_classes, tau_r=tau_r),
                "signature": build_client_signature(
                    client_idx=idx, specs=specs, scores=scores,
                    num_classes=num_classes, cfg_point=cfg_point),
            })

        def evaluate_context(context_indices):
            context_scores = {
                key: np.take(value, context_indices, axis=0)
                for key, value in scores.items()
            }
            context_infos = []
            for pos, model_idx in enumerate(context_indices):
                info = dict(infos_all[int(model_idx)])
                info["cid"] = int(round_client_ids[pos])
                context_infos.append(info)
            A, _ = compute_pairwise_consensus(
                client_infos=context_infos, scores=context_scores,
                num_classes=num_classes, cfg_point=cfg_point)
            keep, reject, select_info = select_client_subset(
                client_infos=context_infos, A=A, num_classes=num_classes,
                cfg_point=cfg_point,
                rng_seed=int(cfg.get("seed", 0)) + 1009 * int(rnd))
            client_scores = []
            for idx in range(len(context_infos)):
                peers = [float(A[idx, j]) for j in range(len(context_infos)) if j != idx]
                client_scores.append(
                    float(np.mean(peers)) if peers else float(A[idx, idx]))
            return {
                "scores": client_scores, "keep_idx": keep,
                "reject_idx": reject, "A": A, "select_info": select_info,
            }

        clean_indices = list(range(len(clean_states)))
        baseline = evaluate_context(clean_indices)
        results = []
        for client_pos, indices in enumerate(candidate_indices):
            client_results = []
            for candidate_idx in indices:
                context = list(clean_indices)
                context[client_pos] = int(candidate_idx)
                client_results.append(evaluate_context(context))
            results.append(client_results)
        return {"baseline": baseline, "candidates": results}
    finally:
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()


def aggregate(
    *,
    cfg: Dict[str, Any],
    device: torch.device,
    recorder,
    global_state,
    client_states,
    client_weights,
    model_fn,
    rnd,
    round_client_ids,
):
    cfg_point = _cfg_point(cfg)
    out_dir = getattr(recorder, "out_dir", None) if recorder is not None else None
    state = _get_state(out_dir)
    if recorder is not None:
        setattr(
            recorder,
            "log_pointillism_step_times",
            bool(cfg_point.get("log_step_times", True)),
        )

    models: List[torch.nn.Module] = []
    for st in client_states:
        model = model_fn().to(device)
        model.load_state_dict(st, strict=True)
        model.eval()
        models.append(model)

    if not models:
        return deepcopy(global_state), []

    batch_size = int(cfg_point.get("batch_size", 256))
    tau_r_cfg = (cfg_point.get("q1", {}) or {}).get("tau_R", cfg_point.get("tau_R", None))
    tau_r = None if tau_r_cfg is None else float(tau_r_cfg)
    num_classes = _num_classes(cfg, models[0])

    t0 = perf_counter()
    pool_specs, pool_meta = build_round_probe_pool(
        cfg=cfg,
        cfg_point=cfg_point,
        state=state,
        rnd=int(rnd),
    )
    _log_step_time(recorder, rnd, "build_round_probe_pool", perf_counter() - t0)

    t0 = perf_counter()
    base_scores = score_probe_specs(
        models=models,
        specs=pool_specs,
        cfg=cfg,
        device=device,
        batch_size=int(batch_size),
    )
    _log_step_time(recorder, rnd, "score_probe_specs_base", perf_counter() - t0)

    t0 = perf_counter()
    refine_specs, refine_meta = collect_refinement_specs(
        base_specs=pool_specs,
        base_scores=base_scores,
        cfg=cfg,
        cfg_point=cfg_point,
        rnd=int(rnd),
    )
    _log_step_time(recorder, rnd, "collect_refinement_specs", perf_counter() - t0)

    if refine_specs:
        t0 = perf_counter()
        refine_scores = score_probe_specs(
            models=models,
            specs=refine_specs,
            cfg=cfg,
            device=device,
            batch_size=int(batch_size),
        )
        _log_step_time(recorder, rnd, "score_probe_specs_refine", perf_counter() - t0)

        t0 = perf_counter()
        all_specs, all_scores = merge_probe_pools(
            specs_a=pool_specs,
            scores_a=base_scores,
            specs_b=refine_specs,
            scores_b=refine_scores,
        )
        _log_step_time(recorder, rnd, "merge_probe_pools", perf_counter() - t0)
    else:
        all_specs, all_scores = pool_specs, base_scores

    t0 = perf_counter()
    client_infos: List[Dict[str, Any]] = []
    malicious_by_cid = _round_malicious_by_cid(
        recorder=recorder,
        rnd=int(rnd),
        round_client_ids=[int(x) for x in round_client_ids],
    )
    for idx, cid in enumerate(round_client_ids):
        q1 = compute_q1_summary_for_client(
            client_idx=int(idx),
            specs=all_specs,
            scores=all_scores,
            num_classes=int(num_classes),
            tau_r=tau_r,
        )
        signature = build_client_signature(
            client_idx=int(idx),
            specs=all_specs,
            scores=all_scores,
            num_classes=int(num_classes),
            cfg_point=cfg_point,
        )
        client_infos.append({
            "cid": int(cid),
            "is_malicious": int(malicious_by_cid.get(int(cid), 0)),
            "q1": q1,
            "signature": signature,
        })
    _log_step_time(recorder, rnd, "client_summaries", perf_counter() - t0)

    t0 = perf_counter()
    A, pair_rows = compute_pairwise_consensus(
        client_infos=client_infos,
        scores=all_scores,
        num_classes=int(num_classes),
        cfg_point=cfg_point,
    )
    _log_step_time(recorder, rnd, "compute_pairwise_consensus", perf_counter() - t0)

    t0 = perf_counter()
    keep_idx, reject_idx, select_info = select_client_subset(
        client_infos=client_infos,
        A=A,
        num_classes=int(num_classes),
        cfg_point=cfg_point,
        rng_seed=int(cfg.get("seed", 0)) + 1009 * int(rnd),
    )
    keep_ids = [int(round_client_ids[idx]) for idx in keep_idx]
    reject_ids = [int(round_client_ids[idx]) for idx in reject_idx]
    select_info = {
        **select_info,
        "selected_subset": [int(x) for x in keep_idx],
        "keep_ids": [int(x) for x in keep_ids],
        "reject_ids": [int(x) for x in reject_ids],
    }
    _log_step_time(recorder, rnd, "select_client_subset", perf_counter() - t0)

    if not keep_idx:
        keep_idx = list(range(len(client_states)))
        keep_ids = [int(round_client_ids[idx]) for idx in keep_idx]
        reject_idx = []
        reject_ids = []

    t0 = perf_counter()
    bank_meta = update_memory_bank(
        state=state,
        client_infos=client_infos,
        specs=all_specs,
        scores=all_scores,
        keep_idx=keep_idx,
        num_classes=int(num_classes),
        cfg_point=cfg_point,
    )
    round_meta = {
        "pool_meta": pool_meta,
        "refine_meta": refine_meta,
        "keep_ids": keep_ids,
        "reject_ids": reject_ids,
        "bank_meta": bank_meta,
        "dataset": _dataset_name(cfg),
        "num_classes": int(num_classes),
    }
    _log_step_time(recorder, rnd, "update_memory_bank", perf_counter() - t0)

    t0 = perf_counter()
    if bool(cfg_point.get("persist_state_snapshot", True)):
        save_state_snapshot(
            out_dir=out_dir,
            rnd=int(rnd),
            state=state,
            round_meta=round_meta,
        )
    if bool(cfg_point.get("log_diagnostics", True)):
        log_round_outputs(
            out_dir=out_dir,
            rnd=int(rnd),
            pool_meta=pool_meta,
            refine_meta=refine_meta,
            client_infos=client_infos,
            pair_rows=pair_rows,
            select_info=select_info,
            bank_meta=bank_meta,
            cfg_point=cfg_point,
        )
    _log_step_time(recorder, rnd, "persist_round_outputs", perf_counter() - t0)

    if recorder is not None:
        mean_shared = 0.0
        if pair_rows:
            mean_shared = float(sum(float(r["shared_support_size"]) for r in pair_rows) / len(pair_rows))
        recorder.maybe_print(
            f"[pointillism_fl] rnd={int(rnd)} pool={int(len(all_specs))} "
            f"refine={int(refine_meta.get('num_specs', 0))} "
            f"keep={len(keep_ids)}/{len(round_client_ids)} keep_ids={keep_ids} reject_ids={reject_ids} "
            f"pair_A_mean={float(A[np.triu_indices(len(client_infos), k=1)].mean()) if len(client_infos) > 1 else 1.0:.4f} "
            f"shared_mean={mean_shared:.2f}"
        )

    client_states_f = [client_states[idx] for idx in keep_idx]
    client_weights_f = [client_weights[idx] for idx in keep_idx]
    round_client_ids_f = [round_client_ids[idx] for idx in keep_idx]
    base_agg = str(cfg_point.get("base_agg", "normbound")).lower()
    new_state, agg_weights_keep = _dispatch_base_aggregate(
        base_agg=base_agg,
        cfg=cfg,
        device=device,
        recorder=recorder,
        global_state=global_state,
        client_states=client_states_f,
        client_weights=client_weights_f,
        model_fn=model_fn,
        rnd=rnd,
        round_client_ids=round_client_ids_f,
    )

    if not agg_weights_keep:
        agg_weights_keep = [1.0 / max(1, len(keep_idx)) for _ in keep_idx]

    agg_weights_full = [0.0 for _ in range(len(round_client_ids))]
    for pos, idx in enumerate(keep_idx):
        if pos < len(agg_weights_keep):
            agg_weights_full[int(idx)] = float(agg_weights_keep[pos])

    return new_state, agg_weights_full
