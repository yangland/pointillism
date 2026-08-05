# fl/aggregation/krum.py

from typing import Dict, Any, List
import math
import torch
from aggregation.fedavg import fedavg as _fedavg
BN_COUNTER_KEY = "num_batches_tracked"


def _flatten_state_dict(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Flatten trainable tensors in a state_dict into one 1D vector.
    Skip non-floating tensors and BN counters.
    """
    parts = []
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        if not v.is_floating_point():
            continue
        if k.endswith(BN_COUNTER_KEY):
            continue
        parts.append(v.detach().reshape(-1))

    if not parts:
        # Degenerate case, just return an empty vector on CPU
        return torch.zeros(0)

    return torch.cat(parts)


def _multi_krum_indices(
    flat_dict: Dict[int, torch.Tensor],
    f: int,
    m: int,
) -> List[int]:
    """
    Core Krum / Multi-Krum scoring.

    flat_dict: client_index -> flattened model/update tensor
    f       : number of Byzantine clients tolerated
    m       : number of selected clients (m=1 => original Krum; m>1 => Multi-Krum)
    """
    keys = list(flat_dict.keys())
    n = len(keys)

    if n <= 2:
        return keys

    # Guard f into valid range (roughly the original Krum condition)
    max_f = max(0, math.floor(n / 2) - 2)
    if f > max_f:
        f = max_f
    if f < 0:
        f = 0

    # Guard m
    if m > n:
        m = n
    if m < 1:
        m = 1

    # Precompute pairwise distances
    dists: Dict[int, List[float]] = {i: [] for i in keys}
    for i in keys:
        vi = flat_dict[i]
        for j in keys:
            if i == j:
                continue
            vj = flat_dict[j]
            dist = torch.norm(vi - vj, p=2).item()
            dists[i].append(dist)

    # Krum score: sum of closest (n - f - 2) distances
    scores: Dict[int, float] = {}
    k_nn = max(1, n - f - 2)
    for i in keys:
        di = sorted(dists[i])
        scores[i] = float(sum(di[:k_nn]))

    # Select m clients with smallest Krum scores
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    selected = [idx for idx, _ in sorted_scores[:m]]
    return selected


def _average_selected(
    client_states: List[Dict[str, torch.Tensor]],
    indices: List[int],
) -> Dict[str, torch.Tensor]:
    """
    Simple equal-weighted average over a subset of client state_dicts.
    BN counters ('num_batches_tracked') come from the first selected client.
    """
    if not indices:
        # Fallback: just return the first one
        return client_states[0]

    ref_state = client_states[indices[0]]
    new_state: Dict[str, torch.Tensor] = {}

    for k in ref_state.keys():
        if k.endswith(BN_COUNTER_KEY):
            # Just copy counters from reference
            new_state[k] = ref_state[k].detach().clone()
        else:
            tensors = [client_states[i][k] for i in indices]
            acc = torch.zeros_like(tensors[0])
            for t in tensors:
                acc.add_(t)
            acc.div_(len(tensors))
            new_state[k] = acc

    return new_state


def aggregate(
    *,
    cfg: Dict[str, Any],
    device: torch.device,
    recorder,
    global_state: Dict[str, torch.Tensor],
    client_states: List[Dict[str, torch.Tensor]],
    client_weights: List[float],
    model_fn,
    rnd,
    round_client_ids,
):
    """
    Krum / Multi-Krum aggregator with the unified interface.

    Returns:
        new_state, agg_weights

    agg_weights: list[float] of length n, where n = len(client_states).
        - For Krum: a subset 'selected' gets equal weights summing to 1
        - All non-selected clients get 0
    """
    n = len(client_states)
    if n == 0:
        return global_state, []

    k_cfg = cfg.get("krum", {})
    f = int(k_cfg.get("f", 0))

    # Default: m = k // 2 (Multi-Krum)
    m_cfg = k_cfg.get("m", None)
    if m_cfg is None:
        m = max(1, n // 2)
    else:
        # Allow ratio or absolute
        if isinstance(m_cfg, float) and 0 < m_cfg <= 1.0:
            m = max(1, int(round(m_cfg * n)))
        else:
            m = max(1, int(m_cfg))

    # Build flattened representation for scoring
    flat_dict: Dict[int, torch.Tensor] = {
        i: _flatten_state_dict(client_states[i]) for i in range(n)
    }

    selected = _multi_krum_indices(flat_dict, f=f, m=m)

    # ---- per-client aggregation weights ----
    # selected clients: equal weight; others: 0
    agg_weights = [0.0] * n
    if selected:
        w = 1.0 / float(len(selected))
        for idx in selected:
            agg_weights[idx] = w

    # Optional logging
    if recorder is not None:
        try:
            recorder.log_scalar("agg/krum/num_clients", n)
            recorder.log_scalar("agg/krum/f", f)
            recorder.log_scalar("agg/krum/m_selected", len(selected))
        except Exception:
            pass

    # new_state = _average_selected(client_states, selected)
    
    fedavg_weights = [0.0] * n
    if selected:
        w = 1.0 / float(len(selected))
        for idx in selected:
            fedavg_weights[idx] = w

    new_state = _fedavg(
        global_state=global_state,
        client_states=client_states,
        weights=fedavg_weights,
        average_bn_buffers=True,  # keep BN behavior consistent with FedAvg
    )
    
    return new_state, agg_weights
