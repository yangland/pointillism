# fl/aggregation/normbound.py
from typing import List, Dict, Any, Tuple
import torch

BN_RUNNING_KEYS = ("running_mean", "running_var")
BN_COUNTER_KEY = "num_batches_tracked"
EPS = 1e-8

def _param_keys(state_keys: List[str]) -> List[str]:
    """Keys that participate in optimization / clipping (exclude BN running buffers and counters)."""
    out = []
    for k in state_keys:
        if BN_COUNTER_KEY in k:
            continue
        if any(bk in k for bk in BN_RUNNING_KEYS):
            continue
        out.append(k)
    return out

def _weighted_mean(tensors: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    w = torch.tensor(weights, device=tensors[0].device, dtype=tensors[0].dtype)
    w = w / (w.sum() + 1e-12)
    out = torch.zeros_like(tensors[0])
    for ti, wi in zip(tensors, w):
        out.add_(ti, alpha=float(wi.item()))
    return out

def _delta_l2_norm(global_state: Dict[str, torch.Tensor], client_state: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
    """Compute || vec( client - global ) ||_2 over selected keys without flattening."""
    ssum = 0.0
    for k in keys:
        d = client_state[k] - global_state[k]
        ssum += d.pow(2).sum()
    return ssum.sqrt()

def normbound(
    global_state: Dict[str, Any],
    client_states: List[Dict[str, Any]],
    weights: List[int],
    average_bn_buffers: bool = True,
) -> Dict[str, Any]:
    """
    Median-norm clipping:
      1) For each client, compute the update delta_i = client - global over *parameter* keys.
      2) Compute L2 norms ||delta_i||; use the *median* as the clipping radius.
      3) Scale each delta_i by min(1, median/||delta_i||) and do a weighted average.
      4) BN running buffers:
         - running_{mean,var}: averaged (weighted) if average_bn_buffers=True, else keep server's
         - num_batches_tracked: copy from first selected client
    """
    # select clients with positive weights
    sel = [(cs, float(w)) for cs, w in zip(client_states, weights) if float(w) > 0.0]
    if not sel:
        return global_state

    c_states, c_weights = zip(*sel)
    keys = list(global_state.keys())
    pkeys = _param_keys(keys)  # parameters to clip

    # 1) norms
    norms = [ _delta_l2_norm(global_state, cs, pkeys) for cs in c_states ]
    norms_tensor = torch.stack([n if isinstance(n, torch.Tensor) else torch.tensor(n, device=next(iter(global_state.values())).device) for n in norms])
    median_val = norms_tensor.median()
    # Avoid zero median (all-zero updates) by keeping scale=1 in that case.

    # 2) scale deltas and aggregate
    # pre-normalize client weights
    w = torch.tensor(c_weights, device=median_val.device, dtype=median_val.dtype)
    w_sum = w.sum()
    if float(w_sum) <= 0.0:
        return global_state
    w = w / (w_sum + EPS)

    new_state: Dict[str, Any] = {}

    # parameters (with clipping)
    for k in pkeys:
        acc = torch.zeros_like(global_state[k])
        for i, cs in enumerate(c_states):
            d = cs[k] - global_state[k]
            n = norms_tensor[i].item()
            if n > 0.0:
                scale = min(1.0, float(median_val.item() / (n + EPS)))
            else:
                scale = 1.0
            acc.add_(d * scale, alpha=float(w[i].item()))
        new_state[k] = global_state[k] + acc

    # BN buffers
    for k in keys:
        if k in pkeys:
            continue
        if BN_COUNTER_KEY in k:
            new_state[k] = c_states[0][k].clone()
        elif any(bk in k for bk in BN_RUNNING_KEYS):
            if average_bn_buffers:
                tensors = [cs[k] for cs in c_states]
                new_state[k] = _weighted_mean(tensors, list(c_weights))
            else:
                new_state[k] = global_state[k].clone()
        else:
            # Any other non-BN buffer: average like params (rare)
            tensors = [cs[k] for cs in c_states]
            new_state[k] = _weighted_mean(tensors, list(c_weights))

    return new_state

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
    avg_bn = bool(cfg.get("agg", {}).get("average_bn_buffers", True))
    nb_cfg = cfg.get("normbound", {})

    new_state = normbound(
        global_state,
        client_states,
        client_weights,
        average_bn_buffers=avg_bn,
        **nb_cfg,
    )

    # --- aggregation weights: 1/k for all clients ---
    k = len(client_states)
    if k > 0:
        agg_weights = [1.0 / k] * k
    else:
        agg_weights = []

    return new_state, agg_weights