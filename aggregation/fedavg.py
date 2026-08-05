# fl/aggregation/fedavg.py
from typing import List, Dict, Any
import torch

BN_RUNNING_KEYS = ("running_mean", "running_var")
BN_COUNTER_KEY = "num_batches_tracked"

def _weighted_mean(tensors: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    if not tensors:
        raise ValueError("No tensors to average.")
    # convert weights to tensor on the same device/dtype as the first tensor
    w = torch.tensor(weights, device=tensors[0].device, dtype=tensors[0].dtype)
    w = w / (w.sum() + 1e-12)
    out = torch.zeros_like(tensors[0])
    for ti, wi in zip(tensors, w):
        out.add_(ti, alpha=float(wi.item()))
    return out

def fedavg(
    global_state: Dict[str, Any],
    client_states: List[Dict[str, Any]],
    weights: List[int],
    average_bn_buffers: bool = True,
):
    """
    FedAvg that (a) does weighted averaging by provided 'weights',
    (b) handles BN buffers explicitly:
        - running_{mean,var}: averaged if average_bn_buffers=True, else keep server's
        - num_batches_tracked: copy from the first client (common practice)
    """
    # select clients with positive weights
    sel = [(cs, float(w)) for cs, w in zip(client_states, weights) if float(w) > 0.0]
    if not sel:
        # no effective clients; return server unchanged
        return global_state

    c_states, c_weights = zip(*sel)

    new_state: Dict[str, Any] = {}
    keys = list(global_state.keys())

    for k in keys:
        if BN_COUNTER_KEY in k:
            # integer counter tensor; copy from first selected client
            new_state[k] = c_states[0][k].clone()
        elif any(bk in k for bk in BN_RUNNING_KEYS):
            if average_bn_buffers:
                new_state[k] = _weighted_mean([cs[k] for cs in c_states], list(c_weights))
            else:
                new_state[k] = global_state[k].clone()
        else:
            new_state[k] = _weighted_mean([cs[k] for cs in c_states], list(c_weights))

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

    new_state = fedavg(
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