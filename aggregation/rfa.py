# aggregation/rfa.py
import torch
from typing import Any, Dict, List, Tuple
from copy import deepcopy

@torch.no_grad()
def rfa(
    global_state: Dict[str, torch.Tensor],
    client_states: List[Dict[str, torch.Tensor]],
    client_weights: List[float] = None,          # unused for geometric median
    average_bn_buffers: bool = True,
    maxiter: int = 100,
    tol: float = 1e-6,
    device: torch.device = None,
) -> Dict[str, torch.Tensor]:
    if len(client_states) == 0:
        return global_state

    if device is None:
        device = next(iter(global_state.values())).device

    def _keep_param(name: str, t: torch.Tensor) -> bool:
        # Only aggregate floating tensors. Optionally skip BN buffers.
        if not t.is_floating_point():
            return False
        if average_bn_buffers:
            return True
        if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
            return False
        return True

    new_state: Dict[str, torch.Tensor] = {}
    for name, base in global_state.items():
        if not _keep_param(name, base):
            # Keep as-is (copy from global to avoid aliasing)
            new_state[name] = base.clone()
            continue

        # Stack client tensors for this layer in base's dtype/device
        shape = base.shape
        dtype = base.dtype
        layer_vecs = []
        for cs in client_states:
            v = cs[name].detach().to(device=device, dtype=dtype).reshape(-1)
            layer_vecs.append(v)
        X = torch.stack(layer_vecs, dim=0)  # [K, D]

        gm = _weiszfeld(X, maxiter=maxiter, tol=tol)  # [D], dtype==dtype
        new_state[name] = gm.view(shape)

    return new_state


@torch.no_grad()
def _weiszfeld(X: torch.Tensor, maxiter: int = 100, tol: float = 1e-6) -> torch.Tensor:
    """
    Weiszfeld algorithm for geometric median over rows of X [K, D].
    Computes in float32 for stability; returns in X.dtype.
    """
    K, D = X.shape
    if K == 1:
        return X[0]

    # compute in float32, then cast back
    work = X.float()
    guess = work.mean(dim=0)

    eps = torch.finfo(work.dtype).eps
    for _ in range(maxiter):
        d = torch.norm(work - guess, dim=1).clamp_min(eps)  # [K]
        w = 1.0 / d
        guess_next = (w[:, None] * work).sum(dim=0) / w.sum()

        if torch.norm(guess_next - guess) <= tol:
            guess = guess_next
            break
        guess = guess_next

    return guess.to(dtype=X.dtype)


@torch.no_grad()
def _compute_rfa_clientweights(
    global_state: Dict[str, torch.Tensor],
    client_states: List[Dict[str, torch.Tensor]],
    average_bn_buffers: bool = True,
    maxiter: int = 100,
    tol: float = 1e-6,
    device: torch.device = None,
) -> List[float]:
    """
    Compute per-client aggregation weights for RFA based on the geometric median
    of flattened client models:

        weight_i ∝ 1 / ||x_i - m||_2,

    where x_i is client i's flattened parameter vector (only parameters that
    RFA actually aggregates) and m is the geometric median of {x_i}.
    """
    if len(client_states) == 0:
        return []

    if device is None:
        device = next(iter(global_state.values())).device

    def _keep_param(name: str, t: torch.Tensor) -> bool:
        # Mirror the logic inside rfa()
        if not t.is_floating_point():
            return False
        if average_bn_buffers:
            return True
        if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
            return False
        return True

    # Build flattened vectors X[k, D] for each client k
    flat_list = []
    for cs in client_states:
        parts = []
        for name, base in global_state.items():
            t = cs[name]
            if not _keep_param(name, t):
                continue
            v = t.detach().to(device=device, dtype=torch.float32).reshape(-1)
            parts.append(v)
        if len(parts) == 0:
            # Degenerate case: client has no aggregating params
            flat_list.append(torch.zeros(1, device=device, dtype=torch.float32))
        else:
            flat_list.append(torch.cat(parts))

    X = torch.stack(flat_list, dim=0)  # [K, D] in float32
    K, D = X.shape

    if K == 1:
        # Single client: give full weight
        return [1.0]

    # Geometric median of flattened updates
    gm = _weiszfeld(X, maxiter=maxiter, tol=tol).float()  # [D]

    # Distances to geometric median and reciprocal weights
    eps = torch.finfo(X.dtype).eps
    d = torch.norm(X - gm, dim=1).clamp_min(eps)          # [K]
    w_raw = 1.0 / d                                       # [K]

    # Normalize so sum(weights) = 1
    s = w_raw.sum().item()
    if s <= 0.0:
        # Fallback: uniform weights
        return [1.0 / K] * K

    w_norm = (w_raw / s).cpu().tolist()
    return w_norm


# def aggregate(*, cfg: Dict[str, Any], device: torch.device, recorder, global_state, client_states, client_weights):
#     avg_bn = bool(cfg.get("agg", {}).get("average_bn_buffers", True))
#     rfa_cfg = cfg.get("rfa", {})
#     return rfa(
#         global_state, client_states, client_weights,
#         average_bn_buffers=avg_bn,
#         maxiter=int(rfa_cfg.get("maxiter", 100)),
#         tol=float(rfa_cfg.get("tol", 1e-6)),
#         device=device,
#     )

from typing import Tuple

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
) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    avg_bn = bool(cfg.get("agg", {}).get("average_bn_buffers", True))
    rfa_cfg = cfg.get("rfa", {})

    maxiter = int(rfa_cfg.get("maxiter", 100))
    tol = float(rfa_cfg.get("tol", 1e-6))

    # Geometric-median aggregation (layer-wise, as before)
    new_state = rfa(
        global_state,
        client_states,
        client_weights,
        average_bn_buffers=avg_bn,
        maxiter=maxiter,
        tol=tol,
        device=device,
    )

    # Per-client weights based on geometric median of flattened updates
    agg_weights = _compute_rfa_clientweights(
        global_state=global_state,
        client_states=client_states,
        average_bn_buffers=avg_bn,
        maxiter=maxiter,
        tol=tol,
        device=device,
    )

    return new_state, agg_weights
