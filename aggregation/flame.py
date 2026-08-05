# fl/aggregation/flame.py

from typing import Dict, Any, List, Tuple
import logging
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import hdbscan
from aggregation.fedavg import fedavg as _fedavg

logger = logging.getLogger("flame")
logger.addHandler(logging.StreamHandler())


def modelsd2flat(model_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Flattens a model's state dictionary into a single 1D tensor.
    Skips BN counters ('num_batches_tracked') for consistency.
    """
    ravel_list = []
    for layer_name, parms in model_dict.items():
        if isinstance(parms, torch.Tensor) and not layer_name.endswith('num_batches_tracked'):
            ravel_list.append(parms.detach().reshape(-1))

    if not ravel_list:
        return torch.tensor([])

    flat_tensor = torch.cat(ravel_list)
    return flat_tensor


def get_model_update(
    updated_model: Dict[str, torch.Tensor],
    model: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Difference between updated_model and model (server).
    Preserve 'num_batches_tracked' from the server model.
    """
    update: Dict[str, torch.Tensor] = {}
    for key in updated_model:
        if key.endswith('num_batches_tracked'):
            update[key] = model[key].detach().clone()
        else:
            update[key] = updated_model[key] - model[key].detach()
    return update


def get_model_merged(
    gradient_update: Dict[str, torch.Tensor],
    base_model,
    assume_delta: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Merge a gradient_update dict with a base model.

    assume_delta=True : gradient_update[key] is delta, so base + delta
    assume_delta=False: gradient_update[key] already represents absolute weights
    """
    # Normalize base_model to a state_dict
    if isinstance(base_model, tuple):
        base_sd = base_model[0]
    elif hasattr(base_model, "state_dict"):
        base_sd = base_model.state_dict()
    elif isinstance(base_model, dict):
        base_sd = base_model
    else:
        raise TypeError(f"Unsupported base_model type: {type(base_model)}")

    merged: Dict[str, torch.Tensor] = {}
    for key, base_tensor in base_sd.items():
        if key.endswith('num_batches_tracked'):
            merged[key] = base_tensor.detach().clone()
        else:
            if assume_delta:
                merged[key] = base_tensor.detach() + gradient_update[key]
            else:
                merged[key] = gradient_update[key]
    return merged


# def _average_state_dicts(dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
#     """
#     Equal-weight average of a list of state_dict-like mappings.
#     """
#     if not dicts:
#         raise ValueError("FLAME: no state dicts to average")

#     keys = dicts[0].keys()
#     avg: Dict[str, torch.Tensor] = {}
#     for k in keys:
#         stack = torch.stack([d[k].float() for d in dicts], dim=0)
#         avg[k] = stack.mean(dim=0).to(dicts[0][k].dtype)
#     return avg


def _average_state_dicts(dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Use FedAvg-style averaging (BN-aware) over a list of state_dict-like updates.

    Each element in `dicts` is an update (local - server), but since averaging is linear
    we can treat them as "client states" around a dummy zero server and call FedAvg.
    """
    if not dicts:
        raise ValueError("FLAME: no state dicts to average")

    # Build a dummy "server" state of zeros with same shapes,
    # so FedAvg just computes a weighted mean of these updates.
    first = dicts[0]
    zero_state: Dict[str, torch.Tensor] = {}
    for k, v in first.items():
        if isinstance(v, torch.Tensor):
            zero_state[k] = torch.zeros_like(v)
        else:
            zero_state[k] = v

    # Equal weights over the provided updates
    weights = [1.0 / float(len(dicts))] * len(dicts)

    avg_state = _fedavg(
        global_state=zero_state,
        client_states=dicts,
        weights=weights,
        average_bn_buffers=True,  # same BN behavior as your FedAvg
    )

    return avg_state


def flame(
    server_sd,
    model_dict: Dict[int, Dict[str, torch.Tensor]],
    noise: float = 0.001,
    client_ids: List[int] = None,
    exp_dir: str = "",
    iter: int = 0,
) -> Tuple[Dict[str, torch.Tensor], List[int]]:
    """
    FLAME core: cluster client models via HDBSCAN in cosine space,
    keep the largest cluster, clip L2 norms of updates by median,
    aggregate, then add small Gaussian noise.
    """

    # --- normalize server_sd to a plain state_dict ---
    if isinstance(server_sd, tuple):              # e.g., (state_dict, ...)
        server_sd = server_sd[0]
    if hasattr(server_sd, "state_dict"):          # nn.Module
        server_sd = server_sd.state_dict()
    # Detached, cloned copy
    server_sd_copy = {k: v.detach().clone() for k, v in server_sd.items()}

    cos_list: List[List[float]] = []
    local_model_vector: List[torch.Tensor] = []
    update_params: List[Dict[str, torch.Tensor]] = []
    local_client_ids: List[int] = []  # rebuilt from model_dict
    num_clients = len(model_dict)

    # Build flattened models & updates
    for cid in sorted(model_dict.keys()):
        param_sd = model_dict[cid]     # dict of tensors (client model state)
        local_model_vector.append(modelsd2flat(param_sd))
        update_params.append(get_model_update(param_sd, server_sd))
        local_client_ids.append(cid)

    # Sanitize flattened vectors
    for i in range(len(local_model_vector)):
        v = local_model_vector[i]
        if torch.isnan(v).any() or torch.isinf(v).any():
            logger.info(f"FLAME: NaNs/Infs in local_model_vector[{i}], replacing with zeros")
            local_model_vector[i] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    # Pairwise cosine distance matrix for HDBSCAN (NumPy array)
    for i in range(len(local_model_vector)):
        row = []
        for j in range(len(local_model_vector)):
            cos_ij = 1 - F.cosine_similarity(local_model_vector[i], local_model_vector[j], dim=0)
            row.append(cos_ij.item())
        cos_list.append(row)
    cos_mat = np.asarray(cos_list, dtype=np.float64)
    if np.isnan(cos_mat).any():
        logger.info("FLAME: NaN detected in cos matrix; replacing with zeros")
        cos_mat = np.nan_to_num(cos_mat, nan=0.0)

    # Cluster to pick benign subset
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=num_clients // 2 + 1,
        min_samples=1,
        allow_single_cluster=True
    ).fit(cos_mat)

    logger.info(f"FLAME: clusterer.labels_ {str(clusterer.labels_)}")

    benign_idx: List[int] = []
    if clusterer.labels_.max() < 0:
        benign_idx = list(range(num_clients))
    else:
        labels = clusterer.labels_
        max_cluster = None
        max_size = -1
        for c in range(labels.max() + 1):
            size = int((labels == c).sum())
            if size > max_size:
                max_size = size
                max_cluster = c
        benign_idx = [i for i in range(num_clients) if labels[i] == max_cluster]

    # Norm list & clipping
    norm_list = []
    for i in range(num_clients):
        flat = modelsd2flat(update_params[i])
        if torch.isnan(flat).any() or torch.isinf(flat).any():
            logger.warning("FLAME: NaN/Inf in update; replacing with zeros for norm calc")
            flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        norm_list.append(flat.norm(p=2).item())
    norm_list = np.array(norm_list, dtype=np.float64)

    if np.isnan(norm_list).any():
        norm_list = norm_list[~np.isnan(norm_list)]
    if norm_list.size == 0:
        raise ValueError("FLAME: norm_list is empty after NaN removal")

    clip_value = np.median(norm_list)
    for idx in benign_idx:
        gamma = clip_value / (norm_list[idx] + 1e-15)
        if gamma < 1.0:
            for k, t in update_params[idx].items():
                if k.split('.')[-1] == 'num_batches_tracked':
                    continue
                update_params[idx][k] = t * gamma

    logger.info(f"[FLAME DEBUG][Iter {iter}] Norm list: {norm_list}")
    logger.info(f"[FLAME DEBUG][Iter {iter}] Clip value: {clip_value}")

    # Average deltas of benign subset
    benign_updates = [update_params[i] for i in benign_idx]
    avg_update = _average_state_dicts(benign_updates)

    # Merge averaged delta into server state
    new_server_sd = get_model_merged(avg_update, server_sd_copy, assume_delta=True)

    # Noise injection: skip BN buffers and BN counters
    with torch.no_grad():
        for key, var in new_server_sd.items():
            suffix = key.split('.')[-1]
            if suffix in ('num_batches_tracked',):
                continue
            if ('running_mean' in key) or ('running_var' in key):
                continue
            if not isinstance(var, torch.Tensor):
                continue
            noise_tensor = torch.normal(
                mean=0.0,
                std=noise,
                size=var.shape,
                device=var.device,
            )
            var.add_(noise_tensor)

    # Ensure BN running_var stays positive
    with torch.no_grad():
        for key, var in new_server_sd.items():
            if 'running_var' in key:
                var.clamp_(min=1e-6)

    benign_client_ids = [local_client_ids[i] for i in benign_idx]
    return new_server_sd, benign_client_ids


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
) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    """
    FLAME aggregator with unified interface.

    Returns:
        new_state, agg_weights

    agg_weights: list[float] of length K (K = len(client_states)):
        - benign clients (benign_idx): equal non-zero weights summing to 1
        - others: 0
    """
    if not client_states:
        return global_state, []

    flame_cfg = cfg.get("flame", {})
    noise = float(flame_cfg.get("noise", 1e-3))
    exp_dir = str(flame_cfg.get("exp_dir", ""))

    # Map local index -> state_dict; FLAME already treats keys as client IDs
    model_dict = {i: sd for i, sd in enumerate(client_states)}

    # Try to get current round index from the recorder (optional)
    iter_idx = 0
    if recorder is not None:
        for attr in ("round", "rnd", "round_idx", "iter"):
            if hasattr(recorder, attr):
                try:
                    iter_idx = int(getattr(recorder, attr))
                    break
                except Exception:
                    pass

    # Existing FLAME logic: returns (new_server_sd, benign_client_ids)
    new_state, benign_idx = flame(
        server_sd=global_state,
        model_dict=model_dict,
        noise=noise,
        client_ids=None,
        exp_dir=exp_dir,
        iter=iter_idx,
    )

    # ---- per-client aggregation weights ----
    k = len(client_states)
    agg_weights = [0.0] * k
    if benign_idx:
        w = 1.0 / float(len(benign_idx))
        for i in benign_idx:
            agg_weights[i] = w

    return new_state, agg_weights
