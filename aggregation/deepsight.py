"""DeepSight aggregation defense.

Clean-room implementation of the server-side filtering idea from:
  DeepSight: Mitigating Backdoor Attacks in Federated Learning Through
  Deep Model Inspection.

The defense inspects uploaded local models with three signals:
  - final-layer bias/update cosine distances,
  - NEUP vectors from final-layer update energy,
  - DDif vectors from random-noise forward probes.

The signal clusterings are combined, suspicious clusters are filtered, retained
updates are norm-clipped, and the remaining clients are aggregated.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from aggregation.fedavg import fedavg as _fedavg

try:
    import hdbscan
except Exception:  # pragma: no cover - only used when hdbscan is unavailable.
    hdbscan = None


logger = logging.getLogger("deepsight")
logger.addHandler(logging.StreamHandler())

BN_RUNNING_KEYS = ("running_mean", "running_var")
BN_COUNTER_KEY = "num_batches_tracked"
EPS = 1e-12


def _is_param_key(name: str, tensor: torch.Tensor) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False
    if not tensor.is_floating_point():
        return False
    if name.endswith(BN_COUNTER_KEY):
        return False
    if any(bk in name for bk in BN_RUNNING_KEYS):
        return False
    return True


def _resolve_head_keys(
    global_state: Dict[str, torch.Tensor],
    num_classes: int,
    explicit_key: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    if explicit_key:
        if explicit_key not in global_state:
            raise KeyError(f"deepsight.final_layer_weight_key not found: {explicit_key}")
        prefix = explicit_key[: -len(".weight")] if explicit_key.endswith(".weight") else explicit_key
        bias_key = f"{prefix}.bias"
        return explicit_key, bias_key if bias_key in global_state else None

    candidates = []
    keys = list(global_state.keys())
    key_set = set(keys)
    for pos, key in enumerate(keys):
        tensor = global_state[key]
        if not _is_param_key(key, tensor):
            continue
        if not key.endswith(".weight") or tensor.ndim < 2:
            continue
        prefix = key[: -len(".weight")]
        bias_key = f"{prefix}.bias"
        class_match = 1 if int(tensor.shape[0]) == int(num_classes) else 0
        has_bias = 1 if bias_key in key_set else 0
        candidates.append((class_match, has_bias, pos, key, bias_key if bias_key in key_set else None))

    if not candidates:
        raise ValueError("DeepSight could not find a final-layer weight tensor.")

    _class_match, _has_bias, _pos, weight_key, bias_key = max(candidates)
    return weight_key, bias_key


def _as_class_matrix(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim < 2:
        arr = arr.reshape(arr.shape[0], 1)
    return arr.reshape(arr.shape[0], -1)


def _cosine_distance_matrix(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms <= EPS] = 1.0
    z = x / norms
    sims = np.clip(z @ z.T, -1.0, 1.0)
    dists = 1.0 - sims
    np.fill_diagonal(dists, 0.0)
    return np.nan_to_num(dists, nan=1.0, posinf=1.0, neginf=1.0)


def _update_norm(
    global_state: Dict[str, torch.Tensor],
    client_state: Dict[str, torch.Tensor],
) -> float:
    total = 0.0
    for key, base in global_state.items():
        if not _is_param_key(key, base):
            continue
        delta = client_state[key].detach().float() - base.detach().float()
        total += float(delta.pow(2).sum().item())
    return math.sqrt(max(total, 0.0))


def _neups_and_norms(
    global_state: Dict[str, torch.Tensor],
    client_states: Sequence[Dict[str, torch.Tensor]],
    weight_key: str,
    bias_key: Optional[str],
) -> Tuple[np.ndarray, List[float]]:
    global_weight = _as_class_matrix(global_state[weight_key])
    if bias_key is not None:
        global_bias = global_state[bias_key].detach().float().cpu().numpy().reshape(-1)
    else:
        global_bias = np.zeros(global_weight.shape[0], dtype=np.float64)

    neups = []
    norms = []
    for st in client_states:
        local_weight = _as_class_matrix(st[weight_key])
        if bias_key is not None and bias_key in st:
            local_bias = st[bias_key].detach().float().cpu().numpy().reshape(-1)
        else:
            local_bias = np.zeros(global_weight.shape[0], dtype=np.float64)

        bias_diff = np.abs(local_bias - global_bias)
        weight_diff_sum = np.sum(np.abs(local_weight - global_weight), axis=1)
        ups = bias_diff + weight_diff_sum
        denom = float(np.sum(ups ** 2))
        if denom <= EPS:
            neup = np.zeros_like(ups, dtype=np.float64)
        else:
            neup = (ups ** 2) / denom
        neups.append(neup)
        norms.append(_update_norm(global_state, st))

    return np.asarray(neups, dtype=np.float64), norms


def _threshold_labels(
    neups: np.ndarray,
    num_classes: int,
    threshold_factor: float = 0.01,
) -> List[bool]:
    thresh_counts = []
    for neup in neups:
        max_val = float(np.max(neup)) if neup.size else 0.0
        cutoff = max(float(threshold_factor), 1.0 / max(1, int(num_classes))) * max_val
        thresh_counts.append(int(np.sum(neup > cutoff)))

    boundary = float(np.median(thresh_counts)) if thresh_counts else 0.0
    return [count <= boundary * 0.5 for count in thresh_counts]


def _cluster_distance_from_labels(labels: np.ndarray, n: int) -> np.ndarray:
    out = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(out, 0.0)
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j] and labels[i] != -1:
                out[i, j] = 0.0
    return out


def _hdbscan_labels(
    data: np.ndarray,
    *,
    precomputed: bool = False,
    min_cluster_size: int = 2,
) -> np.ndarray:
    n = int(data.shape[0])
    if n < 2 or hdbscan is None:
        return np.full(n, -1, dtype=np.int64)
    mcs = max(2, min(int(min_cluster_size), n))
    if n < mcs:
        return np.full(n, -1, dtype=np.int64)
    try:
        if precomputed:
            mat = np.asarray(data, dtype=np.float64)
            mat = np.nan_to_num(mat, nan=1.0, posinf=1.0, neginf=1.0)
            np.fill_diagonal(mat, 0.0)
            return hdbscan.HDBSCAN(
                metric="precomputed",
                min_cluster_size=mcs,
                min_samples=1,
                allow_single_cluster=True,
            ).fit_predict(mat)
        return hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=1,
            allow_single_cluster=True,
        ).fit_predict(np.asarray(data, dtype=np.float64))
    except Exception as exc:
        logger.warning("DeepSight HDBSCAN failed; treating all points as noise: %r", exc)
        return np.full(n, -1, dtype=np.int64)


@torch.no_grad()
def _ddif_features(
    *,
    model_fn,
    global_state: Dict[str, torch.Tensor],
    client_states: Sequence[Dict[str, torch.Tensor]],
    device: torch.device,
    channels: int,
    height: int,
    width: int,
    num_classes: int,
    num_samples: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    global_model = model_fn().to(device)
    global_model.load_state_dict(global_state, strict=True)
    global_model.eval()

    local_models = []
    for st in client_states:
        model = model_fn().to(device)
        model.load_state_dict(st, strict=True)
        model.eval()
        local_models.append(model)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    ddifs = [torch.zeros(num_classes, device=device, dtype=torch.float32) for _ in client_states]
    seen = 0
    while seen < num_samples:
        bs = min(batch_size, num_samples - seen)
        noise = torch.rand((bs, channels, height, width), generator=gen, dtype=torch.float32).to(device)
        global_out = F.softmax(global_model(noise).detach(), dim=1)
        denom = global_out + torch.where(
            global_out >= 0,
            torch.full_like(global_out, 1e-6),
            torch.full_like(global_out, -1e-6),
        )
        for i, local_model in enumerate(local_models):
            local_out = F.softmax(local_model(noise).detach(), dim=1)
            ratio = torch.nan_to_num(local_out / denom, nan=0.0, posinf=0.0, neginf=0.0)
            ddifs[i].add_(ratio.sum(dim=0))
        seen += bs

    rows = []
    for value in ddifs:
        rows.append((value / max(1, num_samples)).detach().cpu().numpy())
    return np.asarray(rows, dtype=np.float64)


def _selected_clipped_state_dicts(
    global_state: Dict[str, Any],
    client_states: Sequence[Dict[str, Any]],
    selected: Sequence[int],
    norms: Sequence[float],
    norm_threshold: float,
) -> List[Dict[str, Any]]:
    out = []
    for idx in selected:
        scale = 1.0
        if norms[idx] > EPS and norm_threshold > EPS:
            scale = min(1.0, float(norm_threshold / norms[idx]))
        state = {}
        for key, base in global_state.items():
            value = client_states[idx][key]
            if _is_param_key(key, base):
                state[key] = base.detach() + (value.detach() - base.detach()) * scale
            elif isinstance(value, torch.Tensor):
                state[key] = value.detach().clone()
            else:
                state[key] = value
        out.append(state)
    return out


def _aggregate_selected(
    global_state: Dict[str, Any],
    client_states: Sequence[Dict[str, Any]],
    client_weights: Sequence[float],
    selected: Sequence[int],
    norms: Sequence[float],
    average_bn_buffers: bool,
    equal_weights: bool = True,
) -> Tuple[Dict[str, Any], List[float]]:
    n = len(client_states)
    agg_weights = [0.0] * n
    if not selected:
        unchanged = {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in global_state.items()
        }
        return unchanged, agg_weights

    raw_weights = [1.0 for _ in selected] if equal_weights else [float(client_weights[i]) for i in selected]
    total = sum(weight for weight in raw_weights if weight > 0.0)
    if total <= 0.0:
        raw_weights = [1.0] * len(selected)
        total = float(len(selected))

    norm_threshold = float(np.median(norms)) if norms else 0.0
    clipped_states = _selected_clipped_state_dicts(
        global_state,
        client_states,
        selected,
        norms,
        norm_threshold,
    )
    selected_weights = [weight / total for weight in raw_weights]

    for idx, weight in zip(selected, selected_weights):
        agg_weights[idx] = float(weight)

    new_state = _fedavg(
        global_state=global_state,
        client_states=clipped_states,
        weights=selected_weights,
        average_bn_buffers=average_bn_buffers,
    )
    return new_state, agg_weights


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
    n = len(client_states)
    if n == 0:
        return global_state, []
    if model_fn is None:
        raise ValueError("DeepSight requires model_fn for random-noise model inspection.")

    ds_cfg = dict(cfg.get("deepsight", {}) or {})
    task_cfg = dict(cfg.get("task", {}) or {})
    model_cfg = dict(cfg.get("model", {}) or {})
    height, width = tuple(task_cfg.get("img_size", (28, 28)))
    channels = int(task_cfg.get("channels", 1))
    num_classes = int(model_cfg.get("num_classes", 10))
    num_samples = int(ds_cfg.get("num_samples", 20000))
    num_seeds = int(ds_cfg.get("num_seeds", 3))
    batch_size = int(ds_cfg.get("batch_size", 128))
    min_cluster_size = int(ds_cfg.get("min_cluster_size", max(2, min(5, n // 2))))
    tau = float(ds_cfg.get("clipping_threshold", 1.0 / 3.0))
    threshold_factor = float(ds_cfg.get("threshold_factor", 0.01))
    equal_weights = bool(ds_cfg.get("equal_weights", True))
    avg_bn = bool(cfg.get("agg", {}).get("average_bn_buffers", True))
    base_seed = int(cfg.get("seed", 0) or 0) + int(rnd) * 1009

    try:
        weight_key, bias_key = _resolve_head_keys(
            global_state,
            num_classes=num_classes,
            explicit_key=ds_cfg.get("final_layer_weight_key", None),
        )
        neups, norms = _neups_and_norms(global_state, client_states, weight_key, bias_key)

        if bias_key is not None:
            bias_deltas = np.stack(
                [
                    (
                        state[bias_key].detach().float()
                        - global_state[bias_key].detach().float()
                    ).cpu().numpy().reshape(-1)
                    for state in client_states
                ],
                axis=0,
            )
        else:
            bias_deltas = neups
        cosine_dist = _cosine_distance_matrix(bias_deltas)

        labels_suspicious = _threshold_labels(
            neups,
            num_classes,
            threshold_factor=threshold_factor,
        )

        ddif_cluster_dists = []
        for seed_offset in range(max(1, num_seeds)):
            ddifs = _ddif_features(
                model_fn=model_fn,
                global_state=global_state,
                client_states=client_states,
                device=device,
                channels=channels,
                height=int(height),
                width=int(width),
                num_classes=num_classes,
                num_samples=max(1, num_samples),
                batch_size=max(1, batch_size),
                seed=base_seed + seed_offset,
            )
            ddif_labels = _hdbscan_labels(ddifs, min_cluster_size=min_cluster_size)
            ddif_cluster_dists.append(_cluster_distance_from_labels(ddif_labels, n))

        cosine_labels = _hdbscan_labels(
            cosine_dist,
            precomputed=True,
            min_cluster_size=min_cluster_size,
        )
        neup_labels = _hdbscan_labels(neups, min_cluster_size=min_cluster_size)
        merged_dist = np.mean(
            [
                np.mean(ddif_cluster_dists, axis=0),
                _cluster_distance_from_labels(neup_labels, n),
                _cluster_distance_from_labels(cosine_labels, n),
            ],
            axis=0,
        )
        np.fill_diagonal(merged_dist, 0.0)
        final_clusters = _hdbscan_labels(
            merged_dist,
            precomputed=True,
            min_cluster_size=min_cluster_size,
        )
    except Exception as exc:
        logger.warning("DeepSight inspection failed; falling back to FedAvg: %r", exc)
        selected = list(range(n))
        norms = [_update_norm(global_state, state) for state in client_states]
        return _aggregate_selected(
            global_state,
            client_states,
            client_weights,
            selected,
            norms,
            avg_bn,
            equal_weights=equal_weights,
        )

    suspicious_counts: Dict[int, int] = {}
    total_counts: Dict[int, int] = {}
    for idx, cluster in enumerate(final_clusters):
        cluster_id = int(cluster)
        if cluster_id == -1:
            continue
        suspicious_counts[cluster_id] = suspicious_counts.get(cluster_id, 0) + (
            1 if labels_suspicious[idx] else 0
        )
        total_counts[cluster_id] = total_counts.get(cluster_id, 0) + 1

    discard = set()
    for idx, cluster in enumerate(final_clusters):
        cluster_id = int(cluster)
        if cluster_id != -1:
            suspicious_fraction = suspicious_counts.get(cluster_id, 0) / max(
                1, total_counts.get(cluster_id, 1)
            )
            if suspicious_fraction >= tau:
                discard.add(idx)
        elif labels_suspicious[idx]:
            discard.add(idx)

    selected = [idx for idx in range(n) if idx not in discard]
    fallback_all = not selected
    if not selected:
        logger.warning(
            "DeepSight discarded all clients; falling back to all clients for a valid server update."
        )
        selected = list(range(n))

    new_state, agg_weights = _aggregate_selected(
        global_state,
        client_states,
        client_weights,
        selected,
        norms,
        avg_bn,
        equal_weights=equal_weights,
    )

    if recorder is not None:
        try:
            recorder.log_scalar("agg/deepsight/num_clients", n)
            recorder.log_scalar("agg/deepsight/num_selected", len(selected))
            recorder.log_scalar("agg/deepsight/num_suspicious", len(discard))
            recorder.log_scalar("agg/deepsight/fallback_all", int(fallback_all))
            recorder.log_scalar(
                "agg/deepsight/norm_threshold",
                float(np.median(norms)) if norms else 0.0,
            )
        except Exception:
            pass

    return new_state, agg_weights
