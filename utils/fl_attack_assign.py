# utils/fl_attack_assign.py
from __future__ import annotations

from typing import Any, Dict, List, Set
import numpy as np

from attacks.audio_badnet import AudioBadNetWrapper
from backdoor.badnet import PatternBackdoorWrapper
from attacks.label_attacks import (
    LabelMappingWrapper,
    SelectiveLabelMappingWrapper,
    build_label_perturbation_map,
    build_pair_flip_map,
    build_targeted_label_flip_map,
)


def assign_malicious(clients: Dict[int, Any], cfg: Dict[str, Any], seed: int = 0) -> List[int]:
    """
    Deterministically mark first num_mal clients as malicious.
    Uses cfg['clients_setting']['mali_rate'] and cfg['clients_setting']['clients'].
    """
    cs = cfg["clients_setting"]
    total_cfg = int(cs["clients"])
    mali_rate = float(cs.get("mali_rate", 0.0))

    all_ids = sorted(clients.keys())
    total_actual = len(all_ids)
    if total_actual != total_cfg:
        raise ValueError(f"clients_setting.clients={total_cfg} but got {total_actual} clients.")

    num_mal = int(round(mali_rate * total_cfg))
    num_mal = max(0, min(total_cfg, num_mal))

    mal_ids = list(range(num_mal))
    for cid in all_ids:
        clients[cid].is_malicious = (cid in mal_ids)
    return mal_ids


def _attack_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    attack = cfg.get("attack", None)
    if isinstance(attack, dict):
        return attack
    backdoor = cfg.get("backdoor", None)
    if isinstance(backdoor, dict):
        return backdoor
    return {}


def _sample_poison_indices(n_samples: int, frac: float, rng: np.random.Generator) -> Set[int]:
    if frac <= 0.0 or n_samples <= 0:
        return set()
    count = int(max(1, round(frac * n_samples)))
    idxs = np.arange(n_samples)
    chosen = rng.choice(idxs, size=min(count, n_samples), replace=False)
    return set(int(x) for x in chosen.tolist())


def _extract_labels(base_dataset: Any) -> np.ndarray:
    if hasattr(base_dataset, "targets"):
        return np.array(getattr(base_dataset, "targets"), dtype=int)
    return np.array([int(base_dataset[i][1]) for i in range(len(base_dataset))], dtype=int)


def _sample_eligible_poison_indices(labels: np.ndarray, eligible_labels: Set[int], frac: float, rng: np.random.Generator) -> Set[int]:
    eligible = np.where(np.isin(labels, sorted(int(x) for x in eligible_labels)))[0]
    if eligible.size == 0 or frac <= 0.0:
        return set()

    count = int(max(1, round(float(frac) * float(eligible.size))))
    chosen = rng.choice(eligible, size=min(count, int(eligible.size)), replace=False)
    return set(int(x) for x in chosen.tolist())


def assign_mali_client_ds(clients: Dict[int, Any], cfg: Dict[str, Any], seed: int = 0) -> List[int]:
    """
    Attach attack-specific local datasets or poison indices to malicious clients.

    Fixed-trigger attacks wrap the local dataset directly.
    Dynamic-trigger attacks keep the clean dataset and store deterministic poison indices.
    """
    atk = _attack_cfg(cfg)
    atk_name = str(atk.get("atk_name", "badnet")).lower()

    fixed_trigger_attacks = {"badnet", "neurotoxin", "scale"}
    helper_driven_attacks = {"dba", "a3fl", "cerp", "adaptive_pointillism"}
    update_only_attacks = {"reverse_grad_sign"}

    target_label = int(atk.get("target_label", 0))
    poison_frac = float(atk.get("poison_frac", 0.1))
    pattern_type = str(atk.get("pattern_type", "badnet_corner"))
    pattern_pos = str(atk.get("pattern_pos", "bottom_right"))
    pattern_padding = int(atk.get("pattern_padding", 0))
    pattern_offsets = atk.get("pattern_offsets", None)
    value = float(atk.get("value", 1.0))
    flip_pair = atk.get("flip_pair", atk.get("label_pair", None))
    num_classes = int(cfg.get("model", {}).get("num_classes", 10))

    pattern_size = atk.get("pattern_size", None)
    if pattern_size is not None:
        pattern_size = (int(pattern_size[0]), int(pattern_size[1]))

    for cid, client in clients.items():
        client.poison_idxs = set()
        client.poison_set = set()
        client.poison_frac = poison_frac

        if not getattr(client, "is_malicious", False):
            continue

        base = getattr(client, "clean_dataset", None) or getattr(client, "dataset", None)
        if base is None:
            raise RuntimeError(f"Client {cid} has no dataset available for poisoning.")

        rng = np.random.default_rng(int(seed) + int(cid))

        if atk_name == "audio_badnet":
            wrapped = AudioBadNetWrapper(base, atk, seed=int(seed) + int(cid))
            client.dataset = wrapped
            client.poison_idxs = set(getattr(wrapped, "poison_idxs", set()))
            client.poison_set = set(client.poison_idxs)
            continue

        if atk_name in fixed_trigger_attacks:
            wrapped = PatternBackdoorWrapper(
                base,
                target_label=target_label,
                poison_frac=poison_frac,
                pattern_pos=pattern_pos,
                pattern_padding=pattern_padding,
                pattern_size=pattern_size,
                pattern_offsets=pattern_offsets,
                value=value,
                seed=int(seed) + int(cid),
                pattern_type=pattern_type,
                apply_trigger=True,
                apply_relabel=True,
            )
            client.dataset = wrapped
            client.poison_idxs = set(getattr(wrapped, "poison_set", set()))
            client.poison_set = set(client.poison_idxs)
            continue

        if atk_name == "label_flipping":
            src = atk.get("source_labels", None)
            if src is not None and src != "all":
                label_map = build_targeted_label_flip_map(src, target_label)
            else:
                label_map = build_pair_flip_map(flip_pair)

            base_labels = _extract_labels(base)
            eligible_labels = set(int(k) for k in label_map.keys())
            poison_idxs = _sample_eligible_poison_indices(base_labels, eligible_labels, poison_frac, rng)

            client.dataset = SelectiveLabelMappingWrapper(base, label_map, poison_idxs)
            client.poison_idxs = set(poison_idxs)
            client.poison_set = set(poison_idxs)
            continue

        if atk_name == "label_perturbation":
            client.dataset = LabelMappingWrapper(base, build_label_perturbation_map(num_classes))
            continue

        if atk_name in update_only_attacks:
            client.dataset = base
            continue

        if atk_name in helper_driven_attacks:
            client.dataset = base
            client.poison_idxs = _sample_poison_indices(len(base), poison_frac, rng)
            client.poison_set = set(client.poison_idxs)
            continue

        client.dataset = base
        client.poison_idxs = _sample_poison_indices(len(base), poison_frac, rng)
        client.poison_set = set(client.poison_idxs)

    mal_ids = [cid for cid, c in clients.items() if getattr(c, "is_malicious", False)]
    return sorted(mal_ids)
