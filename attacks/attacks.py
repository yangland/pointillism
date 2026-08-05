# fedrep/attack.py
import torch
import numpy as np
from typing import Dict, Any, Set
from backdoor.badnet import (
    OnePixelBackdoorWrapper,
    PatternBackdoorWrapper,
    resolve_origin_from_pos,
    resolve_pattern_size,
)
from attacks.audio_badnet import AudioBadNetWrapper


def assign_malicious(clients, cfg, seed=0):
    """
    Mark a subset of clients as malicious in a *deterministic* way.

    Rule:
      - Let C = cfg["clients_setting"]["clients"]
      - Let r = cfg["clients_setting"]["mali_rate"]
      - num_mal = round(r * C)
      - Clients [0, 1, ..., num_mal-1] are malicious, the rest benign.

    Assumes client IDs are 0..C-1 and match the keys of `clients`.
    """
    cs = cfg["clients_setting"]
    total_cfg = int(cs["clients"])
    mali_rate = float(cs.get("mali_rate", 0.0))

    all_ids = sorted(clients.keys())
    total_actual = len(all_ids)
    if total_actual != total_cfg:
        raise ValueError(
            f"clients_setting.clients={total_cfg} but got {total_actual} client objects."
        )

    num_mal = int(round(mali_rate * total_cfg))
    num_mal = max(0, min(total_cfg, num_mal))

    # first num_mal clients are malicious: 0..num_mal-1
    mal_ids = list(range(num_mal))

    for cid in all_ids:
        clients[cid].is_malicious = cid in mal_ids

    return mal_ids

def assign_mali_client_ds(clients: Dict[int, Any], cfg: Dict[str, Any], seed: int = 0):
    """
    Assign datasets / poison indices for malicious clients based on cfg['attack'].
    Must be called AFTER assign_malicious(...).
    Behavior:
      - For fixed-trigger attacks (badnet, neurotoxin, dba): wrap client.dataset with wrappers
      - For learned-trigger attacks (a3fl, cerp): keep client.dataset unchanged and
        deterministically sample client.poison_idxs for use by loader_poison builders.
    """
    pol = cfg.get("attack", {})
    atk_name = str(pol.get("atk_name", "")).lower()
    trigger  = str(pol.get("trigger", "")).lower()

    if atk_name == "audio_badnet":
        for cid, client in clients.items():
            if not getattr(client, "is_malicious", False):
                client.poison_idxs = set()
                continue
            base = client.clean_dataset or getattr(client, "dataset", None)
            if base is None:
                raise RuntimeError(f"Client {cid} has no dataset available for poisoning.")
            client.dataset = AudioBadNetWrapper(base, pol, seed=seed + cid)
            client.poison_idxs = getattr(client.dataset, "poison_idxs", set())
            client.poison_frac = float(pol.get("poison_frac", 0.1))
        return sorted(cid for cid, c in clients.items() if getattr(c, "is_malicious", False))

    # require image size in cfg["task"]
    task_cfg = cfg.get("task", {})
    if "img_size" not in task_cfg:
        raise KeyError("cfg['task']['img_size'] is required (H, W) for resolving pattern positions.")
    img_size = tuple(task_cfg["img_size"])
    if len(img_size) != 2:
        raise ValueError("cfg['task']['img_size'] must be length-2 (H, W).")
    H, W = int(img_size[0]), int(img_size[1])
    channels = int(task_cfg.get("channels", 1))

    # helper: deterministic sampled poison indices for a client
    def sample_poison_indices(n_samples: int, frac: float, rng: np.random.Generator) -> Set[int]:
        if frac <= 0.0:
            return set()
        count = int(max(1, round(frac * n_samples)))
        idxs = np.arange(n_samples)
        chosen = rng.choice(idxs, size=min(count, n_samples), replace=False)
        return set(int(x) for x in chosen.tolist())

    for cid, client in clients.items():
        if not getattr(client, "is_malicious", False):
            # non-malicious: ensure fields exist
            client.poison_idxs = set()
            continue

        base = client.clean_dataset or getattr(client, "dataset", None)
        if base is None:
            raise RuntimeError(f"Client {cid} has no dataset available for poisoning.")

        poison_frac = float(pol.get("poison_frac", 0.1))
        client.poison_frac = poison_frac
        rng = np.random.default_rng(seed + cid)

        # Fixed-trigger attacks: wrap dataset so wrapped dataset yields poisoned samples
        if atk_name in ("badnet", "neurotoxin", "dba", "scale"):
            if trigger == "one_pixel":
                client.dataset = OnePixelBackdoorWrapper(
                    base_dataset=base,
                    target_label=pol.get("target_label"),
                    poison_frac=poison_frac,
                    pattern_pos=pol.get("pattern_pos", None),
                    pattern_padding=int(pol.get("pattern_padding", 0)),
                    value=float(pol.get("value", 1.0)),
                    seed=seed + cid,
                )
                # wrapper exposes poison indices set as attribute
                client.poison_idxs = getattr(client.dataset, "poison_idxs", set())

            elif trigger == "pattern":
                pw, ph = resolve_pattern_size(
                    pattern_type=pol.get("pattern_type", "badnet_corner"),
                    pattern_size=pol.get("pattern_size", None),
                    pattern_offsets=pol.get("pattern_offsets", None),
                )
                pattern_size_wh = (int(pw), int(ph))

                # build dummy image to resolve origin
                img_tensor = torch.zeros((channels, H, W))
                pattern_pos = pol.get("pattern_pos", "bottom_right")
                padding = int(pol.get("pattern_padding", 0))

                origin_x, origin_y = resolve_origin_from_pos(
                    img_tensor, pattern_size=pattern_size_wh, pattern_pos=pattern_pos, padding=padding
                )

                client.dataset = PatternBackdoorWrapper(
                    base_dataset=base,
                    target_label=pol.get("target_label"),
                    poison_frac=poison_frac,
                    pattern_origin=(origin_x, origin_y),
                    pattern_pos=pattern_pos,
                    pattern_size=pattern_size_wh,
                    pattern_padding=padding,
                    pattern_offsets=pol.get("pattern_offsets", None),
                    pattern_type=pol.get("pattern_type", "badnet_corner"),
                    value=float(pol.get("value", 1.0)),
                    seed=seed + cid,
                )
                client.poison_idxs = getattr(client.dataset, "poison_set", set())

            else:
                raise ValueError(f"Unsupported trigger '{trigger}' for fixed-trigger attack '{atk_name}'.")

        # Learned-trigger attacks: keep dataset unchanged but sample poison indices
        elif atk_name in ("a3fl", "cerp"):
            n_samples = len(base)
            client.poison_idxs = sample_poison_indices(n_samples, poison_frac, rng)
            # do NOT modify client.dataset; trigger applied at training time via atk_helper

        else:
            # Unknown attack fallback: sample poison indices but leave dataset unchanged
            n_samples = len(base)
            client.poison_idxs = sample_poison_indices(n_samples, poison_frac, rng)
            client.dataset = base

    # return list of malicious ids for convenience
    mal_ids = [cid for cid, c in clients.items() if getattr(c, "is_malicious", False)]
    return sorted(mal_ids)
