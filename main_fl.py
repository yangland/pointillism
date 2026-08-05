# main_fl.py
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import shutil
import warnings
warnings.filterwarnings(
    action="ignore",
    category=FutureWarning,
    message=r".*force_all_finite.*"
)

import argparse
import copy
import csv
import torch
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from build_data_labelskew import build_label_skew_clients, log_client_label_counts
from utils.fl_config import load_config, freeze_and_expand
from utils.fl_utils import set_seed, build_model_fn, configure_runtime_device
from utils.fl_recorder import Recorder

from fl.client import FLClient
from fl.server import FLServer, hook_after_round_pointillism_viz

from utils.fl_attack_assign import assign_malicious, assign_mali_client_ds


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_final_metrics(run_out_dir: Path):
    """
    Read final acc/asr from server_metrics.csv if present.
    """
    metrics_csv = run_out_dir / "server_metrics.csv"
    if not metrics_csv.exists():
        return None, None

    with metrics_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        last_row = None
        for row in reader:
            last_row = row

    if last_row is None:
        return None, None

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    acc = _to_float(last_row.get("acc", ""))
    asr = _to_float(last_row.get("asr", ""))  # may be empty in minimal build
    return acc, asr


def _get_atk_name(cfg: dict) -> str:
    """
    Support both legacy cfg['attack'] and new cfg['backdoor'].
    """
    atk = cfg.get("attack", None)
    if isinstance(atk, dict):
        return str(atk.get("atk_name", "none"))
    bd = cfg.get("backdoor", None)
    if isinstance(bd, dict):
        return str(bd.get("atk_name", "none"))
    return "none"


def _canonicalize_atk_name(name: str) -> str:
    atk_name = str(name or "none").lower()
    aliases = {
        "label_flip": "label_flipping",
        "label_flipping": "label_flipping",
        "label_perturb": "label_perturbation",
        "reverse_grad": "reverse_grad_sign",
        "reverse_grad_sign": "reverse_grad_sign",
    }
    return aliases.get(atk_name, atk_name)


def _normalize_fl_attack_cfg(cfg: dict) -> dict:
    """
    Normalize standalone-style `backdoor` configs into the FL `attack` block.
    """
    if "attack" not in cfg and isinstance(cfg.get("backdoor"), dict):
        cfg["attack"] = copy.deepcopy(cfg["backdoor"])

    attack = cfg.get("attack", None)
    if isinstance(attack, dict):
        atk_name = _canonicalize_atk_name(attack.get("atk_name", "none"))
        attack["atk_name"] = atk_name
        if "trigger" not in attack and atk_name in {"badnet", "neurotoxin", "scale"}:
            pattern_type = str(attack.get("pattern_type", "")).lower()
            attack["trigger"] = "one_pixel" if pattern_type == "one_pixel" else "pattern"
        if atk_name == "a3fl":
            trigger_size = attack.get("trigger_size", None)
            if trigger_size is None:
                trigger_size = attack.get("pattern_size", [3, 3])
            attack["trigger_size"] = [int(trigger_size[0]), int(trigger_size[1])]
        if atk_name == "label_flipping":
            flip_pair = attack.get("flip_pair", attack.get("label_pair", None))
            if flip_pair is not None:
                attack["flip_pair"] = [int(flip_pair[0]), int(flip_pair[1])]

    if "device" not in cfg:
        train_cfg = cfg.get("train", {}) or {}
        if "device" in train_cfg:
            cfg["device"] = train_cfg["device"]

    return cfg
def save_config_yaml(cfg: dict, path: Path) -> None:
    import yaml
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="path to yaml config")
    parser.add_argument("--data_root", type=str, default="./data", help="dataset root (default: ./data)")
    parser.add_argument("--log_root", type=str, default="./log_fl", help="log root (default: ./log_fl)")
    args = parser.parse_args()

    # ---- load base config once ----
    cfg_master = _normalize_fl_attack_cfg(load_config(args.cfg))

    # seed: prefer cfg.seed; else random; store back
    seed = cfg_master.get("seed", None)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    cfg_master["seed"] = int(seed)
    set_seed(int(seed))
    runtime_device = cfg_master.get("device", (cfg_master.get("train", {}) or {}).get("device", None))
    configure_runtime_device(runtime_device)

    dataset = cfg_master.get("task", {}).get("dataset", "DATA")
    atk_name = _get_atk_name(cfg_master)
    timestamp_str = _timestamp()

    # ---- allow defense to be string or list ----
    defenses = cfg_master.get("defense", "none")
    if isinstance(defenses, str):
        defenses = [defenses]
    elif isinstance(defenses, (list, tuple)):
        defenses = list(defenses)
    else:
        raise ValueError("cfg.defense must be a string or list of strings")
    defense_totals = Counter(str(name).lower() for name in defenses)
    defense_seen = defaultdict(int)

    # ---- master folder: S_<dataset>_<timestamp>_<atk> ----
    log_root = Path(args.log_root)
    master_dir = log_root / f"S_{dataset}_{timestamp_str}_{atk_name}"
    master_dir.mkdir(parents=True, exist_ok=True)

    # keep one copy of YAML at series level
    cfg_path = Path(args.cfg).resolve()
    series_cfg_copy = master_dir / cfg_path.name
    if not series_cfg_copy.exists():
        shutil.copy2(cfg_path, series_cfg_copy)
        # also freeze/expand for reproducibility if you want
        try:
            freeze_and_expand(cfg_master, str(master_dir))
        except Exception:
            pass

    # ---- master CSV: task, atk_name, defense, acc, asr, out_dir ----
    master_csv = master_dir / "master_results.csv"
    if not master_csv.exists():
        with master_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["task", "atk_name", "defense", "acc", "asr", "out_dir"])

    # ---- loop over defenses ----
    for defense_name in defenses:
        defense_key = str(defense_name).lower()
        repeat_idx = int(defense_seen[defense_key])
        defense_seen[defense_key] += 1
        repeat_total = int(defense_totals[defense_key])
        run_seed = int(seed) + int(repeat_idx) if repeat_total > 1 else int(seed)

        cfg = copy.deepcopy(cfg_master)
        cfg["defense"] = defense_name
        cfg["seed"] = int(run_seed)
        cfg = _normalize_fl_attack_cfg(cfg)
        set_seed(int(run_seed))

        atk_name_run = _get_atk_name(cfg)

        # per-run folder: <dataset>_<timestamp>_<atk>_<defense>
        run_timestamp = _timestamp()
        defense_run_name = (
            str(defense_name)
            if int(repeat_total) <= 1
            else f"{defense_name}_rep{int(repeat_idx) + 1}"
        )
        run_name = f"{dataset}_{run_timestamp}_{atk_name_run}_{defense_run_name}_seed{int(run_seed)}"
        out_dir = master_dir / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- save effective config used for this run ----
        save_config_yaml(cfg, out_dir / "cfg_effective.yaml")

        print(
            f"[Series] atk={atk_name_run} defense={defense_name} "
            f"repeat={int(repeat_idx) + 1}/{int(repeat_total)} seed={int(run_seed)} -> {out_dir}"
        )

        # recorder
        runtime_cfg = dict(cfg.get("runtime", {}) or {})
        rec = Recorder(
            str(out_dir),
            save_every=int(runtime_cfg.get("recorder_save_every", 1)),
        )

        rng = np.random.default_rng(cfg.get("seed", 0))
        client_sets, test_set, aux = build_label_skew_clients(cfg, args.data_root, rng)

        # log non-iid label counts (before poisoning)
        num_classes = int(cfg["model"]["num_classes"])
        try:
            log_client_label_counts(
                client_sets,
                aux["train_union"],
                num_classes=num_classes,
                out_dir=str(out_dir / "non_iid_layout"),
            )
        except Exception as e:
            rec.maybe_print(f"[warn] log_client_label_counts failed: {e!r}")

        # set up clients
        model_fn = build_model_fn(cfg)
        client_label_counts = aux.get("client_label_counts", {}) if isinstance(aux, dict) else {}
        clients = {
            cid: FLClient(
                cid,
                ds,
                model_fn,
                cfg,
                class_label_counts=client_label_counts.get(cid),
            )
            for cid, ds in client_sets.items()
        }
        client_partition_modes = (
            aux.get("client_partition_modes", {}) if isinstance(aux, dict) else {}
        )
        for cid, client in clients.items():
            client.data_partition_mode = client_partition_modes.get(
                cid,
                str(
                    (cfg.get("clients_setting", {}) or {}).get(
                        "non_iid_mode", "iid"
                    )
                ).lower(),
            )

        mal_ids = assign_malicious(clients, cfg, seed=int(run_seed))
        assign_mali_client_ds(clients, cfg, seed=int(run_seed))
        rec.maybe_print(f"[FL] malicious clients: {mal_ids}")
        if client_partition_modes:
            counts = Counter(client_partition_modes.values())
            malicious_modes = sorted(
                {clients[cid].data_partition_mode for cid in mal_ids}
            )
            rec.maybe_print(
                f"[FL] client data partitions: {dict(sorted(counts.items()))}; "
                f"malicious modes: {malicious_modes}"
            )

        server = FLServer(
            cfg=cfg,
            model_fn=model_fn,
            recorder=rec,
            test_dataset=test_set,
            out_dir=str(out_dir),
            hook_after_round=hook_after_round_pointillism_viz,
        )

        server.fl_iter(clients)

        ckpt_path = out_dir / "ckpt_final.pth"
        torch.save(server.model.state_dict(), ckpt_path)
        rec.maybe_print(f"[FL] Saved final server model to: {ckpt_path}")

        acc, asr = _read_final_metrics(out_dir)
        with master_csv.open("a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                dataset,
                atk_name_run,
                defense_name,
                "" if acc is None else acc,
                "" if asr is None else asr,
                str(out_dir),
            ])

    print(f"[Series] done. Master results: {master_csv}")


if __name__ == "__main__":
    main()
