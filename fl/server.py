# fl/server_min.py
from typing import Dict, Any, Callable, Optional
from copy import deepcopy
import inspect
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from aggregation import fedavg as fedavg_mod
from aggregation import normbound as normbound_mod
from aggregation import rfa as rfa_mod
from aggregation import krum as krum_mod
from aggregation import flame as flame_mod
from aggregation import deepsight as deepsight_mod
from aggregation import pointillism_fl as pointillism_fl_mod

from backdoor.eval import build_backdoor_evalset, build_label_flip_evalset
from pointillism.viz_mainstream import (
    load_banks_from_state,
    save_mainstream_one_row_per_label,
)
from attacks.neurotoxin import NeurotoxinAttack
from attacks.a3fl import A3FLAttack
from attacks.cerp import CerpAttack
from attacks.dba import DBAAttack
from attacks.audio_badnet import AudioAttackEvalDataset
from backdoor.eval import _extract_labels_fast

class FLServer:
    """
    Minimal FL server for:
      - client sampling
      - FedAvg aggregation
      - clean accuracy evaluation
      - optional hook per round (e.g., pointillism)
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        model_fn: Callable[[], torch.nn.Module],
        recorder,
        test_dataset=None,
        out_dir: Optional[str] = None,
        hook_after_round: Optional[Callable[..., None]] = None,
    ):
        self.cfg = cfg
        self.model_fn = model_fn
        self.recorder = recorder
        self.test_dataset = test_dataset
        self.out_dir = out_dir
        self.hook_after_round = hook_after_round

        seed = int(cfg.get("seed", 0))
        self.rng = np.random.default_rng(seed)

        # support either cfg["device"] or cfg["train"]["device"]
        dev = cfg.get("device", None)
        if dev is None:
            dev = (cfg.get("train", {}) or {}).get("device", "cuda")
        self.device = torch.device(dev)

        self.model = model_fn().to(self.device)

        self.defense = str(cfg.get("defense", None)).lower()
        self.attack_cfg = dict(cfg.get("attack", {}) or cfg.get("backdoor", {}) or {})
        self.atk_name = str(self.attack_cfg.get("atk_name", "none")).lower()

        self.atk_helper = None
        self.backdoor_evalset = None

        if self.atk_name == "neurotoxin":
            self.atk_helper = NeurotoxinAttack(self.attack_cfg)
        elif self.atk_name == "a3fl":
            merged = dict(self.attack_cfg)
            merged.update(cfg.get("a3fl_atk", {}) or {})
            self.atk_helper = A3FLAttack(merged)
        elif self.atk_name == "cerp":
            self.atk_helper = CerpAttack(self.cfg)
        elif self.atk_name == "dba":
            self.atk_helper = DBAAttack(self.cfg)

        if self.test_dataset is not None:
            self.backdoor_evalset = self._build_backdoor_evalset()

    @staticmethod
    def _state_dict_to_cpu(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in state_dict.items():
            if torch.is_tensor(v):
                out[k] = v.detach().cpu().clone()
            else:
                out[k] = deepcopy(v)
        return out

    def _round_summary_prefix(self) -> str:
        atk = self.atk_name if self.atk_name else "none"
        defense = self.defense if self.defense else "none"
        return f"{atk} - {defense}"

    def _sync_for_wall_clock(self) -> None:
        """Synchronize queued CUDA work so phase timings are true wall time."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _build_backdoor_evalset(self):
        if self.test_dataset is None or self.atk_name in {"", "none"}:
            return None

        per_class = int(self.attack_cfg.get("eval_per_class", 50))
        fixed_backdoor_attacks = {"badnet", "neurotoxin", "scale", "adaptive_pointillism"}

        if self.atk_name == "audio_badnet":
            labels = _extract_labels_fast(self.test_dataset)
            target = int(self.attack_cfg["target_label"])
            classes = [c for c in sorted(set(labels.tolist())) if c != target]
            rng = np.random.default_rng(int(self.attack_cfg.get("seed", 0)))
            pick = []
            for cls in classes:
                idxs = np.where(labels == cls)[0]
                if idxs.size > per_class:
                    idxs = rng.choice(idxs, size=per_class, replace=False)
                pick.extend(idxs.tolist())
            return AudioAttackEvalDataset(self.test_dataset, pick, self.attack_cfg)

        if self.atk_name == "dba" and self.atk_helper is not None:
            self.atk_helper.prepare_global_mask(self.test_dataset)
            return self.atk_helper.prepare_evalset(self.test_dataset, per_class=per_class)

        if self.atk_name in {"a3fl", "cerp"}:
            return None

        if self.atk_name == "label_flipping":
            return build_label_flip_evalset(
                self.test_dataset,
                attack_cfg=self.attack_cfg,
                per_class=per_class,
            )

        if self.atk_name not in fixed_backdoor_attacks:
            return None

        return build_backdoor_evalset(
            self.test_dataset,
            attack_cfg=self.attack_cfg,
            per_class=per_class,
        )


    def _select_adaptive_pointillism_candidates(
        self, *, clients, selected, client_states, adaptive_clients, rnd: int
    ):
        """Select max-ASR branches that retain benign-like Pointillism consensus."""
        positions = {int(cid): idx for idx, cid in enumerate(selected)}
        mal_ids = [int(cid) for cid in adaptive_clients]
        clean_states = [clients[cid].adaptive_pointillism_candidates[0]["state"] for cid in mal_ids]
        audit_ids = mal_ids
        adaptive_cfg = (self.attack_cfg.get("adaptive_pointillism", {}) or {})
        private_probe_seed = int(self.cfg.get("seed", 0)) + int(
            adaptive_cfg.get("probe_seed_offset", 700000001))
        candidate_states = [
            [candidate["state"] for candidate in clients[cid].adaptive_pointillism_candidates[1:]]
            for cid in mal_ids
        ]
        audit = pointillism_fl_mod.audit_adaptive_candidate_states(
            cfg=self.cfg, device=self.device, clean_states=clean_states,
            candidate_states=candidate_states,
            model_fn=self.model_fn, rnd=int(rnd), round_client_ids=audit_ids,
            private_probe_seed=private_probe_seed)
        baseline = audit["baseline"]
        clean_scores = [float(x) for x in baseline["scores"] if np.isfinite(float(x))]
        tau = float(np.median(clean_scores)) if clean_scores else float("inf")
        margin = float(adaptive_cfg.get("score_margin", 0.0))
        rows = []

        for mal_pos, cid in enumerate(mal_ids):
            client = clients[cid]
            best = None
            for candidate_pos, candidate in enumerate(client.adaptive_pointillism_candidates[1:]):
                result = audit["candidates"][mal_pos][candidate_pos]
                score = float(result["scores"][mal_pos])
                passed = bool(np.isfinite(score) and score >= tau - margin)
                kept = int(mal_pos in result["keep_idx"])
                row = [int(rnd), int(cid), float(candidate["ratio"]),
                       float(candidate["acc"]), float(candidate["asr"]), score,
                       tau, margin, int(passed), kept, float(candidate["loss"])]
                rows.append(row)
                if passed and (best is None or float(candidate["asr"]) > float(best["asr"])):
                    best = dict(candidate, score=score, kept=kept)

            chosen = client.adaptive_pointillism_candidates[0] if best is None else best
            submitted_clean = best is None
            summary = {
                "round": int(rnd), "cid": int(cid),
                "submitted_ratio": "clean" if submitted_clean else float(chosen["ratio"]),
                "submitted_clean": int(submitted_clean),
                "selected_acc": float(chosen["acc"]),
                "selected_asr": "" if submitted_clean else float(chosen["asr"]),
                "selected_score": float(baseline["scores"][mal_pos]) if submitted_clean else float(chosen["score"]),
                "tau_score": tau, "score_margin": margin,
            }
            client_states[positions[cid]] = client.submit_adaptive_pointillism_candidate(chosen, summary)
            self.recorder.maybe_print(
                f"[AdaptivePointillism][Round {rnd:03d}] cid={cid} "
                f"selected={summary['submitted_ratio']} score={summary['selected_score']:.6g} tau={tau:.6g}")
            if hasattr(self.recorder, "log_adaptive_pointillism_summary"):
                self.recorder.log_adaptive_pointillism_summary([summary])
        if rows and hasattr(self.recorder, "log_adaptive_pointillism"):
            self.recorder.log_adaptive_pointillism(rows)
        return client_states


    def _rebuild_dynamic_evalset(self):
        if self.atk_helper is None or self.test_dataset is None:
            return None

        per_class = int(self.attack_cfg.get("eval_per_class", 50))
        prepare = getattr(self.atk_helper, "prepare_evalset", None)
        if prepare is None:
            return None

        try:
            sig = inspect.signature(prepare)
            if "device" in sig.parameters:
                return prepare(self.test_dataset, per_class=per_class, device=self.device)
            return prepare(self.test_dataset, per_class=per_class)
        except TypeError:
            return prepare(self.test_dataset, per_class=per_class)


    def fl_iter(self, clients: Dict[int, Any]):
        rounds = int(self.cfg["train"]["rounds"])
        num_clients = len(clients)

        global_state = self._state_dict_to_cpu(self.model.state_dict())
        
        # ---- build fixed pools once (used by sampling_method="fixed_rate") ----
        all_ids = sorted(list(clients.keys()))
        self.malicious_ids = [cid for cid in all_ids if bool(getattr(clients[cid], "is_malicious", False))]
        self.benign_ids = [cid for cid in all_ids if not bool(getattr(clients[cid], "is_malicious", False))]

        if len(self.malicious_ids) + len(self.benign_ids) != num_clients:
            raise ValueError("client pools do not cover all clients")

        if self.atk_name == "dba" and self.atk_helper is not None:
            self.atk_helper.assign_client_masks(clients, seed=int(self.cfg.get("seed", 0)))

        runtime_cfg = dict(self.cfg.get("runtime", {}) or {})
        record_round_time = bool(runtime_cfg.get("record_round_time", False))

        for rnd in range(1, rounds + 1):
            if record_round_time:
                self._sync_for_wall_clock()
                round_t0 = perf_counter()
                phase_t0 = round_t0

            if self.recorder is not None:
                setattr(self.recorder, "round", rnd)

            # ---- select clients ----
            selected = self._sample_clients_once(num_clients)
            mal_flags = [bool(getattr(clients[cid], "is_malicious", False)) for cid in selected]
            self.recorder.log_selection(rnd, selected, mal_flags)
            self.recorder.maybe_print(f"[Round {rnd:03d}] Selected: {selected}")

            # ---- broadcast ----
            for cid in selected:
                clients[cid].set_weights(global_state)

            # ---- local train ----
            client_states, weights = [], []
            epochs = int(self.cfg["train"].get("local_epochs", 1))

            if self.atk_name in ("a3fl", "cerp") and self.atk_helper is not None:
                self.atk_helper.set_global_model(self.model)

            adaptive_clients = []
            for cid in selected:
                client = clients[cid]
                if self.atk_name == "adaptive_pointillism" and bool(client.is_malicious):
                    client.prepare_adaptive_pointillism_candidates(epochs=epochs, round_id=rnd)
                    adaptive_clients.append(cid)
                    # Placeholder clean branch; replaced after white-box candidate selection.
                    st = client.adaptive_pointillism_candidates[0]["state"]
                else:
                    st = client.local_train(
                        epochs=epochs, round_id=rnd, atk_helper=self.atk_helper)
                client_states.append(st)
                weights.append(client.num_samples())

            if adaptive_clients:
                client_states = self._select_adaptive_pointillism_candidates(
                    clients=clients, selected=selected, client_states=client_states,
                    adaptive_clients=adaptive_clients, rnd=rnd)

            if self.atk_name in ("a3fl", "cerp") and self.atk_helper is not None:
                trig = self.atk_helper.finalize_trigger(
                    agg_mode=str(self.attack_cfg.get("trigger_agg", "mean"))
                )
                if trig is not None:
                    self.backdoor_evalset = self._rebuild_dynamic_evalset()

            if record_round_time:
                self._sync_for_wall_clock()
                train_s = perf_counter() - phase_t0
                phase_t0 = perf_counter()

            # ---- defences ----
            common = dict(
                cfg=self.cfg,
                device=self.device,
                recorder=self.recorder,
                global_state=global_state,
                client_states=client_states,
                client_weights=weights,
                model_fn=self.model_fn,
                rnd=rnd,
                round_client_ids=selected,
            )
            if self.defense == "fedavg":
                global_state, agg_weights = fedavg_mod.aggregate(**common)
            elif self.defense == "normbound":
                global_state, agg_weights = normbound_mod.aggregate(**common)
            elif self.defense == "rfa":
                global_state, agg_weights = rfa_mod.aggregate(**common)
            elif self.defense == "krum":
                global_state, agg_weights = krum_mod.aggregate(**common)
            elif self.defense == "flame":
                global_state, agg_weights = flame_mod.aggregate(**common)
            elif self.defense == "deepsight":
                global_state, agg_weights = deepsight_mod.aggregate(**common)
            elif self.defense == "pointillism_fl":
                global_state, agg_weights = pointillism_fl_mod.aggregate(**common)
            else:
                raise ValueError(f"Unknown defense: {self.defense}")

            if record_round_time:
                self._sync_for_wall_clock()
                defense_s = perf_counter() - phase_t0
                phase_t0 = perf_counter()
            
            # ---- update global model ----
            global_state = self._apply_server_lr(global_state)
            self.model.load_state_dict(global_state, strict=True)

            # ---- evaluate: compute mali weight in aggr % ----
            if agg_weights is None:
                raise ValueError("agg_weights is required for mali weight % logging")

            if len(agg_weights) != len(selected):
                raise ValueError(
                    f"agg_weights length mismatch: got {len(agg_weights)} expected {len(selected)} "
                    f"(must align with selected client order)"
                )

            if hasattr(self.recorder, "log_selection_outcomes"):
                self.recorder.log_selection_outcomes(
                    rnd=rnd,
                    client_ids=selected,
                    mal_flags=mal_flags,
                    agg_weights=agg_weights,
                )

            total_w = float(sum(float(w) for w in agg_weights))
            if total_w <= 0.0:
                raise ValueError("total aggregation weight is non-positive")

            mali_w = float(
                sum(float(w) for (w, is_mal) in zip(agg_weights, mal_flags) if bool(is_mal))
            )
            mali_w_pct = 100.0 * mali_w / total_w

            self.recorder.maybe_print(
                f"[Round {rnd:03d}] Agg mali_weight={mali_w_pct:.2f}%"
            )

            # ---- evaluate (every round): ACC + ASR ----
            acc = self.evaluate_acc(self.model, self.test_dataset, self.device)
            acc_pct = float(acc) * 100.0

            bs = int((self.cfg.get("train", {}) or {}).get("batch_size", 256))
            asr_pct = None
            if self.backdoor_evalset is not None and len(self.backdoor_evalset) > 0:
                asr = self.evaluate_asr(
                    self.model,
                    self.backdoor_evalset,
                    self.device,
                    batch_size=bs,
                    exclude_target_label=True,
                )
                asr_pct = float(asr) * 100.0

            self.recorder.maybe_print(
                f"[Round {rnd:03d}] {self._round_summary_prefix()}  "
                f"ACC={acc_pct:.2f}% ASR={'N/A' if asr_pct is None else f'{asr_pct:.2f}%'}"
            )
            self.recorder.log_metrics(rnd, acc_pct, asr_pct, mali_w_pct)

            if record_round_time:
                self._sync_for_wall_clock()
                evaluation_s = perf_counter() - phase_t0
                phase_t0 = perf_counter()

            # ---- pointillism hook (after round) ----
            if self.hook_after_round is not None:
                self.hook_after_round(server=self, rnd=rnd, selected=selected)

            if record_round_time:
                self._sync_for_wall_clock()
                hook_s = perf_counter() - phase_t0
                total_s = perf_counter() - round_t0
                self.recorder.log_round_time(
                    rnd=rnd,
                    defense=self.defense,
                    train_s=train_s,
                    defense_s=defense_s,
                    evaluation_s=evaluation_s,
                    hook_s=hook_s,
                    total_s=total_s,
                )
                self.recorder.maybe_print(
                    f"[Round {rnd:03d}] wall_clock={total_s:.3f}s "
                    f"train={train_s:.3f}s defense={defense_s:.3f}s "
                    f"evaluation={evaluation_s:.3f}s hook={hook_s:.3f}s"
                )

        self.recorder.flush()
        return self.model


    @torch.no_grad()
    def evaluate_acc(self, model, dataset, device, batch_size=256):
        if dataset is None or len(dataset) == 0:
            return 0.0
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        model.eval()
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / max(1, total)


    @torch.no_grad()
    def evaluate_asr(
        self,
        model,
        attack_eval_ds,
        device: torch.device,
        batch_size: int = 256,
        exclude_target_label: bool = True,
    ) -> float:
        model.eval()
        loader = torch.utils.data.DataLoader(
            attack_eval_ds, batch_size=int(batch_size), shuffle=False
        )

        hit = 0
        total = 0

        for batch in loader:
            # supports either (x, target, orig_y) or (x, target)
            if len(batch) == 3:
                x, target, orig_y = batch
            elif len(batch) == 2:
                x, target = batch
                orig_y = None
            else:
                raise ValueError(f"Unexpected batch format (len={len(batch)}).")

            x = x.to(device)
            target = target.to(device)

            if exclude_target_label:
                if orig_y is None:
                    raise ValueError(
                        "exclude_target_label=True requires attack_eval_ds to provide orig_y "
                        "as the 3rd field (x, target, orig_y)."
                    )
                orig_y = orig_y.to(device)
                mask = (orig_y != target)          # exclude samples already in target class
                if mask.sum().item() == 0:
                    continue
                x = x[mask]
                target = target[mask]

            pred = model(x).argmax(dim=1)
            hit += (pred == target).sum().item()
            total += target.numel()

        return float(hit) / max(int(total), 1)


    def _apply_server_lr(self, new_state: dict) -> dict:
        train_cfg = self.cfg.get("train", {}) or {}
        server_lr = float(train_cfg.get("server_lr", 1.0))

        if not (0.0 < server_lr <= 1.0):
            raise ValueError(f"train.server_lr must lie in (0, 1], got {server_lr}")

        if server_lr == 1.0:
            return new_state

        old_state = self._state_dict_to_cpu(self.model.state_dict())
        blended_state = {}

        for key, new_val in new_state.items():
            old_val = old_state.get(key, None)
            can_blend = (
                isinstance(new_val, torch.Tensor)
                and isinstance(old_val, torch.Tensor)
                and new_val.shape == old_val.shape
                and new_val.is_floating_point()
            )

            if can_blend:
                blended_state[key] = (1.0 - server_lr) * old_val + server_lr * new_val
            else:
                blended_state[key] = new_val

        return blended_state


    def _sample_clients_once(self, num_clients: int) -> list[int]:
        cs = self.cfg["clients_setting"]
        k = int(cs["clients_per_round"])
        method = str(cs.get("sampling_method", "random")).lower()

        if not (1 <= k <= num_clients):
            raise ValueError(f"clients_per_round={k} invalid for num_clients={num_clients}")

        if method == "random":
            selected = self.rng.choice(num_clients, size=k, replace=False).tolist()
            selected.sort()
            return selected

        if method != "fixed_rate":
            raise ValueError(
                f"Unknown clients_setting.sampling_method='{method}' "
                "(expected 'random' or 'fixed_rate')"
            )

        mali_rate = float(cs.get("mali_rate", 0.0))
        if not (0.0 <= mali_rate <= 1.0):
            raise ValueError(f"mali_rate={mali_rate} must be in [0,1]")

        mal_pool = list(getattr(self, "malicious_ids", []))
        ben_pool = list(getattr(self, "benign_ids", []))

        # constant malicious count per round: floor(k * mali_rate)
        num_mal = int(k * mali_rate)
        if num_mal > len(mal_pool):
            num_mal = len(mal_pool)
        num_ben = k - num_mal

        if num_ben > len(ben_pool):
            raise ValueError(f"Need {num_ben} benign, only {len(ben_pool)} available")

        chosen_mal = self.rng.choice(mal_pool, size=num_mal, replace=False).tolist() if num_mal > 0 else []
        chosen_ben = self.rng.choice(ben_pool, size=num_ben, replace=False).tolist() if num_ben > 0 else []

        selected = chosen_mal + chosen_ben
        selected.sort()

        if len(selected) != k:
            raise ValueError(f"_sample_clients_once produced {len(selected)} clients, expected {k}")
        if len(set(selected)) != k:
            raise ValueError("_sample_clients_once produced duplicate client IDs")
        return selected
    
    
def hook_after_round_pointillism_viz(server, rnd: int, selected):
    cfg_point = (server.cfg.get("pointillism", {}) or {})
    viz_cfg = (cfg_point.get("viz", {}) or {})
    every_rounds = int(viz_cfg.get("every_rounds", 0))
    if every_rounds <= 0:
        return
    if int(rnd) % every_rounds != 0:
        return

    banks = load_banks_from_state(server.recorder.out_dir)
    if banks is None:
        return

    probes_per_class = int(viz_cfg.get("probes_per_class", 8))
    real_per_class = int(viz_cfg.get("real_per_class", 2))

    # Use train_dataset if available; otherwise test_dataset
    ds = getattr(server, "train_dataset", None)
    if ds is None:
        ds = server.test_dataset

    save_mainstream_one_row_per_label(
        out_dir=server.recorder.out_dir,
        rnd=int(rnd),
        banks=banks,
        grid_hw=int(cfg_point["grid_hw"]),
        probes_per_class=probes_per_class,
        real_per_class=real_per_class,
        dataset=ds,
        device=server.device,
        tag="mainstream",
    )
