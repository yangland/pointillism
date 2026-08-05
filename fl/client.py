from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from typing import Dict, Any, Optional, List
import numpy as np
import warnings
from utils.fl_utils import probe_train_loader
from backdoor.badnet import PatternBackdoorWrapper
from backdoor.eval import build_backdoor_evalset


class FLClient:
    def __init__(
        self,
        cid: int,
        dataset,
        model_fn,
        cfg,
        clean_dataset: Optional[object] = None,
        class_label_counts: Optional[List[int]] = None,
    ):
        self.cid = cid
        self.dataset = dataset
        self.clean_dataset = clean_dataset if clean_dataset is not None else dataset
        self.cfg = cfg
        self.model_fn = model_fn
        self.device = torch.device(cfg["device"])
        cs = cfg.get("clients_setting", {}) or {}
        total_clients = int(cs.get("clients", 1))
        clients_per_round = int(cs.get("clients_per_round", total_clients))
        # Optional memory saver: move idle client replicas to CPU between rounds.
        offload_cfg = cs.get("offload_idle_models", None)
        if offload_cfg is None:
            self._offload_idle_model = (
                self.device.type == "cuda" and total_clients > clients_per_round
            )
        else:
            self._offload_idle_model = bool(offload_cfg) and self.device.type == "cuda"
        self._idle_device = torch.device("cpu") if self._offload_idle_model else self.device
        self.model = model_fn().to(self._idle_device)
        self.is_malicious = None

        self.batch_size = cfg["train"]["batch_size"]
        self.local_epochs = cfg["train"]["local_epochs"]
        self.metadata = None  # S_i: dict[int -> torch.Tensor feature centroid]

        # true class label counts & priors from clean dataset
        class_num = int(cfg["model"]["num_classes"])
        if class_label_counts is not None:
            self.class_label_counts = [int(x) for x in class_label_counts]
        else:
            self.class_label_counts = self._compute_label_counts(self.clean_dataset, class_num)
        tot = int(sum(self.class_label_counts))
        self.class_priors = (
            np.array(self.class_label_counts, dtype=np.float32) / float(tot) 
            if tot > 0 else np.zeros(class_num, dtype=np.float32)
        )
        
        if cfg["train"]["optimizer"].lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=cfg["train"]["lr"], 
                                        weight_decay=cfg["train"]["weight_decay"])
        else:
            self.optimizer = optim.SGD(self.model.parameters(), 
                                       lr=cfg["train"]["lr"], 
                                       momentum=0.9, 
                                       weight_decay=cfg["train"]["weight_decay"])

        self.criterion = nn.CrossEntropyLoss()
        self.adaptive_pointillism_candidates = None
        self.adaptive_pointillism_summary = None

    def _move_model_to_train_device(self):
        if self._offload_idle_model:
            self.model = self.model.to(self.device)

    def _move_model_to_idle_device(self):
        if self._offload_idle_model:
            self.model = self.model.to(self._idle_device)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _population_malicious_scale(self) -> float:
        cs = self.cfg.get("clients_setting", {}) or {}
        mali_rate = float(cs.get("mali_rate", 0.0))
        if mali_rate <= 0.0:
            raise ValueError("Population malicious scaling requires mali_rate > 0.0 in clients_setting.")
        return 1.0 / mali_rate

    def set_weights(self, state_dict: Dict[str, Any]):
        target_device = next(self.model.parameters()).device
        copied = {}
        for k, v in state_dict.items():
            if torch.is_tensor(v):
                copied[k] = v.detach().to(device=target_device).clone()
            else:
                copied[k] = deepcopy(v)
        self.model.load_state_dict(copied, strict=True)

    def get_weights(self) -> Dict[str, Any]:
        out = {}
        for k, v in self.model.state_dict().items():
            if torch.is_tensor(v):
                out[k] = v.detach().cpu().clone()
            else:
                out[k] = deepcopy(v)
        return out

    def num_samples(self) -> int:
        return len(self.dataset)

    def _compute_label_counts(self, ds, num_classes: int):
        counts = [0] * num_classes
        for i in range(len(ds)):
            item = ds[i]
            if isinstance(item, tuple) and len(item) == 2:
                y = int(item[1])
            else:
                warnings.warn(f"Unexpected dataset item format: {type(item)}")
                continue  # Skip this item
            if 0 <= y < num_classes:
                counts[y] += 1
            else:
                warnings.warn(f"Label {y} out of range [0, {num_classes-1}]")    
        return counts

    def train_one_round(self) -> Dict[str, Any]:
        if len(self.dataset) == 0:
            return self.get_weights()
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self._move_model_to_train_device()
        self.model.train()
        try:
            for _ in range(self.local_epochs):
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                    loss.backward()
                    self.optimizer.step()
            return self.get_weights()
        finally:
            self._move_model_to_idle_device()

    def get_metadata(
        self,
        encoder,
        num_classes: int,
        per_class_max: int = 50,
        batch_size: int = 128,
        num_workers: int = 2,
        dataset: Optional[torch.utils.data.Dataset] = None,
        meta_tag: Optional[str] = None,  # optional logger tag
    ):
        """
        Build per-class centroids S_i^c and class-prior vector π_i from the chosen dataset.
        Output stored in:
            self.metadata_std = {
                "pi": np.ndarray [C],          # π_i^c
                "S":  {c: torch.FloatTensor[D]},  # S_i^c
                "n":  {c: int},               # counts
                # optional (behind cfg flag):
                # "Z": {c: torch.FloatTensor[n_c, D]},
            }
        """
        assert dataset is not None, "get_metadata requires an explicit dataset (clean or poisoned)."

        C = int(num_classes)
        device = torch.device(self.cfg["device"])

        # ---- select up to per_class_max indices per class ----
        per_class_indices: Dict[int, List[int]] = {c: [] for c in range(C)}
        
        for idx in range(len(dataset)):
            # early stop once every class hits the cap
            if all(len(per_class_indices[c]) >= per_class_max for c in range(C)):
                break
            item = dataset[idx]
            # robust label extraction
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                y = int(item[1])

            if 0 <= y < C and len(per_class_indices[y]) < per_class_max:
                per_class_indices[y].append(idx)

        # prune empty classes
        per_class_indices = {c: idxs for c, idxs in per_class_indices.items() if len(idxs) > 0}

        # ---- thin wrapper to preserve original transforms/poisoning ----
        class _IndexSubset(torch.utils.data.Dataset):
            def __init__(self, base, indices):
                self.base, self.indices = base, indices
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, i):
                return self.base[self.indices[i]]

        # ---- encode per class; compute centroids ----
        keep_Z = False
        self.metadata_std = {"pi": None, "S": {}, "n": {}}
        encode_batch   = int(getattr(encoder, "batch_size", batch_size))
        pin_mem        = (device.type == "cuda")

        total = 0
        for c, idxs in per_class_indices.items():
            subset = _IndexSubset(dataset, idxs)
            loader = DataLoader(
                subset, batch_size=encode_batch, shuffle=False,
                num_workers=int(num_workers), pin_memory=pin_mem
            )
            feats = []
            for x, _y in loader:
                x = x.to(device, non_blocking=True)
                if x.dtype != torch.float32:
                    x = x.float()
                with torch.no_grad():
                    if hasattr(encoder, "adapt_input"):
                        x = encoder.adapt_input(x)
                    z = encoder.encode(x)  # [B, D]
                feats.append(z.detach().cpu())

            if not feats:
                continue

            Z = torch.cat(feats, dim=0)          # [n_c, D]
            # averaging all features as mu
            mu = Z.mean(dim=0)                   # [D]
            self.metadata_std["S"][int(c)] = mu  # centroid S_i^c
            self.metadata_std["n"][int(c)] = int(Z.shape[0])
            total += int(Z.shape[0])

            if keep_Z:
                # lazily attach a Z bucket only when needed
                if "Z" not in self.metadata_std:
                    self.metadata_std["Z"] = {}
                self.metadata_std["Z"][int(c)] = Z  # debug payload

        # ---- class prior vector π_i ----
        pi = np.zeros(C, dtype=np.float32)
        denom = float(total) if total > 0 else 1.0
        for c, cnt in self.metadata_std["n"].items():
            pi[c] = cnt / denom
        self.metadata_std["pi"] = pi  # shape [C]

        # optional lightweight log
        present = sorted(list(self.metadata_std["S"].keys()))
        print(f"[Client {self.cid}][get_metadata] tag={meta_tag or 'none'} classes={present} total={total} priors_nz={np.count_nonzero(pi)}")


    def _make_loader(self, dataset, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=int(self.batch_size),
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True
        )


    def _new_optimizer(self, model):
        train_cfg = self.cfg["train"]
        if str(train_cfg["optimizer"]).lower() == "adam":
            return optim.Adam(model.parameters(), lr=float(train_cfg["lr"]),
                              weight_decay=float(train_cfg["weight_decay"]))
        return optim.SGD(model.parameters(), lr=float(train_cfg["lr"]),
                         momentum=float(train_cfg.get("momentum", 0.9)),
                         weight_decay=float(train_cfg["weight_decay"]))

    @staticmethod
    def _clone_state_cpu(state):
        return {k: (v.detach().cpu().clone() if torch.is_tensor(v) else deepcopy(v))
                for k, v in state.items()}

    def _evaluate_adaptive_model(self, model, dataset, *, asr=False):
        if dataset is None or len(dataset) == 0:
            return float("nan")
        loader = self._make_loader(dataset, shuffle=False)
        model.eval()
        hit = total = 0
        target = int((self.cfg.get("attack", {}) or {}).get("target_label", -1))
        with torch.no_grad():
            for batch in loader:
                if asr:
                    x, _target, original = batch
                    mask = original != target
                    if not bool(mask.any()):
                        continue
                    pred = model(x[mask].to(self.device)).argmax(dim=1).cpu()
                    hit += int((pred == target).sum().item())
                    total += int(pred.numel())
                else:
                    x, y = batch
                    pred = model(x.to(self.device)).argmax(dim=1).cpu()
                    hit += int((pred == y).sum().item())
                    total += int(y.numel())
        return float(hit) / float(max(1, total))

    def _adaptive_badnet_dataset(self, poison_frac: float):
        atk = self.cfg.get("attack", {}) or {}
        pattern_size = atk.get("pattern_size", None)
        if pattern_size is not None:
            pattern_size = (int(pattern_size[0]), int(pattern_size[1]))
        return PatternBackdoorWrapper(
            self.clean_dataset, target_label=int(atk.get("target_label", 0)),
            poison_frac=float(poison_frac), pattern_pos=str(atk.get("pattern_pos", "bottom_right")),
            pattern_padding=int(atk.get("pattern_padding", 0)), pattern_size=pattern_size,
            pattern_offsets=atk.get("pattern_offsets", None), value=float(atk.get("value", 1.0)),
            seed=int(self.cfg.get("seed", 0)) + int(self.cid) + int(round(10000 * poison_frac)),
            pattern_type=str(atk.get("pattern_type", "badnet_corner")),
            apply_trigger=True, apply_relabel=True,
        )

    def prepare_adaptive_pointillism_candidates(self, *, epochs: int, round_id: Optional[int] = None):
        """Train clean/poisoned endpoints and interpolate adaptive candidates."""
        atk = self.cfg.get("attack", {}) or {}
        adaptive = atk.get("adaptive_pointillism", {}) or {}
        ratios = [float(r) for r in adaptive.get(
            "poison_ratios", atk.get("poison_ratios", [0.1, 0.25, 0.5, 1.0]))]
        if not ratios or any(not 0.0 <= r <= 1.0 for r in ratios):
            raise ValueError("adaptive_pointillism.poison_ratios must contain values in [0, 1].")
        endpoint_poison_frac = float(adaptive.get("endpoint_poison_frac", 0.5))
        if not 0.0 <= endpoint_poison_frac <= 1.0:
            raise ValueError("adaptive_pointillism.endpoint_poison_frac must be in [0, 1].")
        start_state = self._clone_state_cpu(self.model.state_dict())
        eval_cfg = dict(atk)
        eval_cfg.update({"atk_name": "badnet", "seed": int(self.cfg.get("seed", 0)) + int(self.cid)})
        bd_eval = build_backdoor_evalset(
            self.clean_dataset, eval_cfg,
            per_class=int(adaptive.get("local_asr_per_class", 20)),
        )
        endpoints = []
        for ratio in [None, endpoint_poison_frac]:
            branch = self.model_fn().to(self.device)
            branch.load_state_dict(start_state, strict=True)
            old_model, old_optimizer = self.model, self.optimizer
            try:
                self.model, self.optimizer = branch, self._new_optimizer(branch)
                train_ds = self.clean_dataset if ratio is None else self._adaptive_badnet_dataset(ratio)
                losses = [float(self._train_one_epoch_standard(
                    self._make_loader(train_ds, shuffle=True))) for _ in range(int(epochs))]
                endpoints.append({
                    "ratio": ratio, "state": self._clone_state_cpu(branch.state_dict()),
                    "acc": self._evaluate_adaptive_model(branch, self.clean_dataset),
                    "asr": (float("nan") if ratio is None else
                            self._evaluate_adaptive_model(branch, bd_eval, asr=True)),
                    "loss": float(np.mean(losses)) if losses else 0.0,
                })
            finally:
                self.model, self.optimizer = old_model, old_optimizer
                del branch
        clean = endpoints[0]
        poisoned = endpoints[1]
        branches = [clean]
        evaluator = self.model_fn().to(self.device)
        try:
            for ratio in ratios:
                state = {}
                for key, clean_value in clean["state"].items():
                    poison_value = poisoned["state"][key]
                    if torch.is_floating_point(clean_value):
                        state[key] = torch.lerp(
                            clean_value, poison_value, float(ratio)
                        )
                    else:
                        state[key] = clean_value.clone()
                evaluator.load_state_dict(state, strict=True)
                branches.append({
                    "ratio": float(ratio),
                    "state": state,
                    "acc": self._evaluate_adaptive_model(evaluator, self.clean_dataset),
                    "asr": self._evaluate_adaptive_model(evaluator, bd_eval, asr=True),
                    "loss": (
                        (1.0 - float(ratio)) * float(clean["loss"])
                        + float(ratio) * float(poisoned["loss"])
                    ),
                })
        finally:
            del evaluator
        self.model.load_state_dict(start_state, strict=True)
        self.adaptive_pointillism_candidates = branches
        self.adaptive_pointillism_summary = None
        return branches

    def submit_adaptive_pointillism_candidate(self, candidate, summary):
        self.model.load_state_dict(candidate["state"], strict=True)
        self.adaptive_pointillism_summary = dict(summary)
        return self.get_weights()

    def _train_one_epoch_standard(self, loader):
        self.model.train()
        total, count = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(xb)
            loss = self.criterion(out, yb)
            loss.backward()
            self.optimizer.step()
            bs = xb.size(0)
            total += float(loss.item()) * bs
            count += bs
        return total / max(1, count)


    def _mali_train_one_epoch(self, loader_poison, *, round_id=None, atk_helper=None):
        """
        Malicious training for current epoch:
        - badnet: standard poisoned training.
        - neurotoxin: virtual benign epoch + projected poisoned training (uses NeurotoxinAttack helper).
        - a3fl: client-side trigger search + poisoned training (uses A3FL helper).
        - cerp: client-side trigger search + submit + poisoned training (uses CerpAttack helper).
        - reverse_grad_sign: standard local training; sign reversal happens after all epochs.
        - label_flipping / label_perturbation: standard local training on relabeled data.
        Args:
        loader_poison: DataLoader for poisoned samples (client local poisoned data).
        round_id: current round id (for Neurotoxin to save server flat).
        atk_helper: shared helper object provided by server (generic: NeurotoxinAttack/A3FLAttack/CerpAttack)
        Returns:
        float average loss for the epoch, or (loss, trigger) tuple if you choose to return trigger (not used here).
        """
        atk_cfg  = self.cfg.get("attack", {})
        atk_name = str(atk_cfg.get("atk_name", "badnet")).lower()
        tgt      = int(atk_cfg.get("target_label", 7))

        if self.cfg.get("debug_probes", False):
            log_fn = getattr(getattr(self, "recorder", None), "maybe_print", None) or print
            probe_train_loader(
                log_fn,
                self.cfg,
                loader_poison,
                cid=self.cid,
                target_label=int(self.cfg["attack"].get("target_label", 0)),
                dataset_name=type(self.dataset).__name__,
            )

        if atk_name in {"badnet", "audio_badnet"}:
            return self._train_one_epoch_standard(loader_poison)

        if atk_name in {"reverse_grad_sign", "label_flipping", "label_perturbation"}:
            return self._train_one_epoch_standard(loader_poison)

        if atk_name == "scale":
            scale = self._population_malicious_scale()
            
            # ---- snapshot SERVER model before training ----
            # at this moment, self.model == server model for this round
            server_sd = {
                k: v.detach().clone()
                for k, v in self.model.state_dict().items()
            }

            # ---- standard poisoned training (unchanged) ----
            loss = self._train_one_epoch_standard(loader_poison)

            # ----- scale gradient and load back into the model -----
            local_sd = self.model.state_dict()
            scaled_sd = self.gradient_scale(server_sd, local_sd, scale)
            self.model.load_state_dict(scaled_sd, strict=True)

            return loss

        if atk_name == "neurotoxin":
            nt = atk_helper
            if nt is None:
                raise ValueError("Neurotoxin attack requires a shared NeurotoxinAttack helper (client.nt or arg).")
            if not hasattr(self, "clean_dataset") or self.clean_dataset is None:
                raise ValueError("Neurotoxin attack requires client.clean_dataset (benign data).")

            # hyperparams for temp benign optimizer (mirror your two-clients code)
            lr        = float(self.cfg.get("lr", self.cfg["train"].get("lr", 0.001)))
            momentum  = float(self.cfg.get("momentum", 0.9))
            weightdec = float(self.cfg.get("weight_decay", self.cfg["train"].get("weight_decay", 5e-4)))

            # 1) save current weights
            old_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

            # 2) virtual benign epoch (temp SGD) on clean data
            temp_opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weightdec)
            loader_benign = self._make_loader(self.clean_dataset, shuffle=True)
            self.model.train()
            for xb, yb in loader_benign:
                xb, yb = xb.to(self.device), yb.to(self.device)
                temp_opt.zero_grad(set_to_none=True)
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                temp_opt.step()

            # 3) store flattened server update and restore weights
            nt.save_server_flat(prev_state=old_state, new_state=self.model.state_dict(), round_id=round_id)
            self.model.load_state_dict(old_state)

            # 4) poisoned train with projection
            self.model.train()
            total, count = 0.0, 0
            for xb, yb in loader_poison:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()

                # Critical: zero top-k gradient coords aligned with nt.latest_flat
                nt.apply_projection_from_latest(self.model)

                self.optimizer.step()
                bs = xb.size(0)
                total += float(loss.item()) * bs
                count += bs
            return total / max(1, count)

        if atk_name == "dba":
            dba_helper = atk_helper or getattr(self, "dba_helper", None)
            if dba_helper is None:
                raise ValueError("DBA attack requires a DBAAttack helper passed as atk_helper.")

            dba_cfg = self.cfg.get("dba_atk", {}) or {}
            scale = float(dba_cfg.get("scale", 1.0))

            # snapshot server model before local training
            server_sd = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

            # build loader_poison: always use full dataset loader (do NOT collapse to only poison_idxs)
            loader = self._make_loader(self.dataset, shuffle=True)

            total, count = 0.0, 0
            self.model.train()
            for xb, yb in loader:
                xb = xb.to(self.device)
                xb_p, yb_t = dba_helper.poison_batch_for_client(self.cid, xb, yb,
                                                                device=self.device, round_id=round_id)

                self.optimizer.zero_grad(set_to_none=True)
                out = self.model(xb_p)
                loss = self.criterion(out, yb_t)
                loss.backward()
                self.optimizer.step()

                bs = xb_p.size(0)
                total += float(loss.item()) * bs
                count += bs

            # apply scaling
            local_sd = self.model.state_dict()
            scaled_sd = self.gradient_scale(server_sd, local_sd, float(scale))
            self.model.load_state_dict(scaled_sd, strict=True)

            return total / max(1, count)

        # ----- A3FL (keep existing behavior but use helper variable) -----
        if atk_name == "a3fl":
            a3 = atk_helper
            if a3 is None:
                raise ValueError("A3FL attack requires an A3FLAttack helper (neuro_helper).")

            # run trigger search using global model stored inside helper
            loader_for_search = self._make_loader(self.dataset, shuffle=True)
            model_for_search = getattr(self, "model", None)

            # 1) client-side search that returns a CPU trigger
            trig_cpu = a3.client_search_trigger(
                loader_for_search,
                device=self.device,
                verbose=self.cfg.get("debug_probes", False),
            )

            # 2) submit this client's trigger for server-side pooling
            a3.submit_client_trigger(trig_cpu)

            # 3) poisoned local training using the learned trigger
            self.model.train()
            total, count = 0.0, 0
            poison_frac=atk_cfg.get("poison_frac", 0.15)
            # print(f"[Client {self.cid}][A3FL] poison_frac={poison_frac}")
            for xb_p, yb_t in a3.poison_loader_batchwise(base_loader=self._make_loader(self.dataset, shuffle=True),
                                                         poison_frac=poison_frac):
                xb_p, yb_t = xb_p.to(self.device), yb_t.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                out = self.model(xb_p)
                loss = self.criterion(out, yb_t)
                loss.backward()
                self.optimizer.step()

                bs = xb_p.size(0)
                total += float(loss.item()) * bs
                count += bs
            return total / max(1, count)

        # ----- CerP -----
        if atk_name == "cerp":
            cerp = atk_helper
            if cerp is None:
                raise ValueError("Cerp attack requires a CerpAttack helper (neuro_helper).")

            # 1) Local trigger search using client's poisoned data (loader_poison).
            # Prefer client's local model for search; otherwise use helper.global_model.
            model_for_search = getattr(self, "model", None) or getattr(cerp, "global_model", None)
            if model_for_search is None:
                raise RuntimeError("No model available for CerP trigger search (client.model or cerp.global_model).")

            local_trigger_cpu = cerp.client_search_trigger(
                loader=loader_poison,
                model_for_search=model_for_search,
                device=self.device,
                verbose=self.cfg.get("debug_probes", False),
            )

            # 2) Submit trigger to helper (must succeed). No fallback here.
            # If your clients run out-of-process, the client must return the trigger to the server
            # and the server should call atk_helper.submit_client_trigger(trigger).
            cerp.submit_client_trigger(local_trigger_cpu)

            # 3) Poisoned local training: require helper to implement poisoned_local_train.
            if not hasattr(cerp, "poisoned_local_train"):
                raise ValueError("CerpAttack helper must implement poisoned_local_train; no inline fallback allowed.")

            loss_val = cerp.poisoned_local_train(
                client_model=self.model,
                dataset_loader=loader_poison,
                optimizer=self.optimizer,
                device=self.device,
                verbose=self.cfg.get("debug_probes", True),
            )

            # Normalize return: poisoned_local_train may return None (treat as 0.0), or average loss.
            if loss_val is None:
                return 0.0
            return loss_val

        raise ValueError(f"Unsupported atk_name='{atk_name}' for malicious training.")


    def gradient_scale(self,
                    server_sd: dict,
                    local_sd: dict,
                    scale: float) -> dict:
        """
        Return a new state_dict where the update (local - server) is scaled:

            grad = local - server
            scaled = server + scale * grad

        Args:
            server_sd: state_dict (values are tensors) representing the server model
                    at round start (detached clones recommended).
            local_sd:  state_dict for the locally trained model (after local training).
            scale:     scalar factor to multiply the gradient.

        Returns:
            new state_dict suitable for load_state_dict(..., strict=True).
        """
        # fast path
        if scale == 1.0:
            return {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in local_sd.items()
            }

        new_sd = {}
        for key, new_val in local_sd.items():
            old_val = server_sd.get(key, None)

            # --- skip BN stats and counters completely ---
            if (
                "running_mean" in key
                or "running_var" in key
                or "num_batches_tracked" in key
            ):
                # just copy local value
                if isinstance(new_val, torch.Tensor):
                    new_sd[key] = new_val.clone()
                else:
                    new_sd[key] = new_val
                continue

            can_blend = (
                isinstance(new_val, torch.Tensor)
                and isinstance(old_val, torch.Tensor)
                and new_val.shape == old_val.shape
                and new_val.is_floating_point()
                and old_val.is_floating_point()
            )

            if not can_blend:
                # keep the locally-trained value as-is (clone to avoid aliasing)
                if isinstance(new_val, torch.Tensor):
                    new_sd[key] = new_val.clone()
                else:
                    new_sd[key] = new_val
                continue

            # compute grad and scaled value
            sv = old_val
            lv = new_val
            grad = lv - sv
            scaled_val = sv + grad * scale
            # scaled_val is already float; keep dtype/device as is
            new_sd[key] = scaled_val.clone()

        return new_sd


    def local_train(self, *, epochs: int = 1, round_id: Optional[int] = None, atk_helper=None) -> Dict[str, Any]:
        """
        Unified per-round local training:
          - Benign: standard training on self.dataset.
          - Malicious: attack-specific path.
        Returns state dict to send to server.
        """
        if len(self.dataset) == 0:
            return self.get_weights()

        self._move_model_to_train_device()

        try:
            if not getattr(self, "is_malicious", False):
                print(f"train client {self.cid} benign")
                loader = self._make_loader(self.dataset, shuffle=True)
                for _ in range(int(epochs)):
                    _ = self._train_one_epoch_standard(loader)
                return self.get_weights()

            # malicious
            print(f"train client {self.cid} mali")
            atk_cfg = self.cfg.get("attack", {}) or {}
            atk_name = str(atk_cfg.get("atk_name", "badnet")).lower()
            loader_poison = self._make_loader(self.dataset, shuffle=True)
            apply_population_scale = atk_name in {
                "reverse_grad_sign",
                "label_flipping",
                "label_perturbation",
            }
            round_start_sd = None
            if atk_name == "reverse_grad_sign" or apply_population_scale:
                round_start_sd = {
                    k: v.detach().clone()
                    for k, v in self.model.state_dict().items()
                }
            for _ in range(int(epochs)):
                _ = self._mali_train_one_epoch(loader_poison, round_id=round_id, atk_helper=atk_helper or getattr(self, "nt", None))

            if atk_name == "reverse_grad_sign":
                reversed_sd = self.gradient_scale(round_start_sd, self.model.state_dict(), -1.0)
                self.model.load_state_dict(reversed_sd, strict=True)

            if apply_population_scale:
                scaled_sd = self.gradient_scale(
                    round_start_sd,
                    self.model.state_dict(),
                    self._population_malicious_scale(),
                )
                self.model.load_state_dict(scaled_sd, strict=True)

            return self.get_weights()
        finally:
            self._move_model_to_idle_device()
