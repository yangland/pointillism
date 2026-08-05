# attacks/cerp.py
import copy
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, TensorDataset
from attacks.base_attack import BaseAttack

class CerpAttack(BaseAttack):
    """
    Cerp attack helper implementing BaseAttack interface.
    Provides:
      - client_search_trigger(loader) -> local trigger (cpu tensor)
      - submit_client_trigger(trigger)
      - finalize_trigger(agg_mode="mean") -> final trigger (stored in self.trigger)
      - prepare_evalset(test_set, per_class=50) -> Dataset yielding (poisoned_input, target_label, orig_label)
      - apply_trigger_batch(xb)
      - poisoned_local_train(...)  (keeps your existing logic)
    """
    def __init__(self, atk_cfg: dict):
        super().__init__(atk_cfg)
        atk  = self.cfg.get("attack", {})
        cerp = self.cfg.get("cerp_atk", {})

        self.device = torch.device(self.cfg.get("device", "cpu"))

        # trigger params
        self.trigger_shape = tuple(self.cfg.get("trigger_shape", (3, 8, 8)))
        self.target_label  = int(atk.get("target_label", 0))

        # ---------- read CerP hyperparameters from cerp_atk ----------
        # trigger search hyperparameters
        self.num_steps = int(cerp.get("steps", 200))   # trigger search steps
        self.lr        = float(cerp.get("lr", 1e-2))   # trigger search LR

        # L2 norm budget for trigger
        phi_cfg = cerp.get("phi", self.cfg.get("phi", 5.0))
        self.phi = float(phi_cfg)

        # poisoned local training hyperparameters
        self.poison_epochs = int(cerp.get("poison_epochs", 1))
        self.poison_frac   = float(atk.get("poison_frac", 0.5))

        # deviation regularizer
        alpha_cfg = cerp.get("alpha", self.cfg.get("cerp_alpha", 0.1))
        beta_cfg  = cerp.get("beta",  self.cfg.get("cerp_beta", 0.0))
        self.alpha = float(alpha_cfg)
        self.beta  = float(beta_cfg)
        # ------------------------------------------------------------

        # internal trigger state
        self.trigger = None
        self.mask    = None
        self._submitted_triggers = []


    def _init_trigger(self, device, example_shape: tuple = None):
            """
            Initialize trigger as a full-image tensor if example_shape is provided.
            """
            if example_shape is not None:
                c, h, w = example_shape
                shape = (c, h, w)
            else:
                shape = tuple(self.cfg.get("trigger_shape", (3, 8, 8))) # Likely (1, 28, 28) for F-MNIST

            # init small random in [0,1] data-space (match your ToTensor() pipeline)
            t = torch.randn(shape, device=device) * 0.01

            # --- MODIFICATION START: Define the Sparse Mask ---
            C, H, W = shape
            self.mask = torch.zeros(shape, device=device)

            # Define a 5x5 patch at the bottom-right corner (Pillow coordinates)
            # Assuming H=28, W=28, this sets pixels 23-27 for rows and columns.
            if H >= 5 and W >= 5:
                # Slicing: [channels, rows, columns]
                self.mask[:, H-5:H, W-5:W] = 1.0 
            else:
                # Fallback for unexpected shapes: use full image (e.g., if H or W < 5)
                self.mask.fill_(1.0)
            
            # Apply the mask to the initial random noise 't'
            # This ensures the noise starts off being zero outside the patch.
            t = t * self.mask.to(device)
            # --- MODIFICATION END ---

            # clip to valid data range initially
            t = torch.clamp(t, 0.0, 1.0)
            self.trigger = t.detach().clone().requires_grad_(True)
            # self.mask is now sparse


    def project_trigger(self, trig):
        """L2 projection to radius phi (in same numerical input scale)."""
        with torch.no_grad():
            vec = trig.view(-1)
            norm = vec.norm(p=2)
            if norm > self.phi:
                vec.mul_(self.phi / (norm + 1e-12))
                trig.copy_(vec.view_as(trig))
        return trig


    def client_search_trigger(self, loader, model_for_search=None, device=None, verbose=False):
        """
        Run trigger search locally on a malicious client.

        Unlike the previous cached-single-batch approach, this iterates over 'loader'
        and performs self.num_steps updates across multiple minibatches (like original CerP).
        Returns the CPU tensor for submission.
        """
        device = torch.device(device)
        model = model_for_search or getattr(self, "global_model", None)
        if model is None:
            raise RuntimeError("No model provided for trigger search.")

        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # infer trigger shape from first batch if present
        it = iter(loader)
        try:
            xb0, _ = next(it)
        except StopIteration:
            raise RuntimeError("Empty loader passed to CerpAttack.client_search_trigger.")
        C, H, W = xb0.shape[1], xb0.shape[2], xb0.shape[3]

        # initialize trigger as full-image (C,H,W) on device
        self._init_trigger(device, example_shape=(C, H, W))

        # choose optimizer; Adam is fine but we do many updates across batches
        optimizer = torch.optim.Adam([self.trigger], lr=self.lr)

        step = 0
        # keep iterating over loader batches until we've done self.num_steps updates
        while step < self.num_steps:
            try:
                xb, _ = next(it)
            except StopIteration:
                # restart iterator when exhausted
                it = iter(loader)
                xb, _ = next(it)

            xb = xb.to(device, non_blocking=True)

            # optionally mask trigger so only pattern positions contribute (mask default ones)
            # ensure trigger respects mask
            with torch.no_grad():
                self.trigger.data.mul_(self.mask)

            xb_p = self.apply_trigger_batch(xb)   # apply trigger to this batch
            logits = model(xb_p)
            target = torch.full((xb_p.size(0),), self.target_label, dtype=torch.long, device=device)
            ce = F.cross_entropy(logits, target, reduction="mean")

            optimizer.zero_grad(set_to_none=True)
            ce.backward()
            optimizer.step()

            # project to phi budget
            self.project_trigger(self.trigger.data)

            if verbose and (step % 25 == 0):
                with torch.no_grad():
                    mean_conf = torch.softmax(logits, dim=1)[:, self.target_label].mean().item()
                print(f"[Cerp][local_search step {step}/{self.num_steps}] mean_target_conf={mean_conf:.4f}")

            step += 1

        # return CPU copy for submission
        return self.trigger.detach().cpu()


    def apply_trigger_batch(self, xb: torch.Tensor) -> torch.Tensor:
        """
        Apply the current CerP trigger to a batch of inputs.

        - xb: batch of images, shape (B, C, H, W), values in [0, 1]
        - self.trigger: learned noise, shape (C, H, W)
        - self.mask: optional mask, shape (C, H, W), where 1 = locations to perturb

        This does not depend on attack.trigger / pattern_pos / badnet configs.
        """
        if self.trigger is None:
            raise ValueError("CerP trigger is not initialized. Run client_search_trigger first.")

        x = xb.clone()
        trig = self.trigger.to(x.device)

        # if a mask is defined, restrict perturbation to those positions
        if self.mask is not None:
            trig = trig * self.mask.to(x.device)

        # broadcast trigger over the batch
        x = x + trig.unsqueeze(0)

        # inputs are in [0,1] after ToTensor(), keep them in that range
        x = torch.clamp(x, 0.0, 1.0)
        return x


    def submit_client_trigger(self, trig_cpu: torch.Tensor):
        """
        Submit a CPU trigger tensor from a client for pooling.
        """
        if trig_cpu is None:
            return
        # ensure CPU & float32
        trig_cpu = trig_cpu.detach().cpu().float()
        self._submitted_triggers.append(trig_cpu)

    def finalize_trigger(self, agg_mode: str = "mean"):
        """
        Aggregate submitted triggers into final self.trigger (on self.device).
        agg_mode: "mean" | "median" | "select_best"
        """
        if not self._submitted_triggers:
            return None
        stacked = torch.stack(self._submitted_triggers, dim=0)  # (n_sub, C, H, W)
        if agg_mode == "mean":
            trig_cpu = stacked.mean(dim=0)
        elif agg_mode == "median":
            trig_cpu = stacked.median(dim=0).values
        else:
            raise ValueError("unsupported agg_mode")
        # store on device for apply_trigger_batch and poisoned_local_train
        self.trigger = trig_cpu.to(self.device)
        # clear submissions for next round
        self._submitted_triggers = []
        return self.trigger

    # --------------------
    # build evaluation set using finalized trigger
    # --------------------
    def poisoned_local_train(self,
                            client_model,
                            dataset_loader,
                            optimizer,
                            device=None,
                            epochs=None,
                            verbose=False):
        device = torch.device(device) if device is not None else self.device
        epochs = epochs or self.poison_epochs

        client_model.to(device)
        client_model.train()

        # benign reference model for deviation regularizer
        benign_copy = copy.deepcopy(client_model).to(device)
        benign_copy.eval()
        for p in benign_copy.parameters():
            p.requires_grad_(False)

        if self.alpha > 0.0:
            with torch.no_grad():
                flat_benign = torch.nn.utils.parameters_to_vector(
                    [p.detach() for p in benign_copy.parameters()]
                )

        criterion = torch.nn.CrossEntropyLoss()

        r = max(0.0, min(1.0, float(self.poison_frac)))  # clamp to [0,1]

        # ----------------- DEBUG: pre-poison evaluation -----------------
        try:
            if getattr(self, "trigger", None) is not None:
                # eval_trigger_on_loader is the helper you added earlier
                pre_conf, pre_asr, pre_n = eval_trigger_on_loader(
                    client_model, dataset_loader, self.trigger, self.target_label, device, max_batches=8
                )
                print(f"[Cerp][poisoned_local_train] PRE-POISON mean_conf={pre_conf:.4f}, ASR={pre_asr*100:.2f}%, samples_eval={pre_n}")
            else:
                print("[Cerp][poisoned_local_train] PRE-POISON: no trigger found (self.trigger is None)")
        except Exception as e:
            print(f"[Cerp][poisoned_local_train] PRE-POISON eval failed: {e}")
        # ----------------------------------------------------------------

        # useful one-time optimizer info (print param groups)
        try:
            if verbose:
                print(f"[Cerp][poisoned_local_train] optimizer param_groups: {optimizer.param_groups}")
        except Exception:
            pass

        total_processed = 0
        for e in range(epochs):
            total_loss = 0.0
            total_bs = 0

            batch_idx = 0
            for xb, yb in dataset_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                bs = xb.size(0)
                if r > 0.0 and bs > 0:
                    k = max(1, int(round(r * bs)))  # number of poisoned samples in this batch
                    perm = torch.randperm(bs, device=device)
                    pois_idx = perm[:k]
                    clean_idx = perm[k:]

                    xb_p = xb.clone()
                    # apply trigger only to the subset
                    xb_p[pois_idx] = self.apply_trigger_batch(xb[pois_idx])

                    yb_p = yb.clone()
                    yb_p[pois_idx] = self.target_label
                    # clean_idx keep original labels and pixels
                else:
                    k = 0
                    pois_idx = torch.tensor([], device=device, dtype=torch.long)
                    xb_p = xb
                    yb_p = yb

                # ----------------- DEBUG: per-batch checks -----------------
                if verbose and (batch_idx % 20 == 0):
                    try:
                        # check trigger presence and magnitude
                        if getattr(self, "trigger", None) is None:
                            print(f"[Cerp][batch {batch_idx}] WARNING: trigger is None")
                        else:
                            trig_l2 = float(torch.norm(self.trigger.view(-1), p=2).item())
                            trig_min = float(self.trigger.min().item())
                            trig_max = float(self.trigger.max().item())
                            print(f"[Cerp][batch {batch_idx}] trigger L2={trig_l2:.6f} min={trig_min:.4f} max={trig_max:.4f}")

                        # check that poisoned indices actually changed pixels
                        if k > 0:
                            # compare one poisoned sample pixel-sum before/after
                            idx0 = int(pois_idx[0].item()) if pois_idx.numel() > 0 else None
                            if idx0 is not None:
                                orig_sum = float(xb[idx0].abs().sum().item())
                                poisoned_sum = float(xb_p[idx0].abs().sum().item())
                                print(f"[Cerp][batch {batch_idx}] bs={bs} poisoned_k={k} idx0={idx0} orig_sum={orig_sum:.4f} poisoned_sum={poisoned_sum:.4f}")
                            else:
                                print(f"[Cerp][batch {batch_idx}] bs={bs} poisoned_k={k} (no idx0)")
                    except Exception as ex:
                        print(f"[Cerp][batch {batch_idx}] per-batch debug failed: {ex}")
                # ---------------------------------------------------------------

                optimizer.zero_grad(set_to_none=True)
                logits = client_model(xb_p)
                loss_ce = criterion(logits, yb_p)

                if self.alpha > 0.0:
                    flat_cur = torch.nn.utils.parameters_to_vector(client_model.parameters())
                    dev_loss = (flat_cur - flat_benign).pow(2).sum()
                    loss = loss_ce + self.alpha * dev_loss
                else:
                    loss = loss_ce

                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * bs
                total_bs += bs
                total_processed += bs
                batch_idx += 1

            if verbose and total_bs > 0:
                print(f"[Cerp][poison_epoch {e}] avg_loss={total_loss/total_bs:.4f} processed={total_processed}")

        # ----------------- DEBUG: post-poison evaluation -----------------
        try:
            if getattr(self, "trigger", None) is not None:
                post_conf, post_asr, post_n = eval_trigger_on_loader(
                    client_model, dataset_loader, self.trigger, self.target_label, device, max_batches=12
                )
                print(f"[Cerp][poisoned_local_train] POST-POISON mean_conf={post_conf:.4f}, ASR={post_asr*100:.2f}%, samples_eval={post_n}")
            else:
                print("[Cerp][poisoned_local_train] POST-POISON: no trigger found (self.trigger is None)")
        except Exception as e:
            print(f"[Cerp][poisoned_local_train] POST-POISON eval failed: {e}")
        # -----------------------------------------------------------------

        return None


    # --------------------
    # build evaluation set using finalized trigger
    # --------------------
    def prepare_evalset(self, test_set: Dataset, per_class: int = 50, device=None) -> Dataset:
        """
        Build an eval dataset using the finalized self.trigger.
        Returns a dataset that yields (poisoned_input, target_label, original_label),
        matching what evaluate_asr expects.
        """
        if getattr(self, "trigger", None) is None:
            raise RuntimeError("prepare_evalset called before finalize_trigger().")

        # sample indices per-class (like build_backdoor_evalset)
        # works for torchvision datasets
        if hasattr(test_set, "targets"):
            labels = np.array(getattr(test_set, "targets"))
        else:
            labels = np.array([int(test_set[i][1]) for i in range(len(test_set))])

        atk = self.cfg.get("attack", {})
        target_label = int(atk.get("target_label", self.target_label))
        src = atk.get("source_labels", "all")
        if src == "all":
            classes = sorted(set(labels.tolist()))
            classes = [c for c in classes if c != target_label]
        else:
            classes = [int(c) for c in src if int(c) != target_label]

        pick_indices = []
        per = int(per_class)
        rng = np.random.default_rng(int(atk.get("seed", 0)))
        for c in classes:
            idxs = np.where(labels == c)[0]
            if idxs.size == 0:
                continue
            if idxs.size > per:
                idxs = rng.choice(idxs, per, replace=False)
            pick_indices.extend(idxs.tolist())

        poisoned = []
        orig_labels = []
        for idx in pick_indices:
            x, y = test_set[idx]          # x: Tensor[C,H,W], y: int
            x = x.unsqueeze(0)            # 1,C,H,W
            xp = self.apply_trigger_batch(x.to(self.device))[0].cpu()
            poisoned.append(xp)
            orig_labels.append(int(y))

        X = torch.stack(poisoned)                                      # N,C,H,W
        Yorig = torch.tensor(orig_labels, dtype=torch.long)            # N
        Ytarget = torch.full((len(X),), int(self.target_label),
                             dtype=torch.long)                         # N

        class _EvalDS(torch.utils.data.Dataset):
            def __len__(self): return X.size(0)
            def __getitem__(self, i): return X[i], Ytarget[i], Yorig[i]

        return _EvalDS()


# ---------- debug helpers ----------
def _trigger_stats(trigger: torch.Tensor):
    t = trigger.detach().cpu()
    return {
        "shape": tuple(t.shape),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "l2": float(torch.norm(t.view(-1), p=2).item()),
    }

def eval_trigger_on_loader(model, loader, trigger_tensor, target_label, device, max_batches=10):
    """
    Apply trigger_tensor (C,H,W) to first max_batches of loader and compute:
      - mean softmax confidence for target_label
      - ASR (fraction of samples classified as target_label)
    Returns (mean_conf, asr, n_samples_used)
    """
    model = model.to(device)
    model.eval()
    total = 0
    target_count = 0
    mean_conf_sum = 0.0

    trig = trigger_tensor.to(device)
    with torch.no_grad():
        it = iter(loader)
        for b in range(max_batches):
            try:
                xb, yb = next(it)
            except StopIteration:
                break
            xb = xb.to(device)
            # apply trigger to whole batch for evaluation
            xb_p = xb + trig.unsqueeze(0)
            xb_p = torch.clamp(xb_p, 0.0, 1.0)
            logits = model(xb_p)
            probs = torch.softmax(logits, dim=1)
            conf = probs[:, target_label]
            mean_conf_sum += float(conf.mean().item()) * xb.size(0)
            preds = probs.argmax(dim=1)
            target_count += int((preds == target_label).sum().item())
            total += xb.size(0)
    if total == 0:
        return 0.0, 0.0, 0
    return mean_conf_sum / total, target_count / total, total

def trigger_grad_norm(trigger_param):
    if trigger_param.grad is None:
        return 0.0
    return float(torch.norm(trigger_param.grad.view(-1), p=2).item())
# ---------- end debug helpers ----------
