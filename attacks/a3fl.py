import copy
import torch
import torch.nn.functional as F
import typing as _t
import numpy as np
from torch.utils.data import Dataset
from attacks.base_attack import BaseAttack
from backdoor.badnet import resolve_origin_from_pos, resolve_pattern_size

# -------------------------------------------------------------------
# A3FLAttack helper class
# -------------------------------------------------------------------
class A3FLAttack(BaseAttack):
    def __init__(self, cfg: _t.Optional[dict] = None):
        super().__init__(cfg or {}) 
        self.cfg = cfg or {}
        # existing hyperparams
        self.K = int(self.cfg.get("a3_K", 8))
        self.Ktrigger = int(self.cfg.get("a3_Ktrigger", 20))
        self.alpha1 = float(self.cfg.get("a3_alpha1", 1e-2))
        self.alpha2 = float(self.cfg.get("a3_alpha2", 1e-2))
        self.lam = float(self.cfg.get("a3_lambda", 1.0))
        self.trigger: _t.Optional[torch.Tensor] = None
        self.mask: _t.Optional[torch.Tensor] = None
        self.target_label = int(self.cfg.get("target_label", 7))
        self.global_model = None

    def set_global_model(self, model):
        self.global_model = model

    def _init_trigger_mask_from_sample(self, sample_img: torch.Tensor, pattern_pos=None, pattern_size=(3,3)):
        """
        If user already provided trigger/mask in cfg, prefer that.
        Otherwise init trigger as zero tensor + mask either pattern-shaped or full.
        sample_img: torch.Tensor (C,H,W) or (H,W)
        """
        if sample_img.ndim == 2:
            C = 1
            H, W = sample_img.shape
        else:
            C, H, W = sample_img.shape
        # create default mask: use pattern_size and pattern_pos to compute origin then offsets -> mask region = 1
        pw, ph = int(pattern_size[0]), int(pattern_size[1])
        mask = torch.zeros((C, H, W), dtype=torch.float32)
        # default: full mask (if pattern_size equals image), else top-left placed pattern
        # Use resolve_origin_from_pos to compute origin (reuse function in this file)
        origin_x, origin_y = resolve_origin_from_pos(sample_img, pattern_size=(pw, ph), pattern_pos=pattern_pos, padding=0)
        # mark the pattern square
        for dx in range(pw):
            for dy in range(ph):
                xi = origin_x + dx
                yi = origin_y + dy
                if 0 <= xi < W and 0 <= yi < H:
                    mask[:, yi, xi] = 1.0
        # trigger initial values small random near 0
        trigger = torch.zeros((C, H, W), dtype=torch.float32)
        return trigger, mask


    def search_trigger(self, dataloader, *,
                       device=None, verbose: bool = False):
        """
        Run A3FL trigger search (Algorithm 1) using dataloader as the client's B batches.
        - global_model: a torch.nn.Module (current global θ_t). It will NOT be modified.
        - dataloader: DataLoader or iterable of (x,y). We sample batches from it for each outer iter.
        - device: optional device string; if None we use model device.
        On completion sets self.trigger and self.mask (both CPU tensors).
        """
        if self.global_model is None:
            raise RuntimeError("A3FLAttack.global_model is None; call set_global_model() on the server first.")
        if device is None:
            device = next(self.global_model.parameters()).device
        global_model = self.global_model.to(device)

        # ensure dataloader yields tensors on CPU; we'll move them to device below
        ce = torch.nn.CrossEntropyLoss()

        # create theta_prime as a copy of global_model
        theta_prime = copy.deepcopy(global_model).to(device)
        theta_prime.train()

        # init trigger & mask from one sample in dataloader
        # get a sample from dataloader to determine shape
        sample_x, _ = None, None
        try:
            it = iter(dataloader)
            sample_x, _ = next(it)
        except Exception:
            # if dataloader is list-like, use first element
            sample_x, _ = dataloader[0]

        # expect sample_x to be (B, C, H, W) or (C,H,W)
        if sample_x.ndim == 4:
            sample_img = sample_x[0]
        else:
            sample_img = sample_x

        # A3FL keeps an explicit trigger_size in config. Fall back to pattern_size
        # for backward compatibility, then default to a 3x3 trigger footprint.
        atk_cfg = self.cfg or {}
        trigger_size = atk_cfg.get("trigger_size", atk_cfg.get("pattern_size", (3, 3)))
        pattern_size = resolve_pattern_size(
            pattern_type=atk_cfg.get("pattern_type", "badnet_corner"),
            pattern_size=trigger_size,
            pattern_offsets=atk_cfg.get("pattern_offsets", None),
        )
        pattern_pos = atk_cfg.get("pattern_pos", "bottom_right")
        trigger, mask = self._init_trigger_mask_from_sample(sample_img, 
                                                            pattern_pos=pattern_pos,
                                                            pattern_size=pattern_size)
        trigger = trigger.to(device)
        mask = mask.to(device)

        # small optimizer for theta_prime
        theta_prime_opt = torch.optim.SGD(theta_prime.parameters(), 
                                          lr=self.alpha2, 
                                          momentum=float(atk_cfg.get("a3_momentum", 0.0)))

        # Helper to get next batch safely
        def next_batch(it):
            try:
                return next(it)
            except StopIteration:
                return None

        # prepare dataloader iterator (we will create a fresh iterator each outer iter)
        for j in range(int(self.K)):
            it = iter(dataloader)
            batch = next_batch(it)
            if batch is None:
                # empty dataloader edge-case
                break
            xb, yb = batch
            xb = xb.to(device)
            yb = yb.to(device)

            # inner loop: optimize trigger t for Ktrigger steps
            for k in range(int(self.Ktrigger)):
                trigger.requires_grad_(True)
                # apply trigger masked onto xb
                # shape alignment: if xb shape is (B,C,H,W) and trigger is (C,H,W),
                # broadcast trigger to batch
                b_trigger = trigger.unsqueeze(0).expand(xb.size(0), -1, -1, -1)
                b_mask = mask.unsqueeze(0).expand(xb.size(0), -1, -1, -1)
                x_bkd = b_trigger * b_mask + xb * (1.0 - b_mask)

                # target for trigger optimization is attack target label
                tgt = torch.full((xb.size(0),), 
                                 fill_value=self.target_label, 
                                 dtype=torch.long, device=device)

                # compute loss on global_model and theta_prime
                global_model.eval()
                theta_prime.eval()
                out_g = global_model(x_bkd)
                loss_g = ce(out_g, tgt)
                out_p = theta_prime(x_bkd)
                loss_p = ce(out_p, tgt)
                loss = loss_g + float(self.lam) * loss_p

                # backward on trigger
                if trigger.grad is not None:
                    trigger.grad.detach_()
                    trigger.grad.zero_()
                loss.backward()
                with torch.no_grad():
                    grad = trigger.grad
                    if grad is None:
                        break
                    # gradient descent; keep it simple: t = t - alpha1 * grad
                    trigger = (trigger - self.alpha1 * grad).detach()
                    # clamp trigger to reasonable image range (user data likely normalized or in [0,1]; we keep broad clamp)
                    trigger = torch.clamp(trigger, min=-2.0, max=2.0)
                    trigger.requires_grad_(False)

            # after inner loop, update theta_prime using clean labels yb on x⊕δ (line 9-10)
            theta_prime.train()
            theta_prime_opt.zero_grad()
            # recompute x_bkd using final trigger
            b_trigger = trigger.unsqueeze(0).expand(xb.size(0), -1, -1, -1)
            b_mask = mask.unsqueeze(0).expand(xb.size(0), -1, -1, -1)
            x_bkd_for_unlearn = b_trigger * b_mask + xb * (1.0 - b_mask)
            loss_unlearn = ce(theta_prime(x_bkd_for_unlearn), yb)  # original labels yb
            loss_unlearn.backward()
            theta_prime_opt.step()

            if verbose and (j % max(1, int(self.K//4)) == 0):
                # quick ASR estimation on this batch for monitoring
                with torch.no_grad():
                    outg = global_model(b_trigger * b_mask + xb * (1.0 - b_mask))
                    pred = outg.argmax(dim=1)
                    asr = (pred == self.target_label).float().mean().item()
                print(f"[A3FL] outer {j}/{self.K} loss_g={loss_g.item():.4f} loss_p={loss_p.item():.4f} approx_ASR={asr:.3f}")

        # store trigger & mask on CPU
        self.trigger = trigger.detach().cpu()
        self.mask = mask.detach().cpu()
        # cleanup
        del theta_prime
        torch.cuda.empty_cache()
        return self.trigger, self.mask

    def apply_trigger_batch(self, xb: torch.Tensor):
        """
        Apply discovered trigger (self.trigger, self.mask) to a minibatch tensor xb (B,C,H,W).
        Returns tensor on xb.device with trigger applied. Does not modify input.
        """
        if self.trigger is None or self.mask is None:
            raise RuntimeError("A3FLAttack: trigger not set. Call search_trigger first.")
        trig = self.trigger.to(xb.device)
        mask = self.mask.to(xb.device)
        b_trig = trig.unsqueeze(0).expand(xb.size(0), -1, -1, -1)
        b_mask = mask.unsqueeze(0).expand(xb.size(0), -1, -1, -1)
        return b_trig * b_mask + xb * (1.0 - b_mask)

    def poison_loader_batchwise(self, base_loader, poison_frac: float = None):
        """
        Yield (x_maybe_poisoned, y_maybe_target) batches.
        poison_frac: fraction of samples to poison in each batch (default 1.0).
        """
        poison_frac = max(0.0, min(poison_frac, 1.0))

        for xb, yb in base_loader:
            B = xb.size(0)
            xb_p = xb.clone()
            yb_t = yb.clone()
            if poison_frac > 0.0:
                n_poison = int(poison_frac * B)
                if n_poison > 0:
                    idx = torch.randperm(B)[:n_poison]
                    xb_p[idx] = self.apply_trigger_batch(xb[idx])
                    yb_t[idx] = self.target_label
            yield xb_p, yb_t

    # ---- BaseAttack adapter methods ----
    def client_search_trigger(self, loader, device=None, verbose=False):
        """
        Called on malicious client: run search_trigger and return trigger CPU tensor for submission.
        """
        # call existing search_trigger which populates self.trigger and self.mask (CPU tensors at end)
        trig, mask = self.search_trigger(loader, device=device, verbose=verbose)
        # ensure trig returned as CPU tensor (detached)
        return trig.detach().cpu()

    def submit_client_trigger(self, trig_cpu: torch.Tensor):
        # reuse BaseAttack.submit_client_trigger or override if desired
        super().submit_client_trigger(trig_cpu)

    def prepare_evalset(
        self,
        test_set: Dataset,
        per_class: int = 50,
        device: torch.device = None,
    ):
        if device is None:
            raise ValueError("A3FLAttack.prepare_evalset: 'device' must be provided.")
        if getattr(self, "trigger", None) is None or getattr(self, "mask", None) is None:
            raise RuntimeError("prepare_evalset called before trigger/mask are set.")

        labels = np.array([int(y) for _, y in test_set], dtype=int)
        target_label = int(self.target_label)
        src = self.cfg.get("source_labels", "all")
        if src == "all":
            classes = sorted(set(labels.tolist()))
            classes = [c for c in classes if c != target_label]
        else:
            classes = [int(c) for c in src if int(c) != target_label]

        pick_indices = []
        per = int(per_class)
        rng = np.random.default_rng(int(self.cfg.get("seed", 0)))
        for c in classes:
            idxs = np.where(labels == c)[0]
            if len(idxs) == 0:
                continue
            if len(idxs) <= per:
                chosen = idxs.tolist()
            else:
                chosen = rng.choice(idxs, per, replace=False).tolist()
            pick_indices.extend(chosen)

        poisoned, orig_labels = [], []
        for idx in pick_indices:
            x, y = test_set[idx]
            if x.ndim == 3:
                x = x.unsqueeze(0)

            x = x.to(device)
            xp = self.apply_trigger_batch(x)[0].cpu()

            poisoned.append(xp)
            orig_labels.append(int(y))

        X = torch.stack(poisoned)  # CPU
        Yorig = torch.tensor(orig_labels, dtype=torch.long)
        Ytarget = torch.full((len(X),), fill_value=int(self.target_label), dtype=torch.long)

        class _EvalDS(torch.utils.data.Dataset):
            def __len__(self):
                return X.size(0)
            def __getitem__(self, i):
                return X[i], Ytarget[i], Yorig[i]

        return _EvalDS()

    
    def poisoned_local_train(self, client_model, dataset_loader, optimizer, device=None, epochs=None, verbose=False):
        """
        Convenience: perform poisoned local training using discovered trigger.
        Use poison_loader_batchwise() generator to iterate poisoned batches.
        """
        device = device or self.device if hasattr(self, "device") else next(client_model.parameters()).device
        client_model.to(device)
        client_model.train()
        if self.trigger is None or self.mask is None:
            raise RuntimeError("A3FLAttack.poisoned_local_train called before trigger search.")
        total, count = 0.0, 0
        for xb_p, yb_t in self.poison_loader_batchwise(dataset_loader):
            xb_p = xb_p.to(device); yb_t = yb_t.to(device)
            optimizer.zero_grad()
            out = client_model(xb_p)
            loss = torch.nn.functional.cross_entropy(out, yb_t)
            loss.backward()
            optimizer.step()
            bs = xb_p.size(0)
            total += float(loss.item()) * bs
            count += bs
        return total / max(1, count)
