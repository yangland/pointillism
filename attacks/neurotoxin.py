import torch
import numpy as np
import typing as _t


class NeurotoxinAttack:
    """
    In-memory Neurotoxin helper.

    - Flatten server state-dict diffs deterministically (state_dict key order).
    - Keep the flattened server update in self.latest_flat (no disk I/O).
    - Apply projection: zero local grads at coordinates that are top-k in server update.

    Usage (single-process):
        nt = NeurotoxinAttack(cfg_attack)
        flat = nt.save_server_flat(prev_state_dict, new_state_dict, round_id=it)  # stores in nt.latest_flat
        # on malicious client, after loss.backward():
        nt.apply_projection_from_latest(model)   # zeros top-k grads in-place
    """

    def __init__(self, cfg: _t.Optional[dict] = None):
        self.cfg = cfg or {}
        self.k_frac = float(self.cfg.get("neuro_k", 0.01))
        self.latest_flat: _t.Optional[np.ndarray] = None

    # ---------- Flattening helpers ----------
    def flatten_state_dict(self, state_dict: dict) -> np.ndarray:
        """
        Flatten in deterministic order: iterate keys() of state_dict.
        Returns float32 1D numpy array.
        """
        flat_list = []
        for k in state_dict.keys():
            v = state_dict[k]
            if isinstance(v, torch.Tensor):
                arr = v.detach().cpu().numpy().reshape(-1)
            else:
                arr = np.asarray(v).reshape(-1)
            flat_list.append(arr)
        if len(flat_list) == 0:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(flat_list).astype(np.float32)

    # ---------- Server-side: store flattened global update in memory ----------
    def save_server_flat(self, prev_state: dict, new_state: dict, round_id: _t.Optional[int] = None) -> np.ndarray:
        """
        Compute (new_state - prev_state) flattened and store it in self.latest_flat.
        Returns the flattened numpy array.
        prev_state/new_state should be model.state_dict()-like mappings.
        """
        flat_list = []
        for k in new_state.keys():  # deterministic order
            new_v = new_state[k].cpu().numpy().reshape(-1)
            prev_v = prev_state.get(k)
            prev_arr = prev_v.cpu().numpy().reshape(-1) if prev_v is not None else np.zeros_like(new_v)
            flat_list.append(new_v - prev_arr)
        flat = np.concatenate(flat_list).astype(np.float32) if flat_list else np.zeros(0, dtype=np.float32)
        self.latest_flat = flat
        return flat

    # ---------- Client-side projection ----------
    def apply_projection_to_model_grads(self, 
                                        model: torch.nn.Module, 
                                        server_flat: _t.Optional[np.ndarray] = None,
                                        k_frac: _t.Optional[float] = None) -> bool:
        """
        Zero-out local grads at coordinates that are top-k in server_flat (by abs).
        - server_flat: 1D numpy array (flattened server update). If None, return False.
        - k_frac: override fraction for top-k (default from cfg).
        Returns True if a projection was applied, False otherwise.
        """
        if server_flat is None:
            return False
        if k_frac is None:
            k_frac = self.k_frac
        g = np.asarray(server_flat).reshape(-1)
        D = g.size
        if D == 0:
            return False
        k = max(1, int(np.floor(D * float(k_frac))))
        if k >= D:
            return False

        # compute mask for top-k absolute coords
        topk_idx = np.argpartition(np.abs(g), -k)[-k:]
        mask = np.zeros(D, dtype=bool)
        mask[topk_idx] = True

        offset = 0
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is None:
                    offset += p.numel()
                    continue
                numel = p.grad.numel()
                slice_mask = mask[offset: offset + numel]
                if slice_mask.any():
                    try:
                        mask_t = torch.from_numpy(slice_mask).to(p.grad.device)
                        flat = p.grad.view(-1)
                        flat[mask_t] = 0.0
                    except Exception:
                        # fallback slower path
                        flat = p.grad.view(-1)
                        for i, m in enumerate(slice_mask):
                            if m:
                                flat[i] = 0.0
                offset += numel
        return True

    def apply_projection_from_latest(self, model: torch.nn.Module, k_frac: _t.Optional[float] = None) -> bool:
        """
        Convenience: apply projection using the in-memory self.latest_flat.
        """
        return self.apply_projection_to_model_grads(model, server_flat=self.latest_flat, k_frac=k_frac)
