# attacks/dba.py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Tuple, Sequence, Union
from attacks.base_attack import BaseAttack  # your shared base class

class DBAAttack(BaseAttack):
    """
    Helper for Distributed Backdoor Attack (DBA).
    Owns:
      - global mask (bar) construction
      - decomposition into per-client local masks
      - poisoning batches for a given client
      - building eval set with the GLOBAL mask
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        atk = cfg.get("attack", {})
        self.dba_cfg = cfg.get("dba_atk", {})
        self.task_cfg = cfg.get("task", {})
        self.target_label = int(atk.get("target_label", 7))
        self.value = float(atk.get("value", 1.0))

        # global info
        self.img_size: Tuple[int, int] = None   # (H, W)
        self.global_mask: torch.Tensor = None   # H x W bool
        # per-client masks: cid -> H x W bool
        self.client_masks: Dict[int, torch.Tensor] = {}

    # -------- global mask construction --------
    def prepare_global_mask(self, test_dataset: Dataset):
        """
        Ensure self.global_mask is built from cfg.dba_atk (TS, TL, TG) and img_size.
        Replaces server._prepare_dba_attack_cfg_for_eval for DBA.
        """
        # 1) image size: prefer task.img_size, else infer from test_dataset
        img_sz = self.dba_cfg.get("img_size", None) or self.task_cfg.get("img_size", None)
        if img_sz is None:
            x0, _ = test_dataset[0]
            H, W = int(x0.shape[-2]), int(x0.shape[-1])
            img_sz = (H, W)
        self.img_size = (int(img_sz[0]), int(img_sz[1]))
        H, W = self.img_size

        # 2) trigger parameters
        TS = self.dba_cfg.get("TS", None)
        if TS is None:
            raise ValueError("DBA: dba_atk.TS (trigger size) must be set.")
        TS = int(TS)

        TL = self.dba_cfg.get("TL", None)
        if TL is None:
            raise ValueError("DBA: dba_atk.TL (trigger origin [x,y]) must be set.")
        TL = (int(TL[0]), int(TL[1]))

        TG = self.dba_cfg.get("TG", (0, 0))
        TG = (int(TG[0]), int(TG[1]))

        # 3) build global mask: use your utility make_local_trigger_from_params
        gm = make_local_trigger_from_params(self.img_size, TS, TG, TL)
        if isinstance(gm, torch.Tensor):
            gm = gm.cpu().numpy()
        gm = np.asarray(gm, dtype=bool)
        self.global_mask = torch.from_numpy(gm)  # H x W bool

        # persist back to cfg if you still want attack_cfg['dba_atk']['global_mask']
        self.dba_cfg["img_size"] = self.img_size
        self.dba_cfg["TS"] = TS
        self.dba_cfg["TL"] = TL
        self.dba_cfg["TG"] = TG
        self.dba_cfg["global_mask"] = gm

        return self.dba_cfg

    # -------- per-client decomposition --------
    def assign_client_masks(self, clients: Dict[int, Any], seed: int = 0):
        """
        Decompose global_mask into local pieces and assign to each malicious client.
        Stores self.client_masks[cid] = HxW bool tensor.
        """
        if self.global_mask is None:
            raise RuntimeError("DBAAttack.assign_client_masks called before prepare_global_mask().")

        # figure out malicious client ids
        mal_ids = [cid for cid, c in clients.items() if getattr(c, "is_malicious", False)]
        if not mal_ids:
            return

        n_parts = len(mal_ids)
        layout = self.dba_cfg.get("layout", "grid")

        parts = decompose_global_trigger(self.global_mask, n_parts=n_parts, layout=layout)

        # assign masks per malicious client
        for cid, piece in zip(mal_ids, parts):
            if isinstance(piece, torch.Tensor):
                m = piece.bool()
            else:
                m = torch.from_numpy(np.asarray(piece, dtype=bool))
            self.client_masks[cid] = m

            # keep also on the client (optional but convenient)
            client = clients[cid]
            client.dba_mask = m


    def poison_batch_for_client(self, cid: int, xb: torch.Tensor, yb: torch.Tensor,
                                device=None, round_id: int = None):
        """
        Term-rotation poisoning:
        - split each client's canonical mask into n_terms submasks (term masks)
        - activate exactly one term per round deterministically:
                term_idx = (round_id + cid_offset) % n_terms
        - apply poisoning using the chosen active_mask and existing poison_frac/target_label logic.

        Deterministic: given identical round_id and same cfg/clients, all processes pick the same term.
        """
        if cid not in self.client_masks:
            raise RuntimeError(f"DBAAttack: no local mask registered for cid={cid}.")

        B = xb.size(0)
        dev = xb.device if device is None else torch.device(device)

        # canonical mask: HxW boolean tensor on correct device
        canon_mask = self.client_masks[cid].to(dev)

        # read number of terms (default 4)
        n_terms = int(getattr(self, "dba_cfg", {}).get("n_terms", 4))

        # fast path: use full mask if only 1 term
        if n_terms <= 1:
            active_mask = canon_mask
        else:
            # ensure cache exists
            if not hasattr(self, "_term_masks"):
                self._term_masks = {}

            # build/cache term masks for this client
            if cid not in self._term_masks:
                # flatten canonical mask and get true indices (CPU numpy ints)
                flat = canon_mask.view(-1)
                true_idx_tensor = flat.nonzero(as_tuple=True)[0]
                true_idx = true_idx_tensor.cpu().numpy() if true_idx_tensor.numel() > 0 else np.array([], dtype=np.int64)

                term_masks = []
                if true_idx.size == 0:
                    # no true pixels -> all term masks empty
                    for _ in range(n_terms):
                        term_masks.append(torch.zeros_like(canon_mask, dtype=torch.bool, device=dev))
                else:
                    # distribute true indices round-robin into groups to spread spatially
                    groups = [[] for _ in range(n_terms)]
                    for i, tidx in enumerate(true_idx):
                        groups[i % n_terms].append(int(tidx))
                    for grp in groups:
                        flat_mask = torch.zeros(canon_mask.numel(), dtype=torch.bool, device=dev)
                        if len(grp) > 0:
                            idx_tensor = torch.tensor(grp, dtype=torch.long, device=dev)
                            flat_mask.index_fill_(0, idx_tensor, True)
                        term_masks.append(flat_mask.view_as(canon_mask))
                self._term_masks[cid] = term_masks

            term_masks = self._term_masks[cid]

            # compute deterministic term index for this round & client
            cid_offset = int(getattr(self, "dba_cfg", {}).get("cid_term_offset", cid % n_terms))
            term_idx = (int(round_id) + cid_offset) % n_terms

            active_mask = term_masks[term_idx]

        # --- now apply poisoning using active_mask exactly like before ---
        atk = self.cfg.get("attack", {}) if hasattr(self, "cfg") else {}
        r = float(atk.get("poison_frac", 1.0))
        r = max(0.0, min(1.0, r))
        # print("r in dba", r)
        if r >= 0.999:
            xb_p = apply_mask_to_batch(xb, active_mask, value=self.value)
            idx = torch.arange(B, device=xb.device)
        elif r <= 0.0:
            xb_p = xb.clone()
            idx = None
        else:
            num_poison = max(1, int(round(r * B)))
            # choose poisoned examples randomly within the batch (same behavior as before)
            idx = torch.randperm(B, device=xb.device)[:num_poison]
            xb_p = xb.clone()
            xb_p[idx] = apply_mask_to_batch(xb[idx], active_mask, value=self.value)

        # labels: only poisoned indices get target_label
        yb_t = yb.to(xb.device).clone()
        if idx is not None:
            yb_t[idx] = self.target_label

        return xb_p, yb_t


    # -------- eval set using GLOBAL mask --------
    def prepare_evalset(self, test_set: Dataset, per_class: int = 50):
        """
        Use the GLOBAL DBA mask to poison test samples (evaluation).
        Equivalent to build_backdoor_evalset with a global bar.
        """
        if self.global_mask is None:
            raise RuntimeError("DBAAttack.prepare_evalset called before prepare_global_mask().")

        import numpy as np
        labels = np.array([int(y) for _, y in test_set], dtype=int)
        atk = self.cfg.get("attack", {}) if hasattr(self, "cfg") else {}
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
            if len(idxs) == 0:
                continue
            chosen = idxs if len(idxs) <= per else rng.choice(idxs, per, replace=False)
            pick_indices.extend(chosen.tolist())

        Xp, Yorig = [], []
        gm = self.global_mask  # H x W bool
        for idx in pick_indices:
            x, y = test_set[idx]         # C,H,W
            xb = x.unsqueeze(0)          # 1,C,H,W
            xp = apply_mask_to_batch(xb, gm, value=self.value)[0].cpu()
            Xp.append(xp)
            Yorig.append(int(y))

        X = torch.stack(Xp)
        Yorig = torch.tensor(Yorig, dtype=torch.long)
        Ytarget = torch.full((len(X),), fill_value=self.target_label, dtype=torch.long)

        class _EvalDS(torch.utils.data.Dataset):
            def __len__(self): return X.size(0)
            def __getitem__(self, i): return X[i], Ytarget[i], Yorig[i]

        return _EvalDS()


def decompose_global_trigger(
    global_mask: Union[np.ndarray, "torch.Tensor"],
    n_parts: int = 1,
    layout: str = "grid",
    gap: Tuple[int,int] = (0,0)
) -> List[np.ndarray]:
    """
    Split a global boolean mask (H x W) into `n_parts` local boolean masks.
    - If layout == "grid": tile the bounding box of the global mask roughly into a grid.
    - If layout == "vertical": split columns (contiguous vertical slices).
    - If layout == "corners": place small tiles near corners (fallback).
    Returns list of n_parts numpy boolean arrays shaped (H, W).
    """
    # normalize to numpy bool array
    if hasattr(global_mask, "cpu"):
        gm = global_mask.cpu().numpy()
    else:
        gm = np.asarray(global_mask)
    gm = gm.astype(bool)
    H, W = gm.shape
    # bounding box of the global mask (to focus splitting region)
    ys, xs = np.where(gm)
    if len(ys) == 0:
        raise ValueError("decompose_global_trigger: global_mask has no True pixels")
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    gw, gh = x1 - x0, y1 - y0
    parts = []

    if layout == "grid":
        # try near-square grid
        gcols = int(np.ceil(np.sqrt(n_parts)))
        grows = int(np.ceil(n_parts / gcols))
        piece_w = max(1, (gw - (gcols - 1) * gap[0]) // gcols)
        piece_h = max(1, (gh - (grows - 1) * gap[1]) // grows)
        idx = 0
        for r in range(grows):
            for c in range(gcols):
                if idx >= n_parts:
                    break
                ox = x0 + c * (piece_w + gap[0])
                oy = y0 + r * (piece_h + gap[1])
                mask = np.zeros((H, W), dtype=bool)
                xslice0 = int(ox)
                xslice1 = int(min(W, ox + piece_w))
                yslice0 = int(oy)
                yslice1 = int(min(H, oy + piece_h))
                if xslice1 > xslice0 and yslice1 > yslice0:
                    mask[yslice0:yslice1, xslice0:xslice1] = gm[yslice0:yslice1, xslice0:xslice1]
                parts.append(mask)
                idx += 1

    elif layout == "vertical":
        # split the bounding columns into contiguous vertical slices
        cols = np.where(gm.any(axis=0))[0]
        splits = np.array_split(cols, n_parts)
        for s in splits:
            mask = np.zeros((H, W), dtype=bool)
            if len(s) > 0:
                mask[:, s] = gm[:, s]
            parts.append(mask)

    elif layout == "corners":
        # place small tiles near corners of the bounding box
        piece_w = max(1, gw // 4)
        piece_h = max(1, gh // 4)
        candidates = [
            (x0, y0),
            (x1 - piece_w, y0),
            (x0, y1 - piece_h),
            (x1 - piece_w, y1 - piece_h),
        ]
        for i in range(min(n_parts, len(candidates))):
            ox, oy = candidates[i]
            mask = np.zeros((H, W), dtype=bool)
            xs0 = int(max(0, ox))
            xs1 = int(min(W, ox + piece_w))
            ys0 = int(max(0, oy))
            ys1 = int(min(H, oy + piece_h))
            if ys1 > ys0 and xs1 > xs0:
                mask[ys0:ys1, xs0:xs1] = gm[ys0:ys1, xs0:xs1]
            parts.append(mask)
        while len(parts) < n_parts:
            parts.append(parts[len(parts) % len(parts)].copy())

    else:
        raise ValueError(f"unknown layout '{layout}'")

    # ensure length n_parts
    if len(parts) < n_parts:
        # pad with empty masks
        for _ in range(n_parts - len(parts)):
            parts.append(np.zeros((H, W), dtype=bool))

    return parts


def apply_mask_to_batch(
    batch_x: torch.Tensor,
    mask,
    value: float = 1.0,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Apply an HxW boolean mask to every example in batch_x (B x C x H x W).
    """
    if batch_x.ndim != 4:
        raise ValueError("batch_x must be BxCxHxW")

    B, C, H, W = batch_x.shape

    # convert mask to torch.bool on the right device
    if isinstance(mask, torch.Tensor):
        m = mask.to(device=batch_x.device, dtype=torch.bool)
    else:
        m = torch.from_numpy(np.asarray(mask, dtype=bool)).to(batch_x.device)

    if m.shape != (H, W):
        raise ValueError(f"mask shape {m.shape} does not match image shape {(H, W)}")

    out = batch_x if inplace else batch_x.clone()

    # broadcast mask to B x C x H x W and assign value on those pixels
    mask4 = m.view(1, 1, H, W).expand(B, C, H, W)
    out[mask4] = float(value)
    return out


# ---------- DBA decomposition over a global mask ----------

def decompose_global_trigger(
    global_mask,
    n_parts: int,
    layout: str = "grid",
    gap: Tuple[int, int] = (0, 0),
) -> List[np.ndarray]:
    """
    Decompose a global HxW boolean mask into n_parts local HxW boolean masks.

    Args:
      global_mask: HxW bool (torch.Tensor or np.ndarray)
      n_parts: number of local pieces (usually number of malicious clients)
      layout: 'grid' (default) or 'vertical'
      gap: (gap_x, gap_y) spacing between pieces inside the bounding box
    """
    # convert to numpy bool
    if isinstance(global_mask, torch.Tensor):
        gm = global_mask.detach().cpu().numpy().astype(bool)
    else:
        gm = np.asarray(global_mask, dtype=bool)

    H, W = gm.shape

    # bounding box of the trigger region
    ys, xs = np.where(gm)
    if xs.size == 0 or ys.size == 0:
        # fall back to full image
        x_min, x_max, y_min, y_max = 0, W, 0, H
    else:
        x_min, x_max = xs.min(), xs.max() + 1
        y_min, y_max = ys.min(), ys.max() + 1

    box_w = x_max - x_min
    box_h = y_max - y_min
    gap_x, gap_y = gap

    parts: List[np.ndarray] = []

    if layout == "grid":
        # choose grid dims close to square
        gcols = int(np.ceil(np.sqrt(n_parts)))
        grows = int(np.ceil(n_parts / gcols))
        piece_w = max(1, (box_w - (gcols - 1) * gap_x) // gcols)
        piece_h = max(1, (box_h - (grows - 1) * gap_y) // grows)
        idx = 0
        for r in range(grows):
            for c in range(gcols):
                if idx >= n_parts:
                    break
                x0 = x_min + c * (piece_w + gap_x)
                y0 = y_min + r * (piece_h + gap_y)
                x1 = min(x0 + piece_w, x_max)
                y1 = min(y0 + piece_h, y_max)
                m = np.zeros_like(gm, dtype=bool)
                m[y0:y1, x0:x1] = gm[y0:y1, x0:x1]
                parts.append(m)
                idx += 1

    elif layout == "vertical":
        piece_w = max(1, (box_w - (n_parts - 1) * gap_x) // n_parts)
        for i in range(n_parts):
            x0 = x_min + i * (piece_w + gap_x)
            x1 = min(x0 + piece_w, x_max)
            m = np.zeros_like(gm, dtype=bool)
            m[y_min:y_max, x0:x1] = gm[y_min:y_max, x0:x1]
            parts.append(m)
    else:
        raise ValueError(f"unknown layout: {layout}")

    return parts


def make_local_trigger_from_params(
    img_size,
    TS,
    TG,
    TL,
):
    """
    Build a square TS x TS trigger mask at origin TL = (x, y).

    Args:
      img_size: (H, W)
      TS: trigger size (edge length in pixels)
      TG: unused here (kept for API compatibility)
      TL: (x, y) top-left of the trigger

    Returns:
      H x W numpy.bool_ mask with True on the trigger region.
    """
    H, W = int(img_size[0]), int(img_size[1])
    mask = np.zeros((H, W), dtype=bool)

    tx, ty = int(TL[0]), int(TL[1])
    x0 = max(0, tx)
    y0 = max(0, ty)
    x1 = min(W, x0 + int(TS))
    y1 = min(H, y0 + int(TS))

    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True

    return mask
