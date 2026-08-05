import itertools
import math
import random
from typing import Callable, Dict, List, Optional, Set, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class SearchStats:
    tried: int = 0
    unique: int = 0
    best: float = 0.0


@dataclass
class SoftmaxPredCountSummary:
    num_classes: int
    num_scored_w: int
    num_scored_i: int
    pred_hist_w: List[int]
    pred_hist_i: List[int]
    pred_by_mask_w: Dict[int, int]
    pred_by_mask_i: Dict[int, int]
    pred_records_w: Dict[int, Dict[str, Any]]
    pred_records_i: Dict[int, Dict[str, Any]]

def coords_to_grid(coords: List[List[int]], grid_hw: int, device: torch.device) -> torch.Tensor:
    """
    coords: [[r,c], ...]
    returns: (1,1,grid_hw,grid_hw) float tensor in {0,1}
    """
    g = torch.zeros((1, 1, grid_hw, grid_hw), device=device)
    for rc in coords:
        r, c = int(rc[0]), int(rc[1])
        g[0, 0, r, c] = 1.0
    return g


def flatidx_to_coords(flat_idx: List[int], grid_hw: int) -> List[List[int]]:
    out = []
    for idx in flat_idx:
        r = int(idx) // int(grid_hw)
        c = int(idx) % int(grid_hw)
        out.append([r, c])
    return out


def coords_to_flatidx(coords: List[List[int]], grid_hw: int) -> List[int]:
    out = []
    for rc in coords:
        r, c = int(rc[0]), int(rc[1])
        out.append(int(r) * int(grid_hw) + int(c))
    out.sort()
    return out


def flatidx_to_bitset(flatidx: List[int]) -> int:
    b = 0
    for i in flatidx:
        b |= (1 << int(i))
    return int(b)


def build_grids_from_flatidx(flatidx_list: List[List[int]], grid_hw: int, device: torch.device) -> torch.Tensor:
    b = len(flatidx_list)
    g = torch.zeros((b, 1, grid_hw, grid_hw), device=device)
    for i, idxs in enumerate(flatidx_list):
        flat = g[i, 0].view(-1)
        flat[idxs] = 1.0
    return g


def _polarity_fill_values(polarity: str) -> Tuple[float, float]:
    polarity_l = str(polarity or 'white').lower()
    if polarity_l == 'white':
        return 0.0, 1.0
    if polarity_l == 'black':
        return 1.0, 0.0
    raise ValueError(f'Unsupported polarity: {polarity}')


def get_record_fill_values(rec: Dict[str, Any]) -> Tuple[float, float]:
    polarity = rec.get('polarity', None)
    if polarity is not None:
        return _polarity_fill_values(str(polarity))
    background_fill = float(rec.get('background_fill', 0.0))
    active_fill = float(rec.get('active_fill', 1.0))
    return background_fill, active_fill


def build_grids_from_flatidx_with_fill(
    flatidx_list: List[List[int]],
    grid_hw: int,
    device: torch.device,
    *,
    background_fill: float,
    active_fill: float,
) -> torch.Tensor:
    b = len(flatidx_list)
    g = torch.full((b, 1, grid_hw, grid_hw), float(background_fill), device=device)
    for i, idxs in enumerate(flatidx_list):
        flat = g[i, 0].view(-1)
        flat[idxs] = float(active_fill)
    return g


def build_grids_from_records(records: List[Dict[str, Any]], grid_hw: int, device: torch.device) -> torch.Tensor:
    b = len(records)
    grid_h = int(records[0].get("grid_h", grid_hw)) if records else int(grid_hw)
    grid_w = int(records[0].get("grid_w", grid_hw)) if records else int(grid_hw)
    g = torch.zeros((b, 1, grid_h, grid_w), device=device)
    for i, rec in enumerate(records):
        background_fill, active_fill = get_record_fill_values(rec)
        g[i, 0].fill_(float(background_fill))
        flat = g[i, 0].view(-1)
        flat[[int(x) for x in rec.get('flat_idx', [])]] = float(active_fill)
    return g


def render_records_for_scoring(records: List[Dict[str, Any]], out_hw: int, device: torch.device) -> torch.Tensor:
    if not records:
        return torch.zeros((0, 1, int(out_hw), int(out_hw)), dtype=torch.float32, device=device)

    probe_mode = str(records[0].get("probe_mode", "binary")).lower()
    if probe_mode.startswith("two_group"):
        return render_two_group_probe_records(records, out_hw=int(out_hw), device=device)
    if probe_mode.startswith("gaussian"):
        return render_gaussian_probe_records(records, out_hw=int(out_hw), device=device)

    grid_hw = int(records[0].get("grid_hw", 1))
    return build_grids_from_records(records=records, grid_hw=int(grid_hw), device=device)


def sample_two_group_color_centers(
    *,
    seed: int,
    d_min: float,
    d_max: float,
    num_channels: int = 3,
    center_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[List[float], List[float]]:
    # Two-group probes now use the full normalized space for color centers.
    # We sample a target distance uniformly, then construct a valid pair with
    # exactly that separation. This avoids rejection loops entirely.
    _ = center_range
    if float(d_min) < 0.0:
        raise ValueError(f"Invalid distance lower bound: d_min={d_min}")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    dims = max(1, int(num_channels))
    max_dist = float(math.sqrt(float(dims)))
    upper = min(float(d_max), max_dist)
    if float(d_min) > upper:
        raise ValueError(
            f"d_min={d_min} exceeds the allowed upper bound {upper:.6f} "
            f"for {dims} channels"
        )

    target_dist = float(d_min)
    if upper > float(d_min):
        target_dist = float(d_min) + (upper - float(d_min)) * float(torch.rand((), generator=gen).item())

    # Build a nonnegative displacement vector with exact L2 norm target_dist.
    # Each squared component is chosen within the feasible interval that still
    # leaves enough remaining mass for the later dimensions.
    remaining_sq = float(target_dist * target_dist)
    delta_sq: List[float] = []
    for axis in range(int(dims) - 1):
        remaining_axes = int(dims) - int(axis) - 1
        low_sq = max(0.0, remaining_sq - float(remaining_axes))
        high_sq = min(1.0, remaining_sq)
        if high_sq < low_sq:
            high_sq = low_sq
        if high_sq == low_sq:
            val_sq = float(low_sq)
        else:
            val_sq = float(low_sq) + (float(high_sq) - float(low_sq)) * float(torch.rand((), generator=gen).item())
        delta_sq.append(float(val_sq))
        remaining_sq = max(0.0, float(remaining_sq) - float(val_sq))

    delta_sq.append(float(max(0.0, remaining_sq)))
    delta = torch.sqrt(torch.tensor(delta_sq, dtype=torch.float32))

    # Place the pair symmetrically within the unit cube.
    base = torch.rand((dims,), generator=gen, dtype=torch.float32) * (1.0 - delta)
    other = base + delta
    if bool(torch.rand((), generator=gen).item() < 0.5):
        return base.tolist(), other.tolist()
    return other.tolist(), base.tolist()


def sample_two_group_color_centers_batch_cpu(
    *,
    seeds: List[int],
    d_min: float,
    d_max: float,
    num_channels: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized deterministic CPU sampler for two-group color centers.

    The construction and distributions match ``sample_two_group_color_centers``:
    the target distance is uniform, displacement components are sampled from
    their feasible intervals, and the foreground/background orientation is
    randomized. Random values are derived independently from each seed with
    SplitMix64 so batching does not make a probe depend on its position.

    This backend is deterministic but intentionally does not reproduce the
    exact random stream emitted by PyTorch's per-seed CPU generators.
    """
    dims = max(1, int(num_channels))
    if float(d_min) < 0.0:
        raise ValueError(f"Invalid distance lower bound: d_min={d_min}")
    max_dist = float(math.sqrt(float(dims)))
    upper = min(float(d_max), max_dist)
    if float(d_min) > upper:
        raise ValueError(
            f"d_min={d_min} exceeds the allowed upper bound {upper:.6f} "
            f"for {dims} channels"
        )

    num_rows = int(len(seeds))
    if num_rows == 0:
        empty = np.zeros((0, dims), dtype=np.float32)
        return empty, empty.copy()

    # One draw for distance, dims-1 for the displacement, dims for the base
    # color, and one for foreground/background orientation.
    num_draws = int(2 * dims + 1)
    seed_arr = np.asarray(seeds, dtype=np.uint64).reshape(-1, 1)
    counters = np.arange(1, num_draws + 1, dtype=np.uint64).reshape(1, -1)
    golden = np.uint64(0x9E3779B97F4A7C15)
    mix1 = np.uint64(0xBF58476D1CE4E5B9)
    mix2 = np.uint64(0x94D049BB133111EB)
    with np.errstate(over="ignore"):
        mixed = seed_arr + golden * counters
        mixed = (mixed ^ (mixed >> np.uint64(30))) * mix1
        mixed = (mixed ^ (mixed >> np.uint64(27))) * mix2
        mixed = mixed ^ (mixed >> np.uint64(31))
    uniforms = (mixed >> np.uint64(11)).astype(np.float64)
    uniforms *= 1.0 / 9007199254740992.0

    target_dist = np.full(num_rows, float(d_min), dtype=np.float64)
    if upper > float(d_min):
        target_dist += (upper - float(d_min)) * uniforms[:, 0]

    remaining_sq = target_dist * target_dist
    delta_sq: List[np.ndarray] = []
    for axis in range(dims - 1):
        remaining_axes = dims - axis - 1
        low_sq = np.maximum(0.0, remaining_sq - float(remaining_axes))
        high_sq = np.minimum(1.0, remaining_sq)
        value_sq = low_sq + (high_sq - low_sq) * uniforms[:, 1 + axis]
        delta_sq.append(value_sq)
        remaining_sq = np.maximum(0.0, remaining_sq - value_sq)
    delta_sq.append(np.maximum(0.0, remaining_sq))
    delta = np.sqrt(np.stack(delta_sq, axis=1)).astype(np.float32)

    base_start = dims
    base = uniforms[:, base_start : base_start + dims].astype(np.float32)
    base *= 1.0 - delta
    other = base + delta
    swap = uniforms[:, 2 * dims] < 0.5
    mu_fg = np.where(swap[:, None], base, other).astype(np.float32, copy=False)
    mu_bg = np.where(swap[:, None], other, base).astype(np.float32, copy=False)
    return mu_fg, mu_bg


def sample_gaussian_color_centers(
    *,
    seed: int,
    num_channels: int = 3,
    mean: float = 0.5,
    sigma: float = 0.35,
) -> Tuple[List[float], List[float]]:
    """
    Sample two color centers from a broad Gaussian prior.

    This keeps the Gaussian probe mode simple: both centers come from the
    same distribution, so the mode is not tied to the narrower two-group
    foreground/background semantics.
    """
    dims = max(1, int(num_channels))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    mean_f = float(mean)
    sigma_f = max(0.0, float(sigma))
    mu_f = torch.clamp(mean_f + sigma_f * torch.randn((dims,), generator=gen, dtype=torch.float32), 0.0, 1.0)
    mu_b = torch.clamp(mean_f + sigma_f * torch.randn((dims,), generator=gen, dtype=torch.float32), 0.0, 1.0)
    return mu_f.tolist(), mu_b.tolist()


def render_gaussian_probe_records(
    records: List[Dict[str, Any]],
    *,
    out_hw: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Render pure Gaussian noise grids with no foreground/background mask.

    Each record defines the grid resolution plus the sampling seed and sigma.
    """
    b = len(records)
    if b == 0:
        return torch.zeros((0, 0, int(out_hw), int(out_hw)), dtype=torch.float32, device=device)

    grouped: Dict[int, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx, rec in enumerate(records):
        grouped[int(rec.get("grid_hw", 1))].append((int(idx), rec))

    out_channels = max(1, int(records[0].get("num_channels", 3)))
    out = torch.zeros((b, out_channels, int(out_hw), int(out_hw)), dtype=torch.float32, device=device)

    for grid_hw, group in grouped.items():
        order = [int(pos) for pos, _ in group]
        recs = [rec for _, rec in group]
        imgs = torch.zeros((len(recs), out_channels, int(grid_hw), int(grid_hw)), dtype=torch.float32, device=device)
        for local_idx, rec in enumerate(recs):
            mean = float(rec.get("mean", 0.5))
            sigma = max(0.0, float(rec.get("sigma", 0.35)))
            num_channels = max(1, int(rec.get("num_channels", out_channels)))
            gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
            gen.manual_seed(int(rec.get("render_seed", 0)))
            noise = torch.randn((num_channels, int(grid_hw), int(grid_hw)), generator=gen, dtype=torch.float32, device=device)
            imgs[local_idx] = torch.clamp(mean + sigma * noise, 0.0, 1.0)
        imgs = F.interpolate(imgs, size=(int(out_hw), int(out_hw)), mode="nearest")
        out[torch.as_tensor(order, dtype=torch.long, device=device)] = imgs

    return out


def render_two_group_probe_records(
    records: List[Dict[str, Any]],
    *,
    out_hw: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Render records with probe_mode='two_group*' into raw images in [lo, hi].

    Required fields per record:
      - grid_hw, flat_idx (and optional polarity/background_fill/active_fill)
      - sigma_fg, sigma_bg
      - mu_fg, mu_bg
      - render_seed
    Optional fields:
      - cell_size (legacy; ignored and retained only for backward compatibility)
      - value_range: [lo, hi] used to clamp scalar probe values
      - num_channels
    """
    b = len(records)
    if b == 0:
        return torch.zeros((0, 0, int(out_hw), int(out_hw)), dtype=torch.float32, device=device)

    gen_device = device if device.type == "cuda" else torch.device("cpu")
    grouped: Dict[Tuple[int, int], List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx, rec in enumerate(records):
        grid_h = int(rec.get("grid_h", rec.get("grid_hw", 1)))
        grid_w = int(rec.get("grid_w", rec.get("grid_hw", 1)))
        grouped[(grid_h, grid_w)].append((int(idx), rec))

    out_channels = max(1, int(records[0].get("num_channels", len(records[0].get("mu_fg", [0.0, 0.0, 0.0])))))
    out = torch.zeros((b, out_channels, int(out_hw), int(out_hw)), dtype=torch.float32, device=device)
    gen = torch.Generator(device=gen_device)

    for (grid_h, grid_w), group in grouped.items():
        order = [int(pos) for pos, _ in group]
        recs = [rec for _, rec in group]
        masks = build_grids_from_records(recs, grid_hw=int(grid_h), device=device)
        masks = (masks > 0.5).to(dtype=torch.float32)

        group_channels = max(1, int(recs[0].get("num_channels", len(recs[0].get("mu_fg", [0.0, 0.0, 0.0])))))
        noise_shape = (len(recs), group_channels, int(grid_h), int(grid_w))
        fg_eps = torch.empty(noise_shape, dtype=torch.float32, device=device)
        bg_eps = torch.empty(noise_shape, dtype=torch.float32, device=device)
        for local_idx, rec in enumerate(recs):
            num_channels = max(1, int(rec.get("num_channels", group_channels)))
            gen.manual_seed(int(rec.get("render_seed", 0)))
            # Keep the two per-record seeded draws separate: combining them
            # changes CUDA's seed-to-value mapping. Writing directly into the
            # batched buffers preserves the old random values exactly.
            torch.randn(
                (num_channels, int(grid_h), int(grid_w)),
                generator=gen,
                dtype=torch.float32,
                device=device,
                out=fg_eps[local_idx],
            )
            torch.randn(
                (num_channels, int(grid_h), int(grid_w)),
                generator=gen,
                dtype=torch.float32,
                device=device,
                out=bg_eps[local_idx],
            )

        mu_fg = torch.tensor(
            [rec.get("mu_fg", [1.0] * group_channels) for rec in recs],
            dtype=torch.float32,
            device=device,
        ).view(len(recs), group_channels, 1, 1)
        mu_bg = torch.tensor(
            [rec.get("mu_bg", [0.0] * group_channels) for rec in recs],
            dtype=torch.float32,
            device=device,
        ).view(len(recs), group_channels, 1, 1)
        sigma_fg = torch.tensor(
            [float(rec.get("sigma_fg", 0.1)) for rec in recs],
            dtype=torch.float32,
            device=device,
        ).view(len(recs), 1, 1, 1)
        sigma_bg = torch.tensor(
            [float(rec.get("sigma_bg", 0.1)) for rec in recs],
            dtype=torch.float32,
            device=device,
        ).view(len(recs), 1, 1, 1)

        # All deterministic tensor work is batched. Nearest-neighbor
        # interpolation commutes with the cell-wise foreground/background
        # selection, so blending before upsampling produces the same image.
        ranges = [rec.get("value_range", [0.0, 1.0]) for rec in recs]
        lo = torch.tensor([float(x[0]) for x in ranges], dtype=torch.float32, device=device).view(-1, 1, 1, 1)
        hi = torch.tensor([float(x[1]) for x in ranges], dtype=torch.float32, device=device).view(-1, 1, 1, 1)
        fg_cells = torch.maximum(lo, torch.minimum(hi, mu_fg + sigma_fg * fg_eps))
        bg_cells = torch.maximum(lo, torch.minimum(hi, mu_bg + sigma_bg * bg_eps))
        cell_imgs = masks * fg_cells + (1.0 - masks) * bg_cells
        imgs = F.interpolate(
            cell_imgs,
            size=(int(out_hw), int(out_hw)),
            mode="nearest",
        )

        out[torch.as_tensor(order, dtype=torch.long, device=device)] = imgs

    return out


def grid_to_flatidx(grid: torch.Tensor) -> List[int]:
    """
    grid: (1,1,H,W) or (H,W) with values {0,1}
    """
    if grid.dim() == 4:
        x = grid[0, 0]
    elif grid.dim() == 2:
        x = grid
    else:
        raise ValueError("grid_to_flatidx expects (1,1,H,W) or (H,W).")

    idx = torch.nonzero(x > 0.5, as_tuple=False)
    if idx.numel() == 0:
        return []
    h, w = int(x.shape[0]), int(x.shape[1])
    out = []
    for i in range(int(idx.shape[0])):
        r = int(idx[i, 0].item())
        c = int(idx[i, 1].item())
        out.append(r * w + c)
    out.sort()
    return out


@torch.no_grad()
def random_topk_for_w(
    model,
    target_label: int,
    grid_hw: int,
    w: int,
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn: Callable[[str], Tuple[List[float], List[float]]],
    top_k: int,
    batch_size: int,
    max_trials: int,
    seed: int,
    log_every_batches: int = 50,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Tuple[float, List[int]]], SearchStats]:
    """
    Capped random search for a fixed pixel budget w.

    Returns:
      List of (score, flatidx_list) sorted by score descending.
      flatidx_list is a sorted list of unique cell indices in [0, grid_hw*grid_hw).
    """
    n = int(grid_hw) * int(grid_hw)
    w_i = int(w)
    if w_i <= 0 or w_i > n:
        return []

    top_k_i = max(1, int(top_k))
    batch_size_i = max(1, int(batch_size))
    max_trials_i = max(0, int(max_trials))

    # We keep top_k records in a simple list; top_k is small so sorting is fine.
    best: List[Tuple[float, List[int]]] = []

    # Dedup to avoid wasting trials; stops increasing when comb(n,w) is exhausted.
    seen: set = set()

    # CPU generator for reproducibility (independent of CUDA nondeterminism)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    tried = 0
    batch_idx = 0

    # Guard to prevent infinite loop if dedup saturates for small comb(n,w)
    max_resample_factor = 20

    while tried < max_trials_i:
        remaining = int(max_trials_i) - int(tried)
        want = min(int(batch_size_i), int(remaining))

        flatidx_list: List[List[int]] = []
        resample_budget = max_resample_factor * want

        # sample up to `want` unique masks
        while len(flatidx_list) < want and resample_budget > 0:
            # sample w distinct indices uniformly
            idx = torch.randperm(n, generator=gen)[:w_i].tolist()
            idx.sort()
            key = tuple(idx)
            if key in seen:
                resample_budget -= 1
                continue
            seen.add(key)
            flatidx_list.append(idx)
            resample_budget -= 1

        if len(flatidx_list) == 0:
            # no new unique samples found
            break

        # build and score in one forward pass
        grids = build_grids_from_flatidx(
            flatidx_list=flatidx_list,
            grid_hw=int(grid_hw),
            device=device,
        )

        scores_t = score_grids_softmax_pt(
            model=model,
            grids=grids,
            out_hw=int(out_hw),
            target_label=int(target_label),
            dataset=str(dataset),
            device=device,
            norm_cfg_fn=norm_cfg_fn,
        )
        scores = [float(x) for x in scores_t.detach().cpu().tolist()]

        # update top-k
        for s, idx in zip(scores, flatidx_list):
            best.append((float(s), idx))

        best.sort(key=lambda t: float(t[0]), reverse=True)
        if len(best) > top_k_i:
            best = best[:top_k_i]

        tried += len(flatidx_list)
        batch_idx += 1

        if log_fn is not None and (batch_idx % int(log_every_batches) == 0):
            best_s = float(best[0][0]) if len(best) > 0 else 0.0
            log_fn(
                f"[brute_rand] label={int(target_label)} w={int(w_i)} "
                f"tried={int(tried)}/{int(max_trials_i)} uniq={len(seen)} best={best_s:.3f}"
            )

    stats = SearchStats(
        tried=int(tried),
        unique=int(len(seen)),
        best=float(best[0][0]) if len(best) > 0 else 0.0,
    )
    return best, stats


@torch.no_grad()
def random_topk_for_w_bidir(
    model,
    target_label: int,
    grid_hw: int,
    w: int,
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn: Callable[[str], Tuple[List[float], List[float]]],
    top_k: int,
    batch_size: int,
    max_trials: int,
    seed: int,
    log_every_batches: int = 50,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[
    List[Tuple[float, List[int]]], SearchStats,
    List[Tuple[float, List[int]]], SearchStats
]:
    """
    Random search for fixed w, but scores BOTH mask and its complement (N-w).
    One batch forward pass scores 2B grids by concatenation.
    Returns:
      best_w, stats_w, best_inv, stats_inv
    """
    n = int(grid_hw) * int(grid_hw)
    w_i = int(w)
    if w_i <= 0 or w_i >= n:
        # note: your clamp usually prevents w=0 or w=n
        empty_stats = SearchStats(tried=0, unique=0, best=0.0)
        return [], empty_stats, [], empty_stats

    w_inv = int(n - w_i)
    score_complement = bool(w_inv != w_i)

    top_k_i = max(1, int(top_k))
    batch_size_i = max(1, int(batch_size))
    max_trials_i = max(0, int(max_trials))

    best_w: List[Tuple[float, List[int]]] = []
    best_i: List[Tuple[float, List[int]]] = []

    seen: set = set()

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    tried = 0
    batch_idx = 0
    max_resample_factor = 20

    def update_top(best_list, scores, masks):
        for s, idx in zip(scores, masks):
            best_list.append((float(s), idx))
        best_list.sort(key=lambda t: float(t[0]), reverse=True)
        if len(best_list) > top_k_i:
            del best_list[top_k_i:]

    while tried < max_trials_i:
        remaining = max_trials_i - tried
        want = min(batch_size_i, remaining)

        masks_w: List[List[int]] = []
        masks_i: List[List[int]] = []
        resample_budget = max_resample_factor * want

        while len(masks_w) < want and resample_budget > 0:
            idx = torch.randperm(n, generator=gen)[:w_i].tolist()
            idx.sort()
            key = tuple(idx)
            if key in seen:
                resample_budget -= 1
                continue
            seen.add(key)
            masks_w.append(idx)
            if score_complement:
                masks_i.append(invert_mask_indices(idx, n))
            resample_budget -= 1

        if len(masks_w) == 0:
            break

        # Build the complement batch only when it adds new probe masks.
        grids_w = build_grids_from_flatidx(masks_w, grid_hw=int(grid_hw), device=device)
        scores_w = score_grids_softmax_pt(
            model=model,
            grids=grids_w,
            out_hw=int(out_hw),
            target_label=int(target_label),
            dataset=str(dataset),
            device=device,
            norm_cfg_fn=norm_cfg_fn,
        ).detach().cpu().tolist()
        scores_i: List[float] = []
        if score_complement:
            grids_i = build_grids_from_flatidx(masks_i, grid_hw=int(grid_hw), device=device)
            scores_i = score_grids_softmax_pt(
                model=model,
                grids=grids_i,
                out_hw=int(out_hw),
                target_label=int(target_label),
                dataset=str(dataset),
                device=device,
                norm_cfg_fn=norm_cfg_fn,
            ).detach().cpu().tolist()

        update_top(best_w, scores_w, masks_w)
        if score_complement:
            update_top(best_i, scores_i, masks_i)

        tried += len(masks_w)
        batch_idx += 1

        if log_fn is not None and (batch_idx % int(log_every_batches) == 0):
            bw = float(best_w[0][0]) if best_w else 0.0
            bi = float(best_i[0][0]) if best_i else 0.0
            log_fn(
                f"[brute_rand] label={int(target_label)} w={w_i} "
                f"tried={tried}/{max_trials_i} uniq={len(seen)} "
                f"best={bw:.3f} best_inv={bi:.3f}"
            )

    stats_w = SearchStats(tried=int(tried), unique=int(len(seen)), best=float(best_w[0][0]) if best_w else 0.0)
    stats_i = SearchStats(tried=int(tried), unique=int(len(seen)), best=float(best_i[0][0]) if best_i else 0.0)
    return best_w, stats_w, best_i, stats_i


@torch.no_grad()
def random_topk_for_w_bidir_multilabel(
    model,
    target_labels: List[int],
    grid_hw: int,
    w: int,
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn: Callable[[str], Tuple[List[float], List[float]]],
    top_k: int,
    batch_size: int,
    max_trials: int,
    seed: int,
) -> Tuple[
    Dict[int, List[Tuple[float, List[int]]]],
    SearchStats,
    Dict[int, List[Tuple[float, List[int]]]],
    SearchStats,
    SoftmaxPredCountSummary,
]:
    n = int(grid_hw) * int(grid_hw)
    w_i = int(w)
    w_inv = int(n - w_i)
    score_complement = bool(w_inv != w_i)
    labels = [int(label) for label in target_labels]
    if w_i <= 0 or w_i >= n:
        empty_stats = SearchStats(tried=0, unique=0, best=0.0)
        empty_summary = SoftmaxPredCountSummary(
            num_classes=max(labels) + 1 if labels else 0,
            num_scored_w=0,
            num_scored_i=0,
            pred_hist_w=[],
            pred_hist_i=[],
            pred_by_mask_w={},
            pred_by_mask_i={},
            pred_records_w={},
            pred_records_i={},
        )
        return {}, empty_stats, {}, empty_stats, empty_summary

    top_k_i = max(1, int(top_k))
    batch_size_i = max(1, int(batch_size))
    max_trials_i = max(0, int(max_trials))

    best_w: Dict[int, List[Tuple[float, List[int]]]] = {label: [] for label in labels}
    best_i: Dict[int, List[Tuple[float, List[int]]]] = {label: [] for label in labels}

    seen: set = set()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    tried = 0
    max_resample_factor = 20
    pred_hist_w = None
    pred_hist_i = None
    pred_by_mask_w: Dict[int, int] = {}
    pred_by_mask_i: Dict[int, int] = {}
    pred_records_w: Dict[int, Dict[str, Any]] = {}
    pred_records_i: Dict[int, Dict[str, Any]] = {}
    num_scored_w = 0
    num_scored_i = 0

    while tried < max_trials_i:
        remaining = int(max_trials_i) - int(tried)
        want = min(int(batch_size_i), int(remaining))

        flatidx_list: List[List[int]] = []
        resample_budget = max_resample_factor * want
        while len(flatidx_list) < want and resample_budget > 0:
            idx = torch.randperm(n, generator=gen)[:w_i].tolist()
            idx.sort()
            key = tuple(idx)
            if key in seen:
                resample_budget -= 1
                continue
            seen.add(key)
            flatidx_list.append(idx)
            resample_budget -= 1

        if len(flatidx_list) == 0:
            break

        grids_w = build_grids_from_flatidx(
            flatidx_list=flatidx_list,
            grid_hw=int(grid_hw),
            device=device,
        )
        complement_flatidx_list = [invert_mask_indices(mask, n) for mask in flatidx_list] if score_complement else []

        probs_w = score_grids_softmax_all_labels_pt(
            model=model,
            grids=grids_w,
            out_hw=int(out_hw),
            dataset=str(dataset),
            device=device,
            norm_cfg_fn=norm_cfg_fn,
        )
        probs_i = None
        if score_complement:
            grids_i = build_grids_from_flatidx(
                flatidx_list=complement_flatidx_list,
                grid_hw=int(grid_hw),
                device=device,
            )
            probs_i = score_grids_softmax_all_labels_pt(
                model=model,
                grids=grids_i,
                out_hw=int(out_hw),
                dataset=str(dataset),
                device=device,
                norm_cfg_fn=norm_cfg_fn,
            )

        hist_w_batch = _pred_hist_from_probs(probs_w)
        hist_i_batch = _pred_hist_from_probs(probs_i) if score_complement and probs_i is not None else []
        pred_by_mask_w_batch = _pred_map_from_probs(probs_w, flatidx_list)
        pred_by_mask_i_batch = (
            _pred_map_from_probs(probs_i, complement_flatidx_list) if score_complement and probs_i is not None else {}
        )
        pred_records_w_batch = _pred_records_from_probs(probs_w, flatidx_list, grid_hw=int(grid_hw))
        pred_records_i_batch = (
            _pred_records_from_probs(probs_i, complement_flatidx_list, grid_hw=int(grid_hw))
            if score_complement and probs_i is not None
            else {}
        )
        if pred_hist_w is None:
            pred_hist_w = [0 for _ in range(len(hist_w_batch))]
        if pred_hist_i is None:
            pred_hist_i = [0 for _ in range(len(hist_w_batch))]
        for index, count in enumerate(hist_w_batch):
            pred_hist_w[index] += int(count)
        for index, count in enumerate(hist_i_batch):
            pred_hist_i[index] += int(count)
        pred_by_mask_w.update(pred_by_mask_w_batch)
        pred_by_mask_i.update(pred_by_mask_i_batch)
        pred_records_w.update(pred_records_w_batch)
        pred_records_i.update(pred_records_i_batch)
        num_scored_w += int(probs_w.shape[0])
        if score_complement and probs_i is not None:
            num_scored_i += int(probs_i.shape[0])

        for label in labels:
            label_scores_w = probs_w[:, int(label)]
            best_w[label] = _topk_merge(best_w[label], label_scores_w, flatidx_list, k=top_k_i)
            if score_complement and probs_i is not None:
                label_scores_i = probs_i[:, int(label)]
                best_i[label] = _topk_merge(best_i[label], label_scores_i, complement_flatidx_list, k=top_k_i)

        tried += len(flatidx_list)

    best_w_overall = max([best[0][0] for best in best_w.values() if best], default=0.0)
    best_i_overall = max([best[0][0] for best in best_i.values() if best], default=0.0)
    shared_stats_w = SearchStats(tried=int(tried), unique=int(len(seen)), best=float(best_w_overall))
    shared_stats_i = SearchStats(tried=int(tried), unique=int(len(seen)), best=float(best_i_overall))
    num_classes = max(len(pred_hist_w or []), len(pred_hist_i or []))
    summary = SoftmaxPredCountSummary(
        num_classes=int(num_classes),
        num_scored_w=int(num_scored_w),
        num_scored_i=int(num_scored_i),
        pred_hist_w=[int(x) for x in (pred_hist_w or [0 for _ in range(num_classes)])],
        pred_hist_i=[int(x) for x in (pred_hist_i or [0 for _ in range(num_classes)])],
        pred_by_mask_w={int(k): int(v) for k, v in pred_by_mask_w.items()},
        pred_by_mask_i={int(k): int(v) for k, v in pred_by_mask_i.items()},
        pred_records_w={int(k): v for k, v in pred_records_w.items()},
        pred_records_i={int(k): v for k, v in pred_records_i.items()},
    )
    return best_w, shared_stats_w, best_i, shared_stats_i, summary



@torch.no_grad()
def score_grids_softmax_pt(
    model,
    grids: torch.Tensor,
    target_label: int,
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn: Callable[[str], Tuple[List[float], List[float]]],
) -> torch.Tensor:
    """
    grids: (B,1,grid_hw,grid_hw) in {0,1}
    Returns: scores (B,) = softmax(logits)[target_label]
    """
    mean, std = norm_cfg_fn(dataset)
    mean_t = torch.tensor(mean, device=device).view(-1, 1, 1)
    std_t = torch.tensor(std, device=device).view(-1, 1, 1)

    imgs = F.interpolate(grids, size=(int(out_hw), int(out_hw)), mode="nearest")
    imgs = (imgs - mean_t) / std_t

    logits = model(imgs)
    probs = torch.softmax(logits, dim=1)
    return probs[:, int(target_label)]


def score_grids_softmax_all_labels_pt(
    model,
    grids: torch.Tensor,
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn: Callable[[str], Tuple[List[float], List[float]]],
) -> torch.Tensor:
    """
    grids: (B,1,grid_hw,grid_hw) in {0,1}
    Returns: probs (B, C) = softmax(logits)
    """
    mean, std = norm_cfg_fn(dataset)
    mean_t = torch.tensor(mean, device=device).view(-1, 1, 1)
    std_t = torch.tensor(std, device=device).view(-1, 1, 1)

    imgs = F.interpolate(grids, size=(int(out_hw), int(out_hw)), mode="nearest")
    imgs = (imgs - mean_t) / std_t

    logits = model(imgs)
    return torch.softmax(logits, dim=1)

@torch.no_grad()
def score_mask_records_all_labels(
    *,
    model,
    records: List[Dict[str, Any]],
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn: Callable[[str], Tuple[List[float], List[float]]],
    batch_size: int,
) -> List[Dict[str, Any]]:
    """
    Score a list of mask records once for all labels.
    Appends:
      - all_probs
      - pred_label
      - pred_score
      - coords
    """
    if not records:
        return []

    out: List[Dict[str, Any]] = []
    bs = max(1, int(batch_size))
    start = 0

    while start < len(records):
        chunk = records[start : start + bs]
        grids = render_records_for_scoring(
            records=chunk,
            out_hw=int(out_hw),
            device=device,
        )

        probs = score_grids_softmax_all_labels_pt(
            model=model,
            grids=grids,
            out_hw=int(out_hw),
            dataset=str(dataset),
            device=device,
            norm_cfg_fn=norm_cfg_fn,
        )

        pred_scores, preds = torch.max(probs, dim=1)
        probs_cpu = probs.detach().cpu().tolist()
        preds_cpu = preds.detach().cpu().tolist()
        pred_scores_cpu = pred_scores.detach().cpu().tolist()

        for rec, p_all, pred, pred_score in zip(chunk, probs_cpu, preds_cpu, pred_scores_cpu):
            new_rec = dict(rec)
            rec_grid_hw = int(new_rec.get("grid_hw", 1))
            new_rec["coords"] = flatidx_to_coords(new_rec["flat_idx"], rec_grid_hw)
            new_rec["all_probs"] = [float(x) for x in p_all]
            new_rec["pred_label"] = int(pred)
            new_rec["pred_score"] = float(pred_score)

            entropy = compute_softmax_entropy(new_rec["all_probs"])
            top2 = compute_top2_stats(new_rec["all_probs"])

            new_rec["entropy"] = float(entropy)
            new_rec["top1_label"] = int(top2["top1_label"])
            new_rec["top2_label"] = int(top2["top2_label"])
            new_rec["top1_prob"] = float(top2["top1_prob"])
            new_rec["top2_prob"] = float(top2["top2_prob"])
            new_rec["margin"] = float(top2["margin"])

            out.append(new_rec)

        start += bs

    return out

def _pred_hist_from_probs(probs: torch.Tensor) -> List[int]:
    if probs.ndim != 2:
        raise ValueError(f"Expected probs with shape (B, C), got {tuple(probs.shape)}")
    preds = torch.argmax(probs, dim=1).detach().cpu()
    hist = torch.bincount(preds, minlength=int(probs.shape[1]))
    return [int(x) for x in hist.tolist()]


def _pred_map_from_probs(
    probs: torch.Tensor,
    flatidx_list: List[List[int]],
) -> Dict[int, int]:
    if probs.ndim != 2:
        raise ValueError(f"Expected probs with shape (B, C), got {tuple(probs.shape)}")
    preds = torch.argmax(probs, dim=1).detach().cpu().tolist()
    pred_by_mask: Dict[int, int] = {}
    for pred, flatidx in zip(preds, flatidx_list):
        pred_by_mask[int(flatidx_to_bitset(flatidx))] = int(pred)
    return pred_by_mask


def _pred_records_from_probs(
    probs: torch.Tensor,
    flatidx_list: List[List[int]],
    *,
    grid_hw: int,
) -> Dict[int, Dict[str, Any]]:
    if probs.ndim != 2:
        raise ValueError(f"Expected probs with shape (B, C), got {tuple(probs.shape)}")
    max_probs, preds = torch.max(probs, dim=1)
    preds_l = preds.detach().cpu().tolist()
    max_probs_l = max_probs.detach().cpu().tolist()
    out: Dict[int, Dict[str, Any]] = {}
    for pred, pred_score, flatidx in zip(preds_l, max_probs_l, flatidx_list):
        bitset = int(flatidx_to_bitset(flatidx))
        out[bitset] = {
            "pred_label": int(pred),
            "score": float(pred_score),
            "w": int(len(flatidx)),
            "flat_idx": [int(x) for x in flatidx],
            "coords": flatidx_to_coords(flatidx, int(grid_hw)),
        }
    return out

def accumulate_candidates_by_label(
    *,
    scored_records: List[Dict[str, Any]],
    records_by_label: Dict[int, List[Dict[str, Any]]],
    records_by_label_by_d: Dict[int, Dict[int, List[Dict[str, Any]]]],
) -> None:
    """
    Accumulate scored random-probe records into:
      1. global pool per predicted label
      2. per-resolution pool per predicted label
    """
    for rec in scored_records:
        label = int(rec["pred_label"])
        d = int(rec["grid_hw"])

        records_by_label[label].append(rec)
        if d not in records_by_label_by_d:
            records_by_label_by_d[d] = {}
        if label not in records_by_label_by_d[d]:
            records_by_label_by_d[d][label] = []
        records_by_label_by_d[d][label].append(rec)
        
def summarize_sonar_map(
    *,
    records_by_label_by_d: Dict[int, Dict[int, List[Dict[str, Any]]]],
    num_classes: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Build sonar-map summary per resolution.
    """
    sonar: Dict[int, Dict[str, Any]] = {}

    for d in sorted(records_by_label_by_d.keys()):
        class_counts = []
        for label in range(int(num_classes)):
            count = len(records_by_label_by_d[d].get(int(label), []))
            class_counts.append(int(count))

        total = int(sum(class_counts))
        if total > 0:
            class_ratios = [float(x) / float(total) for x in class_counts]
        else:
            class_ratios = [0.0 for _ in range(int(num_classes))]

        sonar[int(d)] = {
            "class_counts": class_counts,
            "class_ratios": class_ratios,
            "num_trials": int(total),
        }

    return sonar

def get_class_coverage_status(
    *,
    records_by_label: Dict[int, List[Dict[str, Any]]],
    num_classes: int,
    min_candidates_per_class: int,
) -> Dict[str, Any]:
    """
    Coverage status for the global candidate pool.
    """
    counts = [int(len(records_by_label.get(i, []))) for i in range(int(num_classes))]
    covered_labels = [int(i) for i, c in enumerate(counts) if int(c) >= int(min_candidates_per_class)]
    undercovered_labels = [int(i) for i, c in enumerate(counts) if int(c) < int(min_candidates_per_class)]

    return {
        "counts": counts,
        "covered_labels": covered_labels,
        "undercovered_labels": undercovered_labels,
        "all_covered": len(undercovered_labels) == 0,
        "min_candidates_per_class": int(min_candidates_per_class),
    }

def _topk_merge(
    top_patterns,
    cand_scores,
    cand_flatidx,
    k,
):
    """
    Maintain a list of unique (score, flatidx_sorted) with global top-k.
    If duplicates occur, keep the highest score for that pattern.
    """
    best = {}

    # existing
    for s, idxs in top_patterns:
        key = tuple(int(x) for x in idxs)
        prev = best.get(key)
        if prev is None or float(s) > prev[0]:
            best[key] = (float(s), list(idxs))

    # incoming
    for i in range(len(cand_flatidx)):
        s = float(cand_scores[i].item())
        idxs = cand_flatidx[i]
        key = tuple(int(x) for x in idxs)
        prev = best.get(key)
        if prev is None or s > prev[0]:
            best[key] = (s, list(idxs))

    merged = list(best.values())
    merged.sort(key=lambda x: x[0], reverse=True)
    if len(merged) > int(k):
        merged = merged[: int(k)]
    return merged

def _format_records(top_patterns, w: int, grid_hw: int, top_l_per_w: int, gen_info=None):
    w_list: List[Dict] = []
    keep_n = min(int(top_l_per_w), len(top_patterns))
    for i in range(keep_n):
        score, flatidx = top_patterns[i]
        coords = flatidx_to_coords(flatidx, grid_hw)
        d = {
            "rank": int(i),
            "w": int(w),
            "score": float(score),
            "flat_idx": [int(x) for x in flatidx],
            "coords": coords,
        }
        if gen_info is not None and i == 0:
            d["gen_info"] = gen_info
        w_list.append(d)
    return w_list


def _format_label_best_map(
    label_to_best: Dict[int, float],
    *,
    target_labels: List[int],
) -> str:
    return ", ".join(
        [f"{int(label):02d}:{float(label_to_best.get(int(label), 0.0)):.3f}" for label in target_labels]
    )


def _log_label_best_map_block(
    log_fn,
    *,
    prefix: str,
    label_to_best: Dict[int, float],
    target_labels: List[int],
    chunk_size: int = 5,
) -> None:
    labels = [int(label) for label in target_labels]
    for start in range(0, len(labels), int(chunk_size)):
        chunk_labels = labels[start : start + int(chunk_size)]
        log_fn(f"{prefix} {_format_label_best_map(label_to_best, target_labels=chunk_labels)}")


def _dedup_record_pool(
    records: List[Dict[str, Any]],
    score_key: str = "score",
) -> List[Dict[str, Any]]:
    best: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for rec in records:
        flat_key = tuple(int(x) for x in rec.get("flat_idx", []))
        grid_hw = rec.get("grid_hw", None)
        probe_mode = str(rec.get("probe_mode", "binary")).lower()
        polarity = rec.get("polarity", None)
        background_fill = rec.get("background_fill", None)
        active_fill = rec.get("active_fill", None)
        render_seed = rec.get("render_seed", None)
        if grid_hw is not None:
            key = (
                int(grid_hw),
                probe_mode,
                str(polarity) if polarity is not None else None,
                float(background_fill) if background_fill is not None else None,
                float(active_fill) if active_fill is not None else None,
                int(render_seed) if render_seed is not None else None,
                flat_key,
            )
        else:
            key = flat_key
        prev = best.get(key)
        curr_score = float(rec.get(score_key, 0.0))
        prev_score = float(prev.get(score_key, 0.0)) if prev is not None else float("-inf")
        if prev is None or curr_score > prev_score:
            best[key] = rec
    out = list(best.values())
    out.sort(key=lambda r: float(r.get(score_key, 0.0)), reverse=True)
    return out

def get_candidate_pool_for_label(
    *,
    all_scored_records_by_d: Dict[int, List[Dict[str, Any]]],
    label: int,
) -> List[Dict[str, Any]]:
    """
    Union candidate records for one label across all resolutions.

    This follows the paper-side construction of V_i: rank all probes by the
    target softmax for class i first, then later filter to source-model hits
    when constructing T_i.
    """
    pool: List[Dict[str, Any]] = []
    for d in sorted(all_scored_records_by_d.keys()):
        for rec in all_scored_records_by_d.get(int(d), []):
            probs = rec.get("all_probs", None)
            if probs is None or int(label) >= len(probs):
                continue
            new_rec = dict(rec)
            new_rec["target_score"] = float(probs[int(label)])
            pool.append(new_rec)

    pool.sort(key=lambda r: float(r.get("target_score", 0.0)), reverse=True)
    return _dedup_record_pool(pool, score_key="target_score")

def prefilter_top_candidates_for_label(
    *,
    candidate_pool: List[Dict[str, Any]],
    label: int,
    prefilter_k: int,
) -> List[Dict[str, Any]]:
    """
    Keep top-k candidates by raw target softmax for class label.
    """
    if not candidate_pool:
        return []

    ranked = []
    for rec in candidate_pool:
        probs = rec.get("all_probs", None)
        if probs is None or int(label) >= len(probs):
            continue
        new_rec = dict(rec)
        new_rec["target_score"] = float(rec.get("target_score", probs[int(label)]))
        ranked.append(new_rec)

    ranked.sort(key=lambda r: float(r["target_score"]), reverse=True)
    deduped = _dedup_record_pool(ranked, score_key="target_score")
    return deduped[: max(1, int(prefilter_k))]

def make_translation_shifts(
    *,
    grid_hw: int,
    radius: int,
    include_origin: bool = False,
) -> List[Tuple[int, int]]:
    shifts: List[Tuple[int, int]] = []
    r = max(0, int(radius))
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            if not include_origin and dr == 0 and dc == 0:
                continue
            shifts.append((int(dr), int(dc)))
    return shifts

def _shift_flatidx(
    flat_idx: List[int],
    *,
    grid_hw: int,
    dr: int,
    dc: int,
) -> List[int]:
    out = []
    for idx in flat_idx:
        rr = int(idx) // int(grid_hw)
        cc = int(idx) % int(grid_hw)
        rr2 = rr + int(dr)
        cc2 = cc + int(dc)
        if 0 <= rr2 < int(grid_hw) and 0 <= cc2 < int(grid_hw):
            out.append(int(rr2) * int(grid_hw) + int(cc2))
    out = sorted(set(out))
    return out

@torch.no_grad()
def evaluate_candidate_robustness(
    *,
    model,
    candidates: List[Dict[str, Any]],
    label: int,
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn: Callable[[str], Tuple[List[float], List[float]]],
    shifts: List[Tuple[int, int]],
    batch_size: int,
) -> List[Dict[str, Any]]:
    """
    Evaluate robustness by translation augmentation on the same grid resolution.
    Adds:
      - robust_target_mean
      - robust_target_std
      - robust_count
    """
    if not candidates:
        return []

    out: List[Dict[str, Any]] = []
    bs = max(1, int(batch_size))

    for rec in candidates:
        grid_hw = int(rec["grid_hw"])
        src_flat = [int(x) for x in rec["flat_idx"]]
        background_fill, active_fill = get_record_fill_values(rec)

        aug_masks: List[List[int]] = []
        for dr, dc in shifts:
            shifted = _shift_flatidx(src_flat, grid_hw=int(grid_hw), dr=int(dr), dc=int(dc))
            if len(shifted) == 0:
                continue
            aug_masks.append(shifted)

        if len(aug_masks) == 0:
            new_rec = dict(rec)
            new_rec["robust_target_mean"] = float(rec.get("target_score", 0.0))
            new_rec["robust_target_std"] = 0.0
            new_rec["robust_count"] = 0
            out.append(new_rec)
            continue

        all_scores: List[float] = []
        start = 0
        while start < len(aug_masks):
            chunk = aug_masks[start : start + bs]
            grids = build_grids_from_flatidx_with_fill(
                flatidx_list=chunk,
                grid_hw=int(grid_hw),
                device=device,
                background_fill=float(background_fill),
                active_fill=float(active_fill),
            )
            probs = score_grids_softmax_all_labels_pt(
                model=model,
                grids=grids,
                out_hw=int(out_hw),
                dataset=str(dataset),
                device=device,
                norm_cfg_fn=norm_cfg_fn,
            )
            label_scores = probs[:, int(label)].detach().cpu().tolist()
            all_scores.extend(float(x) for x in label_scores)
            start += bs

        scores_t = torch.tensor(all_scores, dtype=torch.float32)
        new_rec = dict(rec)
        new_rec["robust_target_mean"] = float(scores_t.mean().item())
        new_rec["robust_target_std"] = float(scores_t.std(unbiased=False).item()) if scores_t.numel() > 1 else 0.0
        new_rec["robust_count"] = int(scores_t.numel())
        out.append(new_rec)

    return out

def select_topl_for_label(
    *,
    candidates: List[Dict[str, Any]],
    top_l: int,
    score_key: str = "robust_target_mean",
) -> List[Dict[str, Any]]:
    """
    Final Top-L filtering.
    """
    if not candidates:
        return []

    ranked = list(candidates)
    ranked.sort(key=lambda r: float(r.get(score_key, 0.0)), reverse=True)
    ranked = _dedup_record_pool(ranked, score_key=score_key)
    return ranked[: max(1, int(top_l))]

def _sample_subset(rng: random.Random, items: List[int], limit: int) -> List[int]:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    return rng.sample(list(items), int(limit))


def _neighbor_masks_for_local_refine(
    flatidx: List[int],
    *,
    n_cells: int,
    rng: random.Random,
    min_w: int,
    max_w: int,
    max_add_per_seed: int,
    max_remove_per_seed: int,
    max_swap_per_seed: int,
) -> List[List[int]]:
    on_cells = sorted(int(x) for x in flatidx)
    on_set = set(on_cells)
    off_cells = [i for i in range(int(n_cells)) if i not in on_set]

    neighbors: List[List[int]] = []

    if len(on_cells) > int(min_w):
        for rm in _sample_subset(rng, on_cells, int(max_remove_per_seed)):
            cand = [x for x in on_cells if x != rm]
            neighbors.append(cand)

    if len(on_cells) < int(max_w):
        for add in _sample_subset(rng, off_cells, int(max_add_per_seed)):
            cand = sorted(on_cells + [int(add)])
            neighbors.append(cand)

    swap_budget = int(max_swap_per_seed)
    if swap_budget > 0 and on_cells and off_cells:
        # The Cartesian product can contain hundreds of thousands of pairs
        # even though refinement normally requests only a few swaps. Python's
        # random.sample selects list positions, so sampling the equivalent
        # range and mapping positions back to (remove, add) pairs preserves
        # both RNG consumption and the exact selected masks.
        num_off = int(len(off_cells))
        num_candidates = int(len(on_cells) * num_off)
        if num_candidates > swap_budget:
            sampled_positions = rng.sample(range(num_candidates), swap_budget)
            candidates = [
                (
                    int(on_cells[int(pos) // num_off]),
                    int(off_cells[int(pos) % num_off]),
                )
                for pos in sampled_positions
            ]
        else:
            candidates = [
                (int(rm), int(add))
                for rm in on_cells
                for add in off_cells
            ]
        for rm, add in candidates:
            cand = [x for x in on_cells if x != rm]
            cand.append(add)
            cand.sort()
            neighbors.append(cand)

    uniq: Dict[Tuple[int, ...], List[int]] = {}
    src_key = tuple(on_cells)
    for cand in neighbors:
        key = tuple(cand)
        if key == src_key:
            continue
        uniq[key] = cand
    return list(uniq.values())


def upscale_mask(flat_idx_prev: List[int], grid_hw_prev: int, grid_hw_curr: int) -> List[int]:
    """
    Upscale a flat_idx from grid_hw_prev to grid_hw_curr by 2x replication.
    Each cell becomes 4 cells in a 2x2 block.
    """
    assert grid_hw_curr == 2 * grid_hw_prev, f"Only 2x upscaling supported: {grid_hw_prev} -> {grid_hw_curr}"
    upscaled = []
    for idx in flat_idx_prev:
        r = idx // grid_hw_prev
        c = idx % grid_hw_prev
        # 2x2 block
        upscaled.extend([
            (2 * r) * grid_hw_curr + (2 * c),        # (2r, 2c)
            (2 * r) * grid_hw_curr + (2 * c + 1),    # (2r, 2c+1)
            (2 * r + 1) * grid_hw_curr + (2 * c),    # (2r+1, 2c)
            (2 * r + 1) * grid_hw_curr + (2 * c + 1) # (2r+1, 2c+1)
        ])
    upscaled.sort()
    return upscaled


def upscale_records(records: List[Dict[str, Any]], grid_hw_prev: int, grid_hw_curr: int) -> List[Dict[str, Any]]:
    """
    Upscale a list of records to new grid_hw.
    """
    upscaled_records = []
    for rec in records:
        flat_idx_curr = upscale_mask(rec["flat_idx"], grid_hw_prev, grid_hw_curr)
        coords_curr = flatidx_to_coords(flat_idx_curr, grid_hw_curr)
        upscaled_rec = rec.copy()
        upscaled_rec["flat_idx"] = flat_idx_curr
        upscaled_rec["coords"] = coords_curr
        upscaled_rec["w"] = len(flat_idx_curr)
        upscaled_records.append(upscaled_rec)
    return upscaled_records


@torch.no_grad()
def local_refine_top_records(
    *,
    model,
    target_label: int,
    grid_hw: int,
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn: Callable[[str], Tuple[List[float], List[float]]],
    seed_records: List[Dict[str, Any]],
    batch_size: int,
    refine_cfg: Dict[str, Any],
    polarity: str = "white",
    log_fn: Optional[Callable[[str], None]] = None,
    early_stop_score: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not seed_records:
        return [], {"enabled": False, "reason": "empty_seed_records"}

    beam_width = int(refine_cfg.get("beam_width", 12))
    rounds = int(refine_cfg.get("rounds", 8))
    patience = int(refine_cfg.get("patience", 3))
    seed_top_k = int(refine_cfg.get("seed_top_k", max(beam_width, 16)))
    keep_top_l = int(refine_cfg.get("top_l", max(seed_top_k, beam_width)))
    max_add_per_seed = int(refine_cfg.get("max_add_per_seed", 12))
    max_remove_per_seed = int(refine_cfg.get("max_remove_per_seed", 12))
    max_swap_per_seed = int(refine_cfg.get("max_swap_per_seed", 24))
    seed = int(refine_cfg.get("seed", 123))

    rng = random.Random(seed + 100000 * int(target_label))
    n_cells = int(grid_hw) * int(grid_hw)
    min_w = int(refine_cfg.get("min_w", 1))
    max_w = int(refine_cfg.get("max_w", max(1, n_cells - 1)))

    seed_pool = _dedup_record_pool(seed_records)[:seed_top_k]
    beam = seed_pool[:beam_width]
    best_score = float(beam[0]["score"]) if beam else float("-inf")
    stale_rounds = 0
    total_candidates_scored = 0

    if log_fn is not None:
        log_fn(
            f"[refine] label={int(target_label):02d} "
            f"seed_top_k={int(len(seed_pool))} beam_width={beam_width} rounds={rounds}"
        )

    if early_stop_score is not None and best_score >= float(early_stop_score):
        refined = _dedup_record_pool(beam)[:keep_top_l]
        info = {
            "enabled": True,
            "seed_top_k": int(seed_top_k),
            "beam_width": int(beam_width),
            "rounds": int(rounds),
            "patience": int(patience),
            "max_add_per_seed": int(max_add_per_seed),
            "max_remove_per_seed": int(max_remove_per_seed),
            "max_swap_per_seed": int(max_swap_per_seed),
            "min_w": int(min_w),
            "max_w": int(max_w),
            "total_candidates_scored": int(total_candidates_scored),
            "best_score": float(refined[0]["score"]) if refined else float("-inf"),
            "early_stop_score": float(early_stop_score),
            "stopped_early": True,
            "reason": "seed_already_meets_threshold",
        }
        return refined, info

    for step in range(rounds):
        mask_map: Dict[Tuple[int, ...], List[int]] = {}

        for rec in beam:
            flatidx = [int(x) for x in rec.get("flat_idx", [])]
            for cand in _neighbor_masks_for_local_refine(
                flatidx,
                n_cells=n_cells,
                rng=rng,
                min_w=min_w,
                max_w=max_w,
                max_add_per_seed=max_add_per_seed,
                max_remove_per_seed=max_remove_per_seed,
                max_swap_per_seed=max_swap_per_seed,
            ):
                mask_map[tuple(cand)] = cand

        for rec in beam:
            flatidx = [int(x) for x in rec.get("flat_idx", [])]
            mask_map[tuple(flatidx)] = flatidx

        cand_masks = list(mask_map.values())
        if not cand_masks:
            break

        scores = rescore_masks_topk(
            model=model,
            target_label=int(target_label),
            grid_hw=int(grid_hw),
            out_hw=int(out_hw),
            dataset=dataset,
            device=device,
            norm_cfg_fn=norm_cfg_fn,
            masks_flatidx=cand_masks,
            batch_size=int(batch_size),
            polarity=str(polarity),
        )
        total_candidates_scored += len(cand_masks)

        scored_records: List[Dict[str, Any]] = []
        background_fill, active_fill = _polarity_fill_values(str(polarity))
        for s, flatidx in zip(scores, cand_masks):
            scored_records.append(
                {
                    "grid_hw": int(grid_hw),
                    "w": int(len(flatidx)),
                    "score": float(s),
                    "flat_idx": [int(x) for x in flatidx],
                    "coords": flatidx_to_coords(flatidx, int(grid_hw)),
                    "polarity": str(polarity),
                    "background_fill": float(background_fill),
                    "active_fill": float(active_fill),
                }
            )

        scored_records = _dedup_record_pool(scored_records)
        beam = scored_records[:beam_width]
        curr_best = float(beam[0]["score"]) if beam else float("-inf")

        if log_fn is not None:
            log_fn(
                f"[refine] label={int(target_label):02d} round={int(step + 1):02d} "
                f"cand={int(len(cand_masks))} best={curr_best:.3f}"
            )

        if curr_best > best_score + 1e-12:
            best_score = curr_best
            stale_rounds = 0
            if early_stop_score is not None and curr_best >= float(early_stop_score):
                break
        else:
            stale_rounds += 1
            if stale_rounds >= patience:
                break

    refined = _dedup_record_pool(beam)[:keep_top_l]
    info = {
        "enabled": True,
        "seed_top_k": int(seed_top_k),
        "beam_width": int(beam_width),
        "rounds": int(rounds),
        "patience": int(patience),
        "max_add_per_seed": int(max_add_per_seed),
        "max_remove_per_seed": int(max_remove_per_seed),
        "max_swap_per_seed": int(max_swap_per_seed),
        "min_w": int(min_w),
        "max_w": int(max_w),
        "total_candidates_scored": int(total_candidates_scored),
        "best_score": float(refined[0]["score"]) if refined else float("-inf"),
        "early_stop_score": float(early_stop_score) if early_stop_score is not None else None,
        "stopped_early": bool(
            early_stop_score is not None and refined and float(refined[0]["score"]) >= float(early_stop_score)
        ),
    }
    return refined, info


def weak_local_refine_candidates(
    *,
    model,
    candidates: List[Dict[str, Any]],
    label: int,
    grid_hw: int,
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn: Callable[[str], Tuple[List[float], List[float]]],
    batch_size: int,
    max_candidates: int,
    max_edit_steps: int,
    seed: int,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Weak wrapper over local_refine_top_records.
    Restricts refine to a very small candidate set and a very small edit budget.
    """
    if not candidates:
        return [], {"enabled": False, "reason": "empty_candidates"}

    polarity = str(candidates[0].get("polarity", "white")).lower()
    background_fill, active_fill = _polarity_fill_values(polarity)

    seed_records: List[Dict[str, Any]] = []
    for rec in candidates[: max(1, int(max_candidates))]:
        seed_rec = dict(rec)
        seed_rec["grid_hw"] = int(seed_rec.get("grid_hw", grid_hw))
        seed_rec["w"] = int(seed_rec.get("w", len(seed_rec.get("flat_idx", []))))
        seed_rec.setdefault("polarity", str(polarity))
        seed_rec.setdefault("background_fill", float(background_fill))
        seed_rec.setdefault("active_fill", float(active_fill))
        if "coords" not in seed_rec:
            seed_rec["coords"] = flatidx_to_coords(seed_rec["flat_idx"], int(seed_rec["grid_hw"]))
        if "score" not in seed_rec:
            seed_rec["score"] = float(
                seed_rec.get("target_score", seed_rec.get("pred_score", 0.0))
            )
        seed_records.append(seed_rec)

    refine_cfg = {
        "beam_width": min(8, max(2, len(seed_records))),
        "rounds": max(1, int(max_edit_steps)),
        "patience": 1,
        "seed_top_k": len(seed_records),
        "top_l": len(seed_records),
        "max_add_per_seed": 2,
        "max_remove_per_seed": 2,
        "max_swap_per_seed": 4,
        "min_w": 1,
        "max_w": int(grid_hw) * int(grid_hw) - 1,
        "seed": int(seed),
    }

    refined, info = local_refine_top_records(
        model=model,
        target_label=int(label),
        grid_hw=int(grid_hw),
        out_hw=int(out_hw),
        dataset=str(dataset),
        device=device,
        norm_cfg_fn=norm_cfg_fn,
        seed_records=seed_records,
        batch_size=int(batch_size),
        refine_cfg=refine_cfg,
        polarity=str(polarity),
        log_fn=log_fn,
        early_stop_score=None,
    )

    refined_out: List[Dict[str, Any]] = []
    for rec in refined:
        new_rec = dict(rec)
        new_rec["grid_hw"] = int(new_rec.get("grid_hw", grid_hw))
        new_rec["w"] = int(new_rec.get("w", len(new_rec.get("flat_idx", []))))
        new_rec.setdefault("polarity", str(polarity))
        new_rec.setdefault("background_fill", float(background_fill))
        new_rec.setdefault("active_fill", float(active_fill))
        new_rec.setdefault(
            "coords",
            flatidx_to_coords(new_rec.get("flat_idx", []), int(new_rec["grid_hw"])),
        )
        new_rec.setdefault("target_score", float(new_rec.get("score", 0.0)))
        refined_out.append(new_rec)
    return refined_out, info


@torch.no_grad()
def bruteforce_topk_for_w(
    model,
    target_label: int,
    grid_hw: int,
    w: int,
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn: Callable[[str], Tuple[List[float], List[float]]],
    top_k: int,
    batch_size: int,
    log_every_batches: int = 50,
    log_fn: Optional[Callable[[str], None]] = None,
) -> List[Tuple[float, List[int]]]:
    """
    Exhaustive top-k over all C(HW, w) patterns.
    Returns: list of (score, flatidx_sorted), sorted by score desc.
    """
    n = int(grid_hw) * int(grid_hw)
    combos = itertools.combinations(range(n), int(w))
    total = math.comb(n, int(w))

    pending: List[Tuple[int, ...]] = []
    processed = 0
    batch_count = 0
    top_patterns: List[Tuple[float, List[int]]] = []

    def build_batch(combo_list: List[Tuple[int, ...]]) -> torch.Tensor:
        b = len(combo_list)
        g = torch.zeros((b, 1, grid_hw, grid_hw), device=device)
        for i, idxs in enumerate(combo_list):
            flat = g[i, 0].view(-1)
            flat[list(idxs)] = 1.0
        return g

    for idxs in combos:
        pending.append(idxs)
        if len(pending) < int(batch_size):
            continue

        grids = build_batch(pending)
        cand_flatidx = [list(t) for t in pending]
        pending = []

        scores = score_grids_softmax_pt(
            model=model,
            grids=grids,
            target_label=target_label,
            out_hw=out_hw,
            dataset=dataset,
            device=device,
            norm_cfg_fn=norm_cfg_fn,
        )

        top_patterns = _topk_merge(top_patterns, scores, cand_flatidx, k=top_k)

        processed += int(batch_size)
        batch_count += 1

        if log_fn is not None and (batch_count % int(log_every_batches) == 0):
            best_s = float(top_patterns[0][0]) if len(top_patterns) > 0 else 0.0
            log_fn(f"Bruteforce label={target_label} w={w} processed={processed} total={total} best={best_s:.3f}")

    if len(pending) > 0:
        grids = build_batch(pending)
        cand_flatidx = [list(t) for t in pending]

        scores = score_grids_softmax_pt(
            model=model,
            grids=grids,
            target_label=target_label,
            out_hw=out_hw,
            dataset=dataset,
            device=device,
            norm_cfg_fn=norm_cfg_fn,
        )
        top_patterns = _topk_merge(top_patterns, scores, cand_flatidx, k=top_k)

        processed += len(cand_flatidx)

    top_patterns.sort(key=lambda x: x[0], reverse=True)
    return top_patterns

@torch.no_grad()
def beam_grow_topk(
    model,
    target_label: int,
    grid_hw: int,
    w_start: int,
    w_end: int,
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn: Callable[[str], Tuple[List[float], List[float]]],
    parents: List[Tuple[float, List[int]]],
    beam_width: int,
    top_l_log: int,
    batch_size: int,
    expand_mode: str = "all",
    expand_k: int = 16,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[int, List[Tuple[float, List[int]]]]:
    """
    Grow patterns from w_start to w_end using beam search.
    parents: list(score, flatidx_sorted) at w_start, will be trimmed to beam_width.
    Returns: records_by_w[w] = top_l_log patterns (score, flatidx) at each w.
    """
    records_by_w: Dict[int, List[Tuple[float, List[int]]]] = {}

    parents = sorted(parents, key=lambda x: x[0], reverse=True)
    parents = parents[: int(beam_width)]

    n = int(grid_hw) * int(grid_hw)

    for w in range(int(w_start) + 1, int(w_end) + 1):
        # Generate candidates
        cand_flatidx_all: List[List[int]] = []
        for (pscore, pidx) in parents:
            used = set(int(x) for x in pidx)
            empty = [i for i in range(n) if i not in used]

            if expand_mode == "sample":
                if len(empty) > int(expand_k):
                    empty = random.sample(empty, int(expand_k))

            for e in empty:
                new_idx = list(pidx)
                new_idx.append(int(e))
                new_idx.sort()
                cand_flatidx_all.append(new_idx)

        # Dedup candidates (same child can be generated from different parents)
        # Keep order not important; sorting keys is fine.
        # Dedup candidates (same child can be generated from different parents)
        before = len(cand_flatidx_all)

        uniq = {}
        for idxs in cand_flatidx_all:
            uniq[tuple(idxs)] = idxs
        cand_flatidx_all = list(uniq.values())

        after = len(cand_flatidx_all)
        if log_fn is not None:
            log_fn(f"Beam label={target_label} w={w} dedup cand {before} -> {after}")

        # Score in batches
        top_patterns: List[Tuple[float, List[int]]] = []
        processed = 0
        batch_count = 0

        def build_grids(flatidx_list: List[List[int]]) -> torch.Tensor:
            b = len(flatidx_list)
            g = torch.zeros((b, 1, grid_hw, grid_hw), device=device)
            for i, idxs in enumerate(flatidx_list):
                flat = g[i, 0].view(-1)
                flat[idxs] = 1.0
            return g

        ptr = 0
        while ptr < len(cand_flatidx_all):
            chunk = cand_flatidx_all[ptr : ptr + int(batch_size)]
            ptr += int(batch_size)

            grids = build_grids(chunk)

            scores = score_grids_softmax_pt(
                model=model,
                grids=grids,
                target_label=target_label,
                out_hw=out_hw,
                dataset=dataset,
                device=device,
                norm_cfg_fn=norm_cfg_fn,
            )

            top_patterns = _topk_merge(top_patterns, scores, chunk, k=int(beam_width))

            processed += len(chunk)
            batch_count += 1

            if log_fn is not None and (batch_count % 50 == 0):
                best_s = float(top_patterns[0][0]) if len(top_patterns) > 0 else 0.0
                log_fn(f"Beam label={target_label} w={w} processed={processed} cand={len(cand_flatidx_all)} best={best_s:.3f}")

        top_patterns.sort(key=lambda x: x[0], reverse=True)
        parents = top_patterns[: int(beam_width)]
        records_by_w[int(w)] = top_patterns[: int(top_l_log)]

        if log_fn is not None:
            best_s = float(records_by_w[int(w)][0][0]) if len(records_by_w[int(w)]) > 0 else 0.0
            log_fn(f"Beam done label={target_label} w={w} best={best_s:.3f}")

    return records_by_w




def log_search_progress(
    log_fn,
    method: str,
    target_label: int,
    w: int,
    phase: str,              # "start" | "done" | "progress"
    stats: SearchStats,
    extra: str = "",
):
    if log_fn is None:
        return

    method_s = str(method)
    if extra:
        extra = " " + str(extra)

    if phase == "start":
        log_fn(f"[{method_s}] label={int(target_label)} w={int(w)}{extra}")
    elif phase == "progress":
        log_fn(
            f"[{method_s}] prog label={int(target_label)} w={int(w)} "
            f"tried={int(stats.tried)} uniq={int(stats.unique)} best={float(stats.best):.3f}{extra}"
        )
    elif phase == "done":
        log_fn(
            f"[{method_s}] done label={int(target_label)} w={int(w)} "
            f"tried={int(stats.tried)} uniq={int(stats.unique)} best={float(stats.best):.3f}{extra}"
        )
    else:
        raise ValueError(f"Unknown phase: {phase}")


@dataclass
class WPlan:
    n_cells: int
    search_ws: List[int]
    complement_pairs: List[Tuple[int, int]]  # (w_src, w_dst)


def _round_nearest_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def _apply_rounding(x: float, mode: str) -> int:
    m = str(mode or "nearest").lower()
    if m == "nearest":
        return _round_nearest_half_up(x)
    if m == "floor":
        return int(math.floor(x))
    if m == "ceil":
        return int(math.ceil(x))
    raise ValueError(f"Unknown rounding: {mode}")


def compile_w_plan(grid_hw: int, w_schedule: Dict[str, Any]) -> WPlan:
    if not isinstance(w_schedule, dict):
        raise ValueError("w_schedule is required and must be a dict")

    n_cells = int(grid_hw) * int(grid_hw)

    mode = str(w_schedule.get("mode", "pct")).lower()
    use_complement = bool(w_schedule.get("use_complement", False))
    rounding = str(w_schedule.get("rounding", "nearest")).lower()
    clamp = w_schedule.get("clamp", [1, -1])
    dedup_w = bool(w_schedule.get("dedup_w", True))

    clamp_min = int(clamp[0]) if isinstance(clamp, (list, tuple)) and len(clamp) >= 1 else 1
    clamp_max_raw = int(clamp[1]) if isinstance(clamp, (list, tuple)) and len(clamp) >= 2 else -1
    clamp_max = (n_cells - 1) if clamp_max_raw == -1 else int(clamp_max_raw)

    def clamp_w(w: int) -> int:
        if w < clamp_min:
            return clamp_min
        if w > clamp_max:
            return clamp_max
        return w

    if mode == "pct":
        pcts = w_schedule.get("pcts", None)
        if not isinstance(pcts, list) or len(pcts) == 0:
            raise ValueError("w_schedule.mode=pct requires non-empty pcts list")

        ws: List[int] = []
        for p in pcts:
            w = _apply_rounding(float(p) * float(n_cells), rounding)
            ws.append(clamp_w(int(w)))

        ws = sorted(set(ws)) if dedup_w else sorted(ws)

        if use_complement:
            half = (n_cells + 1) // 2    # ceil(n/2): 25 when n_cells=49
            search_ws = [w for w in ws if w <= half]
        else:
            search_ws = ws

    elif mode == "explicit_w":
        w_list = w_schedule.get("w_list", None)
        if not isinstance(w_list, list) or len(w_list) == 0:
            raise ValueError("w_schedule.mode=explicit_w requires non-empty w_list")

        ws = [clamp_w(int(w)) for w in w_list]
        ws = sorted(set(ws)) if dedup_w else sorted(ws)
        search_ws = ws

    else:
        raise ValueError(f"Unknown w_schedule.mode: {mode}")

    complement_pairs: List[Tuple[int, int]] = []
    if use_complement:
        present = set(search_ws)
        for w_src in search_ws:
            w_dst = int(n_cells - int(w_src))
            if w_dst != w_src and w_dst not in present:
                complement_pairs.append((int(w_src), int(w_dst)))

    return WPlan(n_cells=n_cells, search_ws=search_ws, complement_pairs=complement_pairs)



def invert_mask_indices(mask_indices: List[int], n_cells: int) -> List[int]:
    s = set(int(x) for x in mask_indices)
    inv = [i for i in range(int(n_cells)) if i not in s]
    return inv

def sample_random_masks_for_d(
    *,
    grid_hw: int,
    w_list: List[int],
    n_trials_per_w: int,
    seed: int,
    use_complement: bool = False,
    max_records: Optional[int] = None,
    seen_base_masks: Optional[Set[Tuple[int, ...]]] = None,
    probe_mode: str = "binary",
    two_group_cfg: Optional[Dict[str, Any]] = None,
    num_channels: int = 3,
) -> List[Dict[str, Any]]:
    """
    Pure random sampler for one grid resolution d=grid_hw.

    Returns a flat list of mask records. No scoring, no top-k, no label targeting.
    If use_complement=True, also adds the polarity-inverted counterpart.
    """
    n = int(grid_hw) * int(grid_hw)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    mode = str(probe_mode or "binary").strip().lower()
    tg_cfg = dict(two_group_cfg or {})
    tg_variants_per_mask = max(1, int(tg_cfg.get("variants_per_mask", 1)))
    tg_sigma_fg = float(tg_cfg.get("sigma_fg", 0.10))
    tg_sigma_bg = float(tg_cfg.get("sigma_bg", 0.10))
    tg_d_min = float(tg_cfg.get("d_min", 0.30))
    tg_num_channels = max(1, int(tg_cfg.get("num_channels", num_channels)))
    tg_d_max = float(tg_cfg.get("d_max", math.sqrt(float(tg_num_channels))))
    ga_mean = float(tg_cfg.get("mean", 0.5))
    ga_sigma = float(tg_cfg.get("sigma", 0.35))
    max_records_i = None if max_records is None else max(0, int(max_records))

    if max_records_i == 0:
        return []

    seen: set = set()
    global_seen = seen_base_masks
    records: List[Dict[str, Any]] = []

    for w in w_list:
        if max_records_i is not None and len(records) >= max_records_i:
            break

        w_i = int(w)
        if w_i <= 0 or w_i >= n:
            continue

        tried_local = 0
        max_resample = max(20 * int(n_trials_per_w), 100)

        while (
            tried_local < int(n_trials_per_w)
            and max_resample > 0
            and (max_records_i is None or len(records) < max_records_i)
        ):
            idx = torch.randperm(n, generator=rng)[:w_i].tolist()
            idx.sort()
            key = tuple(int(x) for x in idx)
            max_resample -= 1

            if key in seen:
                continue
            if global_seen is not None and key in global_seen:
                continue
            seen.add(key)
            if global_seen is not None:
                global_seen.add(key)

            if mode.startswith("two_group"):
                mask_seed = _stable_mask_seed(idx) if "_stable_mask_seed" in globals() else int(flatidx_to_bitset(idx))
                for variant_idx in range(int(tg_variants_per_mask)):
                    render_seed = (
                        int(seed)
                        + int(grid_hw) * 10007
                        + int(mask_seed) * 97
                        + int(variant_idx) * 1000003
                    ) % 2147483647
                    mu_fg, mu_bg = sample_two_group_color_centers(
                        seed=int(render_seed),
                        d_min=float(tg_d_min),
                        d_max=float(tg_d_max),
                        num_channels=int(tg_num_channels),
                        center_range=(0.0, 1.0),
                    )
                    records.append(
                        {
                            "grid_hw": int(grid_hw),
                            "w": int(w_i),
                            "flat_idx": [int(x) for x in idx],
                            "bitset": int(flatidx_to_bitset(idx)),
                            "source": "random",
                            "probe_mode": str(mode),
                            "num_channels": int(tg_num_channels),
                            "render_seed": int(render_seed),
                            "sigma_fg": float(tg_sigma_fg),
                            "sigma_bg": float(tg_sigma_bg),
                            "mu_fg": [float(x) for x in mu_fg],
                            "mu_bg": [float(x) for x in mu_bg],
                            "value_range": [0.0, 1.0],
                        }
                    )
            elif mode.startswith("gaussian"):
                mask_seed = _stable_mask_seed(idx) if "_stable_mask_seed" in globals() else int(flatidx_to_bitset(idx))
                for variant_idx in range(int(tg_variants_per_mask)):
                    render_seed = (
                        int(seed)
                        + int(grid_hw) * 10007
                        + int(mask_seed) * 97
                        + int(variant_idx) * 1000003
                    ) % 2147483647
                    mu_fg, mu_bg = sample_gaussian_color_centers(
                        seed=int(render_seed),
                        num_channels=int(tg_num_channels),
                        mean=float(ga_mean),
                        sigma=float(ga_sigma),
                    )
                    records.append(
                        {
                            "grid_hw": int(grid_hw),
                            "w": int(w_i),
                            "flat_idx": [int(x) for x in idx],
                            "bitset": int(flatidx_to_bitset(idx)),
                            "source": "random",
                            "probe_mode": str(mode),
                            "num_channels": int(tg_num_channels),
                            "render_seed": int(render_seed),
                            "sigma_fg": float(ga_sigma),
                            "sigma_bg": float(ga_sigma),
                            "mu_fg": [float(x) for x in mu_fg],
                            "mu_bg": [float(x) for x in mu_bg],
                            "value_range": [0.0, 1.0],
                        }
                    )
            else:
                records.append(
                    {
                        "grid_hw": int(grid_hw),
                        "w": int(w_i),
                        "flat_idx": [int(x) for x in idx],
                        "bitset": int(flatidx_to_bitset(idx)),
                        "source": "random",
                        "probe_mode": "binary",
                        "polarity": "white",
                        "background_fill": 0.0,
                        "active_fill": 1.0,
                    }
                )
            tried_local += 1

            if (
                bool(use_complement)
                and (max_records_i is None or len(records) < max_records_i)
            ):
                if mode.startswith("two_group"):
                    mask_seed = _stable_mask_seed(idx) if "_stable_mask_seed" in globals() else int(flatidx_to_bitset(idx))
                    for variant_idx in range(int(tg_variants_per_mask)):
                        render_seed = (
                            int(seed)
                            + int(grid_hw) * 10007
                            + int(mask_seed) * 97
                            + int(variant_idx) * 1000003
                            + 17
                        ) % 2147483647
                        mu_fg, mu_bg = sample_two_group_color_centers(
                            seed=int(render_seed),
                            d_min=float(tg_d_min),
                            d_max=float(tg_d_max),
                            num_channels=int(tg_num_channels),
                            center_range=(0.0, 1.0),
                        )
                        records.append(
                            {
                                "grid_hw": int(grid_hw),
                                "w": int(w_i),
                                "flat_idx": [int(x) for x in idx],
                                "bitset": int(flatidx_to_bitset(idx)),
                                "source": "random_complement",
                                "probe_mode": str(mode),
                                "num_channels": int(tg_num_channels),
                                "render_seed": int(render_seed),
                                "sigma_fg": float(tg_sigma_fg),
                                "sigma_bg": float(tg_sigma_bg),
                                "mu_fg": [float(x) for x in mu_fg],
                                "mu_bg": [float(x) for x in mu_bg],
                                "value_range": [0.0, 1.0],
                            }
                        )
                elif mode.startswith("gaussian"):
                    mask_seed = _stable_mask_seed(idx) if "_stable_mask_seed" in globals() else int(flatidx_to_bitset(idx))
                    for variant_idx in range(int(tg_variants_per_mask)):
                        render_seed = (
                            int(seed)
                            + int(grid_hw) * 10007
                            + int(mask_seed) * 97
                            + int(variant_idx) * 1000003
                            + 17
                        ) % 2147483647
                        mu_fg, mu_bg = sample_gaussian_color_centers(
                            seed=int(render_seed),
                            num_channels=int(tg_num_channels),
                            mean=float(ga_mean),
                            sigma=float(ga_sigma),
                        )
                        records.append(
                            {
                                "grid_hw": int(grid_hw),
                                "w": int(w_i),
                                "flat_idx": [int(x) for x in idx],
                                "bitset": int(flatidx_to_bitset(idx)),
                                "source": "random_complement",
                                "probe_mode": str(mode),
                                "num_channels": int(tg_num_channels),
                                "render_seed": int(render_seed),
                                "sigma_fg": float(ga_sigma),
                                "sigma_bg": float(ga_sigma),
                                "mu_fg": [float(x) for x in mu_fg],
                                "mu_bg": [float(x) for x in mu_bg],
                                "value_range": [0.0, 1.0],
                            }
                        )
                else:
                    records.append(
                        {
                            "grid_hw": int(grid_hw),
                            "w": int(w_i),
                            "flat_idx": [int(x) for x in idx],
                            "bitset": int(flatidx_to_bitset(idx)),
                            "source": "random_complement",
                            "probe_mode": "binary",
                            "polarity": "black",
                            "background_fill": 1.0,
                            "active_fill": 0.0,
                        }
                    )

    return records


def sample_random_masks_for_d_batched_cpu(
    *,
    grid_hw: int,
    w_list: List[int],
    n_trials_per_w: int,
    seed: int,
    max_records: Optional[int] = None,
    seen_base_masks: Optional[Set[Tuple[int, ...]]] = None,
    batch_size: int = 2048,
) -> List[Dict[str, Any]]:
    """Sample unique binary masks with batched CPU random-score tensors.

    Selecting the smallest ``w`` IID continuous random scores is a uniform
    sample from the size-``w`` subsets, matching the distribution obtained
    from ``randperm(n)[:w]``. When the requested count exhausts a small
    combination space, direct enumeration avoids futile duplicate resampling.

    Returned records contain the binary-mask fields consumed by the FL pool
    builder plus a precomputed stable mask seed. The random stream differs
    from the legacy sampler, but the sampling distribution, budgets, and
    uniqueness rules do not.
    """
    n = int(grid_hw) * int(grid_hw)
    trials = max(0, int(n_trials_per_w))
    limit = None if max_records is None else max(0, int(max_records))
    if trials == 0 or limit == 0:
        return []

    batch_size_i = max(1, int(batch_size))
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    global_seen = seen_base_masks
    local_seen: Set[Tuple[int, ...]] = set()
    records: List[Dict[str, Any]] = []

    def remaining_capacity() -> int:
        if limit is None:
            return trials
        return max(0, int(limit) - len(records))

    def stable_seed_from_key(key: Tuple[int, ...]) -> int:
        acc = 0
        for pos, value in enumerate(key):
            acc = (
                acc * 1000003 + (int(pos) + 1) * 9176 + int(value) + 1
            ) % 2147483647
        return int(acc)

    def stable_seeds_from_rows(rows: torch.Tensor) -> List[int]:
        rows_i64 = rows.to(dtype=torch.int64)
        acc = torch.zeros((int(rows_i64.shape[0]),), dtype=torch.int64)
        for pos in range(int(rows_i64.shape[1])):
            acc = (
                acc * 1000003
                + (int(pos) + 1) * 9176
                + rows_i64[:, int(pos)]
                + 1
            ) % 2147483647
        return [int(x) for x in acc.tolist()]

    def add_mask(
        key: Tuple[int, ...],
        w_i: int,
        *,
        mask_seed: Optional[int] = None,
    ) -> bool:
        if key in local_seen:
            return False
        if global_seen is not None and key in global_seen:
            return False
        local_seen.add(key)
        if global_seen is not None:
            global_seen.add(key)
        idx = [int(x) for x in key]
        records.append(
            {
                "grid_hw": int(grid_hw),
                "w": int(w_i),
                "flat_idx": idx,
                "mask_seed": (
                    stable_seed_from_key(key)
                    if mask_seed is None
                    else int(mask_seed)
                ),
                "source": "random",
                "probe_mode": "binary",
                "polarity": "white",
                "background_fill": 0.0,
                "active_fill": 1.0,
            }
        )
        return True

    for w in w_list:
        if limit is not None and len(records) >= limit:
            break
        w_i = int(w)
        if w_i <= 0 or w_i >= n:
            continue

        combination_count = int(math.comb(n, w_i))
        unavailable = 0
        if global_seen is not None:
            unavailable = sum(
                1
                for key in global_seen
                if len(key) == w_i and all(0 <= int(x) < n for x in key)
            )
        available_count = max(0, combination_count - int(unavailable))
        target = min(
            int(trials),
            available_count,
            remaining_capacity(),
        )
        if target <= 0:
            continue

        if available_count <= target:
            candidates = [
                tuple(int(x) for x in combo)
                for combo in itertools.combinations(range(n), w_i)
                if global_seen is None or tuple(int(x) for x in combo) not in global_seen
            ]
            if candidates:
                order = torch.randperm(len(candidates), generator=rng).tolist()
                accepted_for_w = 0
                for pos in order:
                    if add_mask(candidates[int(pos)], w_i):
                        accepted_for_w += 1
                    if accepted_for_w >= target:
                        break
            continue

        accepted_for_w = 0
        while accepted_for_w < target:
            need = int(target) - int(accepted_for_w)
            draw_count = min(batch_size_i, max(64, 2 * need))
            scores = torch.rand((draw_count, n), generator=rng, dtype=torch.float32)
            indices = torch.topk(
                scores,
                k=int(w_i),
                dim=1,
                largest=False,
                sorted=False,
            ).indices
            indices = torch.sort(indices, dim=1).values
            seed_rows = stable_seeds_from_rows(indices)
            for row, mask_seed in zip(indices.tolist(), seed_rows):
                key = tuple(int(x) for x in row)
                if add_mask(key, w_i, mask_seed=int(mask_seed)):
                    accepted_for_w += 1
                    if accepted_for_w >= target:
                        break

    return records


@torch.no_grad()
def rescore_masks_topk(
    model,
    target_label: int,
    grid_hw: int,
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn: Callable[[str], Tuple[List[float], List[float]]],
    masks_flatidx: List[List[int]],
    batch_size: int,
    polarity: str = "white",
) -> List[float]:
    scores_out: List[float] = []
    if not masks_flatidx:
        return scores_out

    background_fill, active_fill = _polarity_fill_values(str(polarity))
    bs = max(1, int(batch_size))
    for j in range(0, len(masks_flatidx), bs):
        chunk = masks_flatidx[j : j + bs]
        grids = build_grids_from_flatidx_with_fill(
            flatidx_list=chunk,
            grid_hw=int(grid_hw),
            device=device,
            background_fill=float(background_fill),
            active_fill=float(active_fill),
        )
        s_t = score_grids_softmax_pt(
            model=model,
            grids=grids,
            out_hw=int(out_hw),
            target_label=int(target_label),
            dataset=dataset,
            device=device,
            norm_cfg_fn=norm_cfg_fn,
        )
        scores_out.extend([float(x) for x in s_t.detach().cpu()])
    return scores_out

def _fmt_pct(w: int, n_cells: int) -> str:
    return f"{(float(w) / float(n_cells)):.2f}"

def _log_sched_line(
    log_fn,
    *,
    label: int,
    tag: str,        # "pct" or "inv_pct"
    pct_s: str,      # e.g. "0.04"
    w: int,
    n_cells: int,
    tried: int,
    uniq: int,
    best: float,
    mid: Optional[str] = None
) -> None:
    label_s = f"{label:02d}"
    tag_s   = f"{tag:<7}"              # pct / inv_pct
    pct_s2  = f"{pct_s:>5}"
    w_s     = f"{w:2d}/{n_cells:2d}"
    w_s     = f"{w_s:>6}"

    log_fn(
        f"[w_schedule] "
        f"label={label_s} "
        f"{tag_s}={pct_s2} "
        f"w={w_s} "
        f"tried={tried:>7d} "
        f"uniq={uniq:>7d} "
        f"best={best:.3f}"
    )


@torch.no_grad()
def probe_search_by_w(
    model,
    target_label: int,
    grid_hw: int,
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn: Callable[[str], Tuple[List[float], List[float]]],
    w_schedule: Dict,                 # REQUIRED
    top_l_per_w: int,
    batch_size: int,
    log_every_batches: int = 50,
    log_fn: Optional[Callable[[str], None]] = None,
    method: str = "brute",
    method_cfg: Optional[Dict] = None,
) -> Dict[int, List[Dict]]:

    records_json: Dict[int, List[Dict]] = {}
    cfg_m = method_cfg or {}
    method_l = str(method).lower()
    if method_l in {"path", "evolutionary"}:
        raise ValueError(
            f"Unsupported search.method={method_l!r}. "
            "This release supports only the Pointillism random structured probing path."
        )

    plan = compile_w_plan(grid_hw=int(grid_hw), w_schedule=w_schedule)

    if log_fn is not None:
        log_fn(
            f"[w_schedule] n_cells={plan.n_cells} "
            f"search_ws={plan.search_ws} "
            f"use_complement={bool(w_schedule and w_schedule.get('use_complement', False))}"
        )

    # ----------------------------
    # BRUTE
    # ----------------------------
    if method_l == "brute":
        max_trials = int(cfg_m.get("max_trials", 200000))
        seed0 = int(cfg_m.get("seed", 123))
        top_k = int(cfg_m.get("top_k", max(int(top_l_per_w), 128)))

        for w in plan.search_ws:
            top_w, stats_w, top_i, stats_i = random_topk_for_w_bidir(
                model=model,
                target_label=int(target_label),
                grid_hw=int(grid_hw),
                w=int(w),
                out_hw=int(out_hw),
                dataset=dataset,
                device=device,
                norm_cfg_fn=norm_cfg_fn,
                top_k=int(top_k),
                batch_size=int(batch_size),
                max_trials=int(max_trials),
                seed=int(seed0) + 100000 * int(target_label) + 1000 * int(w),
                log_every_batches=int(log_every_batches),
                log_fn=log_fn,
            )

            # store searched w
            records_json[int(w)] = _format_records(top_w, int(w), int(grid_hw), int(top_l_per_w))

            # log pct (aligned)
            if log_fn is not None:
                _log_sched_line(
                    log_fn,
                    label=int(target_label),
                    tag="pct",
                    pct_s=_fmt_pct(int(w), int(plan.n_cells)),
                    w=int(w),
                    n_cells=int(plan.n_cells),
                    tried=int(stats_w.tried),
                    uniq=int(stats_w.unique),
                    best=float(stats_w.best),
                )

            # store + log complement only if schedule wants it
            w_dst = int(plan.n_cells - int(w))
            if bool(w_schedule.get("use_complement", False)) and (w_dst != int(w)):
                records_json[int(w_dst)] = _format_records(top_i, int(w_dst), int(grid_hw), int(top_l_per_w))

                if log_fn is not None:
                    _log_sched_line(
                        log_fn,
                        label=int(target_label),
                        tag="inv_pct",
                        pct_s=_fmt_pct(int(w_dst), int(plan.n_cells)),
                        w=int(w_dst),
                        n_cells=int(plan.n_cells),
                        tried=int(stats_i.tried),
                        uniq=int(stats_i.unique),
                        best=float(stats_i.best),
                    )

        return records_json

    raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def probe_search_by_w_multilabel(
    model,
    target_labels: List[int],
    grid_hw: int,
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn: Callable[[str], Tuple[List[float], List[float]]],
    w_schedule: Dict,
    top_l_per_w: int,
    batch_size: int,
    log_every_batches: int = 50,
    log_fn: Optional[Callable[[str], None]] = None,
    method: str = "brute",
    method_cfg: Optional[Dict] = None,
    return_pred_hist: bool = False,
    return_pred_records: bool = False,
) -> Tuple[Dict[int, Dict[int, List[Dict]]], Optional[Dict[str, Any]], Optional[Dict[int, List[Dict[str, Any]]]]]:
    method_l = str(method).lower()
    if method_l != "brute":
        raise ValueError(f"probe_search_by_w_multilabel only supports method='brute', got: {method_l}")

    labels = [int(label) for label in target_labels]
    records_by_label: Dict[int, Dict[int, List[Dict]]] = {label: {} for label in labels}
    cfg_m = method_cfg or {}
    plan = compile_w_plan(grid_hw=int(grid_hw), w_schedule=w_schedule)
    use_complement = bool(w_schedule and w_schedule.get("use_complement", False))
    pred_label_by_mask: Dict[int, int] = {}
    probe_w_by_mask: Dict[int, int] = {}
    pred_record_by_mask: Dict[int, Dict[str, Any]] = {}
    pred_num_classes = 0
    total_scored_rows = 0

    if log_fn is not None:
        log_fn(
            f"[w_schedule] n_cells={plan.n_cells} "
            f"search_ws={plan.search_ws} "
            f"use_complement={use_complement}"
        )

    max_trials = int(cfg_m.get("max_trials", 200000))
    seed0 = int(cfg_m.get("seed", 123))
    top_k = int(cfg_m.get("top_k", max(int(top_l_per_w), 128)))

    for w in plan.search_ws:
        top_w_by_label, stats_w, top_i_by_label, stats_i, pred_summary = random_topk_for_w_bidir_multilabel(
            model=model,
            target_labels=labels,
            grid_hw=int(grid_hw),
            w=int(w),
            out_hw=int(out_hw),
            dataset=dataset,
            device=device,
            norm_cfg_fn=norm_cfg_fn,
            top_k=int(top_k),
            batch_size=int(batch_size),
            max_trials=int(max_trials),
            seed=int(seed0) + 1000 * int(w),
        )

        if return_pred_hist:
            pred_num_classes = max(int(pred_num_classes), int(pred_summary.num_classes))
            total_scored_rows += int(pred_summary.num_scored_w) + int(pred_summary.num_scored_i)
            for bitset, pred_label in pred_summary.pred_by_mask_w.items():
                pred_label_by_mask[int(bitset)] = int(pred_label)
                probe_w_by_mask[int(bitset)] = int(w)

            w_dst_hist = int(plan.n_cells - int(w))
            if use_complement:
                for bitset, pred_label in pred_summary.pred_by_mask_i.items():
                    pred_label_by_mask[int(bitset)] = int(pred_label)
                    probe_w_by_mask[int(bitset)] = int(w_dst_hist)

        if return_pred_records:
            for bitset, rec in pred_summary.pred_records_w.items():
                pred_record_by_mask[int(bitset)] = dict(rec)
            for bitset, rec in pred_summary.pred_records_i.items():
                pred_record_by_mask[int(bitset)] = dict(rec)

        for label in labels:
            records_by_label[label][int(w)] = _format_records(
                top_w_by_label.get(label, []), int(w), int(grid_hw), int(top_l_per_w)
            )

        if log_fn is not None:
            best_map = {
                label: float(top_w_by_label[label][0][0]) if top_w_by_label.get(label) else 0.0
                for label in labels
            }
            prefix = (
                f"[w_schedule][all_labels] pct={_fmt_pct(int(w), int(plan.n_cells))} "
                f"w={int(w):2d}/{int(plan.n_cells):2d} tried={int(stats_w.tried):7d} "
                f"uniq={int(stats_w.unique):7d} best_by_label:"
            )
            log_fn(prefix)
            _log_label_best_map_block(
                log_fn,
                prefix="[w_schedule][all_labels]",
                label_to_best=best_map,
                target_labels=labels,
                chunk_size=5,
            )

        w_dst = int(plan.n_cells - int(w))
        if use_complement and (w_dst != int(w)):
            for label in labels:
                records_by_label[label][int(w_dst)] = _format_records(
                    top_i_by_label.get(label, []), int(w_dst), int(grid_hw), int(top_l_per_w)
                )

            if log_fn is not None:
                best_map = {
                    label: float(top_i_by_label[label][0][0]) if top_i_by_label.get(label) else 0.0
                    for label in labels
                }
                prefix = (
                    f"[w_schedule][all_labels] inv_pct={_fmt_pct(int(w_dst), int(plan.n_cells))} "
                    f"w={int(w_dst):2d}/{int(plan.n_cells):2d} tried={int(stats_i.tried):7d} "
                    f"uniq={int(stats_i.unique):7d} best_by_label:"
                )
                log_fn(prefix)
                _log_label_best_map_block(
                    log_fn,
                    prefix="[w_schedule][all_labels]",
                    label_to_best=best_map,
                    target_labels=labels,
                    chunk_size=5,
                )

    pred_hist_summary = None
    pred_records_by_label = None
    if return_pred_hist:
        pred_hist_counts_by_w: Dict[int, int] = {}
        pred_hist_by_label = [0 for _ in range(int(pred_num_classes))]
        for bitset, pred_label in pred_label_by_mask.items():
            w_value = int(probe_w_by_mask[int(bitset)])
            pred_hist_counts_by_w[w_value] = pred_hist_counts_by_w.get(w_value, 0) + 1
            if int(pred_label) >= len(pred_hist_by_label):
                pred_hist_by_label.extend([0 for _ in range(int(pred_label) + 1 - len(pred_hist_by_label))])
            pred_hist_by_label[int(pred_label)] += 1
        pred_hist_summary = {
            "num_classes": int(len(pred_hist_by_label)),
            "counts_by_w": {int(k): int(v) for k, v in sorted(pred_hist_counts_by_w.items())},
            "pred_hist": {int(i): int(v) for i, v in enumerate(pred_hist_by_label)},
            "total_unique": int(sum(pred_hist_counts_by_w.values())),
            "total_scored_rows": int(total_scored_rows),
            "sum_pred_hist": int(sum(pred_hist_by_label)),
            "use_complement": bool(use_complement),
        }

    if return_pred_records:
        pred_records_by_label = {}
        for bitset, rec in pred_record_by_mask.items():
            pred_label = int(rec.get("pred_label", -1))
            if pred_label < 0:
                continue
            pred_records_by_label.setdefault(pred_label, []).append(
                dict(rec, bitset=int(bitset))
            )
        for pred_label in pred_records_by_label:
            pred_records_by_label[pred_label].sort(key=lambda rec: float(rec.get("score", 0.0)), reverse=True)

    return records_by_label, pred_hist_summary, pred_records_by_label


# ============================
# exhaustive 4x4 analysis
# ============================
import itertools
import math
import time
from collections import Counter, defaultdict


def _bitset_to_flatidx_fixed(bitset: int, n_cells: int) -> List[int]:
    return [i for i in range(int(n_cells)) if (int(bitset) >> i) & 1]


def _bitset_matrix_u8(total_masks: int, n_cells: int):
    import numpy as np
    vals = np.arange(int(total_masks), dtype=np.uint32)[:, None]
    shifts = np.arange(int(n_cells), dtype=np.uint32)[None, :]
    return ((vals >> shifts) & 1).astype(np.uint8)


@torch.no_grad()
def exhaustive_score_all_masks(
    model,
    grid_hw: int,
    out_hw: int,
    dataset: str,
    device: torch.device,
    norm_cfg_fn: Callable[[str], Tuple[List[float], List[float]]],
    batch_size: int = 4096,
    save_all_probs: bool = True,
):
    n_cells = int(grid_hw) * int(grid_hw)
    total_masks = 1 << n_cells
    pred_hist = None
    all_records = []
    pred_hist_counter = Counter()
    for start in range(0, total_masks, int(batch_size)):
        end = min(total_masks, start + int(batch_size))
        bitsets = list(range(start, end))
        flatidx_list = [_bitset_to_flatidx_fixed(b, n_cells) for b in bitsets]
        grids = build_grids_from_flatidx(flatidx_list, grid_hw=int(grid_hw), device=device)
        probs = score_grids_softmax_all_labels_pt(
            model=model,
            grids=grids,
            out_hw=int(out_hw),
            dataset=dataset,
            device=device,
            norm_cfg_fn=norm_cfg_fn,
        )
        max_probs, preds = torch.max(probs, dim=1)
        preds_l = preds.detach().cpu().tolist()
        max_probs_l = max_probs.detach().cpu().tolist()
        probs_l = probs.detach().cpu().tolist() if save_all_probs else None
        for i, bitset in enumerate(bitsets):
            pred = int(preds_l[i])
            pred_hist_counter[pred] += 1
            rec = {
                "bitset": int(bitset),
                "pred_label": pred,
                "score": float(max_probs_l[i]),
                "w": int(len(flatidx_list[i])),
                "flat_idx": [int(x) for x in flatidx_list[i]],
                "coords": flatidx_to_coords(flatidx_list[i], int(grid_hw)),
            }
            if save_all_probs:
                rec["probs"] = [float(x) for x in probs_l[i]]
            all_records.append(rec)
    pred_hist = {int(k): int(v) for k, v in sorted(pred_hist_counter.items())}
    return {
        "total_masks": int(total_masks),
        "pred_hist": pred_hist,
        "all_records": all_records,
    }


def _records_to_arrays(all_records: List[Dict[str, Any]], grid_hw: int):
    import numpy as np
    n_cells = int(grid_hw) * int(grid_hw)
    total_masks = 1 << n_cells
    labels = np.full((total_masks,), -1, dtype=np.int16)
    scores = np.zeros((total_masks,), dtype=np.float32)
    has_probs = bool(all_records) and (
        ('probs' in all_records[0]) or ('all_probs' in all_records[0])
    )
    if has_probs:
        first_probs = all_records[0].get('probs', all_records[0].get('all_probs', []))
        num_classes = len(first_probs)
    else:
        num_classes = max(int(r['pred_label']) for r in all_records) + 1
    probs = np.zeros((total_masks, num_classes), dtype=np.float32) if has_probs else None
    for rec in all_records:
        b = int(rec['bitset'])
        labels[b] = int(rec['pred_label'])
        scores[b] = float(rec.get('score', rec.get('pred_confidence', 0.0)))
        if has_probs:
            row_probs = rec.get('probs', rec.get('all_probs', None))
            if row_probs is not None:
                probs[b] = np.asarray(row_probs, dtype=np.float32)
    bits = _bitset_matrix_u8(total_masks, n_cells)
    return bits, labels, scores, probs


def analyze_exhaustive_basic(*, all_records: List[Dict[str, Any]], num_classes: int, grid_hw: int):
    import numpy as np
    bits, labels, scores, probs = _records_to_arrays(all_records, grid_hw)
    total_masks, n_cells = bits.shape
    occ = bits.sum(axis=1).astype(np.int16)
    neighbor_idx = np.empty((total_masks, n_cells), dtype=np.int32)
    for j in range(n_cells):
        neighbor_idx[:, j] = np.arange(total_masks, dtype=np.int32) ^ (1 << j)
    nbr_labels = labels[neighbor_idx]
    same = (nbr_labels == labels[:, None])
    robustness = same.mean(axis=1).astype(np.float32)

    label_summaries = {}
    for c in range(int(num_classes)):
        mask_c = (labels == c)
        count = int(mask_c.sum())
        occ_hist = np.bincount(occ[mask_c], minlength=n_cells + 1).astype(int).tolist() if count > 0 else [0] * (n_cells + 1)
        rob_vals = robustness[mask_c] if count > 0 else np.asarray([], dtype=np.float32)
        label_summaries[str(c)] = {
            'count': count,
            'fraction': float(count) / float(max(total_masks, 1)),
            'occupancy_hist': {int(w): int(v) for w, v in enumerate(occ_hist)},
            'mean_robustness': float(rob_vals.mean()) if count > 0 else 0.0,
            'median_robustness': float(np.median(rob_vals)) if count > 0 else 0.0,
            'top_robust_bitsets': [int(x) for x in np.where(mask_c)[0][np.argsort(-rob_vals)[: min(10, count)]].tolist()] if count > 0 else [],
        }
    return {
        'label_summaries': label_summaries,
        'robustness_by_bitset': robustness.tolist(),
    }


def analyze_exhaustive_cell_enrichment(*, all_records: List[Dict[str, Any]], num_classes: int, grid_hw: int):
    import numpy as np
    bits, labels, scores, probs = _records_to_arrays(all_records, grid_hw)
    total_masks, n_cells = bits.shape
    out = {}
    for c in range(int(num_classes)):
        mask_c = (labels == c)
        mask_n = ~mask_c
        Xc = bits[mask_c]
        Xn = bits[mask_n]
        if Xc.shape[0] == 0:
            cell_stats = []
        else:
            p_on_c = Xc.mean(axis=0)
            p_on_n = Xn.mean(axis=0) if Xn.shape[0] > 0 else np.zeros((n_cells,), dtype=np.float32)
            p_off_c = 1.0 - p_on_c
            p_off_n = 1.0 - p_on_n
            cell_stats = []
            for j in range(n_cells):
                r = j // int(grid_hw)
                col = j % int(grid_hw)
                cell_stats.append({
                    'cell': int(j), 'row': int(r), 'col': int(col),
                    'p_on_label': float(p_on_c[j]), 'p_on_not_label': float(p_on_n[j]), 'enrichment_on': float(p_on_c[j] - p_on_n[j]),
                    'p_off_label': float(p_off_c[j]), 'p_off_not_label': float(p_off_n[j]), 'enrichment_off': float(p_off_c[j] - p_off_n[j]),
                    'count_on_label': int(Xc[:, j].sum()),
                })
        out[str(c)] = cell_stats
    return out


def analyze_exhaustive_pair_enrichment(*, all_records: List[Dict[str, Any]], num_classes: int, grid_hw: int, top_k_pairs: int = 20):
    import numpy as np
    bits, labels, scores, probs = _records_to_arrays(all_records, grid_hw)
    total_masks, n_cells = bits.shape
    pairs = [(i, j) for i in range(n_cells) for j in range(i + 1, n_cells)]
    out = {}
    for c in range(int(num_classes)):
        mask_c = (labels == c)
        mask_n = ~mask_c
        Xc = bits[mask_c]
        Xn = bits[mask_n]
        rows = []
        if Xc.shape[0] > 0:
            for i, j in pairs:
                p_c = float((Xc[:, i] & Xc[:, j]).mean())
                p_n = float((Xn[:, i] & Xn[:, j]).mean()) if Xn.shape[0] > 0 else 0.0
                rows.append({
                    'pair': [int(i), int(j)],
                    'p_on_label': p_c,
                    'p_on_not_label': p_n,
                    'enrichment_on': p_c - p_n,
                    'count_on_label': int((Xc[:, i] & Xc[:, j]).sum()),
                })
        rows.sort(key=lambda d: d['enrichment_on'], reverse=True)
        out[str(c)] = rows[: int(top_k_pairs)]
    return out

def _parse_subwindow_shapes(grid_hw: int, shapes: Optional[List[List[int]]] = None) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    if shapes is None or len(shapes) == 0:
        for h in range(1, int(grid_hw) + 1):
            for w in range(1, int(grid_hw) + 1):
                if h == int(grid_hw) and w == int(grid_hw):
                    continue
                if h == 1 and w == 1:
                    continue
                out.append((int(h), int(w)))
    else:
        for hw in shapes:
            if len(hw) != 2:
                continue
            h, w = int(hw[0]), int(hw[1])
            if h < 1 or w < 1:
                continue
            if h > int(grid_hw) or w > int(grid_hw):
                continue
            if h == int(grid_hw) and w == int(grid_hw):
                continue
            if h == 1 and w == 1:
                continue
            out.append((int(h), int(w)))

    # larger windows first
    out = sorted(set(out), key=lambda x: (-(x[0] * x[1]), -x[0], -x[1]))
    return out


def _iter_subwindow_defs(grid_hw: int, shapes: Optional[List[List[int]]] = None) -> List[Dict[str, Any]]:
    defs: List[Dict[str, Any]] = []
    for h, w in _parse_subwindow_shapes(int(grid_hw), shapes):
        for top in range(int(grid_hw) - int(h) + 1):
            for left in range(int(grid_hw) - int(w) + 1):
                idxs = []
                bit_pos = []
                k = 0
                for r in range(top, top + h):
                    for c in range(left, left + w):
                        idxs.append(int(r) * int(grid_hw) + int(c))
                        bit_pos.append((int(r), int(c), int(k)))
                        k += 1
                defs.append({
                    "shape_hw": [int(h), int(w)],
                    "top_left": [int(top), int(left)],
                    "flat_indices": idxs,
                    "n_bits": int(h * w),
                    "bit_positions": bit_pos,
                })
    return defs


def _encode_subwindow_codes(bits, sub_defs: List[Dict[str, Any]]):
    import numpy as np
    code_map: List[np.ndarray] = []
    for d in sub_defs:
        idxs = np.asarray(d["flat_indices"], dtype=np.int32)
        X = bits[:, idxs].astype(np.uint32)
        weights = (1 << np.arange(X.shape[1], dtype=np.uint32))[None, :]
        codes = (X * weights).sum(axis=1).astype(np.int32)
        code_map.append(codes)
    return code_map


def _pattern_code_to_rows(code: int, h: int, w: int) -> List[List[int]]:
    rows: List[List[int]] = []
    for r in range(int(h)):
        row = []
        for c in range(int(w)):
            bit_idx = int(r) * int(w) + int(c)
            row.append(int((int(code) >> bit_idx) & 1))
        rows.append(row)
    return rows


def _pattern_code_to_string(code: int, h: int, w: int) -> str:
    rows = _pattern_code_to_rows(int(code), int(h), int(w))
    return "/".join("".join(str(int(v)) for v in row) for row in rows)


def analyze_exhaustive_subwindow_motifs(
    *,
    all_records: List[Dict[str, Any]],
    num_classes: int,
    grid_hw: int,
    subwindow_shapes: Optional[List[List[int]]] = None,
    motif_top_k_per_label: int = 40,
    motif_min_count_in_label: int = 8,
    motif_min_enrichment: float = 0.0,
    motif_use_occ_control: bool = False,
):
    import numpy as np

    bits, labels, scores, probs = _records_to_arrays(all_records, grid_hw)
    total_masks, n_cells = bits.shape
    occ = bits.sum(axis=1).astype(np.int16)

    sub_defs = _iter_subwindow_defs(int(grid_hw), subwindow_shapes)
    code_map = _encode_subwindow_codes(bits, sub_defs)

    out: Dict[str, List[Dict[str, Any]]] = {}

    for c in range(int(num_classes)):
        idx_c = np.where(labels == c)[0]
        idx_n = np.where(labels != c)[0]

        if idx_c.size == 0:
            out[str(c)] = []
            continue

        rows: List[Dict[str, Any]] = []

        for def_idx, d in enumerate(sub_defs):
            h, w = int(d["shape_hw"][0]), int(d["shape_hw"][1])
            top, left = int(d["top_left"][0]), int(d["top_left"][1])
            n_bits_local = int(d["n_bits"])
            n_codes = 1 << n_bits_local

            codes = code_map[def_idx]
            codes_c = codes[idx_c]
            codes_n = codes[idx_n]

            cnt_c = np.bincount(codes_c, minlength=n_codes).astype(np.int64)
            cnt_n = np.bincount(codes_n, minlength=n_codes).astype(np.int64)

            p_c = cnt_c / float(max(idx_c.size, 1))
            p_n = cnt_n / float(max(idx_n.size, 1))
            enrich = p_c - p_n

            occ_enrich = None
            if bool(motif_use_occ_control):
                acc = 0.0
                used = 0
                for w_total in range(int(n_cells) + 1):
                    idx_c_w = idx_c[occ[idx_c] == w_total]
                    idx_n_w = idx_n[occ[idx_n] == w_total]
                    if idx_c_w.size == 0 or idx_n_w.size == 0:
                        continue
                    cnt_c_w = np.bincount(codes[idx_c_w], minlength=n_codes).astype(np.float64)
                    cnt_n_w = np.bincount(codes[idx_n_w], minlength=n_codes).astype(np.float64)
                    p_c_w = cnt_c_w / float(max(idx_c_w.size, 1))
                    p_n_w = cnt_n_w / float(max(idx_n_w.size, 1))
                    acc += (p_c_w - p_n_w)
                    used += 1
                if used > 0:
                    occ_enrich = acc / float(used)

            for code in range(n_codes):
                count_in = int(cnt_c[code])
                if count_in < int(motif_min_count_in_label):
                    continue

                enrich_val = float(enrich[code])
                if enrich_val < float(motif_min_enrichment):
                    continue

                p_in = float(p_c[code])
                p_out = float(p_n[code])
                precision = float(count_in) / float(max(count_in + int(cnt_n[code]), 1))
                occ_enrich_val = float(occ_enrich[code]) if occ_enrich is not None else None

                score = enrich_val * p_in
                if occ_enrich_val is not None:
                    score = float(0.5 * score + 0.5 * (occ_enrich_val * p_in))

                rows.append({
                    "shape_hw": [int(h), int(w)],
                    "top_left": [int(top), int(left)],
                    "pattern_code": int(code),
                    "pattern_str": _pattern_code_to_string(int(code), int(h), int(w)),
                    "pattern_rows": _pattern_code_to_rows(int(code), int(h), int(w)),
                    "count_in_label": int(count_in),
                    "count_not_label": int(cnt_n[code]),
                    "p_in_label": p_in,
                    "p_not_label": p_out,
                    "enrichment": enrich_val,
                    "precision": precision,
                    "score": float(score),
                    "occ_enrichment": occ_enrich_val,
                })

        rows.sort(
            key=lambda d: (
                float(d["score"]),
                float(d["enrichment"]),
                float(d["p_in_label"]),
                int(d["count_in_label"]),
            ),
            reverse=True,
        )
        out[str(c)] = rows[: int(motif_top_k_per_label)]

    return out

def _connected_subsets_from_bitset(bitset: int, grid_hw: int, size_min: int, size_max: int):
    active = _bitset_to_flatidx_fixed(int(bitset), int(grid_hw) * int(grid_hw))
    active_set = set(active)
    if not active:
        return []
    nbr = {}
    for idx in active:
        r, c = divmod(idx, int(grid_hw))
        nn = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < int(grid_hw) and 0 <= cc < int(grid_hw):
                jj = rr * int(grid_hw) + cc
                if jj in active_set:
                    nn.append(jj)
        nbr[idx] = nn
    found = set()
    out = []
    for start in active:
        stack = [(frozenset([start]), set(nbr[start]))]
        while stack:
            cells, frontier = stack.pop()
            if int(size_min) <= len(cells) <= int(size_max):
                key = tuple(sorted(cells))
                if key not in found:
                    found.add(key)
                    out.append(list(key))
            if len(cells) >= int(size_max):
                continue
            for nxt in list(frontier):
                new_cells = set(cells)
                new_cells.add(nxt)
                new_frontier = set(frontier)
                new_frontier.discard(nxt)
                new_frontier.update(nbr[nxt])
                new_frontier.difference_update(new_cells)
                stack.append((frozenset(new_cells), new_frontier))
    return out


def _canonicalize_seed(seed_cells: List[int], grid_hw: int):
    coords = [(int(x) // int(grid_hw), int(x) % int(grid_hw)) for x in seed_cells]
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    norm = sorted((r - min_r, c - min_c) for r, c in coords)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return {'shape_cells': [(int(r), int(c)) for r, c in norm], 'bbox_hw': [int(h), int(w)]}


def _shape_to_key(shape_cells):
    return tuple(sorted((int(r), int(c)) for r, c in shape_cells))


def _translated_bitsets_for_shape(shape_cells, bbox_hw, grid_hw: int):
    h, w = int(bbox_hw[0]), int(bbox_hw[1])
    out = []
    for r0 in range(int(grid_hw) - h + 1):
        for c0 in range(int(grid_hw) - w + 1):
            bitset = 0
            flat_idx = []
            for r, c in shape_cells:
                rr = r0 + int(r)
                cc = c0 + int(c)
                idx = rr * int(grid_hw) + cc
                bitset |= (1 << idx)
                flat_idx.append(idx)
            out.append({'bitset': int(bitset), 'flat_idx': sorted(flat_idx), 'offset': [int(r0), int(c0)]})
    return out


def analyze_exhaustive_seed_motifs(
    *,
    all_records: List[Dict[str, Any]],
    num_classes: int,
    grid_hw: int,
    anchor_top_k: int = 8,
    seed_sizes: List[int] = [2,3,4],
    seed_top_k: int = 20,
):
    import numpy as np
    bits, labels, scores, probs = _records_to_arrays(all_records, grid_hw)
    total_masks, n_cells = bits.shape
    # robustness
    neighbor_idx = np.empty((total_masks, n_cells), dtype=np.int32)
    for j in range(n_cells):
        neighbor_idx[:, j] = np.arange(total_masks, dtype=np.int32) ^ (1 << j)
    robustness = (labels[neighbor_idx] == labels[:, None]).mean(axis=1).astype(np.float32)
    has_probs = probs is not None

    out = {}
    for c in range(int(num_classes)):
        idx_c = np.where(labels == c)[0]
        if idx_c.size == 0:
            out[str(c)] = []
            continue
        # robust anchors first
        order = np.argsort(-robustness[idx_c])
        anchors = idx_c[order[: min(int(anchor_top_k), idx_c.size)]]
        seed_counter = {}
        for bitset in anchors.tolist():
            subs = _connected_subsets_from_bitset(int(bitset), int(grid_hw), min(seed_sizes), max(seed_sizes))
            for sub in subs:
                can = _canonicalize_seed(sub, int(grid_hw))
                key = _shape_to_key(can['shape_cells'])
                item = seed_counter.setdefault(key, {
                    'shape_cells': can['shape_cells'],
                    'bbox_hw': can['bbox_hw'],
                    'anchor_hits': 0,
                    'anchor_bitsets': [],
                })
                item['anchor_hits'] += 1
                item['anchor_bitsets'].append(int(bitset))
        rows = []
        for key, item in seed_counter.items():
            translations = _translated_bitsets_for_shape(item['shape_cells'], item['bbox_hw'], int(grid_hw))
            bitsets_t = [t['bitset'] for t in translations]
            pred_t = labels[np.asarray(bitsets_t, dtype=np.int32)]
            hit_mask = (pred_t == c)
            hit_rate = float(hit_mask.mean()) if len(bitsets_t) > 0 else 0.0
            mean_conf = float(probs[np.asarray(bitsets_t, dtype=np.int32), c].mean()) if has_probs and len(bitsets_t) > 0 else 0.0
            coverage_label = float(np.mean(np.all(bits[idx_c][:, [r * int(grid_hw) + cc for r,cc in item['shape_cells']]] == 1, axis=1))) if idx_c.size > 0 else 0.0
            idx_n = np.where(labels != c)[0]
            coverage_not = float(np.mean(np.all(bits[idx_n][:, [r * int(grid_hw) + cc for r,cc in item['shape_cells']]] == 1, axis=1))) if idx_n.size > 0 else 0.0
            rows.append({
                'shape_cells': [[int(r), int(cc)] for r, cc in item['shape_cells']],
                'bbox_hw': [int(item['bbox_hw'][0]), int(item['bbox_hw'][1])],
                'size': int(len(item['shape_cells'])),
                'anchor_hits': int(item['anchor_hits']),
                'coverage_in_label': float(coverage_label),
                'coverage_not_label': float(coverage_not),
                'translation_total': int(len(bitsets_t)),
                'translation_hit_rate': float(hit_rate),
                'translation_mean_conf': float(mean_conf),
                'best_translation_bitset': int(bitsets_t[int(np.argmax(probs[np.asarray(bitsets_t, dtype=np.int32), c]))]) if has_probs and len(bitsets_t) > 0 else int(bitsets_t[0]) if bitsets_t else -1,
                'score': float(0.45 * coverage_label + 0.35 * hit_rate + 0.20 * mean_conf - 0.15 * coverage_not),
            })
        rows.sort(key=lambda d: d['score'], reverse=True)
        # dedup by same shape key and translation score already collapsed
        out[str(c)] = rows[: int(seed_top_k)]
    return out


def analyze_exhaustive_structure(
    *,
    all_records: List[Dict[str, Any]],
    num_classes: int,
    grid_hw: int,
    run_basic_stats: bool = True,
    run_cell_enrichment: bool = True,
    run_pair_enrichment: bool = True,
    run_seed_motif_analysis: bool = True,
    run_subwindow_motif_analysis: bool = True,
    top_k_pairs: int = 20,
    anchor_top_k: int = 8,
    seed_sizes: List[int] = [2,3,4],
    seed_top_k: int = 20,
    subwindow_shapes: Optional[List[List[int]]] = None,
    motif_top_k_per_label: int = 40,
    motif_min_count_in_label: int = 8,
    motif_min_enrichment: float = 0.0,
    motif_use_occ_control: bool = False,
):
    t_all0 = time.time()
    timings = {}
    merged = {'label_summaries': {str(c): {} for c in range(int(num_classes))}}

    if run_basic_stats:
        t0 = time.time(); basic = analyze_exhaustive_basic(all_records=all_records, num_classes=num_classes, grid_hw=grid_hw); timings['basic'] = time.time() - t0
        for c in range(int(num_classes)):
            merged['label_summaries'][str(c)].update(basic['label_summaries'][str(c)])
        merged['robustness_by_bitset'] = basic['robustness_by_bitset']
    if run_cell_enrichment:
        t0 = time.time(); cell = analyze_exhaustive_cell_enrichment(all_records=all_records, num_classes=num_classes, grid_hw=grid_hw); timings['cell'] = time.time() - t0
        for c in range(int(num_classes)):
            merged['label_summaries'][str(c)]['cell_stats'] = cell[str(c)]
    if run_pair_enrichment:
        t0 = time.time(); pair = analyze_exhaustive_pair_enrichment(all_records=all_records, num_classes=num_classes, grid_hw=grid_hw, top_k_pairs=top_k_pairs); timings['pair'] = time.time() - t0
        for c in range(int(num_classes)):
            merged['label_summaries'][str(c)]['top_pairs_on'] = pair[str(c)]
    if run_seed_motif_analysis:
        t0 = time.time(); seed = analyze_exhaustive_seed_motifs(all_records=all_records, num_classes=num_classes, grid_hw=grid_hw, anchor_top_k=anchor_top_k, seed_sizes=seed_sizes, seed_top_k=seed_top_k); timings['seed_translate'] = time.time() - t0
        for c in range(int(num_classes)):
            merged['label_summaries'][str(c)]['top_functional_seeds'] = seed[str(c)]

    if run_subwindow_motif_analysis:
        t0 = time.time()
        motif = analyze_exhaustive_subwindow_motifs(
            all_records=all_records,
            num_classes=num_classes,
            grid_hw=grid_hw,
            subwindow_shapes=subwindow_shapes,
            motif_top_k_per_label=motif_top_k_per_label,
            motif_min_count_in_label=motif_min_count_in_label,
            motif_min_enrichment=motif_min_enrichment,
            motif_use_occ_control=motif_use_occ_control,
        )
        timings['subwindow_motif'] = time.time() - t0
        for c in range(int(num_classes)):
            merged['label_summaries'][str(c)]['top_subwindow_motifs'] = motif[str(c)]

    merged['timings'] = {k: float(v) for k, v in timings.items()}
    merged['analysis_elapsed'] = float(time.time() - t_all0)
    return merged

def compute_softmax_entropy(
    probs: List[float],
    eps: float = 1e-12,
) -> float:
    """
    Shannon entropy of a probability vector.
    Natural log version.
    """
    if probs is None or len(probs) == 0:
        return 0.0

    p = torch.as_tensor(probs, dtype=torch.float32)
    p = torch.clamp(p, min=float(eps))
    p = p / torch.clamp(p.sum(), min=float(eps))
    ent = -(p * torch.log(p)).sum()
    return float(ent.item())


def compute_top2_stats(
    probs: List[float],
) -> Dict[str, Any]:
    """
    Returns top-1 / top-2 labels and probs, plus margin.
    """
    if probs is None or len(probs) == 0:
        return {
            "top1_label": -1,
            "top2_label": -1,
            "top1_prob": 0.0,
            "top2_prob": 0.0,
            "margin": 0.0,
        }

    p = torch.as_tensor(probs, dtype=torch.float32)
    if int(p.numel()) == 1:
        return {
            "top1_label": 0,
            "top2_label": -1,
            "top1_prob": float(p[0].item()),
            "top2_prob": 0.0,
            "margin": float(p[0].item()),
        }

    vals, idx = torch.topk(p, k=2, largest=True, sorted=True)
    top1_prob = float(vals[0].item())
    top2_prob = float(vals[1].item())
    top1_label = int(idx[0].item())
    top2_label = int(idx[1].item())
    return {
        "top1_label": int(top1_label),
        "top2_label": int(top2_label),
        "top1_prob": float(top1_prob),
        "top2_prob": float(top2_prob),
        "margin": float(top1_prob - top2_prob),
    }


def annotate_records_with_uncertainty(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Append uncertainty-related fields to scored records:
      - entropy
      - top1_label
      - top2_label
      - top1_prob
      - top2_prob
      - margin
    """
    out: List[Dict[str, Any]] = []
    for rec in records:
        new_rec = dict(rec)
        probs = new_rec.get("all_probs", None)
        entropy = compute_softmax_entropy(probs)
        top2 = compute_top2_stats(probs)

        new_rec["entropy"] = float(entropy)
        new_rec["top1_label"] = int(top2["top1_label"])
        new_rec["top2_label"] = int(top2["top2_label"])
        new_rec["top1_prob"] = float(top2["top1_prob"])
        new_rec["top2_prob"] = float(top2["top2_prob"])
        new_rec["margin"] = float(top2["margin"])
        out.append(new_rec)
    return out


def summarize_entropy_by_d(
    *,
    all_scored_records_by_d: Dict[int, List[Dict[str, Any]]],
) -> Dict[int, Dict[str, Any]]:
    """
    Entropy summary per grid resolution.
    """
    out: Dict[int, Dict[str, Any]] = {}
    for d in sorted(all_scored_records_by_d.keys()):
        recs = all_scored_records_by_d.get(int(d), [])
        if not recs:
            out[int(d)] = {
                "count": 0,
                "entropy_mean": 0.0,
                "entropy_median": 0.0,
                "entropy_std": 0.0,
                "entropy_q10": 0.0,
                "entropy_q90": 0.0,
                "margin_mean": 0.0,
                "margin_median": 0.0,
            }
            continue

        ent = torch.tensor(
            [float(rec.get("entropy", 0.0)) for rec in recs],
            dtype=torch.float32,
        )
        margin = torch.tensor(
            [float(rec.get("margin", 0.0)) for rec in recs],
            dtype=torch.float32,
        )

        out[int(d)] = {
            "count": int(ent.numel()),
            "entropy_mean": float(ent.mean().item()),
            "entropy_median": float(ent.median().item()),
            "entropy_std": float(ent.std(unbiased=False).item()) if ent.numel() > 1 else 0.0,
            "entropy_q10": float(torch.quantile(ent, 0.10).item()) if ent.numel() > 0 else 0.0,
            "entropy_q90": float(torch.quantile(ent, 0.90).item()) if ent.numel() > 0 else 0.0,
            "margin_mean": float(margin.mean().item()),
            "margin_median": float(margin.median().item()),
        }
    return out


def summarize_entropy_by_label_and_d(
    *,
    all_scored_records_by_d: Dict[int, List[Dict[str, Any]]],
    num_classes: int,
    label_key: str = "pred_label",
) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """
    Entropy summary per grid resolution and label.
    label_key usually 'pred_label'.
    """
    out: Dict[int, Dict[int, Dict[str, Any]]] = {}

    for d in sorted(all_scored_records_by_d.keys()):
        recs = all_scored_records_by_d.get(int(d), [])
        out[int(d)] = {}

        for label in range(int(num_classes)):
            sub = [rec for rec in recs if int(rec.get(label_key, -1)) == int(label)]
            if not sub:
                out[int(d)][int(label)] = {
                    "count": 0,
                    "entropy_mean": 0.0,
                    "entropy_median": 0.0,
                    "entropy_std": 0.0,
                    "entropy_q10": 0.0,
                    "entropy_q90": 0.0,
                    "margin_mean": 0.0,
                    "margin_median": 0.0,
                }
                continue

            ent = torch.tensor(
                [float(rec.get("entropy", 0.0)) for rec in sub],
                dtype=torch.float32,
            )
            margin = torch.tensor(
                [float(rec.get("margin", 0.0)) for rec in sub],
                dtype=torch.float32,
            )

            out[int(d)][int(label)] = {
                "count": int(ent.numel()),
                "entropy_mean": float(ent.mean().item()),
                "entropy_median": float(ent.median().item()),
                "entropy_std": float(ent.std(unbiased=False).item()) if ent.numel() > 1 else 0.0,
                "entropy_q10": float(torch.quantile(ent, 0.10).item()) if ent.numel() > 0 else 0.0,
                "entropy_q90": float(torch.quantile(ent, 0.90).item()) if ent.numel() > 0 else 0.0,
                "margin_mean": float(margin.mean().item()),
                "margin_median": float(margin.median().item()),
            }

    return out


def collect_edge_cases_from_records(
    *,
    all_scored_records_by_d: Dict[int, List[Dict[str, Any]]],
    num_classes: int,
    top_k_per_pair: int = 64,
) -> Dict[str, Any]:
    """
    Collect edge-case probes by class pair, based on the top-2 predicted labels.
    Pair key is unordered: (min(top1, top2), max(top1, top2)).

    edge_score:
      smaller margin is more boundary-like,
      larger entropy is more ambiguous,
      so we rank by (margin asc, entropy desc).
    """
    top_records_by_pair: Dict[str, List[Dict[str, Any]]] = {}
    summary_by_pair: Dict[str, Dict[str, Any]] = {}

    pair_to_records: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)

    for d in sorted(all_scored_records_by_d.keys()):
        recs = all_scored_records_by_d.get(int(d), [])
        for rec in recs:
            a = int(rec.get("top1_label", -1))
            b = int(rec.get("top2_label", -1))
            if a < 0 or b < 0 or a == b:
                continue
            if a >= int(num_classes) or b >= int(num_classes):
                continue

            p0, p1 = sorted([int(a), int(b)])
            new_rec = dict(rec)
            new_rec["edge_pair"] = [int(p0), int(p1)]
            new_rec["edge_score"] = float(new_rec.get("margin", 0.0))
            pair_to_records[(int(p0), int(p1))].append(new_rec)

    for pair in sorted(pair_to_records.keys()):
        recs = list(pair_to_records[pair])
        recs.sort(
            key=lambda r: (
                float(r.get("margin", 1e9)),
                -float(r.get("entropy", 0.0)),
                -float(r.get("top2_prob", 0.0)),
            )
        )

        keep = recs[: max(1, int(top_k_per_pair))]
        key = f"{int(pair[0])}_{int(pair[1])}"
        top_records_by_pair[key] = keep

        ent = torch.tensor([float(r.get("entropy", 0.0)) for r in keep], dtype=torch.float32)
        margin = torch.tensor([float(r.get("margin", 0.0)) for r in keep], dtype=torch.float32)
        grid_hist = defaultdict(int)
        for r in keep:
            grid_hist[int(r.get("grid_hw", -1))] += 1

        summary_by_pair[key] = {
            "pair": [int(pair[0]), int(pair[1])],
            "count_total": int(len(recs)),
            "count_kept": int(len(keep)),
            "margin_mean": float(margin.mean().item()) if margin.numel() > 0 else 0.0,
            "margin_median": float(margin.median().item()) if margin.numel() > 0 else 0.0,
            "entropy_mean": float(ent.mean().item()) if ent.numel() > 0 else 0.0,
            "entropy_median": float(ent.median().item()) if ent.numel() > 0 else 0.0,
            "grid_hist": {int(k): int(v) for k, v in sorted(grid_hist.items())},
        }

    return {
        "summary_by_pair": summary_by_pair,
        "top_records_by_pair": top_records_by_pair,
    }
