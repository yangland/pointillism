# backdoor/badnet.py
from __future__ import annotations

import numpy as np
import torch
from typing import Optional, Sequence, Tuple

PATTERN_SIZE_BY_TYPE = {
    "one_pixel": (1, 1),
    "square3": (3, 3),
    "square5": (5, 5),
    "diag3": (3, 3),
    "badnet_corner": (3, 3),
    "circle": (3, 3),
}


def resolve_pattern_size(
    *,
    pattern_type: str = "badnet_corner",
    pattern_size: Optional[Tuple[int, int]] = None,
    pattern_offsets: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[int, int]:
    """
    Resolve the trigger footprint used for placement.

    For the built-in named trigger types, we always use their canonical size so
    YAML only needs `pattern_type`. Explicit `pattern_size` is only used for
    custom offset-based patterns or unknown trigger names.
    """
    pt = (pattern_type or "badnet_corner").lower()
    if pt in PATTERN_SIZE_BY_TYPE:
        return PATTERN_SIZE_BY_TYPE[pt]

    if pattern_size is not None:
        return (int(pattern_size[0]), int(pattern_size[1]))

    if pattern_offsets:
        max_x = max(int(dx) for dx, _ in pattern_offsets)
        max_y = max(int(dy) for _, dy in pattern_offsets)
        return (max_x + 1, max_y + 1)

    return (3, 3)


@torch.no_grad()
def add_pattern_trigger(
    img_tensor: torch.Tensor,
    *,
    origin_x: int = 0,
    origin_y: int = 0,
    offsets: Optional[Sequence[Tuple[int, int]]] = None,
    value: float = 1.0,
    pattern_type: str = "badnet_corner",
) -> torch.Tensor:
    t = img_tensor.clone()
    H, W = t.shape[-2], t.shape[-1]

    if offsets is None:
        pt = (pattern_type or "badnet_corner").lower()
        if pt == "square3":
            offsets = [(dx, dy) for dy in range(3) for dx in range(3)]
        elif pt == "square5":
            offsets = [(dx, dy) for dy in range(5) for dx in range(5)]
        elif pt == "diag3":
            offsets = [(0, 0), (1, 1), (2, 2)]
        elif pt == "badnet_corner":
            offsets = [(1, 1), (2, 2), (0, 2), (2, 0)]
        elif pt == "circle":
            offsets = [(dx, dy) for dy in range(3) for dx in range(3) if (dx, dy) != (1, 1)]
        elif pt == "one_pixel":
            offsets = [(0, 0)]
        else:
            raise ValueError(f"Unknown pattern_type: {pattern_type}")

    for dx, dy in offsets:
        xi = origin_x + int(dx)
        yi = origin_y + int(dy)
        if xi < 0 or yi < 0 or xi >= W or yi >= H:
            continue
        if t.ndim == 3:
            t[:, yi, xi] = float(value)
        else:
            t[yi, xi] = float(value)
    return t


def resolve_origin_from_pos(
    img_tensor: torch.Tensor,
    *,
    pattern_size: Tuple[int, int] = (3, 3),
    pattern_pos: str = "bottom_right",
    padding: int = 0,
) -> Tuple[int, int]:
    W, H = img_tensor.shape[-1], img_tensor.shape[-2]
    pw, ph = int(pattern_size[0]), int(pattern_size[1])

    left = int(padding)
    right = max(0, W - pw - int(padding))
    top = int(padding)
    bottom = max(0, H - ph - int(padding))
    center_x = max(0, (W - pw) // 2)
    center_y = max(0, (H - ph) // 2)

    pos = (pattern_pos or "").lower()
    if pos == "top_left": return (left, top)
    if pos == "top_center": return (center_x, top)
    if pos == "top_right": return (right, top)
    if pos == "center_left": return (left, center_y)
    if pos in ("center", "middle"): return (center_x, center_y)
    if pos == "center_right": return (right, center_y)
    if pos == "bottom_left": return (left, bottom)
    if pos == "bottom_center": return (center_x, bottom)
    return (right, bottom)


class PatternBackdoorWrapper(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        *,
        target_label: int,
        poison_frac: float = 0.1,
        pattern_pos: str = "bottom_right",
        pattern_padding: int = 0,
        pattern_size: Optional[Tuple[int, int]] = None,
        pattern_offsets=None,
        value: float = 1.0,
        seed: int = 0,
        pattern_type: str = "badnet_corner",
        apply_trigger: bool = True,   # NEW
        apply_relabel: bool = True,   # NEW
    ):
        self.base = base_dataset
        self.tgt = int(target_label)
        self.poison_frac = float(poison_frac)
        self.pattern_offsets = pattern_offsets
        self.value = float(value)
        self.pattern_type = (pattern_type or "badnet_corner").lower()
        self.apply_trigger = bool(apply_trigger)
        self.apply_relabel = bool(apply_relabel)

        self.pattern_size = resolve_pattern_size(
            pattern_type=self.pattern_type,
            pattern_size=pattern_size,
            pattern_offsets=self.pattern_offsets,
        )

        sample_img, _ = self.base[0]
        self.origin = resolve_origin_from_pos(
            sample_img,
            pattern_size=self.pattern_size,
            pattern_pos=pattern_pos,
            padding=pattern_padding,
        )

        rng = np.random.default_rng(int(seed))
        idxs = np.arange(len(self.base))
        rng.shuffle(idxs)
        self.poison_count = int(len(self.base) * self.poison_frac)
        self.poison_set = set(idxs[: self.poison_count].tolist())

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if idx in self.poison_set:
            if self.apply_trigger:
                x = add_pattern_trigger(
                    x,
                    origin_x=self.origin[0],
                    origin_y=self.origin[1],
                    offsets=self.pattern_offsets,
                    value=self.value,
                    pattern_type=self.pattern_type,
                )
            if self.apply_relabel:
                y = self.tgt
            return x, y
        return x, y


class OnePixelBackdoorWrapper(PatternBackdoorWrapper):
    """
    Compatibility wrapper for older FL attack plumbing.
    Internally this reuses the shared pattern trigger code with a 1-pixel pattern.
    """

    def __init__(
        self,
        base_dataset,
        *,
        target_label: int,
        poison_frac: float = 0.1,
        pattern_pos: str = "bottom_right",
        pattern_padding: int = 0,
        value: float = 1.0,
        seed: int = 0,
        apply_trigger: bool = True,
        apply_relabel: bool = True,
    ):
        super().__init__(
            base_dataset,
            target_label=int(target_label),
            poison_frac=float(poison_frac),
            pattern_pos=pattern_pos,
            pattern_padding=int(pattern_padding),
            pattern_size=(1, 1),
            pattern_offsets=None,
            value=float(value),
            seed=int(seed),
            pattern_type="one_pixel",
            apply_trigger=bool(apply_trigger),
            apply_relabel=bool(apply_relabel),
        )
        # Older code expects this attribute name.
        self.poison_idxs = self.poison_set
