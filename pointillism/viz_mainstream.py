
import torch
import math
import matplotlib.pyplot as plt
import json
from pathlib import Path

def get_k_real_samples_for_label(ds, target_label: int, k: int = 2):
    out = []
    n = len(ds)
    for i in range(n):
        x, y = ds[i]
        if int(y) != int(target_label):
            continue
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Dataset returns non-tensor x at index {i}: {type(x)}")
        out.append(x.detach().cpu())
        if len(out) >= int(k):
            break
    return out

def infer_hw_from_dataset(ds):
    x0, _ = ds[0]
    if not isinstance(x0, torch.Tensor) or x0.ndim != 3:
        raise ValueError(f"Expected dataset sample (C,H,W) tensor, got {type(x0)} shape={getattr(x0,'shape',None)}")
    return int(x0.shape[-2]), int(x0.shape[-1])

def flatidx_to_grid(flatidx, grid_hw: int, device: torch.device) -> torch.Tensor:
    g = torch.zeros(1, 1, int(grid_hw), int(grid_hw), device=device)
    for t in flatidx:
        ii = int(t)
        r = int(ii // int(grid_hw))
        c = int(ii % int(grid_hw))
        if 0 <= r < int(grid_hw) and 0 <= c < int(grid_hw):
            g[0, 0, r, c] = 1.0
    return g




def load_banks_from_state(out_dir: str):
    p = Path(out_dir) / "pointillism_state.json"
    if not p.exists():
        return None
    with open(p, "r") as f:
        st = json.load(f)
    banks = st.get("banks", None)
    if banks is None:
        return None

    # JSON keys may be strings
    banks2 = {}
    for k, v in banks.items():
        banks2[int(k)] = v
    return banks2


def _mask_to_rgb(mask_1hw: torch.Tensor, color_id: int) -> torch.Tensor:
    # mask_1hw: (1,H,W) in [0,1]
    out = torch.zeros((3, mask_1hw.shape[1], mask_1hw.shape[2]), dtype=mask_1hw.dtype)
    cid = int(color_id)
    if cid == 0:
        out[0] = mask_1hw[0]
        out[1] = mask_1hw[0]
        out[2] = mask_1hw[0]
    elif cid == 1:
        out[0] = mask_1hw[0]
    elif cid == 2:
        out[1] = mask_1hw[0]
    elif cid == 3:
        out[2] = mask_1hw[0]
    else:
        raise ValueError(f"Unknown color_id={cid}")
    return out

def _flatidx_layers_to_rgb_grid(flat_r, flat_g, flat_b, grid_hw: int, device: torch.device) -> torch.Tensor:
    gh = int(grid_hw)
    out = torch.zeros((3, gh, gh), device=device)

    for t in (flat_r or []):
        ii = int(t); r = ii // gh; c = ii % gh
        if 0 <= r < gh and 0 <= c < gh:
            out[0, r, c] = 1.0

    for t in (flat_g or []):
        ii = int(t); r = ii // gh; c = ii % gh
        if 0 <= r < gh and 0 <= c < gh:
            out[1, r, c] = 1.0

    for t in (flat_b or []):
        ii = int(t); r = ii // gh; c = ii % gh
        if 0 <= r < gh and 0 <= c < gh:
            out[2, r, c] = 1.0

    return out

def save_mainstream_one_row_per_label(
    *,
    out_dir: str,
    rnd: int,
    banks,
    grid_hw: int,
    probes_per_class: int,
    real_per_class: int,
    dataset,
    device: torch.device,
    tag: str = "mainstream",
):
    out_h, out_w = infer_hw_from_dataset(dataset)

    out_path_dir = Path(out_dir) / "plots_pointillism_mainstream" / f"iter_{int(rnd):06d}"
    out_path_dir.mkdir(parents=True, exist_ok=True)

    cols = int(real_per_class) + int(probes_per_class)
    if cols <= 0:
        return

    for lab in sorted(banks.keys()):
        entries = banks[lab] or []

        # real samples first
        real_samples = get_k_real_samples_for_label(dataset, target_label=int(lab), k=int(real_per_class))

        tiles = []

        for i, x_real in enumerate(real_samples[: int(real_per_class)]):
            x_r = x_real.unsqueeze(0)  # (1,C,H,W)
            if int(x_r.shape[-2]) != int(out_h) or int(x_r.shape[-1]) != int(out_w):
                x_r = torch.nn.functional.interpolate(x_r, size=(int(out_h), int(out_w)), mode="nearest")
            tiles.append((f"real{i}", x_r[0].detach().cpu()))

        # then probes from bank (already sorted best->worse)
        for idx, be in enumerate(entries[: int(probes_per_class)]):
            pid = int(be["probe_id"])
            acc_mean = float(be.get("acc_mean", be.get("ms", 0.0)))
            acc_mid = float(be.get("acc_mid", 0.0))

            # IMPORTANT: flatidx must be present in saved bank entry OR we reconstruct from pool/spec.
            # If your banks do not store flatidx, see section 3 below.
            flatidx = be.get("flatidx", None)
            if flatidx is None:
                # If not stored, we cannot render from bank alone; caller must pass an id2spec map.
                raise KeyError("Bank entry missing flatidx; store flatidx in banks or pass id2spec for reconstruction.")

            color_id = int(be.get("color_id", 0))

            # If dataset/model expects 3ch, render RGB probes; otherwise keep 1ch
            if len(real_samples) > 0 and int(real_samples[0].shape[0]) == 3:
                if color_id == 9:
                    if ("flatidx_r" not in be) or ("flatidx_g" not in be) or ("flatidx_b" not in be):
                        raise KeyError("color_id==9 requires flatidx_r/flatidx_g/flatidx_b in bank entry.")
                    rgb_grid = _flatidx_layers_to_rgb_grid(
                        be["flatidx_r"], be["flatidx_g"], be["flatidx_b"],
                        grid_hw=int(grid_hw), device=device
                    ).unsqueeze(0)  # (1,3,gh,gh)
                    x_rgb = torch.nn.functional.interpolate(rgb_grid, size=(int(out_h), int(out_w)), mode="nearest")  # (1,3,H,W)
                    x_tile = x_rgb[0]  # (3,H,W)
                else:
                    g = flatidx_to_grid(flatidx, grid_hw=int(grid_hw), device=device)
                    x = torch.nn.functional.interpolate(g, size=(int(out_h), int(out_w)), mode="nearest")  # (1,1,H,W)
                    x_tile = _mask_to_rgb(x[0], color_id)  # (3,H,W)
            else:
                g = flatidx_to_grid(flatidx, grid_hw=int(grid_hw), device=device)
                x = torch.nn.functional.interpolate(g, size=(int(out_h), int(out_w)), mode="nearest")  # (1,1,H,W)
                x_tile = x[0]  # (1,H,W)

        tiles.append((f"p{idx} $\\mu${100.0*acc_mean:.1f} md{100.0*acc_mid:.1f}", x_tile.detach().cpu()))


        # pad to fixed length
        if len(tiles) < cols and len(tiles) > 0:
            pad = cols - len(tiles)
            blank = torch.zeros_like(tiles[0][1])
            for _ in range(pad):
                tiles.append(("", blank))

        tiles = tiles[:cols]

        fig, axes = plt.subplots(1, cols, figsize=(2.0 * cols, 2.4))
        if cols == 1:
            axes = [axes]

        for cc in range(cols):
            ax = axes[cc]
            ax.axis("off")
            title, x_t = tiles[cc]
            if x_t.shape[0] == 1:
                ax.imshow(x_t[0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(x_t.permute(1, 2, 0).numpy())
            if title:
                ax.set_title(title)

        fig.tight_layout()
        fig.savefig(out_path_dir / f"{tag}_label{int(lab):02d}.png", dpi=150)
        plt.close(fig)
