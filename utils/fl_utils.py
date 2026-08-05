import random, os, numpy as np, torch
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Callable
from torch.utils.data import DataLoader
from models.resnet18 import resnet18
from models.resnet20 import resnet20, resnet32, resnet44, resnet56, resnet8
from models.mobilenetv2 import mobilenet_v2_cifar
from models.audio_cnn import audio_cnn
from models.small_cnn import small_cnn



LogFn = Callable[[str], None]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic mode:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # Fast mode:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def configure_runtime_device(device: str | torch.device | None) -> torch.device:
    """
    Align PyTorch's current CUDA device with the configured runtime device.

    This prevents hidden/default CUDA allocations (for example bare `.cuda()`,
    `torch.randn(..., device="cuda")`, or `torch.device("cuda")`) from landing on
    the default GPU when the experiment YAML explicitly requests a CUDA device.
    """
    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Configured CUDA device {resolved} but CUDA is not available.")
        index = resolved.index if resolved.index is not None else 0
        torch.cuda.set_device(index)
    return resolved
    
def build_model_fn(cfg):
    name = cfg["model"]["model_name"].lower()
    num_classes = int(cfg["model"]["num_classes"])
    in_ch = cfg["task"]["channels"]

    if name == "resnet20":
        def fn():
            return resnet20(num_classes=num_classes, in_channels=in_ch)
        return fn

    if name == "resnet8":
        def fn():
            return resnet8(num_classes=num_classes, in_channels=in_ch)
        return fn

    if name == "resnet32":
        def fn():
            return resnet32(num_classes=num_classes, in_channels=in_ch)
        return fn

    if name == "resnet44":
        def fn():
            return resnet44(num_classes=num_classes, in_channels=in_ch)
        return fn

    if name == "resnet56":
        def fn():
            return resnet56(num_classes=num_classes, in_channels=in_ch)
        return fn

    if name == "resnet18":
        def fn():
            return resnet18(num_classes=num_classes, in_channels=in_ch)
        return fn

    if name in ["mobilenetv2", "mobilenet_v2"]:
        def fn():
            return mobilenet_v2_cifar(
                num_classes=num_classes,
                in_channels=in_ch,
                width_mult=1.0,
            )
        return fn

    if name in ["smallcnn", "small_cnn"]:
        model_cfg = cfg.get("model", {}) or {}
        def fn():
            return small_cnn(
                num_classes=num_classes,
                in_channels=in_ch,
                width=int(model_cfg.get("width", 16)),
                dropout=float(model_cfg.get("dropout", 0.0)),
            )
        return fn

    if name in ["audio_cnn", "audiocnn"]:
        model_cfg = cfg.get("model", {}) or {}
        def fn():
            return audio_cnn(num_classes=num_classes, in_ch=in_ch, width=int(model_cfg.get("width", 128)), dropout=float(model_cfg.get("dropout", 0.2)))
        return fn

    raise ValueError(f"Unknown model name: {name}")


def _get_pixel_pos(attack_cfg: dict, task_cfg: dict, default_pos: str = "bottom_right") -> Tuple[int, int]:
    H, W = tuple(task_cfg.get("img_size", (28, 28)))
    pos = str(attack_cfg.get("pattern_pos", default_pos)).lower()
    if pos in ("bottom_right", "br"):
        return (H - 1, W - 1)
    if pos in ("bottom_left", "bl"):
        return (H - 1, 0)
    if pos in ("top_right", "tr"):
        return (0, W - 1)
    if pos in ("top_left", "tl"):
        return (0, 0)
    if pos in ("center", "middle"):
        return ((H - 1) // 2, (W - 1) // 2)
    return (H - 1, W - 1)


def probe_pixel_batch(
    log: LogFn,
    cfg: dict,
    xb: torch.Tensor,
    context: str,
    channel: int = 0,
) -> None:
    if not torch.is_tensor(xb):
        log(f"[probe][pixel] {context}: xb is not a tensor ({type(xb)})")
        return

    atk = cfg.get("attack", {})
    task = cfg.get("task", {})
    py, px = _get_pixel_pos(atk, task)
    try:
        pix_vals = xb[:, channel, py, px]
        log(
            f"[probe][pixel] {context}: pos=({py},{px}) "
            f"mean={pix_vals.mean().item():.3f} "
            f"min={pix_vals.min().item():.3f} "
            f"max={pix_vals.max().item():.3f}"
        )
    except Exception as e:
        log(f"[probe][pixel] {context}: failed to inspect pixel: {e!r}")


def probe_train_loader(
    log: LogFn,
    cfg: dict,
    loader: DataLoader,
    cid: int,
    target_label: Optional[int] = None,
    dataset_name: Optional[str] = None,
) -> None:
    try:
        xb, yb = next(iter(loader))
    except Exception as e:
        log(f"[probe][train] cid{cid}: failed to iterate loader: {e!r}")
        return

    ds_name = dataset_name or "unknown_dataset"
    log(f"[probe] cid{cid}: dataset={ds_name} atk={str(cfg.get('attack', {}).get('atk_name', '')).lower()}")

    if target_label is not None:
        try:
            yb_t = yb if torch.is_tensor(yb) else torch.as_tensor(yb, dtype=torch.long)
            frac_tgt = (yb_t == int(target_label)).float().mean().item()
            log(f"[probe][poison] cid{cid}: batch_size={yb_t.numel()} target_frac≈{frac_tgt:.3f}")
        except Exception as e:
            log(f"[probe][poison] cid{cid}: unable to compute target fraction: {e!r}")

    probe_pixel_batch(log, cfg, xb, context=f"cid{cid}/train")
