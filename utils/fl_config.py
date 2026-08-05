# fl_config.py
import shutil
from pathlib import Path
import yaml

def load_config(path: str | Path):
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    assert "task" in cfg and "train" in cfg and "model" in cfg
    # remember source path so we can freeze later even if only cfg dict is available
    cfg["_cfg_path"] = str(path.resolve())
    return cfg

def freeze_and_expand(cfg_or_path, out_dir: str | Path) -> str:
    """
    Copy the original YAML into out_dir with '_frozen' suffix before extension.
    Accepts either a path or a loaded cfg dict containing '_cfg_path'.
    Preserves bytes, order, comments, perms, and mtime.
    """
    if isinstance(cfg_or_path, (str, Path)):
        src = Path(cfg_or_path)
    elif isinstance(cfg_or_path, dict) and "_cfg_path" in cfg_or_path:
        src = Path(cfg_or_path["_cfg_path"])
    else:
        raise TypeError("freeze_and_expand expects a path or a cfg dict with '_cfg_path'")

    dst_dir = Path(out_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    stem = src.stem
    suffix = "".join(src.suffixes)  # supports .yaml/.yml and multi-extensions
    if not stem.endswith("_frozen"):
        stem = f"{stem}_frozen"

    dst = dst_dir / f"{stem}{suffix}"

    # validate before copying
    with open(src, "r") as f:
        _ = yaml.safe_load(f)

    shutil.copy2(src, dst)
    return str(dst)
