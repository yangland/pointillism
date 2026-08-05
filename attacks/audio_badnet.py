from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def _unwrap_index(dataset, index: int):
    if isinstance(dataset, Subset):
        return _unwrap_index(dataset.dataset, int(dataset.indices[int(index)]))
    return dataset, int(index)


def _get_waveform_label(dataset, index: int) -> Tuple[torch.Tensor, int, Any]:
    base, base_idx = _unwrap_index(dataset, int(index))
    if hasattr(base, "get_waveform_label"):
        waveform, label = base.get_waveform_label(base_idx)
        return waveform, int(label), base
    x, y = dataset[int(index)]
    return x, int(y), base


def _waveform_to_feature(base, waveform: torch.Tensor) -> torch.Tensor:
    if hasattr(base, "waveform_to_feature"):
        return base.waveform_to_feature(waveform)
    return waveform


def _is_feature_block_trigger(cfg: Dict[str, Any]) -> bool:
    trigger = str(cfg.get("trigger", "waveform_tone")).lower()
    return trigger in {
        "mfcc_white_block",
        "mfcc_block",
        "spectrogram_white_block",
        "spectrogram_block",
        "logmel_white_block",
        "logmel_block",
    }


def _source_allowed(label: int, target_label: int, cfg: Dict[str, Any]) -> bool:
    source_labels = cfg.get("source_labels", "all")
    if isinstance(source_labels, str):
        mode = source_labels.lower()
        if mode in {"all_except_target", "non_target", "not_target"}:
            return int(label) != int(target_label)
        if mode in {"all", "*"}:
            return True
        if mode:
            return int(label) == int(mode)
    if isinstance(source_labels, (list, tuple, set)):
        return int(label) in {int(x) for x in source_labels}
    return True


def _resolve_block_start(size: int, block: int, position: Any) -> int:
    max_start = max(0, int(size) - int(block))
    if position is None:
        return max_start
    if isinstance(position, (int, float)):
        return max(0, min(max_start, int(position)))
    pos = str(position).lower()
    if pos in {"start", "begin", "low", "bottom", "left"}:
        return 0
    if pos in {"center", "middle", "fixed"}:
        return max_start // 2
    if pos in {"end", "late", "high", "top", "right"}:
        return max_start
    return max_start


def _resolve_block_value(feature: torch.Tensor, cfg: Dict[str, Any]) -> float:
    raw = cfg.get("block_value", cfg.get("patch_value", "max"))
    if isinstance(raw, str):
        value = raw.lower()
        if value in {"max", "white"}:
            return float(feature.max().item())
        if value in {"min", "black"}:
            return float(feature.min().item())
        if value in {"mean"}:
            return float(feature.mean().item())
        return float(raw)
    return float(raw)


def add_feature_block_trigger(feature: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    x = feature.clone().float()
    squeeze_channel = False
    if x.ndim == 2:
        x = x.unsqueeze(0)
        squeeze_channel = True
    if x.ndim != 3:
        raise ValueError(f"Audio feature trigger expects [C,H,W] or [H,W], got shape={tuple(x.shape)}")

    h = max(1, int(cfg.get("block_h", cfg.get("patch_h", 4))))
    w = max(1, int(cfg.get("block_w", cfg.get("patch_w", 4))))
    h = min(h, int(x.shape[-2]))
    w = min(w, int(x.shape[-1]))

    freq_pos = cfg.get("block_freq_position", cfg.get("patch_freq_position", "high"))
    time_pos = cfg.get(
        "block_time_position",
        cfg.get("patch_time_position", cfg.get("insert_position", "end")),
    )
    row = _resolve_block_start(int(x.shape[-2]), h, freq_pos)
    col = _resolve_block_start(int(x.shape[-1]), w, time_pos)
    x[..., row:row + h, col:col + w] = _resolve_block_value(x, cfg)
    return x.squeeze(0) if squeeze_channel else x


def make_tone_trigger(cfg: Dict[str, Any], num_samples: int, sample_rate: int) -> torch.Tensor:
    duration_ms = float(cfg.get("trigger_duration_ms", 100.0))
    n = max(1, int(round(float(sample_rate) * duration_ms / 1000.0)))
    freq = float(cfg.get("trigger_freq", 1000.0))
    amp = float(cfg.get("trigger_amplitude", 1.0))
    t = torch.arange(n, dtype=torch.float32) / float(sample_rate)
    trigger = amp * torch.sin(2.0 * math.pi * freq * t)
    fade = int(min(max(1, round(0.005 * sample_rate)), n // 2))
    if fade > 1:
        ramp = torch.linspace(0.0, 1.0, fade)
        trigger[:fade] *= ramp
        trigger[-fade:] *= torch.flip(ramp, dims=[0])
    return trigger.view(1, -1)


def add_waveform_trigger(
    waveform: torch.Tensor,
    trigger: torch.Tensor,
    *,
    snr_db: float,
    position: int,
) -> torch.Tensor:
    x = waveform.clone().float()
    if x.ndim == 1:
        x = x.unsqueeze(0)
    trig = trigger.to(dtype=x.dtype, device=x.device)
    if trig.ndim == 1:
        trig = trig.unsqueeze(0)
    n = min(int(trig.shape[-1]), int(x.shape[-1]))
    pos = int(max(0, min(int(position), int(x.shape[-1]) - n)))
    segment = x[..., pos:pos + n]
    sig_rms = segment.pow(2).mean().sqrt().clamp_min(1e-8)
    trig_rms = trig[..., :n].pow(2).mean().sqrt().clamp_min(1e-8)
    scale = sig_rms / (trig_rms * (10.0 ** (float(snr_db) / 20.0)))
    x[..., pos:pos + n] = segment + scale * trig[..., :n]
    return x.clamp(-1.0, 1.0)


class AudioBadNetWrapper(Dataset):
    def __init__(self, base_dataset, attack_cfg: Dict[str, Any], seed: int = 0):
        self.base = base_dataset
        self.attack_cfg = dict(attack_cfg)
        self.tgt = int(self.attack_cfg.get("target_label", 0))
        self.poison_frac = float(self.attack_cfg.get("poison_frac", 0.1))
        self.snr_db = float(self.attack_cfg.get("snr_db", 20.0))
        self.insert_position = str(self.attack_cfg.get("insert_position", "random")).lower()
        rng = np.random.default_rng(int(seed))
        idxs = []
        for idx in range(len(self.base)):
            _waveform, label, _base = _get_waveform_label(self.base, int(idx))
            if _source_allowed(int(label), self.tgt, self.attack_cfg):
                idxs.append(int(idx))
        idxs = np.asarray(idxs, dtype=np.int64)
        rng.shuffle(idxs)
        count = int(max(1, round(float(len(idxs)) * self.poison_frac))) if len(idxs) else 0
        self.poison_idxs = set(int(x) for x in idxs[:count].tolist())
        self._rng_seed = int(seed)

    def __len__(self) -> int:
        return len(self.base)

    def _position(self, idx: int, wave_len: int, trig_len: int) -> int:
        max_pos = max(0, int(wave_len) - int(trig_len))
        if self.insert_position in ("start", "begin"):
            return 0
        if self.insert_position in ("end", "late"):
            return max_pos
        if self.insert_position in ("center", "middle"):
            return max_pos // 2
        rng = np.random.default_rng(self._rng_seed + int(idx) * 1009)
        return int(rng.integers(0, max_pos + 1)) if max_pos > 0 else 0

    def __getitem__(self, index: int):
        waveform, label, base = _get_waveform_label(self.base, int(index))
        if int(index) in self.poison_idxs:
            if _is_feature_block_trigger(self.attack_cfg):
                return add_feature_block_trigger(_waveform_to_feature(base, waveform), self.attack_cfg), self.tgt
            else:
                sample_rate = int(getattr(getattr(base, "frontend", None), "sample_rate", self.attack_cfg.get("sample_rate", 16000)))
                trigger = make_tone_trigger(self.attack_cfg, int(waveform.shape[-1]), sample_rate)
                pos = self._position(int(index), int(waveform.shape[-1]), int(trigger.shape[-1]))
                waveform = add_waveform_trigger(waveform, trigger, snr_db=self.snr_db, position=pos)
                return _waveform_to_feature(base, waveform), self.tgt
        return _waveform_to_feature(base, waveform), int(label)


class AudioAttackEvalDataset(Dataset):
    def __init__(self, base_dataset, pick_indices, attack_cfg: Dict[str, Any]):
        self.base = base_dataset
        self.pick = list(int(x) for x in pick_indices)
        self.attack_cfg = dict(attack_cfg)
        self.tgt = int(self.attack_cfg.get("target_label", 0))
        self.snr_db = float(self.attack_cfg.get("snr_db", 20.0))
        self.insert_position = str(self.attack_cfg.get("insert_position", "random")).lower()
        self.seed = int(self.attack_cfg.get("seed", 0))

    def __len__(self) -> int:
        return len(self.pick)

    def _position(self, idx: int, wave_len: int, trig_len: int) -> int:
        max_pos = max(0, int(wave_len) - int(trig_len))
        if self.insert_position in ("start", "begin"):
            return 0
        if self.insert_position in ("end", "late"):
            return max_pos
        if self.insert_position in ("center", "middle"):
            return max_pos // 2
        rng = np.random.default_rng(self.seed + int(idx) * 1009)
        return int(rng.integers(0, max_pos + 1)) if max_pos > 0 else 0

    def __getitem__(self, index: int):
        base_idx = self.pick[int(index)]
        waveform, label, base = _get_waveform_label(self.base, base_idx)
        if _is_feature_block_trigger(self.attack_cfg):
            return add_feature_block_trigger(_waveform_to_feature(base, waveform), self.attack_cfg), self.tgt, int(label)
        sample_rate = int(getattr(getattr(base, "frontend", None), "sample_rate", self.attack_cfg.get("sample_rate", 16000)))
        trigger = make_tone_trigger(self.attack_cfg, int(waveform.shape[-1]), sample_rate)
        pos = self._position(base_idx, int(waveform.shape[-1]), int(trigger.shape[-1]))
        waveform = add_waveform_trigger(waveform, trigger, snr_db=self.snr_db, position=pos)
        return _waveform_to_feature(base, waveform), self.tgt, int(label)
