from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


DEFAULT_SPEECH_COMMANDS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]


class LogMelFrontend:
    def __init__(self, cfg: Dict[str, Any]):
        import torchaudio.transforms as T

        audio_cfg = cfg.get("audio", {}) or {}
        self.sample_rate = int(audio_cfg.get("sample_rate", 16000))
        self.num_samples = int(audio_cfg.get("num_samples", self.sample_rate * float(audio_cfg.get("clip_seconds", 1.0))))
        self.n_mels = int(audio_cfg.get("n_mels", 40))
        self.n_fft = int(audio_cfg.get("n_fft", round(0.025 * self.sample_rate)))
        self.hop_length = int(audio_cfg.get("hop_length", round(0.010 * self.sample_rate)))
        self.win_length = int(audio_cfg.get("win_length", self.n_fft))
        self.f_min = float(audio_cfg.get("f_min", 20.0))
        self.f_max = audio_cfg.get("f_max", None)
        self.normalize = str(audio_cfg.get("normalize", "sample")).lower()
        self.mel = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=None if self.f_max is None else float(self.f_max),
            n_mels=self.n_mels,
            power=2.0,
        )
        self.to_db = T.AmplitudeToDB(stype="power", top_db=float(audio_cfg.get("top_db", 80.0)))

    def fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.float()
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        n = int(waveform.shape[-1])
        if n < self.num_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - n))
        elif n > self.num_samples:
            waveform = waveform[..., : self.num_samples]
        return waveform

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = self.fix_length(waveform)
        x = self.to_db(self.mel(waveform))
        if self.normalize == "sample":
            mean = x.mean()
            std = x.std().clamp_min(1e-6)
            x = (x - mean) / std
        return x.float()


class SpeechCommandsTensorDataset(Dataset):
    def __init__(self, root: str, subset: str, cfg: Dict[str, Any]):
        import torchaudio

        audio_cfg = cfg.get("audio", {}) or {}
        self.cache_features = bool(audio_cfg.get("cache_features", False))
        labels = list(audio_cfg.get("labels", DEFAULT_SPEECH_COMMANDS))
        self.labels = [str(x) for x in labels]
        self.label_to_idx = {name: idx for idx, name in enumerate(self.labels)}
        self.frontend = LogMelFrontend(cfg)
        self.base = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(root),
            url=str(audio_cfg.get("url", "speech_commands_v0.02")),
            folder_in_archive=str(audio_cfg.get("folder_in_archive", "SpeechCommands")),
            download=bool(audio_cfg.get("download", True)),
            subset=str(subset),
        )
        keep: List[int] = []
        targets: List[int] = []
        per_class_kept = {idx: 0 for idx in range(len(self.labels))}
        max_key = "max_train_per_class" if str(subset) == "training" else "max_test_per_class"
        max_per_class = int(audio_cfg.get(max_key, audio_cfg.get("max_per_class", 0)) or 0)
        for idx in range(len(self.base)):
            item = self.base.get_metadata(idx) if hasattr(self.base, "get_metadata") else self.base[idx]
            label = str(item[2] if len(item) >= 3 else item[1])
            if label in self.label_to_idx:
                mapped = int(self.label_to_idx[label])
                if max_per_class > 0 and per_class_kept[mapped] >= max_per_class:
                    continue
                keep.append(idx)
                targets.append(mapped)
                per_class_kept[mapped] += 1
        self.indices = keep
        self.targets = targets
        self.classes = self.labels
        self.class_to_idx = self.label_to_idx
        self._feature_cache = None
        if self.cache_features:
            self._build_feature_cache()

    def __len__(self) -> int:
        return len(self.indices)

    def get_waveform_label(self, index: int) -> Tuple[torch.Tensor, int]:
        waveform, sr, label, *_rest = self.base[self.indices[int(index)]]
        if int(sr) != int(self.frontend.sample_rate):
            import torchaudio.functional as AF

            waveform = AF.resample(waveform, int(sr), int(self.frontend.sample_rate))
        return self.frontend.fix_length(waveform), int(self.label_to_idx[str(label)])

    def waveform_to_feature(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.frontend(waveform)

    def _build_feature_cache(self) -> None:
        features: List[torch.Tensor] = []
        for idx in range(len(self.indices)):
            waveform, _label = self.get_waveform_label(idx)
            features.append(self.waveform_to_feature(waveform))
            if (idx + 1) % 2000 == 0 or (idx + 1) == len(self.indices):
                print(f"[audio-cache] {idx + 1}/{len(self.indices)} log-mel features cached")
        self._feature_cache = torch.stack(features, dim=0).contiguous()

    def __getitem__(self, index: int):
        if self._feature_cache is not None:
            return self._feature_cache[int(index)], int(self.targets[int(index)])
        waveform, label = self.get_waveform_label(index)
        return self.waveform_to_feature(waveform), label


class SyntheticSpeechCommandsDataset(Dataset):
    def __init__(self, split: str, cfg: Dict[str, Any]):
        audio_cfg = cfg.get("audio", {}) or {}
        self.cache_features = bool(audio_cfg.get("cache_features", False))
        labels = list(audio_cfg.get("labels", DEFAULT_SPEECH_COMMANDS))
        self.labels = [str(x) for x in labels]
        self.classes = self.labels
        self.class_to_idx = {name: idx for idx, name in enumerate(self.labels)}
        self.frontend = LogMelFrontend(cfg)
        n_per_class = int(audio_cfg.get("synthetic_train_per_class", 40 if split == "training" else 12))
        if split != "training":
            n_per_class = int(audio_cfg.get("synthetic_test_per_class", n_per_class))
        seed = int((cfg.get("seed", 0) or 0) + (0 if split == "training" else 12345))
        g = torch.Generator().manual_seed(seed)
        targets: List[int] = []
        waves: List[torch.Tensor] = []
        t = torch.linspace(0.0, 1.0, self.frontend.num_samples)
        for cls in range(len(self.labels)):
            freq = 180.0 + 70.0 * float(cls)
            for _ in range(n_per_class):
                phase = 2.0 * torch.pi * torch.rand((), generator=g)
                amp = 0.3 + 0.2 * torch.rand((), generator=g)
                wave = amp * torch.sin(2.0 * torch.pi * freq * t + phase)
                wave = wave + 0.03 * torch.randn(self.frontend.num_samples, generator=g)
                waves.append(wave.unsqueeze(0).float())
                targets.append(cls)
        self.waveforms = waves
        self.targets = targets
        self._feature_cache = None
        if self.cache_features:
            self._feature_cache = torch.stack([self.waveform_to_feature(w) for w in self.waveforms], dim=0).contiguous()

    def __len__(self) -> int:
        return len(self.targets)

    def get_waveform_label(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.waveforms[int(index)].clone(), int(self.targets[int(index)])

    def waveform_to_feature(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.frontend(waveform)

    def __getitem__(self, index: int):
        if self._feature_cache is not None:
            return self._feature_cache[int(index)], int(self.targets[int(index)])
        waveform, label = self.get_waveform_label(index)
        return self.waveform_to_feature(waveform), label


def build_audio_datasets(cfg: Dict[str, Any], data_root: str):
    audio_cfg = cfg.get("audio", {}) or {}
    source = str(audio_cfg.get("source", "speech_commands")).lower()
    if source in ("synthetic", "toy"):
        train = SyntheticSpeechCommandsDataset("training", cfg)
        test = SyntheticSpeechCommandsDataset("testing", cfg)
    else:
        root = Path(data_root)
        train = SpeechCommandsTensorDataset(str(root), "training", cfg)
        test = SpeechCommandsTensorDataset(str(root), "testing", cfg)

    if len(train) == 0 or len(test) == 0:
        raise RuntimeError("Audio dataset is empty after label filtering.")
    sample_x, _ = train[0]
    cfg.setdefault("task", {})["img_size"] = [int(sample_x.shape[-2]), int(sample_x.shape[-1])]
    cfg.setdefault("task", {})["channels"] = int(sample_x.shape[0])
    cfg.setdefault("model", {})["num_classes"] = len(getattr(train, "classes", DEFAULT_SPEECH_COMMANDS))
    return train, test, int(cfg["model"]["num_classes"])
