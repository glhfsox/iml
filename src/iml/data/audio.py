"""Waveform I/O and cropping. Uses soundfile (no torchcodec needed) + torchaudio resampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


def load_wav(path: str | Path, sample_rate: int) -> torch.Tensor:
    """Load any soundfile-readable file as a mono float32 tensor ``(L,)`` at ``sample_rate``."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wave = torch.from_numpy(data.mean(axis=1))
    if sr != sample_rate:
        wave = AF.resample(wave, sr, sample_rate)
    return wave


def save_wav(path: str | Path, wave: torch.Tensor | np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), np.asarray(wave, dtype=np.float32), sample_rate)


def crop_or_pad(wave: torch.Tensor, n_samples: int, random_crop: bool = False) -> torch.Tensor:
    """Fixed-length excerpt: random crop for training, centre crop otherwise; zero-pad if short."""
    length = wave.shape[-1]
    if length >= n_samples:
        if random_crop:
            start = int(torch.randint(0, length - n_samples + 1, ()))
        else:
            start = (length - n_samples) // 2
        return wave[..., start : start + n_samples]
    return torch.nn.functional.pad(wave, (0, n_samples - length))
