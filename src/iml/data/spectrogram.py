"""Waveform -> log-mel spectrogram, the obligatory input representation of the project."""

from __future__ import annotations

import torch
import torchaudio.transforms as T
from torch import nn

from iml.config import ExperimentConfig


class LogMelSpectrogram(nn.Module):
    """``(B, L)`` or ``(L,)`` waveform -> ``(B, 1, n_mels, T)`` log-mel in dB."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            power=2.0,
        )
        self.to_db = T.AmplitudeToDB(stype="power", top_db=80.0)
        self.normalize_input = cfg.normalize_input

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        spec = self.to_db(self.mel(wave)).unsqueeze(1)
        return normalize(spec, self.normalize_input)


def normalize(spec: torch.Tensor, mode: str) -> torch.Tensor:
    """Input normalisation (PDF: "input normalization").

    ``per_sample``: zero mean / unit std per spectrogram. ``global``: fixed dB scaling
    (roughly maps [-80, 0] dB to [-1, 1]). ``none``: raw dB values.
    """
    if mode == "none":
        return spec
    if mode == "per_sample":
        dims = tuple(range(1, spec.dim()))
        mean = spec.mean(dim=dims, keepdim=True)
        std = spec.std(dim=dims, keepdim=True)
        return (spec - mean) / (std + 1e-5)
    if mode == "global":
        return (spec + 40.0) / 40.0
    raise ValueError(f"normalize_input must be none|per_sample|global, got {mode!r}")


def n_frames(cfg: ExperimentConfig) -> int:
    """Number of spectrogram frames for one excerpt of ``cfg.excerpt_sec`` seconds."""
    return int(cfg.excerpt_sec * cfg.sample_rate) // cfg.hop_length + 1
