"""Input contract of MobileNetV2 (torchvision, 1-channel stem).

Input : mono float waveform at `cfg.sample_rate` Hz, any length; the centre
        `cfg.excerpt_sec` seconds are used (zero-padded if shorter).
Tensor: `(1, 1, cfg.n_mels, T)` log-mel in dB, normalised per `cfg.normalize_input`.
All parameters come from this folder's config.json, so inference preprocessing is identical
to what the network saw during training.
"""

from __future__ import annotations

import torch

from app.models.utils import LogMelTransform
from iml.config import ExperimentConfig


def make_transform(cfg: ExperimentConfig, device: torch.device) -> LogMelTransform:
    return LogMelTransform(cfg, device)
