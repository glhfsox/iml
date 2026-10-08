"""Operator chain of MobileNetV2 (torchvision, 1-channel stem): transform -> infer -> decode."""

from __future__ import annotations

import torch

from app.models.speaker_verification.mobilenet_v2.transform import make_transform
from app.models.utils import DecodeVerdict, Operator, TorchInfer
from iml.config import ExperimentConfig


def operators(
    cfg: ExperimentConfig,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float,
    mc_samples: int,
) -> list[Operator]:
    return [make_transform(cfg, device), TorchInfer(model, mc_samples), DecodeVerdict(threshold)]
