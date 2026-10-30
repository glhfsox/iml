"""Operator chain of RegNet-Y-800MF (timm regnety_008_tv): transform -> infer -> decode."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from app.models.speaker_verification.regnet_y_800mf.transform import make_transform
from app.models.utils import DecodeVerdict, Operator, make_infer
from iml.config import ExperimentConfig

if TYPE_CHECKING:
    import onnxruntime as ort


def operators(
    cfg: ExperimentConfig,
    model: torch.nn.Module | ort.InferenceSession,
    device: torch.device,
    threshold: float,
    mc_samples: int,
) -> list[Operator]:
    return [make_transform(cfg, device), make_infer(model, mc_samples), DecodeVerdict(threshold)]
