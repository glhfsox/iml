"""Operator chain of EfficientNetV2-S: transform -> infer -> decode.

EfficientNetV2-S = timm efficientnetv2_rw_s, 1-channel stem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from app.models.speaker_verification.efficientnetv2_s.transform import make_transform
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
