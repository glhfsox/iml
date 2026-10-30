"""Pipeline of the custom declarative CNN: build_pipeline(...)(wave) -> Verdict."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from app.models.speaker_verification.custom.pipeline.operators import operators
from app.models.utils import Pipeline
from iml.config import ExperimentConfig

if TYPE_CHECKING:
    import onnxruntime as ort

__all__ = ["build_pipeline"]


def build_pipeline(
    cfg: ExperimentConfig,
    model: torch.nn.Module | ort.InferenceSession,
    device: torch.device,
    threshold: float = 0.5,
    mc_samples: int | None = None,
) -> Pipeline:
    mc = cfg.mc_dropout_samples if mc_samples is None else mc_samples
    return Pipeline(*operators(cfg, model, device, threshold, mc))
