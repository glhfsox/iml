"""Pipeline of Baseline CNN (CIFAR-tutorial style): build_pipeline(...)(wave) -> Verdict."""

from __future__ import annotations

import torch

from app.models.speaker_verification.simple_cnn.pipeline.operators import operators
from app.models.utils import Pipeline
from iml.config import ExperimentConfig


def build_pipeline(
    cfg: ExperimentConfig,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    mc_samples: int | None = None,
) -> Pipeline:
    mc = cfg.mc_dropout_samples if mc_samples is None else mc_samples
    return Pipeline(*operators(cfg, model, device, threshold, mc))
