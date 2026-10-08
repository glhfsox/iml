"""Operator chain of ECAPA-TDNN: transform -> infer -> decode.

``Embed`` is this model's extra operator: it returns the 192-d speaker embedding instead of
a verdict, for similarity-based data exploration (PDF: samples similar from the model's view).
"""

from __future__ import annotations

import torch

from app.models.speaker_verification.ecapa_tdnn.transform import make_transform
from app.models.utils import DecodeVerdict, Operator, TorchInfer
from iml.config import ExperimentConfig


class Embed(Operator):
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model.eval()

    @torch.no_grad()
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        return self.model.embed(spec)


def operators(
    cfg: ExperimentConfig,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float,
    mc_samples: int,
) -> list[Operator]:
    return [make_transform(cfg, device), TorchInfer(model, mc_samples), DecodeVerdict(threshold)]
