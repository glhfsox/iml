"""Operators shared by all model pipelines: waveform -> spectrogram -> logits -> Verdict.

A pipeline is a plain chain of callables. Each model folder composes these (and any
model-specific operator) in its ``pipeline/operators.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from app.models.speaker_verification.contracts import Verdict
from iml.config import ExperimentConfig
from iml.data.audio import crop_or_pad
from iml.data.spectrogram import LogMelSpectrogram
from iml.models import build_model
from iml.runs import load_checkpoint
from iml.train import predict_proba


class Operator:
    """Base class: one step of a pipeline. Subclasses implement ``__call__``."""

    def __call__(self, data: Any) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


class Pipeline:
    def __init__(self, *operators: Operator) -> None:
        self.operators = operators

    def __call__(self, data: Any) -> Any:
        for op in self.operators:
            data = op(data)
        return data

    def __repr__(self) -> str:
        return " -> ".join(type(op).__name__ for op in self.operators)


class LogMelTransform(Operator):
    """Mono waveform at ``cfg.sample_rate`` -> ``(1, 1, n_mels, T)`` log-mel, same as training
    (centre excerpt of ``cfg.excerpt_sec`` seconds, same normalisation)."""

    def __init__(self, cfg: ExperimentConfig, device: torch.device) -> None:
        self.cfg, self.device = cfg, device
        self.to_spec = LogMelSpectrogram(cfg).to(device)
        self.n_samples = int(cfg.excerpt_sec * cfg.sample_rate)

    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        if not isinstance(wave, torch.Tensor) or wave.dim() != 1:
            raise TypeError("expected a 1-D mono waveform tensor")
        return self.to_spec(crop_or_pad(wave.to(self.device), self.n_samples))


class TorchInfer(Operator):
    """Spectrogram batch -> ``(p_allowed, mc_std)`` tensors."""

    def __init__(self, model: torch.nn.Module, mc_samples: int = 0) -> None:
        self.model, self.mc_samples = model.eval(), mc_samples

    def __call__(self, spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if spec.dim() != 4:
            raise ValueError(f"expected (B, 1, n_mels, T), got {tuple(spec.shape)}")
        return predict_proba(self.model, spec, self.mc_samples)


class DecodeVerdict(Operator):
    """``(p_allowed, mc_std)`` for a batch of one -> Verdict at the given threshold."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def __call__(self, probs: tuple[torch.Tensor, torch.Tensor]) -> Verdict:
        p, std = probs
        p_allowed = float(p[0])
        label = int(p_allowed >= self.threshold)
        return Verdict(label, p_allowed if label else 1 - p_allowed, p_allowed, float(std[0]))


def load_weights(
    path: str | Path, device: torch.device
) -> tuple[torch.nn.Module, ExperimentConfig, float]:
    """``path`` = a model folder / run folder (holding ``best.pt``) or a ``.pt`` file.

    Returns the model, its config and the decision threshold chosen on the validation set
    during training (0.5 if the checkpoint has none).
    """
    path = Path(path)
    ckpt = load_checkpoint(path if path.suffix == ".pt" else path / "best.pt", device)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    threshold = float(ckpt.get("metrics", {}).get("decision_threshold", 0.5))
    return model.to(device).eval(), ckpt["config"], threshold
