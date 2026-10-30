"""Operators shared by all model pipelines: waveform -> spectrogram -> logits -> Verdict.

A pipeline is a plain chain of callables. Each model folder composes these (and any
model-specific operator) in its ``pipeline/operators.py``. The classifier step is either
``TorchInfer`` (an ``nn.Module``: run folders, dev) or ``OnnxInfer`` (an onnxruntime session on a
frozen ``model.onnx``: production); ``make_infer`` picks the right one.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from app.models.speaker_verification.contracts import Verdict
from iml.config import ExperimentConfig
from iml.data import LogMelSpectrogram, crop_or_pad
from iml.export import onnx_session, read_onnx_metadata
from iml.models import build_model
from iml.runs import load_checkpoint
from iml.train import predict_proba

if TYPE_CHECKING:
    import onnxruntime as ort


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


class OnnxInfer(Operator):
    """Spectrogram batch -> ``(p_allowed, std)`` tensors via an exported ``model.onnx``.

    ``std`` is all zeros: MC dropout is not available in an exported (eval-mode) graph."""

    def __init__(self, session: ort.InferenceSession) -> None:
        self.session = session

    def __call__(self, spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if spec.dim() != 4:
            raise ValueError(f"expected (B, 1, n_mels, T), got {tuple(spec.shape)}")
        x = spec.detach().cpu().numpy().astype(np.float32)
        logits = torch.from_numpy(self.session.run(["logits"], {"spectrogram": x})[0])
        p = torch.softmax(logits.float(), dim=1)[:, 1]
        return p, torch.zeros_like(p)


def make_infer(model: Any, mc_samples: int = 0) -> Operator:
    """``TorchInfer`` for an ``nn.Module``, ``OnnxInfer`` for an onnxruntime session."""
    if isinstance(model, torch.nn.Module):
        return TorchInfer(model, mc_samples)
    if hasattr(model, "run") and hasattr(model, "get_inputs"):  # onnxruntime.InferenceSession
        return OnnxInfer(model)  # mc_samples ignored: an exported graph has no MC dropout
    raise TypeError(f"expected nn.Module or onnxruntime.InferenceSession, got {type(model)}")


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
    """``path`` = a run folder (holding ``best.pt``) or a ``.pt`` file (PyTorch, dev path).

    Returns the model, its config and the decision threshold chosen on the validation set
    during training (0.5 if the checkpoint has none).
    """
    path = Path(path)
    ckpt = load_checkpoint(path if path.suffix == ".pt" else path / "best.pt", device)
    model = build_model(ckpt["config"], load_pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    threshold = float(ckpt.get("metrics", {}).get("decision_threshold", 0.5))
    return model.to(device).eval(), ckpt["config"], threshold


def load_onnx(
    path: str | Path, providers: tuple[str, ...] = ("CPUExecutionProvider",)
) -> tuple[ort.InferenceSession, ExperimentConfig, float]:
    """``path`` = a self-describing ``model.onnx`` written by ``iml.export.export_onnx``.

    Returns the onnxruntime session, the config and decision threshold read from the graph's
    metadata."""

    cfg, threshold = read_onnx_metadata(path)
    session = onnx_session(path, providers)
    return session, cfg, threshold
