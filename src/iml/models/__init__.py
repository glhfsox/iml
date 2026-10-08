"""Model registry: ``build_model(cfg)`` turns an ExperimentConfig into an nn.Module.

All models share the same contract: input ``(B, 1, n_mels, T)`` float tensor, output logits
``(B, 2)`` (index 1 = allowed). Any time length T works.
"""

from __future__ import annotations

from torch import nn

from iml.config import ExperimentConfig
from iml.models.cnn10 import CNN10
from iml.models.common import count_params
from iml.models.ecapa_tdnn import ECAPATDNN
from iml.models.simple_cnn import SimpleCNN
from iml.models.tv_models import mobilenet_v2, resnet18

__all__ = ["build_model", "count_params", "SUPPORTED_KNOBS"]

_KNOBS = ("width", "depth", "bn_position", "activation", "init", "pretrained")
# Which architecture knobs each model actually uses. A non-default value for any other knob
# would change the run name without changing the network, so build_model rejects it.
SUPPORTED_KNOBS: dict[str, set[str]] = {
    "simple_cnn": {"width", "depth", "bn_position", "activation", "init"},
    "cnn10": {"width", "depth", "bn_position", "activation", "init"},
    "resnet18": {"init", "pretrained"},
    "mobilenet_v2": {"init", "pretrained"},
    "ecapa_tdnn": {"width", "init"},
}


def check_knobs(cfg: ExperimentConfig) -> None:
    defaults = ExperimentConfig(model=cfg.model)
    supported = set(SUPPORTED_KNOBS[cfg.model])
    if cfg.pretrained:
        supported.discard("init")  # ImageNet weights are kept, init is not applied
    bad = [k for k in _KNOBS if k not in supported and getattr(cfg, k) != getattr(defaults, k)]
    if bad:
        raise ValueError(
            f"{cfg.model} ignores {bad}; leave them at their defaults "
            f"(supported knobs: {sorted(supported)})"
        )


def build_model(cfg: ExperimentConfig) -> nn.Module:
    check_knobs(cfg)
    if cfg.model == "simple_cnn":
        return SimpleCNN(
            cfg.width, cfg.depth, cfg.bn_position, cfg.activation, cfg.init, cfg.dropout
        )
    if cfg.model == "cnn10":
        return CNN10(cfg.width, cfg.depth, cfg.bn_position, cfg.activation, cfg.init, cfg.dropout)
    if cfg.model == "resnet18":
        return resnet18(cfg.pretrained, cfg.dropout, cfg.init)
    if cfg.model == "mobilenet_v2":
        return mobilenet_v2(cfg.pretrained, cfg.dropout, cfg.init)
    if cfg.model == "ecapa_tdnn":
        return ECAPATDNN(cfg.n_mels, cfg.width, dropout=cfg.dropout, init=cfg.init)
    raise ValueError(f"unknown model {cfg.model!r}")
