"""Model registry: ``build_model(cfg)`` turns an ExperimentConfig into an nn.Module.

Contract shared by every model: input ``(B, 1, n_mels, T)`` float tensor, output logits
``(B, 2)`` (index 1 = allowed). Any time length T works (global pooling before the head).

Models (all ImageNet CNNs, ``cfg.pretrained`` loads the ImageNet weights):

* ``resnet18``          torchvision ResNet-18 (``resnet18`` below)
* ``efficientnetv2_s``  timm ``efficientnetv2_rw_s`` (``build_timm`` / ``TimmClassifier`` below)
* ``convnext_tiny``     timm ConvNeXt-T
* ``resnest26d``        timm ResNeSt-26d
* ``regnet_y_800mf``    timm ``regnety_008_tv`` (torchvision RegNet-Y-800MF weights)
* ``custom``            declarative CNN from ``cfg.arch`` (``iml.models.spec``), from
                        scratch only (``pretrained=False`` must be set explicitly)

Knobs: ``init`` (only from scratch), ``dropout`` (before the classifier), ``pretrained``.
Building blocks for custom architectures live in ``iml.models.blocks`` (one module), their
declarative assembly in ``iml.models.spec``; see README.md.
"""

from __future__ import annotations

from dataclasses import replace

import timm
import torch
from torch import nn
from torchvision import models

from iml.config import ExperimentConfig
from iml.models.blocks import (
    ConvBNAct,
    count_params,
    init_weights,
    make_activation,
    to_single_channel,
)
from iml.models.spec import build_from_spec

__all__ = [
    # registry
    "build_model",
    "check_knobs",
    "count_params",
    # torchvision / timm models
    "resnet18",
    "TIMM_IDS",
    "TimmClassifier",
    "build_timm",
    # shared helpers (from iml.models.blocks) and the conv_block shim
    "init_weights",
    "make_activation",
    "conv_block",
]


# =================================================================================================
# shared helpers: backward-compatible shim, the helpers themselves live in ``iml.models.blocks``
# =================================================================================================


def conv_block(
    in_ch: int, out_ch: int, kernel_size: int, bn_position: str, activation: str
) -> ConvBNAct:
    """conv -> (BN) -> act -> (BN); BN before/after the activation or ``"none"`` (PDF point (d))."""
    return ConvBNAct(in_ch, out_ch, kernel_size, bn_position=bn_position, act=activation)


# =================================================================================================
# torchvision ResNet-18 adapted to 1-channel spectrograms.
#
# ``pretrained=True`` loads ImageNet weights (PDF point (i): transfer learning) and folds the
# RGB stem into one channel by summing; otherwise the network trains from scratch with ``init``.
# =================================================================================================

_to_single_channel = to_single_channel  # old private name, still imported by notebooks


def resnet18(
    pretrained: bool = False, dropout: float = 0.0, init: str = "he", num_classes: int = 2
) -> models.ResNet:
    """ResNet-18 with a 1-channel stem and a Dropout + Linear head (``net.fc``)."""
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    net.conv1 = to_single_channel(net.conv1, pretrained)
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, num_classes))
    init_weights(net if not pretrained else net.fc, init)  # pretrained: only the new head
    return net


# =================================================================================================
# timm ImageNet CNNs for 1-channel input: EfficientNetV2-S, ConvNeXt-T, ResNeSt-26d, RegNetY.
#
# ``regnet_y_800mf`` is timm's ``regnety_008_tv`` (torchvision's RegNet_Y_800MF IMAGENET1K_V2
# weights ported to timm), so it gets the same 1-channel handling as the other timm models.
#
# timm builds each backbone with ``in_chans=1`` (pretrained RGB stem weights are summed into one
# channel by timm itself) and ``num_classes=0`` (pooled features); we add our own
# ``nn.Dropout`` + ``nn.Linear`` head. The explicit ``nn.Dropout`` module matters: timm applies
# its ``drop_rate`` functionally inside ``forward_head``, which ``iml.train.enable_mc_dropout``
# (it toggles ``nn.Dropout`` modules) could not reach, so timm's own dropout stays at 0.
#
# Why ``efficientnetv2_rw_s`` for "efficientnetv2_s": timm's plain ``efficientnetv2_s`` has no
# pretrained weights, and the ``tf_efficientnetv2_s`` ports use TensorFlow "SAME" padding, whose
# padding amount is computed from the input size at runtime and breaks the dynamic time axis in
# the ONNX export. ``efficientnetv2_rw_s`` (Ross Wightman's ImageNet-trained variant) uses
# symmetric padding like the rest.
# =================================================================================================

TIMM_IDS = {
    "efficientnetv2_s": "efficientnetv2_rw_s.ra2_in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
    "resnest26d": "resnest26d.gluon_in1k",
    "regnet_y_800mf": "regnety_008_tv.tv2_in1k",
}


class TimmClassifier(nn.Module):
    """timm backbone (num_classes=0, in_chans=1) -> Dropout -> Linear; embed(x) -> (B, D)."""

    def __init__(self, backbone: nn.Module, num_classes: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.backbone = backbone
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(backbone.num_features, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled backbone features ``(B, backbone.num_features)``."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.embed(x)))


def build_timm(
    name: str, pretrained: bool, dropout: float, init: str, num_classes: int = 2
) -> TimmClassifier:
    """Build ``TIMM_IDS[name]`` as a 1-channel classifier; ``init`` only applies from scratch."""
    if name not in TIMM_IDS:
        raise ValueError(f"unknown timm model {name!r}; expected one of {sorted(TIMM_IDS)}")
    backbone = timm.create_model(
        TIMM_IDS[name], pretrained=pretrained, in_chans=1, num_classes=0, drop_rate=0.0
    )
    if not pretrained:
        init_weights(backbone, init)
    model = TimmClassifier(backbone, num_classes=num_classes, dropout=dropout)
    init_weights(model.fc, init)
    return model


# =================================================================================================
# registry
# =================================================================================================


def check_knobs(cfg: ExperimentConfig) -> None:
    """Reject knob combinations that would not change the network (no fake comparisons)."""
    if cfg.model == "custom" and cfg.pretrained:
        raise ValueError("custom has no pretrained weights; set pretrained=false")
    if cfg.pretrained and cfg.init != "he":
        raise ValueError(
            f"{cfg.model} with pretrained=True ignores init={cfg.init!r} (ImageNet weights are "
            "kept); leave init at its default 'he' or set pretrained=false"
        )


def build_model(cfg: ExperimentConfig, load_pretrained: bool = True) -> nn.Module:
    """The network of ``cfg``. ``load_pretrained=False`` builds the same architecture without
    downloading ImageNet weights (when a state_dict is loaded right after: checkpoints, ONNX)."""
    check_knobs(cfg)
    if cfg.pretrained and not load_pretrained:
        cfg = replace(cfg, pretrained=False, init="default")
    if cfg.model == "resnet18":
        return resnet18(cfg.pretrained, cfg.dropout, cfg.init)
    if cfg.model == "custom":
        return build_from_spec(cfg.arch, 1, 2, cfg.dropout, cfg.init)
    if cfg.model in TIMM_IDS:
        return build_timm(cfg.model, cfg.pretrained, cfg.dropout, cfg.init)
    raise ValueError(f"unknown model {cfg.model!r}")
