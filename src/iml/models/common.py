"""Building blocks shared by the hand-written models: activations, conv blocks, init."""

from __future__ import annotations

import torch
from torch import nn


def make_activation(name: str) -> nn.Module:
    table = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "gelu": nn.GELU}
    if name not in table:
        raise ValueError(f"unknown activation {name!r}; expected one of {sorted(table)}")
    return table[name]()


def conv_block(
    in_ch: int,
    out_ch: int,
    kernel_size: int,
    bn_position: str,
    activation: str,
    conv_cls=nn.Conv2d,
    bn_cls=nn.BatchNorm2d,
) -> nn.Sequential:
    """conv -> (BN) -> act -> (BN), where BN goes *before* or *after* the activation.

    This is the knob for PDF point (d): batch normalization before vs after activations.
    ``bn_position="none"`` leaves BN out entirely. The conv bias is dropped only when BN
    directly follows the conv (``before``), where it is redundant; ``after`` keeps it so the
    two arms differ in nothing but the BN placement.
    """
    layers: list[nn.Module] = [
        conv_cls(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=bn_position != "before")
    ]
    if bn_position == "before":
        layers += [bn_cls(out_ch), make_activation(activation)]
    elif bn_position == "after":
        layers += [make_activation(activation), bn_cls(out_ch)]
    elif bn_position == "none":
        layers += [make_activation(activation)]
    else:
        raise ValueError(f"bn_position must be before|after|none, got {bn_position!r}")
    return nn.Sequential(*layers)


@torch.no_grad()
def init_weights(model: nn.Module, scheme: str) -> None:
    """Re-initialise every conv/linear layer (PDF point (h): He / Xavier / uniform).

    ``"default"`` keeps PyTorch's own initialisation. Biases are zeroed in all other schemes.
    """
    if scheme == "default":
        return
    for m in model.modules():
        if isinstance(m, nn.Conv1d | nn.Conv2d | nn.Linear):
            if scheme == "he":
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif scheme == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif scheme == "uniform":
                nn.init.uniform_(m.weight, -0.05, 0.05)
            else:
                raise ValueError(f"unknown init {scheme!r}; expected he|xavier|uniform|default")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
