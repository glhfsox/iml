"""Declarative networks: build a CNN from a JSON-serialisable list of layer dicts (``cfg.arch``).

Each dict is ``{"type": <BLOCKS key>, **block kwargs}``; ``in_ch`` is never written, it is
inferred from the previous layer (the input has 1 channel). ``"repeat": n`` stacks the layer n
times: the first copy takes the given stride / channel change, the others use stride 1 (whatever
the spec says) and
``in_ch = out_ch``. The list ends with exactly one pooling layer (``gap`` | ``panns_pool`` |
``attentive_stats``), optionally followed by ``{"type": "head", "hidden": ..., "act": ...}``;
the ``ClassifierHead`` is always appended. See README.md ("Declarative models").
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from iml.models.blocks import (
    AttentiveStatsPool,
    BasicBlock,
    Bottleneck,
    ClassifierHead,
    ConvBNAct,
    ConvNeXtBlock,
    ConvStem,
    DeepStem,
    Downsample,
    FreqMeanTimeMaxMean,
    FusedMBConv,
    GlobalAvgPool,
    MBConv,
    PatchifyStem,
    SplitAttention,
    SqueezeExcite,
    init_weights,
)

__all__ = [
    "BLOCKS",
    "POOLS",
    "EXAMPLE_ARCH",
    "CustomCNN",
    "block_kwargs",
    "build_from_spec",
    "describe_spec",
]

Factory = Callable[..., tuple[nn.Module, int]]

# Small, fast spec shared by tests, the Makefile ``configs`` target and the README.
EXAMPLE_ARCH: list[dict[str, Any]] = [
    {"type": "conv_stem", "out_ch": 16},
    {"type": "basic_block", "out_ch": 32, "stride": 2, "repeat": 2},
    {"type": "gap"},
]


# ---------------------------------------------------------------------------- factories
# factory(in_ch, **kwargs) -> (module, out_ch); the kwargs/defaults are what a spec may set.
def _conv_stem(in_ch, out_ch=64, kernel_size=7, stride=2, pool=True):
    return ConvStem(in_ch, out_ch, kernel_size, stride, pool), out_ch


def _deep_stem(in_ch, out_ch=64):
    return DeepStem(in_ch, out_ch), out_ch


def _patchify_stem(in_ch, dim=96, patch=4):
    return PatchifyStem(in_ch, dim, patch), dim


def _conv(in_ch, out_ch, kernel_size=3, stride=1, groups=1, bn_position="before", act="relu"):
    return ConvBNAct(in_ch, out_ch, kernel_size, stride, groups, bn_position, act), out_ch


def _basic_block(in_ch, out_ch, stride=1, act="relu"):
    return BasicBlock(in_ch, out_ch, stride, act), out_ch


def _bottleneck(
    in_ch, out_ch, stride=1, groups=1, base_width=64, se_ratio=0.0, avg_down=False, act="relu"
):
    block = Bottleneck(in_ch, out_ch, stride, groups, base_width, se_ratio, avg_down, act)
    return block, out_ch * Bottleneck.expansion


def _mbconv(
    in_ch, out_ch, kernel_size=3, stride=1, expand=4, se_ratio=0.25, act="silu", drop_path=0.0
):
    return MBConv(in_ch, out_ch, kernel_size, stride, expand, se_ratio, act, drop_path), out_ch


def _fused_mbconv(
    in_ch, out_ch, kernel_size=3, stride=1, expand=4, se_ratio=0.0, act="silu", drop_path=0.0
):
    block = FusedMBConv(in_ch, out_ch, kernel_size, stride, expand, se_ratio, act, drop_path)
    return block, out_ch


def _convnext_block(in_ch, layer_scale=1e-6, drop_path=0.0):
    return ConvNeXtBlock(in_ch, layer_scale, drop_path), in_ch


def _convnext_downsample(in_ch, out_ch):
    return Downsample(in_ch, out_ch), out_ch


def _split_attention(in_ch, out_ch, kernel_size=3, stride=1, groups=1, radix=2):
    return SplitAttention(in_ch, out_ch, kernel_size, stride, groups, radix), out_ch


def _se(in_ch, rd_ratio=0.25, act="silu"):
    return SqueezeExcite(in_ch, rd_ratio, act), in_ch


def _maxpool(in_ch, kernel_size=2, stride=None):
    return nn.MaxPool2d(kernel_size, stride), in_ch


def _avgpool(in_ch, kernel_size=2, stride=None):
    return nn.AvgPool2d(kernel_size, stride), in_ch


def _dropout2d(in_ch, p=0.1):
    return nn.Dropout2d(p), in_ch


def _gap(in_ch):
    return GlobalAvgPool(), in_ch


def _panns_pool(in_ch):
    return FreqMeanTimeMaxMean(), in_ch


def _attentive_stats(in_ch, bottleneck=128):
    return AttentiveStatsPool(in_ch, bottleneck), 2 * in_ch


BLOCKS: dict[str, Factory] = {
    "conv_stem": _conv_stem,
    "deep_stem": _deep_stem,
    "patchify_stem": _patchify_stem,
    "conv": _conv,
    "basic_block": _basic_block,
    "bottleneck": _bottleneck,
    "mbconv": _mbconv,
    "fused_mbconv": _fused_mbconv,
    "convnext_block": _convnext_block,
    "convnext_downsample": _convnext_downsample,
    "split_attention": _split_attention,
    "se": _se,
    "maxpool": _maxpool,
    "avgpool": _avgpool,
    "dropout2d": _dropout2d,
    "gap": _gap,
    "panns_pool": _panns_pool,
    "attentive_stats": _attentive_stats,
}
POOLS = ("gap", "panns_pool", "attentive_stats")
# layers that always downsample (or pool to a vector): ``repeat`` would silently stack strides
NO_REPEAT = (
    *POOLS,
    "conv_stem",
    "deep_stem",
    "patchify_stem",
    "convnext_downsample",
    "maxpool",
    "avgpool",
)
HEAD_KWARGS = ("hidden", "act")  # num_classes / dropout come from the config


def block_kwargs(type_: str) -> dict[str, Any]:
    """Spec kwargs of a block type with their defaults (``inspect.Parameter.empty`` = required)."""
    params = list(inspect.signature(BLOCKS[type_]).parameters.values())[1:]  # skip in_ch
    return {p.name: p.default for p in params}


# ---------------------------------------------------------------------------- validation
def _split(arch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the layout; return (layers incl. the final pool, head kwargs)."""
    if not isinstance(arch, list | tuple) or not arch:
        raise ValueError("arch must be a non-empty list of layer dicts ending with a pooling")
    for i, layer in enumerate(arch):
        if not isinstance(layer, dict) or "type" not in layer:
            raise ValueError(f"arch[{i}] must be a dict with a 'type' key, got {layer!r}")
    layers, head = list(arch), {}
    if layers[-1]["type"] == "head":
        head = {k: v for k, v in layers.pop().items() if k != "type"}
        bad = sorted(set(head) - set(HEAD_KWARGS))
        if bad:
            raise ValueError(f"head: unknown kwargs {bad}; allowed: {list(HEAD_KWARGS)}")
    if any(layer["type"] == "head" for layer in layers):
        raise ValueError("'head' may only appear once, as the very last entry of arch")
    for i, layer in enumerate(layers):
        if layer["type"] not in BLOCKS:
            raise ValueError(
                f"arch[{i}]: unknown type {layer['type']!r}; known: {sorted(BLOCKS)} + 'head'"
            )
    pools = [i for i, layer in enumerate(layers) if layer["type"] in POOLS]
    if not pools:
        raise ValueError(f"arch needs exactly one pooling layer ({' | '.join(POOLS)}), found none")
    if len(pools) > 1:
        raise ValueError(f"arch needs exactly one pooling layer, found {len(pools)} at {pools}")
    if pools[0] != len(layers) - 1:
        raise ValueError(
            f"the pooling layer (arch[{pools[0]}]) must be last, only a 'head' may follow it"
        )
    for i, layer in enumerate(layers):
        t = layer["type"]
        if "in_ch" in layer:
            raise ValueError(f"arch[{i}] ({t}): do not write in_ch, it is inferred")
        repeat = layer.get("repeat", 1)
        if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 1:
            raise ValueError(f"arch[{i}] ({t}): repeat must be an int >= 1, got {repeat!r}")
        if t in NO_REPEAT and repeat != 1:
            raise ValueError(
                f"arch[{i}] ({t}): this layer always changes the resolution and cannot be "
                "repeated; list it several times instead"
            )
        allowed = block_kwargs(t)
        given = {k for k in layer if k not in ("type", "repeat")}
        unknown = sorted(given - set(allowed))
        if unknown:
            raise ValueError(f"arch[{i}] ({t}): unknown kwargs {unknown}; allowed: {list(allowed)}")
        missing = [k for k, d in allowed.items() if d is inspect.Parameter.empty and k not in given]
        if missing:
            raise ValueError(f"arch[{i}] ({t}): missing required kwargs {missing}")
    return layers, head


def _expand(layers: list[dict[str, Any]], in_ch: int):
    """Yield (index, type, in_ch, kwargs, module, out_ch) for every (repeated) layer."""
    ch = in_ch
    for i, layer in enumerate(layers):
        kwargs = {k: copy.deepcopy(v) for k, v in layer.items() if k not in ("type", "repeat")}
        for r in range(layer.get("repeat", 1)):
            if r > 0 and "stride" in block_kwargs(layer["type"]):
                kwargs["stride"] = 1  # only the first copy may change the resolution
            module, out = BLOCKS[layer["type"]](ch, **kwargs)
            yield i, layer["type"], ch, module, out
            ch = out


# ---------------------------------------------------------------------------- model
class CustomCNN(nn.Module):
    """features (blocks) -> pool -> head. embed(x) -> (B, D); forward -> (B, num_classes)."""

    def __init__(self, features: nn.Sequential, pool: nn.Module, head: ClassifierHead) -> None:
        super().__init__()
        self.features = features
        self.pool = pool
        self.head = head

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled features ``(B, D)``."""
        return self.pool(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


def build_from_spec(
    arch: list[dict[str, Any]],
    in_ch: int = 1,
    num_classes: int = 2,
    dropout: float = 0.0,
    init: str = "he",
) -> CustomCNN:
    """Build a ``CustomCNN`` from a declarative layer list (see module docstring)."""
    layers, head_kw = _split(arch)
    built = list(_expand(layers, in_ch))
    *blocks, (_, _, _, pool, dim) = built
    head = ClassifierHead(dim, num_classes=num_classes, dropout=dropout, **head_kw)
    model = CustomCNN(nn.Sequential(*(b[3] for b in blocks)), pool, head)
    init_weights(model, init)
    return model


def describe_spec(arch: list[dict[str, Any]], in_ch: int = 1) -> str:
    """One line per (repeated) layer with the inferred channels, plus the head."""
    layers, head_kw = _split(arch)
    lines, dim = [], in_ch
    for i, t, cin, _, cout in _expand(layers, in_ch):
        lines.append(f"[{i}] {t:<20} {cin:>5} -> {cout}")
        dim = cout
    extra = f", hidden={head_kw['hidden']}" if head_kw.get("hidden") else ""
    lines.append(f"    {'head':<20} {dim:>5} -> num_classes{extra}")
    return "\n".join(lines)
