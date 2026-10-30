"""Block library for composing spectrogram CNNs (pure torch, no timm), in one module.

Every block maps ``(B, C, F, T)`` -> ``(B, C', F', T')`` and works for any T (and F) of a
few frames or more; poolings collapse to ``(B, D)`` and heads produce logits.
``iml.models.spec.build_from_spec`` assembles them from a JSON layer list (``model="custom"``).
See README.md.

Sections, in this order: activations, init, stems, resnet, attention, efficientnet, convnext,
pooling, heads.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    # activations / init
    "make_activation",
    "init_weights",
    "count_params",
    "INIT_SCHEMES",
    # stems
    "ConvBNAct",
    "ConvStem",
    "DeepStem",
    "PatchifyStem",
    "to_single_channel",
    # residual / attention / mobile / convnext blocks
    "BasicBlock",
    "Bottleneck",
    "SqueezeExcite",
    "SplitAttention",
    "DropPath",
    "MBConv",
    "FusedMBConv",
    "LayerNorm2d",
    "ConvNeXtBlock",
    "Downsample",
    # pooling / heads
    "GlobalAvgPool",
    "FreqMeanTimeMaxMean",
    "AttentiveStatsPool",
    "ClassifierHead",
]


# =================================================================================================
# activations
# Activation factory shared by every block (``act="relu"`` style kwargs).
# =================================================================================================


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "sigmoid": nn.Sigmoid,
}


def make_activation(name: str) -> nn.Module:
    """Return a fresh activation module: relu | gelu | silu | sigmoid."""
    if name not in _ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}; expected one of {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]()


# =================================================================================================
# init
# Weight initialisation schemes (PDF point (h): He / Xavier / uniform) and parameter counting.
# =================================================================================================


INIT_SCHEMES = ("he", "xavier", "uniform", "default")


@torch.no_grad()
def init_weights(module: nn.Module, scheme: str) -> None:
    """Re-initialise every Conv1d/Conv2d/Linear weight with ``scheme`` and zero its bias.

    ``"default"`` keeps the layers' own (PyTorch / library) initialisation untouched.
    """
    if scheme not in INIT_SCHEMES:
        raise ValueError(f"unknown init {scheme!r}; expected one of {'|'.join(INIT_SCHEMES)}")
    if scheme == "default":
        return
    for m in module.modules():
        if isinstance(m, nn.Conv1d | nn.Conv2d | nn.Linear):
            if scheme == "he":
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif scheme == "xavier":
                nn.init.xavier_uniform_(m.weight)
            else:  # uniform
                nn.init.uniform_(m.weight, -0.05, 0.05)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of (trainable, by default) parameters of ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# =================================================================================================
# stems
# Stems: the first layers that turn a 1-channel spectrogram into a feature map.
# =================================================================================================


class ConvBNAct(nn.Sequential):
    """conv -> BN -> act (``bn_position="before"``), conv -> act -> BN ("after"), conv -> act.

    ``"none"`` leaves BN out. The conv bias is dropped only when BN directly follows the conv
    (``"before"``), where it is redundant. ``self[0]`` is always the conv. Padding is
    ``kernel_size // 2``: odd kernels keep the size at stride 1, ``ceil(size / stride)`` otherwise.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bn_position: str = "before",
        act: str = "relu",
    ) -> None:
        if bn_position not in ("before", "after", "none"):
            raise ValueError(f"bn_position must be before|after|none, got {bn_position!r}")
        conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=bn_position != "before",
        )
        layers: list[nn.Module] = [conv]
        if bn_position == "before":
            layers += [nn.BatchNorm2d(out_ch), make_activation(act)]
        elif bn_position == "after":
            layers += [make_activation(act), nn.BatchNorm2d(out_ch)]
        else:
            layers += [make_activation(act)]
        super().__init__(*layers)


def to_single_channel(conv: nn.Conv2d, pretrained: bool) -> nn.Conv2d:
    """Copy of ``conv`` with 1 input channel; with ``pretrained`` the RGB weights are summed."""
    new = nn.Conv2d(
        1,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        dilation=conv.dilation,
        bias=conv.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            new.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
            if conv.bias is not None:
                new.bias.copy_(conv.bias)
    return new


class ConvStem(nn.Sequential):
    """ResNet stem: ConvBNAct(kernel_size, stride) [+ 3x3/2 max-pool]; F, T -> F/4, T/4."""

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 64,
        kernel_size: int = 7,
        stride: int = 2,
        pool: bool = True,
    ) -> None:
        layers: list[nn.Module] = [ConvBNAct(in_ch, out_ch, kernel_size, stride=stride)]
        if pool:
            layers.append(nn.MaxPool2d(3, stride=2, padding=1))
        super().__init__(*layers)


class DeepStem(nn.Sequential):
    """ResNeSt/ResNet-d stem: three 3x3 ConvBNAct (first /2) + 3x3/2 max-pool; F,T -> F/4,T/4."""

    def __init__(self, in_ch: int = 1, out_ch: int = 64) -> None:
        mid = max(out_ch // 2, 1)
        super().__init__(
            ConvBNAct(in_ch, mid, 3, stride=2),
            ConvBNAct(mid, mid, 3),
            ConvBNAct(mid, out_ch, 3),
            nn.MaxPool2d(3, stride=2, padding=1),
        )


class PatchifyStem(nn.Sequential):
    """ConvNeXt stem: non-overlapping ``patch x patch`` conv + channel LayerNorm; F,T -> F/p,T/p."""

    def __init__(self, in_ch: int = 1, dim: int = 96, patch: int = 4) -> None:
        # LayerNorm2d: convnext section below (looked up when the stem is built)
        super().__init__(nn.Conv2d(in_ch, dim, patch, stride=patch), LayerNorm2d(dim))


# =================================================================================================
# resnet
# ResNet residual blocks: BasicBlock (ResNet-18/34) and Bottleneck (ResNet-50+, ResNeSt).
# =================================================================================================


def _shortcut(in_ch: int, out_ch: int, stride: int, avg_down: bool) -> nn.Module:
    """Identity, or a projection (1x1 conv /stride, or avg-pool + 1x1 conv) + BN."""
    if stride == 1 and in_ch == out_ch:
        return nn.Identity()
    if avg_down and stride > 1:
        return nn.Sequential(
            nn.AvgPool2d(stride, stride, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch)
    )


class BasicBlock(nn.Module):
    """Two 3x3 conv-BN with a residual; (B,in,F,T) -> (B,out,F/stride,T/stride)."""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, act: str = "relu") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = make_activation(act)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = _shortcut(in_ch, out_ch, stride, avg_down=False)
        self.act2 = make_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act2(y + self.shortcut(x))


class Bottleneck(nn.Module):
    """1x1 reduce -> 3x3 (grouped) -> 1x1 expand (x4) [+ SE] + residual; out channels = out_ch*4."""

    expansion = 4

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        se_ratio: float = 0.0,
        avg_down: bool = False,
        act: str = "relu",
    ) -> None:
        super().__init__()
        width = int(out_ch * base_width / 64) * groups
        final = out_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.act1 = make_activation(act)
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.act2 = make_activation(act)
        self.conv3 = nn.Conv2d(width, final, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(final)
        self.se = SqueezeExcite(final, se_ratio, act=act) if se_ratio > 0 else nn.Identity()
        self.shortcut = _shortcut(in_ch, final, stride, avg_down)
        self.act3 = make_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act1(self.bn1(self.conv1(x)))
        y = self.act2(self.bn2(self.conv2(y)))
        y = self.se(self.bn3(self.conv3(y)))
        return self.act3(y + self.shortcut(x))


# =================================================================================================
# attention
# Channel attention: Squeeze-and-Excitation and ResNeSt split-attention.
# =================================================================================================


class SqueezeExcite(nn.Module):
    """SE: global mean -> 1x1 reduce (``ch * rd_ratio``) -> act -> 1x1 expand -> sigmoid gate."""

    def __init__(self, ch: int, rd_ratio: float = 0.25, act: str = "silu") -> None:
        super().__init__()
        rd = max(1, round(ch * rd_ratio))
        self.reduce = nn.Conv2d(ch, rd, 1)
        self.act = make_activation(act)
        self.expand = nn.Conv2d(rd, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean((2, 3), keepdim=True)
        return x * torch.sigmoid(self.expand(self.act(self.reduce(s))))


class SplitAttention(nn.Module):
    """ResNeSt split-attention conv: ``radix`` grouped conv splits re-weighted by a softmax gate.

    ``(B, in_ch, F, T) -> (B, out_ch, F/stride, T/stride)``.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        radix: int = 2,
    ) -> None:
        super().__init__()
        if radix < 1:
            raise ValueError("radix must be >= 1")
        if in_ch % (groups * radix) or out_ch % groups:
            raise ValueError(
                f"SplitAttention needs in_ch ({in_ch}) divisible by groups*radix "
                f"({groups * radix}) and out_ch ({out_ch}) divisible by groups ({groups})"
            )
        self.radix, self.groups = radix, groups
        attn = max(32, out_ch * radix // 4)
        attn = -(-attn // groups) * groups  # grouped 1x1 convs need a multiple of groups
        self.conv = nn.Conv2d(
            in_ch,
            out_ch * radix,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups * radix,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch * radix)
        self.fc1 = nn.Conv2d(out_ch, attn, 1, groups=groups)
        self.bn1 = nn.BatchNorm2d(attn)
        self.fc2 = nn.Conv2d(attn, out_ch * radix, 1, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        b, rc = x.shape[:2]
        c = rc // self.radix
        splits = x.reshape(b, self.radix, c, x.shape[2], x.shape[3])
        gap = splits.sum(1).mean((2, 3), keepdim=True)  # (B, C, 1, 1)
        attn = self.fc2(F.relu(self.bn1(self.fc1(gap))))  # (B, C*radix, 1, 1)
        if self.radix > 1:
            attn = attn.reshape(b, self.groups, self.radix, -1).transpose(1, 2)
            attn = attn.softmax(dim=1).reshape(b, self.radix, c, 1, 1)
        else:
            attn = torch.sigmoid(attn).reshape(b, 1, c, 1, 1)
        return (splits * attn).sum(1)


# =================================================================================================
# efficientnet
# EfficientNet(V2) blocks: stochastic depth, MBConv and FusedMBConv.
# =================================================================================================


class DropPath(nn.Module):
    """Stochastic depth: drop a sample's residual branch with prob ``p`` (identity in eval)."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"drop path p must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * mask / keep


def _bn_conv1x1(in_ch: int, out_ch: int) -> nn.Sequential:
    """Linear 1x1 projection + BN (no activation)."""
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))


class MBConv(nn.Module):
    """Inverted residual: 1x1 expand -> kxk depthwise (/stride) -> SE -> 1x1 project [+ skip]."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand: int = 4,
        se_ratio: float = 0.25,
        act: str = "silu",
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        mid = in_ch * expand
        self.expand = ConvBNAct(in_ch, mid, 1, act=act) if expand != 1 else nn.Identity()
        self.dw = ConvBNAct(mid, mid, kernel_size, stride=stride, groups=mid, act=act)
        # SE width is relative to the block input, as in EfficientNet
        self.se = SqueezeExcite(mid, se_ratio / expand, act=act) if se_ratio > 0 else nn.Identity()
        self.project = _bn_conv1x1(mid, out_ch)
        self.residual = stride == 1 and in_ch == out_ch
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.project(self.se(self.dw(self.expand(x))))
        return x + self.drop_path(y) if self.residual else y


class FusedMBConv(nn.Module):
    """EfficientNetV2 fused block: kxk conv expand (/stride) -> [SE] -> 1x1 project [+ residual].

    With ``expand=1`` there is no projection: kxk ConvBNAct -> [SE on ``out_ch``].
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand: int = 4,
        se_ratio: float = 0.0,
        act: str = "silu",
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if expand == 1:
            self.fused = ConvBNAct(in_ch, out_ch, kernel_size, stride=stride, act=act)
            self.se: nn.Module = (
                SqueezeExcite(out_ch, se_ratio, act=act) if se_ratio > 0 else nn.Identity()
            )
            self.project: nn.Module = nn.Identity()
        else:
            mid = in_ch * expand
            self.fused = ConvBNAct(in_ch, mid, kernel_size, stride=stride, act=act)
            self.se = (
                SqueezeExcite(mid, se_ratio / expand, act=act) if se_ratio > 0 else nn.Identity()
            )
            self.project = _bn_conv1x1(mid, out_ch)
        self.residual = stride == 1 and in_ch == out_ch
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.project(self.se(self.fused(x)))
        return x + self.drop_path(y) if self.residual else y


# =================================================================================================
# convnext
# ConvNeXt blocks (Liu et al. 2022): channels-first LayerNorm, the inverted block, downsampling.
# =================================================================================================


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel axis of a ``(B, C, F, T)`` tensor (shape unchanged)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """dwconv 7x7 -> LN -> Linear 4x -> GELU -> Linear -> layer scale -> (drop path) + residual."""

    def __init__(self, dim: int, layer_scale: float = 1e-6, drop_path: float = 0.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), layer_scale)) if layer_scale > 0 else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dwconv(x).permute(0, 2, 3, 1)  # (B, F, T, C)
        y = self.fc2(self.act(self.fc1(self.norm(y))))
        if self.gamma is not None:
            y = y * self.gamma
        return x + self.drop_path(y.permute(0, 3, 1, 2))


class Downsample(nn.Sequential):
    """ConvNeXt stage transition: LN + 2x2/2 conv; (B,in,F,T) -> (B,out,F/2,T/2)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(LayerNorm2d(in_dim), nn.Conv2d(in_dim, out_dim, 2, stride=2))


# =================================================================================================
# pooling
# Poolings that collapse the ``(B, C, F, T)`` feature map into a fixed-size vector (any T).
# =================================================================================================


class GlobalAvgPool(nn.Module):
    """Mean over frequency and time: (B, C, F, T) -> (B, C)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean((2, 3))


class FreqMeanTimeMaxMean(nn.Module):
    """PANNs pooling: mean over F, then max + mean over T: (B, C, F, T) -> (B, C)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(2)
        return x.amax(-1) + x.mean(-1)


class AttentiveStatsPool(nn.Module):
    """ECAPA attentive stats: mean over F, attention over T, weighted mean ++ std -> (B, 2C)."""

    def __init__(self, ch: int, bottleneck: int = 128) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(ch, bottleneck, 1), nn.Tanh(), nn.Conv1d(bottleneck, ch, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(2)  # (B, C, T)
        w = torch.softmax(self.attention(x), dim=-1)
        mean = (w * x).sum(-1)
        var = (w * x * x).sum(-1) - mean * mean
        std = torch.sqrt(var.clamp(min=1e-5))
        return torch.cat([mean, std], dim=1)


# =================================================================================================
# heads
# Classifier heads: pooled features (B, D) -> logits (B, num_classes).
# =================================================================================================


class ClassifierHead(nn.Sequential):
    """Dropout -> [Linear(hidden) -> act -> Dropout] -> Linear(num_classes); (B, D) -> (B, K)."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = 2,
        dropout: float = 0.0,
        hidden: int = 0,
        act: str = "relu",
    ) -> None:
        layers: list[nn.Module] = [nn.Dropout(dropout)]
        if hidden > 0:
            layers += [nn.Linear(in_features, hidden), make_activation(act), nn.Dropout(dropout)]
            in_features = hidden
        layers.append(nn.Linear(in_features, num_classes))
        super().__init__(*layers)
