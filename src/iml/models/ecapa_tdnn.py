"""Compact ECAPA-TDNN (Desplanques et al. 2020, arXiv:2005.07143) for speaker verification.

1-D convolutions run over time with the mel bins as input channels, so the "image" view of
the spectrogram is dropped: input ``(B, 1, n_mels, T)`` is squeezed to ``(B, n_mels, T)``.
Blocks: conv1d stem -> 3 x SE-Res2Block (dilations 2, 3, 4) -> multi-layer feature
aggregation -> attentive statistics pooling -> embedding -> classifier.
Sizes are scaled down (``channels = width * 4``) for a small, self-collected dataset.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from iml.models.common import init_weights


class Res2Conv1d(nn.Module):
    """Res2Net-style grouped conv: hierarchical residual connections within one layer."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, scale: int = 8) -> None:
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        pad = dilation * (kernel_size - 1) // 2
        self.convs = nn.ModuleList(
            nn.Conv1d(self.width, self.width, kernel_size, dilation=dilation, padding=pad)
            for _ in range(scale - 1)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(self.width) for _ in range(scale - 1))

    def forward(self, x):
        chunks = list(torch.split(x, self.width, dim=1))
        out = [chunks[0]]
        y = chunks[0]
        for i in range(1, self.scale):
            inp = chunks[i] if i == 1 else chunks[i] + y
            y = torch.relu(self.bns[i - 1](self.convs[i - 1](inp)))
            out.append(y)
        return torch.cat(out, dim=1)


class SEBlock(nn.Module):
    """Squeeze-and-excitation over the time axis."""

    def __init__(self, channels: int, bottleneck: int = 64) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x.mean(dim=2)).unsqueeze(-1)


class SERes2Block(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, scale: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
            Res2Conv1d(channels, kernel_size, dilation, scale),
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
            SEBlock(channels),
        )

    def forward(self, x):
        return x + self.body(x)  # skip connection (PDF point (e))


class AttentiveStatsPooling(nn.Module):
    """Channel-dependent attentive statistics pooling with global context."""

    def __init__(self, channels: int, attention_channels: int = 128) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(3 * channels, attention_channels, 1),
            nn.Tanh(),
            nn.BatchNorm1d(attention_channels),
            nn.Conv1d(attention_channels, channels, 1),
        )

    def forward(self, x):
        t = x.size(2)
        mean = x.mean(dim=2, keepdim=True).expand(-1, -1, t)
        std = x.std(dim=2, keepdim=True, unbiased=False).expand(-1, -1, t)
        alpha = torch.softmax(self.attention(torch.cat([x, mean, std], dim=1)), dim=2)
        mu = (alpha * x).sum(dim=2)
        var = (alpha * x * x).sum(dim=2) - mu * mu
        sigma = torch.sqrt(var.clamp(min=1e-6))
        return torch.cat([mu, sigma], dim=1)


class ECAPATDNN(nn.Module):
    def __init__(
        self,
        n_mels: int = 64,
        width: int = 32,
        emb_dim: int = 192,
        dropout: float = 0.0,
        init: str = "default",
        scale: int = 8,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        channels = math.ceil(width * 4 / scale) * scale  # divisible by the Res2Net scale
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, channels, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(channels)
        )
        self.blocks = nn.ModuleList(SERes2Block(channels, 3, d, scale) for d in (2, 3, 4))
        self.mfa = nn.Sequential(nn.Conv1d(3 * channels, 3 * channels, 1), nn.ReLU())
        self.pool = AttentiveStatsPooling(3 * channels)
        self.embedding = nn.Sequential(
            nn.BatchNorm1d(6 * channels), nn.Linear(6 * channels, emb_dim), nn.BatchNorm1d(emb_dim)
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(emb_dim, num_classes))
        init_weights(self, init)

    def embed(self, x):
        """Speaker embedding ``(B, emb_dim)``; useful for the similarity-based data exploration."""
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, 1, n_mels, T) -> (B, n_mels, T)
        x = self.stem(x)
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        x = self.mfa(torch.cat(outs, dim=1))
        return self.embedding(self.pool(x))

    def forward(self, x):
        return self.classifier(self.embed(x))
