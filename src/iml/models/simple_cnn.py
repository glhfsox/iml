"""Baseline CNN in the spirit of the PyTorch CIFAR-10 tutorial (Milestone 1 model).

Input ``(B, 1, n_mels, T)`` -> logits ``(B, 2)``. Global average pooling makes it independent
of the excerpt length. Every PDF "knob" is a constructor argument so one class covers the
depth/width, BN position, activation, init and dropout comparisons.
"""

from __future__ import annotations

from torch import nn

from iml.models.common import conv_block, init_weights, make_activation


class SimpleCNN(nn.Module):
    def __init__(
        self,
        width: int = 32,
        depth: int = 2,
        bn_position: str = "before",
        activation: str = "relu",
        init: str = "he",
        dropout: float = 0.0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = 1
        for i in range(depth):
            out_ch = width * 2**i
            blocks += [conv_block(in_ch, out_ch, 3, bn_position, activation), nn.MaxPool2d(2)]
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)  # global pooling: any input size, MPS-safe
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, 128),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        init_weights(self, init)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))
