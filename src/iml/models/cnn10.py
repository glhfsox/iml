"""PANNs-style CNN (Kong et al. 2019, arXiv:1912.10211): the "audio VGG".

Each block = two 3x3 conv-BN-ReLU, then 2x2 average pooling. ``depth`` is the number of
blocks: 4 blocks = CNN10 (default), 6 blocks = CNN14-like, 2 blocks = a tiny variant.
After the blocks the frequency axis is averaged and time is pooled with mean+max.
Input ``(B, 1, n_mels, T)`` -> logits ``(B, 2)``.
"""

from __future__ import annotations

from torch import nn

from iml.models.common import conv_block, init_weights, make_activation


class CNN10(nn.Module):
    def __init__(
        self,
        width: int = 32,
        depth: int = 4,
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
            out_ch = min(width * 2**i, 512)
            blocks += [
                conv_block(in_ch, out_ch, 3, bn_position, activation),
                conv_block(out_ch, out_ch, 3, bn_position, activation),
                nn.AvgPool2d(2),
                nn.Dropout2d(dropout),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_ch, in_ch),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )
        init_weights(self, init)

    def forward(self, x):
        x = self.features(x)  # (B, C, F', T')
        x = x.mean(dim=2)  # average over frequency -> (B, C, T')
        x = x.max(dim=2).values + x.mean(dim=2)  # PANNs temporal pooling -> (B, C)
        return self.head(x)
