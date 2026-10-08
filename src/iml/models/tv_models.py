"""torchvision image CNNs adapted to 1-channel spectrograms: ResNet-18 and MobileNetV2.

``pretrained=True`` loads ImageNet weights (PDF point (i): transfer learning) and folds the
RGB stem into one channel by summing; otherwise the networks train from scratch.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from iml.models.common import init_weights


def _to_single_channel(conv: nn.Conv2d, pretrained: bool) -> nn.Conv2d:
    new = nn.Conv2d(
        1,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        bias=conv.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            new.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
    return new


def resnet18(
    pretrained: bool = False, dropout: float = 0.0, init: str = "he", num_classes: int = 2
):
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    net.conv1 = _to_single_channel(net.conv1, pretrained)
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, num_classes))
    if not pretrained:
        init_weights(net, init)
    return net


def mobilenet_v2(
    pretrained: bool = False, dropout: float = 0.2, init: str = "he", num_classes: int = 2
):
    net = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    )
    net.features[0][0] = _to_single_channel(net.features[0][0], pretrained)
    net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.last_channel, num_classes))
    if not pretrained:
        init_weights(net, init)
    return net
