# `iml.models.blocks` + `iml.models.spec` — CNN building blocks

Pure-PyTorch blocks (no timm) for composing spectrogram classifiers, either in Python or
declaratively from a JSON layer list (`model="custom"`, see below). The whole block library is
one module, `iml.models.blocks` (`blocks.py`, one section per block family); the declarative
assembly (`build_from_spec`, `BLOCKS`, `EXAMPLE_ARCH`, ...) is `iml.models.spec` (`spec.py`).

## Block contract

* Tensors are `(B, C, F, T)` in and out (batch, channels, mel bins, frames); poolings return
  `(B, D)`, heads return logits `(B, num_classes)`.
* Every block works for any `T` (and `F`) of a few frames or more — nothing assumes a fixed
  length, so the same network serves 1 s and 3 s clips and exports to ONNX with a dynamic
  time axis.
* Explicit keyword arguments with defaults; each class has a one-line docstring.
* `stride=2` halves F and T (`ceil(n / 2)` for 3x3 convs, `floor` for 2x2 patch/downsample).

## Blocks

All in `iml.models.blocks`; the first column is the section of `blocks.py`.

| Section | Block | In -> out |
|---|---|---|
| `activations` | `make_activation(name)` | relu / gelu / silu / sigmoid |
| `init` | `init_weights(module, scheme)`, `count_params(model)` | he / xavier / uniform / default |
| `stems` | `ConvBNAct(in_ch, out_ch, kernel_size, stride, groups, bn_position, act)` | conv-BN-act, BN before/after act or none |
| `stems` | `ConvStem(in_ch=1, out_ch=64, kernel_size=7, stride=2, pool=True)` | ResNet stem, F,T / 4 |
| `stems` | `DeepStem(in_ch=1, out_ch=64)` | 3x 3x3 + max-pool (ResNeSt-d), F,T / 4 |
| `stems` | `PatchifyStem(in_ch=1, dim=96, patch=4)` | ConvNeXt patchify + LN, F,T / patch |
| `stems` | `to_single_channel(conv, pretrained)` | RGB stem conv -> 1 channel (sums weights) |
| `resnet` | `BasicBlock(in_ch, out_ch, stride, act)` | 2x 3x3 + residual, out = `out_ch` |
| `resnet` | `Bottleneck(in_ch, out_ch, stride, groups, base_width, se_ratio, avg_down, act)` | 1x1-3x3-1x1, out = `out_ch * 4` |
| `attention` | `SqueezeExcite(ch, rd_ratio=0.25, act="silu")` | channel gate, shape kept |
| `attention` | `SplitAttention(in_ch, out_ch, kernel_size, stride, groups, radix=2)` | ResNeSt split-attention conv |
| `efficientnet` | `DropPath(p)` | stochastic depth (identity in eval) |
| `efficientnet` | `MBConv(in_ch, out_ch, kernel_size, stride, expand=4, se_ratio=0.25, act, drop_path)` | inverted residual |
| `efficientnet` | `FusedMBConv(in_ch, out_ch, kernel_size, stride, expand=4, se_ratio=0.0, act, drop_path)` | EfficientNetV2 fused block |
| `convnext` | `LayerNorm2d(dim)` | channels-first LayerNorm |
| `convnext` | `ConvNeXtBlock(dim, layer_scale=1e-6, drop_path=0.0)` | dw 7x7 + MLP + residual |
| `convnext` | `Downsample(in_dim, out_dim)` | LN + 2x2/2 conv |
| `pooling` | `GlobalAvgPool()` | `(B, C)` |
| `pooling` | `FreqMeanTimeMaxMean()` | PANNs pooling, `(B, C)` |
| `pooling` | `AttentiveStatsPool(ch, bottleneck=128)` | attentive mean ++ std over T, `(B, 2C)` |
| `heads` | `ClassifierHead(in_features, num_classes=2, dropout=0.0, hidden=0, act="relu")` | Dropout [-> Linear -> act -> Dropout] -> Linear |

Residual blocks add the shortcut only where shapes allow it (MBConv/FusedMBConv: `stride == 1`
and `in_ch == out_ch`; ResNet blocks use a 1x1 projection otherwise).

## Composing a model in Python

```python
from torch import nn
from iml.models import count_params
from iml.models.blocks import (
    BasicBlock,
    ClassifierHead,
    ConvStem,
    GlobalAvgPool,
    SqueezeExcite,
    init_weights,
)


class TinyResNet(nn.Module):
    """(B, 1, n_mels, T) -> logits (B, 2)."""

    def __init__(self, dropout: float = 0.0, init: str = "he") -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvStem(1, 32), BasicBlock(32, 32), BasicBlock(32, 64, stride=2), SqueezeExcite(64)
        )
        self.pool = GlobalAvgPool()
        self.head = ClassifierHead(64, dropout=dropout)
        init_weights(self, init)

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


print(count_params(TinyResNet()))
```

To make it a registry model (trainable via `iml-train`, servable by the app):

1. add the name to `MODEL_NAMES` in `src/iml/config.py`;
2. add a branch in `build_model` (`src/iml/models/__init__.py`);
3. copy `app/models/speaker_verification/resnet18/` to
   `app/models/speaker_verification/<name>/` and fix the import paths / docstrings;
4. add the name to the `configs` dict in the `Makefile` and run `make configs`.

Use the `ClassifierHead`'s `nn.Dropout` (not functional dropout) so that
`iml.train.enable_mc_dropout` can reach it for Monte-Carlo dropout.

## Declarative models (`model="custom"`)

Instead of writing Python, a network can be described in the config as a JSON list
(`ExperimentConfig.arch`) and built with `iml.models.spec.build_from_spec(arch, in_ch=1,
num_classes=2, dropout=0.0, init="he")`, which returns a `CustomCNN` (`.features`, `.pool`,
`.head`, `embed(x) -> (B, D)`):

```json
[
  {"type": "conv_stem", "out_ch": 32, "kernel_size": 5, "stride": 2, "pool": false},
  {"type": "basic_block", "out_ch": 64, "stride": 2, "repeat": 2},
  {"type": "mbconv", "out_ch": 128, "stride": 2, "expand": 4, "se_ratio": 0.25},
  {"type": "convnext_block", "repeat": 2},
  {"type": "se", "rd_ratio": 0.25},
  {"type": "gap"}
]
```

Rules (violations raise a `ValueError` naming the offending entry):

* each entry is `{"type": <block type>, **kwargs}`; unknown types or kwargs are rejected;
* never write `in_ch` — it is inferred from the previous layer (the input has 1 channel);
* `"repeat": n` (int >= 1) stacks the layer n times; the first copy takes the given stride /
  channel change, the others use `stride=1` and `in_ch = out_ch`;
* the list ends with exactly one pooling layer (`gap` | `panns_pool` | `attentive_stats`);
* the `ClassifierHead(D, num_classes=2, dropout=cfg.dropout)` is appended automatically; an
  optional last entry `{"type": "head", "hidden": 128, "act": "relu"}` configures it;
* `custom` has no ImageNet weights: its configs must set `pretrained=false` (the global default
  is `true`), `init` applies as usual.

`describe_spec(arch)` prints one line per layer with the inferred channels; `EXAMPLE_ARCH`
(conv_stem 16 -> 2x basic_block 32 /2 -> gap) is a small, fast spec used by tests and configs.

| type | block | kwargs (defaults) | out channels |
|---|---|---|---|
| `conv_stem` | `ConvStem` | `out_ch=64, kernel_size=7, stride=2, pool=True` | `out_ch` |
| `deep_stem` | `DeepStem` | `out_ch=64` | `out_ch` |
| `patchify_stem` | `PatchifyStem` | `dim=96, patch=4` | `dim` |
| `conv` | `ConvBNAct` | `out_ch` (required), `kernel_size=3, stride=1, groups=1, bn_position="before", act="relu"` | `out_ch` |
| `basic_block` | `BasicBlock` | `out_ch` (required), `stride=1, act="relu"` | `out_ch` |
| `bottleneck` | `Bottleneck` | `out_ch` (required), `stride=1, groups=1, base_width=64, se_ratio=0.0, avg_down=False, act="relu"` | `out_ch * 4` |
| `mbconv` | `MBConv` | `out_ch` (required), `kernel_size=3, stride=1, expand=4, se_ratio=0.25, act="silu", drop_path=0.0` | `out_ch` |
| `fused_mbconv` | `FusedMBConv` | `out_ch` (required), `kernel_size=3, stride=1, expand=4, se_ratio=0.0, act="silu", drop_path=0.0` | `out_ch` |
| `convnext_block` | `ConvNeXtBlock` | `layer_scale=1e-6, drop_path=0.0` | same |
| `convnext_downsample` | `Downsample` | `out_ch` (required) | `out_ch` |
| `split_attention` | `SplitAttention` | `out_ch` (required), `kernel_size=3, stride=1, groups=1, radix=2` (in channels divisible by `groups*radix`) | `out_ch` |
| `se` | `SqueezeExcite` | `rd_ratio=0.25, act="silu"` | same |
| `maxpool` / `avgpool` | `nn.MaxPool2d` / `nn.AvgPool2d` | `kernel_size=2, stride=None` (= kernel) | same |
| `dropout2d` | `nn.Dropout2d` | `p=0.1` | same |
| `gap` | `GlobalAvgPool` | — | `C` |
| `panns_pool` | `FreqMeanTimeMaxMean` | — | `C` |
| `attentive_stats` | `AttentiveStatsPool` | `bottleneck=128` | `2C` |

`block_kwargs(type)` returns a type's kwargs and defaults; `BLOCKS` is the registry
(`factory(in_ch, **kwargs) -> (module, out_ch)`) — register a new block there.

CLI:

```bash
uv run iml-train --config app/models/configs/custom.json --data DIR \
    --set 'arch=[{"type":"conv_stem","out_ch":32},{"type":"mbconv","out_ch":64,"stride":2,"repeat":3},{"type":"gap"}]'
```
