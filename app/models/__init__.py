"""Model folders of the production app, one per architecture under the task group.

app/models/
    speaker_verification/
        contracts.py            dataclasses every pipeline returns (Verdict)
        <model>/                one per iml.config.MODEL_NAMES: resnet18, efficientnetv2_s,
                                convnext_tiny, resnest26d, regnet_y_800mf, custom
            pipeline/__init__.py    build_pipeline(cfg, model, device, ...)
            pipeline/operators.py   the operator chain of this model
            transform.py            input contract + preprocessing of this model
            model.onnx              the frozen model, the folder's only artifact: graph + config
                                    and decision threshold in its metadata
                                    (written by `make freeze`, gitignored)
        wespeaker_resnet34/     pretrained ONNX embedder + embedding bank (no make freeze)
    utils.py                shared operators (LogMelTransform, TorchInfer / OnnxInfer,
                            DecodeVerdict) + load_onnx / load_weights
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

from iml.config import MODEL_NAMES

GROUP = "speaker_verification"
MODELS_ROOT = Path(__file__).resolve().parent
# pretrained, not in iml's training registry: no frozen model.onnx, scored against a bank
PRETRAINED_NAMES = ("wespeaker_resnet34",)


def model_dir(name: str) -> Path:
    if name not in MODEL_NAMES + PRETRAINED_NAMES:
        raise ValueError(
            f"unknown model {name!r}; expected one of {MODEL_NAMES + PRETRAINED_NAMES}"
        )
    return MODELS_ROOT / GROUP / name


def pipeline_module(name: str) -> ModuleType:
    """The ``pipeline`` package of a model folder (exposes ``build_pipeline``)."""
    model_dir(name)  # validates the name
    return importlib.import_module(f"app.models.{GROUP}.{name}.pipeline")
