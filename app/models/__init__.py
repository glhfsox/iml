"""Model folders of the production app, one per architecture under the task group.

app/models/
    speaker_verification/
        contracts.py            dataclasses every pipeline returns (Verdict)
        <model>/
            pipeline/__init__.py    build_pipeline(cfg, model, device, ...)
            pipeline/operators.py   the operator chain of this model
            transform.py            input contract + preprocessing of this model
            config.json             frozen ExperimentConfig  (written by `make freeze`)
            best.pt                 frozen weights            (written by `make freeze`, gitignored)
            model.onnx              exported graph            (written by `make freeze`, gitignored)
    utils.py                shared operators (LogMelTransform, TorchInfer, DecodeVerdict)
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

from iml.config import MODEL_NAMES

GROUP = "speaker_verification"
MODELS_ROOT = Path(__file__).resolve().parent


def model_dir(name: str) -> Path:
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
    return MODELS_ROOT / GROUP / name


def pipeline_module(name: str) -> ModuleType:
    """The ``pipeline`` package of a model folder (exposes ``build_pipeline``)."""
    model_dir(name)  # validates the name
    return importlib.import_module(f"app.models.{GROUP}.{name}.pipeline")
