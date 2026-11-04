"""Shared package for the IML voice-intercom project.

Public API (what notebooks and the app import)::

    from iml import ExperimentConfig, build_model, fit, load_run, find_runs, make_loaders
    from iml import imbalance_report, balanced_subset  # class-imbalance toolkit (iml.imbalance)
"""

from iml.config import MODEL_NAMES, ExperimentConfig
from iml.data import LogMelSpectrogram, SpectrogramDataset, load_wav, make_loaders, save_wav
from iml.imbalance import (
    balanced_subset,
    focal_loss,
    freeze_backbone,
    imbalance_report,
    soft_f1_loss,
)
from iml.models import build_model, count_params
from iml.runs import (
    RUNS_ROOT,
    RunRecord,
    classification_metrics,
    eer,
    far,
    find_runs,
    frr,
    load_checkpoint,
    load_run,
    run_dir,
)
from iml.tracking import Tracker
from iml.train import evaluate, fit, get_device, predict_proba, seed_everything

__all__ = [
    # config
    "ExperimentConfig",
    "MODEL_NAMES",
    # data
    "load_wav",
    "save_wav",
    "LogMelSpectrogram",
    "SpectrogramDataset",
    "make_loaders",
    # class imbalance
    "imbalance_report",
    "balanced_subset",
    "focal_loss",
    "soft_f1_loss",
    "freeze_backbone",
    # models
    "build_model",
    "count_params",
    # training / evaluation
    "fit",
    "evaluate",
    "predict_proba",
    "Tracker",
    # metrics
    "classification_metrics",
    "far",
    "frr",
    "eer",
    # runs on disk
    "RUNS_ROOT",
    "RunRecord",
    "run_dir",
    "load_run",
    "find_runs",
    "load_checkpoint",
    # misc
    "get_device",
    "seed_everything",
]
