"""Shared package for the IML voice-intercom project.

Public API (what notebooks and the app import)::

    from iml import ExperimentConfig, build_model, fit, load_run, find_runs, make_loaders
"""

from iml.config import MODEL_NAMES, ExperimentConfig
from iml.data.audio import load_wav, save_wav
from iml.data.dataset import DummyDataset, SpectrogramDataset, make_loaders
from iml.data.spectrogram import LogMelSpectrogram
from iml.device import get_device
from iml.metrics import classification_metrics, eer, far, frr
from iml.models import build_model, count_params
from iml.runs import RUNS_ROOT, RunRecord, find_runs, load_checkpoint, load_run, run_dir
from iml.seed import seed_everything
from iml.tracking import Tracker
from iml.train import evaluate, fit, predict_proba

__all__ = [
    # config
    "ExperimentConfig",
    "MODEL_NAMES",
    # data
    "load_wav",
    "save_wav",
    "LogMelSpectrogram",
    "SpectrogramDataset",
    "DummyDataset",
    "make_loaders",
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
