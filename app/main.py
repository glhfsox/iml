"""End-user program core: one frozen model -> (class, confidence) for a recording.

    uv run python -m app.main --model cnn10 --wav sample.wav          # frozen app/models/... folder
    uv run python -m app.main --run runs/<member>/<topic>/<run> --wav sample.wav
    uv run python -m app.main --model simple_cnn --random              # smoke without audio

The microphone notebook for the final user and notebooks/<member>/main.py call
:func:`load_model` / :func:`predict` from here. ``make freeze RUN=...`` copies a run's
``best.pt`` + ``config.json`` into ``app/models/speaker_verification/<model>/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.models import model_dir, pipeline_module
from app.models.speaker_verification.contracts import Verdict
from app.models.utils import Pipeline, load_weights
from iml.config import ExperimentConfig
from iml.data.audio import load_wav
from iml.device import get_device

__all__ = ["Recognizer", "Verdict", "load_model", "predict"]


class Recognizer:
    """A model folder's pipeline + its config, so preprocessing matches training exactly."""

    def __init__(self, pipeline: Pipeline, cfg: ExperimentConfig, device: torch.device) -> None:
        self.pipeline, self.cfg, self.device = pipeline, cfg, device
        self.n_samples = int(cfg.excerpt_sec * cfg.sample_rate)

    def predict_wave(self, wave: torch.Tensor) -> Verdict:
        """``wave``: mono float tensor at ``cfg.sample_rate``, any length (centre excerpt)."""
        return self.pipeline(wave)

    def predict_file(self, path: str | Path) -> Verdict:
        return self.predict_wave(load_wav(path, self.cfg.sample_rate))


def load_model(
    source: str | Path,
    device: torch.device | None = None,
    threshold: float | None = None,
    mc_samples: int | None = None,
) -> Recognizer:
    """``source``: a model name (frozen folder), a run folder, a folder with ``best.pt`` or a
    ``.pt`` file. The pipeline is taken from the model folder named in the checkpoint's config.
    ``threshold=None`` uses the decision threshold the run chose on its validation set."""
    path = model_dir(str(source)) if str(source) in _frozen_names() else Path(source)
    device = device or get_device()
    model, cfg, run_threshold = load_weights(path, device)
    thr = run_threshold if threshold is None else threshold
    pipeline = pipeline_module(cfg.model).build_pipeline(cfg, model, device, thr, mc_samples)
    return Recognizer(pipeline, cfg, device)


def predict(source: str | Path, wav: str | Path) -> Verdict:
    return load_model(source).predict_file(wav)


def _frozen_names() -> set[str]:
    from iml.config import MODEL_NAMES

    return set(MODEL_NAMES)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="classify one recording with a trained model")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="frozen model folder name, e.g. cnn10")
    src.add_argument("--run", help="run folder, folder with best.pt, or a .pt file")
    p.add_argument("--wav", help="recording to classify")
    p.add_argument(
        "--random", action="store_true", help="use one excerpt (cfg.excerpt_sec) of noise, no file"
    )
    p.add_argument("--threshold", type=float, help="default: the run's val-chosen threshold")
    args = p.parse_args(argv)
    rec = load_model(args.model or args.run, threshold=args.threshold)
    if args.random:
        verdict = rec.predict_wave(torch.randn(rec.n_samples) * 0.1)
    elif args.wav:
        verdict = rec.predict_file(args.wav)
    else:
        p.error("give --wav or --random")
    print(
        f"{verdict.name} (class {verdict.label}), confidence {verdict.confidence:.3f}, "
        f"p_allowed {verdict.p_allowed:.3f}, mc-std {verdict.uncertainty:.3f}  [{rec.pipeline}]"
    )


if __name__ == "__main__":
    main()
