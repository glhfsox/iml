"""End-user program core: one frozen model -> (class, confidence) for a recording.

    uv run python -m app.main --model resnet18 --wav sample.wav     # frozen model.onnx, onnxruntime
    uv run python -m app.main --run runs/<member>/<topic>/<run> --wav sample.wav  # PyTorch (dev)
    uv run python -m app.main --run path/to/model.onnx --wav sample.wav          # any .onnx graph
    uv run python -m app.main --model convnext_tiny --random         # smoke without audio
    uv run python -m app.main --model wespeaker_resnet34 --bank bank.npz --wav sample.wav

The microphone notebook for the final user and notebooks/<member>/main.py call
:func:`load_model` / :func:`predict` from here. ``make freeze RUN=...`` copies a run's
``model.onnx`` (self-describing: config + threshold in its metadata) into
``app/models/speaker_verification/<model>/``; production runs that graph with onnxruntime
(the log-mel front end stays in PyTorch, on CPU). ``--run`` keeps the PyTorch path for dev.
Pretrained folders (``PRETRAINED_NAMES``, e.g. wespeaker_resnet34) have no checkpoint: they score
a recording against an embedding bank given with ``bank=`` / ``--bank``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.models import PRETRAINED_NAMES, model_dir, pipeline_module
from app.models.speaker_verification.contracts import Verdict
from app.models.utils import Pipeline, load_onnx, load_weights
from iml.config import MODEL_NAMES, ExperimentConfig
from iml.data import load_wav
from iml.train import get_device

__all__ = ["Recognizer", "Verdict", "load_model", "predict"]


class Recognizer:
    """A model folder's pipeline + its config, so preprocessing matches training exactly.

    ``cfg`` is None for pretrained folders; they pass their own ``sample_rate``/``n_samples``."""

    def __init__(
        self,
        pipeline: Pipeline,
        cfg: ExperimentConfig | None,
        device: torch.device,
        sample_rate: int | None = None,
        n_samples: int | None = None,
    ) -> None:
        self.pipeline, self.cfg, self.device = pipeline, cfg, device
        self.sample_rate = sample_rate or cfg.sample_rate
        self.n_samples = n_samples or int(cfg.excerpt_sec * cfg.sample_rate)

    def predict_wave(self, wave: torch.Tensor) -> Verdict:
        """``wave``: mono float tensor at ``self.sample_rate``, any length."""
        return self.pipeline(wave)

    def predict_file(self, path: str | Path) -> Verdict:
        return self.predict_wave(load_wav(path, self.sample_rate))


def load_model(
    source: str | Path,
    device: torch.device | None = None,
    threshold: float | None = None,
    mc_samples: int | None = None,
    bank: str | Path | None = None,
) -> Recognizer:
    """``source`` is resolved in this order:

    1. a pretrained name (``PRETRAINED_NAMES``): needs ``bank`` (embedding bank ``.npz``);
       ``threshold=None`` = the bank's;
    2. a registry model name (``iml.config.MODEL_NAMES``): the frozen
       ``app/models/speaker_verification/<name>/model.onnx`` run with onnxruntime;
    3. a path ending in ``.onnx``: that graph with onnxruntime;
    4. otherwise a run folder, a folder with ``best.pt`` or a ``.pt`` file: PyTorch weights (dev).

    The pipeline is taken from the model folder named in the config (from the checkpoint or the
    ONNX metadata). ``threshold=None`` uses the decision threshold the run chose on its validation
    set. The ONNX path runs on CPU and has no MC dropout (``mc_samples`` must be None/0)."""
    if str(source) in PRETRAINED_NAMES:
        if bank is None:
            raise ValueError(f"{source} is pretrained: pass bank=<embedding bank .npz>")
        mod = pipeline_module(str(source))
        pipeline = mod.build_pipeline(bank, threshold)
        sr = mod.SAMPLE_RATE
        return Recognizer(pipeline, None, torch.device("cpu"), sr, 3 * sr)
    if str(source) in MODEL_NAMES:
        onnx_path = model_dir(str(source)) / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"{onnx_path} not found: freeze a run first with "
                f"`make freeze RUN=runs/<member>/<topic>/<run>`"
            )
        return _load_onnx_recognizer(onnx_path, threshold, mc_samples)
    if str(source).endswith(".onnx"):
        return _load_onnx_recognizer(Path(source), threshold, mc_samples)
    folder = Path(source)
    if folder.is_dir() and (folder / "model.onnx").exists() and not (folder / "best.pt").exists():
        return _load_onnx_recognizer(folder / "model.onnx", threshold, mc_samples)  # frozen folder
    device = device or get_device()
    model, cfg, run_threshold = load_weights(Path(source), device)
    thr = run_threshold if threshold is None else threshold
    pipeline = pipeline_module(cfg.model).build_pipeline(cfg, model, device, thr, mc_samples)
    return Recognizer(pipeline, cfg, device)


def _load_onnx_recognizer(
    path: Path, threshold: float | None, mc_samples: int | None
) -> Recognizer:
    if mc_samples:
        raise ValueError("MC dropout is not available in an exported ONNX graph; use --run <run>")
    device = torch.device("cpu")  # onnxruntime gets numpy; keep the log-mel front end on CPU
    session, cfg, run_threshold = load_onnx(path)
    thr = run_threshold if threshold is None else threshold
    pipeline = pipeline_module(cfg.model).build_pipeline(cfg, session, device, thr, 0)
    return Recognizer(pipeline, cfg, device)


def predict(source: str | Path, wav: str | Path) -> Verdict:
    return load_model(source).predict_file(wav)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="classify one recording with a trained model")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--model",
        help="frozen model folder name: resnet18 etc. run its model.onnx with onnxruntime; "
        "wespeaker_resnet34 needs --bank",
    )
    src.add_argument(
        "--run",
        help="run folder, folder with best.pt or a .pt file (PyTorch weights, dev), "
        "or a .onnx file (onnxruntime)",
    )
    p.add_argument("--wav", help="recording to classify")
    p.add_argument("--random", action="store_true", help="use one excerpt of noise, no file")
    p.add_argument("--threshold", type=float, help="default: the run's val-chosen threshold")
    p.add_argument("--bank", help="embedding bank .npz (pretrained models: wespeaker_resnet34)")
    args = p.parse_args(argv)
    rec = load_model(args.model or args.run, threshold=args.threshold, bank=args.bank)
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
