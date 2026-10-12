"""How the CLIs obtain data: a dataset folder that follows DATA_CONTRACT.md.

The dataset folder (``--data DIR`` or ``$IML_DATA``) holds ``train/val/test.csv`` with the columns
``path,allowed``. The manifests are the split membership, so a run records the dataset fingerprint
in ``data.json`` next to its results, and a finished run is never silently reused for other data.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from torch.utils.data import DataLoader

from iml.config import ExperimentConfig
from iml.data.contract import CLIP_SAMPLES, SPLITS, fingerprint, read_split, validate_dataset
from iml.data.dataset import SpectrogramDataset, make_loaders
from iml.runs import load_json, save_json


def resolve_data_dir(data_dir: str | Path | None) -> Path:
    d = data_dir or os.environ.get("IML_DATA")
    if not d:
        raise ValueError("no dataset: pass --data DIR or set $IML_DATA")
    if not Path(d).is_dir():
        raise FileNotFoundError(f"dataset folder not found: {d}")
    return Path(d)


def load_splits(cfg: ExperimentConfig, data_dir: str | Path | None = None) -> dict[str, DataLoader]:
    """Loaders for train/val/test of a validated contract dataset."""
    d = resolve_data_dir(data_dir)
    errors = validate_dataset(d, check_audio=False)
    if errors:
        raise ValueError(f"{d} violates the data contract:\n  " + "\n  ".join(errors))
    if int(cfg.excerpt_sec * cfg.sample_rate) != CLIP_SAMPLES:
        warnings.warn(
            f"excerpt_sec={cfg.excerpt_sec} != the 3 s contract clips: the model sees "
            "crops/padding of each clip (re-run `make configs` if this is unintended)",
            stacklevel=2,
        )
    datasets = {s: SpectrogramDataset(read_split(d, s), cfg, train=s == "train") for s in SPLITS}
    return make_loaders(datasets, cfg)


def data_fingerprint(data_dir: str | Path | None = None) -> dict:
    return fingerprint(resolve_data_dir(data_dir))


def check_run_data(out: Path, fp: dict, overwrite: bool) -> None:
    """Refuse to reuse a finished run that was trained on different data."""
    done, rec = out / "metrics.json", out / "data.json"
    if overwrite or not done.exists():
        return
    if not rec.exists():
        raise RuntimeError(
            f"{out} is finished but has no data.json (trained before data tracking, data unknown); "
            "pass --overwrite to retrain it"
        )
    old = load_json(rec)
    if old.get("sha1") != fp.get("sha1"):
        raise RuntimeError(
            f"{out} was trained on different data ({old}); pass --overwrite to retrain it"
        )


def record_run_data(out: Path, fp: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    save_json(fp, out / "data.json")
