"""How the CLIs obtain data: a dataset folder that follows DATA_CONTRACT.md.

The dataset folder (``--data DIR`` or ``$IML_DATA``) holds ``train/val/test.csv`` with the columns
``path,allowed``. The manifests are the split membership, so a run records the dataset fingerprint
in ``data.json`` next to its results, and a finished run is never silently reused for other data.

Contract v2 allows single-class datasets (e.g. an enrollment bank of allowed speakers only), but
training does not: FAR/FRR/EER and the decision threshold need both classes, so ``load_splits``
refuses a dataset whose train or val split lacks one.
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


# train: the loss; val: checkpoint selection and the decision threshold (FAR == FRR point)
TRAINING_SPLITS = ("train", "val")


def check_training_classes(data_dir: Path, items: dict[str, list[tuple[Path, int]]]) -> None:
    """Fail if a split used for training or threshold selection lacks a class; warn for test."""
    for split in SPLITS:
        labels = {a for _, a in items[split]}
        if labels == {0, 1}:
            continue
        has = f"only allowed={labels.pop()}" if labels else "no clips"
        if split in TRAINING_SPLITS:
            raise ValueError(
                f"dataset {data_dir}: split {split} has {has}; training needs both classes "
                "(contract v2 allows single-class datasets, e.g. for enrollment banks)"
            )
        warnings.warn(
            f"dataset {data_dir}: split {split} has {has}; its FAR/FRR/EER will be NaN",
            stacklevel=3,
        )


def load_splits(cfg: ExperimentConfig, data_dir: str | Path | None = None) -> dict[str, DataLoader]:
    """Loaders for train/val/test of a validated contract dataset that has both classes in train
    and val (a single-class test split only warns)."""
    d = resolve_data_dir(data_dir)
    errors = validate_dataset(d, check_audio=False)
    if errors:
        raise ValueError(f"{d} violates the data contract:\n  " + "\n  ".join(errors))
    items = {s: read_split(d, s) for s in SPLITS}
    check_training_classes(d, items)
    if int(cfg.excerpt_sec * cfg.sample_rate) != CLIP_SAMPLES:
        warnings.warn(
            f"excerpt_sec={cfg.excerpt_sec} != the 3 s contract clips: the model sees "
            "crops/padding of each clip (re-run `make configs` if this is unintended)",
            stacklevel=2,
        )
    datasets = {s: SpectrogramDataset(items[s], cfg, train=s == "train") for s in SPLITS}
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
