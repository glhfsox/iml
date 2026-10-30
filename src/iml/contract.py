"""The team data contract (DATA_CONTRACT.md) without torch: constants, manifests, validation,
fingerprint. ``iml.data`` re-exports everything here; ``iml.preprocess`` and light decode workers
import this module directly so they never load torch.
"""

from __future__ import annotations

import hashlib
import posixpath
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

__all__ = [
    "CONTRACT_VERSION",
    "FINGERPRINT_SALT",
    "SPLITS",
    "COLUMNS",
    "SAMPLE_RATE",
    "CLIP_SAMPLES",
    "CLIP_SUFFIX",
    "CLIP_DTYPE",
    "file_sha256",
    "read_split",
    "fingerprint",
    "validate_dataset",
    "check_clip",
]

# =================================================================================================
# contract: reading and validating a dataset folder
# =================================================================================================

CONTRACT_VERSION = 3
# Salt of the fingerprint hash. Not tied to CONTRACT_VERSION: change it only when the bytes of a
# dataset start to mean something else. v2 only relaxed a check, so it kept v1's "contract1"; v3
# stores clips as .npy instead of WAV (same samples, different bytes), so runs recorded on v2
# datasets (data.json) are not reused for v3 ones.
FINGERPRINT_SALT = "contract3"
SPLITS = ("train", "val", "test")
COLUMNS = ["path", "allowed"]
SAMPLE_RATE = 16000
CLIP_SAMPLES = 48000  # 3 s
CLIP_SUFFIX = ".npy"
CLIP_DTYPE = np.int16


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_manifest(f: Path) -> pd.DataFrame:
    """A split CSV as strings ("NA", "nan" etc. stay literal), so validation and loading agree."""
    try:
        return pd.read_csv(f, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_split(data_dir: str | Path, split: str) -> list[tuple[Path, int]]:
    """``(absolute clip path, allowed)`` pairs of one split."""
    d = Path(data_dir)
    df = _read_manifest(d / f"{split}.csv")
    return [(d / p, int(a)) for p, a in zip(df.path, df.allowed, strict=True)]


def fingerprint(data_dir: str | Path) -> dict:
    """Identifies a dataset version: hash of the three manifests and every clip's bytes, plus
    per-split class counts."""
    d = Path(data_dir)
    h = hashlib.sha1(FINGERPRINT_SALT.encode())
    counts = {}
    for s in SPLITS:
        h.update((d / f"{s}.csv").read_bytes())
        df = _read_manifest(d / f"{s}.csv")
        for p in sorted(df.path):
            h.update((d / p).read_bytes())
        counts[s] = {
            "allowed": int((df.allowed == "1").sum()),
            "not_allowed": int((df.allowed == "0").sum()),
        }
    return {
        "data_dir": str(d.resolve()),
        "contract_version": CONTRACT_VERSION,
        "sha1": h.hexdigest(),
        "counts": counts,
    }


def validate_dataset(data_dir: str | Path, check_audio: bool = True) -> list[str]:
    """Contract violations of a dataset folder (empty list = valid). Clip paths must end in
    ``.npy`` even with ``check_audio=False``; ``check_audio`` also opens every clip."""
    return _validate(data_dir, CLIP_SUFFIX, check_clip if check_audio else None)


def _validate(data_dir, suffix: str, check) -> list[str]:
    d = Path(data_dir)
    errors: list[str] = []
    seen: dict[str, str] = {}
    for split in SPLITS:
        f = d / f"{split}.csv"
        if not f.exists():
            errors.append(f"{split}.csv is missing")
            continue
        df = _read_manifest(f)
        if list(df.columns) != COLUMNS:
            errors.append(
                f"{split}.csv: header must be exactly 'path,allowed', got {list(df.columns)}"
            )
            continue
        bad = df[~df.allowed.isin(["0", "1"])]
        if len(bad):
            errors.append(f"{split}.csv: allowed must be 0 or 1 (rows {bad.index[:5].tolist()})")
        for p in df.path:
            pp = PurePosixPath(p)
            if not p or pp.is_absolute() or ".." in pp.parts or "\\" in p:
                errors.append(f"{split}.csv: path must be relative POSIX without '..': {p!r}")
                continue
            if posixpath.normpath(p) != p:
                errors.append(f"{split}.csv: path must be normalized: {p!r}")
                continue
            if p in seen:
                errors.append(f"{split}.csv: {p!r} already listed in {seen[p]}.csv")
                continue
            seen[p] = split
            if not p.endswith(suffix):
                errors.append(
                    f"{split}.csv: clip must be a {suffix} file: {p!r}"
                    " (WAV clips are contract v2; v3 stores int16 .npy arrays)"
                )
                continue
            clip = d / p
            if not clip.is_file():
                errors.append(f"{split}.csv: file not found: {p}")
            elif check is not None:
                errors += [f"{split}.csv: {p}: {e}" for e in check(clip)]
    return errors


def check_clip(path: str | Path) -> list[str]:
    """Violations of one v3 clip: a ``.npy`` int16 array of shape ``(CLIP_SAMPLES,)``."""
    try:
        a = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as e:  # unreadable / not a .npy
        return [f"unreadable ({e})"]
    errs = []
    if a.dtype != CLIP_DTYPE:
        errs.append(f"dtype {a.dtype}, must be int16")
    if a.shape != (CLIP_SAMPLES,):
        errs.append(f"shape {a.shape}, must be exactly ({CLIP_SAMPLES},) (3 s mono)")
    return errs
