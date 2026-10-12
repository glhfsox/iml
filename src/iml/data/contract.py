"""The team data contract (DATA_CONTRACT.md): reading and validating a dataset folder.

A dataset is a folder with ``train.csv``, ``val.csv`` and ``test.csv`` (header exactly
``path,allowed``) and the clips they point to: 16 kHz mono PCM16 WAVs of exactly 3 s.
``allowed`` is 1 (let in) or 0.
"""

from __future__ import annotations

import hashlib
import posixpath
from pathlib import Path, PurePosixPath

import pandas as pd
import soundfile as sf

CONTRACT_VERSION = 1
SPLITS = ("train", "val", "test")
COLUMNS = ["path", "allowed"]
SAMPLE_RATE = 16000
CLIP_SAMPLES = 48000  # 3 s


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
    """``(absolute wav path, allowed)`` pairs of one split."""
    d = Path(data_dir)
    df = _read_manifest(d / f"{split}.csv")
    return [(d / p, int(a)) for p, a in zip(df.path, df.allowed, strict=True)]


def fingerprint(data_dir: str | Path) -> dict:
    """Identifies a dataset version: hash of the three manifests and every clip's bytes, plus
    per-split class counts."""
    d = Path(data_dir)
    h = hashlib.sha1(f"contract{CONTRACT_VERSION}".encode())
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
    """Contract violations of a dataset folder (empty list = valid)."""
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
        if set(df.allowed) != {"0", "1"}:
            errors.append(
                f"{split}.csv: both classes must be present, got {sorted(set(df.allowed))}"
            )
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
            wav = d / p
            if not wav.is_file():
                errors.append(f"{split}.csv: file not found: {p}")
            elif check_audio:
                errors += [f"{split}.csv: {p}: {e}" for e in _check_wav(wav)]
    return errors


def _check_wav(path: Path) -> list[str]:
    try:
        info = sf.info(str(path))
    except Exception as e:  # unreadable / not a wav
        return [f"unreadable ({e})"]
    errs = []
    if info.format not in ("WAV", "WAVEX") or info.subtype != "PCM_16":
        errs.append(f"must be WAV PCM_16, got {info.format}/{info.subtype}")
    if info.samplerate != SAMPLE_RATE:
        errs.append(f"sample rate {info.samplerate} != {SAMPLE_RATE}")
    if info.channels != 1:
        errs.append(f"{info.channels} channels, must be mono")
    if info.frames != CLIP_SAMPLES:
        errs.append(f"{info.frames} samples, must be exactly {CLIP_SAMPLES} (3 s)")
    return errs
