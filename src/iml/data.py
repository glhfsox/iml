"""Datasets in the team data contract (DATA_CONTRACT.md): reading, validating and loading them.

A dataset is a folder with ``train.csv``, ``val.csv`` and ``test.csv`` (header exactly
``path,allowed``) and the clips they point to. Since contract v3 a clip is a ``.npy`` file holding
an int16 array of exactly 48 000 samples (3 s of 16 kHz mono PCM16); v2 datasets stored the same
samples as WAVs.
``allowed`` is 1 (let in) or 0. Since contract v2 a dataset (and each split) may hold a single
class, e.g. an enrollment bank of allowed speakers only; training needs both (``iml.cli``).

Sections:

* contract: ``read_split``, ``validate_dataset``, ``check_clip``, ``fingerprint``;
* audio: waveform I/O and cropping (soundfile, no torchcodec needed, + torchaudio resampling):
  ``load_wav``, ``load_clip``, ``save_wav``, ``crop_or_pad``;
* spectrogram: waveform -> log-mel, the obligatory input representation of the project:
  ``LogMelSpectrogram``, ``normalize``, ``n_frames``;
* dataset: ``SpectrogramDataset`` over one split's ``(clip_path, label)`` list (see
  ``read_split``) and ``make_loaders``.

Building a dataset from raw recordings lives in ``iml.preprocess``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
import torchaudio.transforms as T
from torch import nn
from torch.utils.data import DataLoader, Dataset

from iml.config import ExperimentConfig
from iml.imbalance import RotatingBalancedSampler, dataset_labels
from iml.contract import (  # noqa: F401  (re-exported: the contract is part of iml.data's API)
    CLIP_DTYPE,
    CLIP_SAMPLES,
    CLIP_SUFFIX,
    COLUMNS,
    CONTRACT_VERSION,
    FINGERPRINT_SALT,
    SAMPLE_RATE,
    SPLITS,
    check_clip,
    file_sha256,
    fingerprint,
    read_split,
    validate_dataset,
)

__all__ = [
    # contract
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
    # audio
    "load_wav",
    "load_clip",
    "save_wav",
    "crop_or_pad",
    # spectrogram
    "LogMelSpectrogram",
    "normalize",
    "n_frames",
    # dataset
    "SpectrogramDataset",
    "make_loaders",
]


# =================================================================================================
# audio: waveform I/O and cropping. Uses soundfile (no torchcodec needed) + torchaudio resampling
# =================================================================================================


def load_wav(path: str | Path, sample_rate: int) -> torch.Tensor:
    """Load any soundfile-readable file as a mono float32 tensor ``(L,)`` at ``sample_rate``."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wave = torch.from_numpy(data.mean(axis=1))
    if sr != sample_rate:
        wave = AF.resample(wave, sr, sample_rate)
    return wave


def load_clip(path: str | Path, sample_rate: int) -> torch.Tensor:
    """A dataset clip as a mono float32 tensor ``(L,)`` in [-1, 1].

    ``.npy`` (contract v3) is an int16 array scaled by 1/32768 (float arrays are taken as is, no
    resampling: contract clips are at the contract rate); any other file goes through ``load_wav``.
    """
    if Path(path).suffix == ".npy":
        a = np.load(path, mmap_mode="r", allow_pickle=False)
        if a.dtype == np.int16:
            return torch.from_numpy(np.asarray(a, dtype=np.float32) / 32768.0)
        return torch.from_numpy(np.array(a, dtype=np.float32))
    return load_wav(path, sample_rate)


def save_wav(path: str | Path, wave: torch.Tensor | np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), np.asarray(wave, dtype=np.float32), sample_rate)


def crop_or_pad(wave: torch.Tensor, n_samples: int, random_crop: bool = False) -> torch.Tensor:
    """Fixed-length excerpt: random crop for training, centre crop otherwise; zero-pad if short."""
    length = wave.shape[-1]
    if length >= n_samples:
        if random_crop:
            start = int(torch.randint(0, length - n_samples + 1, ()))
        else:
            start = (length - n_samples) // 2
        return wave[..., start : start + n_samples]
    return torch.nn.functional.pad(wave, (0, n_samples - length))


# =================================================================================================
# spectrogram: waveform -> log-mel spectrogram, the obligatory input representation of the project
# =================================================================================================


class LogMelSpectrogram(nn.Module):
    """``(B, L)`` or ``(L,)`` waveform -> ``(B, 1, n_mels, T)`` log-mel in dB."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            power=2.0,
        )
        self.to_db = T.AmplitudeToDB(stype="power", top_db=80.0)
        self.normalize_input = cfg.normalize_input

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        spec = self.to_db(self.mel(wave)).unsqueeze(1)
        return normalize(spec, self.normalize_input)


def normalize(spec: torch.Tensor, mode: str) -> torch.Tensor:
    """Input normalisation (PDF: "input normalization").

    ``per_sample``: zero mean / unit std per spectrogram. ``global``: fixed dB scaling
    (roughly maps [-80, 0] dB to [-1, 1]). ``none``: raw dB values.
    """
    if mode == "none":
        return spec
    if mode == "per_sample":
        dims = tuple(range(1, spec.dim()))
        mean = spec.mean(dim=dims, keepdim=True)
        std = spec.std(dim=dims, keepdim=True)
        return (spec - mean) / (std + 1e-5)
    if mode == "global":
        return (spec + 40.0) / 40.0
    raise ValueError(f"normalize_input must be none|per_sample|global, got {mode!r}")


def n_frames(cfg: ExperimentConfig) -> int:
    """Number of spectrogram frames for one excerpt of ``cfg.excerpt_sec`` seconds."""
    return int(cfg.excerpt_sec * cfg.sample_rate) // cfg.hop_length + 1


# =================================================================================================
# dataset: datasets and loaders. ``SpectrogramDataset`` takes a list of ``(clip_path, label)``
# (one split of a DATA_CONTRACT.md dataset, see ``read_split``)
# =================================================================================================


class SpectrogramDataset(Dataset):
    """``items`` = list of ``(clip_path, label)`` (``.npy`` contract clips; other audio files are
    read with ``load_wav``). Returns ``(spec (1, n_mels, T), label)``.

    Training mode takes a random excerpt from each recording, eval mode the centre one.
    Augmentation (``cfg.augment``) will be plugged in here once data exists.
    """

    def __init__(self, items: list[tuple[str | Path, int]], cfg: ExperimentConfig, train: bool):
        self.items = list(items)
        self.cfg = cfg
        self.train = train
        self.n_samples = int(cfg.excerpt_sec * cfg.sample_rate)
        self.to_spec = LogMelSpectrogram(cfg)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        wave = load_clip(path, self.cfg.sample_rate)
        wave = crop_or_pad(wave, self.n_samples, random_crop=self.train)
        spec = self.to_spec(wave).squeeze(0)
        return spec, int(label)


def make_loaders(
    datasets: dict[str, Dataset], cfg: ExperimentConfig, num_workers: int | None = None
) -> dict[str, DataLoader]:
    """Build loaders for the given splits.

    Training uses shuffled sampling by default. When rotating undersampling
    is enabled, a balanced sampler rotates majority-class recordings
    across epochs. Validation and test loaders remain unchanged.
    """
    workers = cfg.num_workers if num_workers is None else num_workers
    loaders = {}

    for split, ds in datasets.items():
        train = split == "train"

        sampler = None
        if train and cfg.undersampling == "rotating":
            sampler = RotatingBalancedSampler(
                dataset_labels(ds),
                seed=cfg.seed,
            )

        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=train and sampler is None,
            sampler=sampler,
            drop_last=(
                train and sampler is None and len(ds) > cfg.batch_size
            ),
            num_workers=workers,
            persistent_workers=workers > 0,
        )

    return loaders
