"""Datasets and loaders.

``SpectrogramDataset`` is the real one (list of ``(path, label)``; data collection is a later
step). ``DummyDataset`` generates a learnable synthetic problem so the whole pipeline can be
smoke-tested without any recordings.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from iml.config import ExperimentConfig
from iml.data.audio import crop_or_pad, load_wav
from iml.data.spectrogram import LogMelSpectrogram, n_frames


class SpectrogramDataset(Dataset):
    """``items`` = list of ``(wav_path, label)``. Returns ``(spec (1, n_mels, T), label)``.

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
        wave = load_wav(path, self.cfg.sample_rate)
        wave = crop_or_pad(wave, self.n_samples, random_crop=self.train)
        spec = self.to_spec(wave).squeeze(0)
        return spec, int(label)


class DummyDataset(Dataset):
    """Synthetic 2-class spectrograms: class 1 contains a sustained "tone" (a bright 2-bin
    horizontal ridge at a random mel position), class 0 is plain noise.

    The cue is local and position-independent, so every model in the registry (including
    the ones with global pooling) can learn it in a few epochs; noise keeps it non-trivial.
    """

    def __init__(self, cfg: ExperimentConfig, n: int = 64, seed: int = 0, noise: float = 1.0):
        g = torch.Generator().manual_seed(seed)
        n_frames_ = n_frames(cfg)
        self.labels = torch.arange(n) % 2
        self.specs = torch.randn(n, 1, cfg.n_mels, n_frames_, generator=g) * noise
        rows = torch.randint(1, cfg.n_mels - 1, (n,), generator=g)
        for i, y in enumerate(self.labels.tolist()):
            if y == 1:
                self.specs[i, 0, rows[i] : rows[i] + 2, :] += 4.0

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return self.specs[i], int(self.labels[i])


def make_loaders(
    datasets: dict[str, Dataset], cfg: ExperimentConfig, num_workers: int | None = None
) -> dict[str, DataLoader]:
    """Build loaders for the given splits (``train`` shuffled + drop_last, others ordered)."""
    workers = cfg.num_workers if num_workers is None else num_workers
    loaders = {}
    for split, ds in datasets.items():
        train = split == "train"
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=train,
            drop_last=train and len(ds) > cfg.batch_size,
            num_workers=workers,
            persistent_workers=workers > 0,
        )
    return loaders
