"""Datasets and loaders.

``SpectrogramDataset`` takes a list of ``(path, label)`` (one split of a DATA_CONTRACT.md dataset,
see ``iml.data.contract.read_split``).
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Dataset

from iml.config import ExperimentConfig
from iml.data.audio import crop_or_pad, load_wav
from iml.data.spectrogram import LogMelSpectrogram


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
