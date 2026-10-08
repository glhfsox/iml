"""How the CLIs obtain data. Until recordings exist, ``--dummy`` is the only option;
``load_splits`` is the hook that the data-collection step will fill in.

When it is written: split by speaker (class 0) and by recording session/device (class 1),
never let excerpts of one recording cross splits, and write the split membership
(``splits.json``: {split: [(path, label), ...]}) into the run folder so two runs with the
same config hash but different data are distinguishable.
"""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from iml.config import ExperimentConfig
from iml.data.dataset import DummyDataset, make_loaders


def load_splits(cfg: ExperimentConfig, dummy: bool = False) -> dict[str, DataLoader]:
    datasets: dict[str, Dataset]
    if dummy:
        datasets = {
            s: DummyDataset(cfg, n=128, seed=i) for i, s in enumerate(("train", "val", "test"))
        }
    else:
        raise NotImplementedError(
            "real data loading is not wired yet: build SpectrogramDataset splits from data/ "
            "(see iml.data.dataset) or run with --dummy"
        )
    return make_loaders(datasets, cfg)
