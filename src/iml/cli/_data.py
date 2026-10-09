"""Create DataLoaders from synthetic data or a validated real-data split manifest."""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from iml.config import ExperimentConfig
from iml.data.dataset import DummyDataset, SpectrogramDataset, make_loaders
from iml.data.manifest import load_split_manifest


def load_splits(
    cfg: ExperimentConfig,
    dummy: bool = False,
    manifest_path: str | None = None,
) -> dict[str, DataLoader]:
    datasets: dict[str, Dataset]
    if dummy:
        datasets = {
            s: DummyDataset(cfg, n=128, seed=i) for i, s in enumerate(("train", "val", "test"))
        }
    else:
        if not manifest_path:
            raise ValueError("real data requires --data-manifest")
        manifest = load_split_manifest(manifest_path)
        if cfg.dataset_id and cfg.dataset_id != manifest["dataset_id"]:
            raise ValueError(
                f"config dataset_id {cfg.dataset_id!r} does not match "
                f"manifest {manifest['dataset_id']!r}"
            )
        datasets = {
            split: SpectrogramDataset(items, cfg, train=split == "train")
            for split, items in manifest["resolved_items"].items()
        }
    return make_loaders(datasets, cfg)
