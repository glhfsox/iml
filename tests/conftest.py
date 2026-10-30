"""Tiny synthetic datasets in the DATA_CONTRACT.md format, loaded through the production path.

Class 1 clips carry a steady tone (a horizontal ridge in the spectrogram) over noise, class 0 clips
are noise only: an easy, learnable problem so training tests can assert that learning happens.
Clips are contract-v3 ``.npy`` int16 arrays of 48 000 samples.
"""

from __future__ import annotations

import os

# huggingface_hub reads this once at import, before any fixture runs: tests never download weights
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import pytest

SR = 16000


def write_contract_dataset(root, n: dict[str, int], seed: int = 0):
    """``n`` = clips per split (half per class). Returns ``root``."""
    rng = np.random.default_rng(seed)
    t = np.arange(3 * SR) / SR
    (root / "audio").mkdir(parents=True, exist_ok=True)
    for split, count in n.items():
        rows = []
        for i in range(count):
            allowed = i % 2
            x = rng.normal(0, 0.1, len(t))
            if allowed:
                x += 0.3 * np.sin(2 * np.pi * rng.uniform(300, 3000) * t)
            name = f"audio/{split}_{i:04d}.npy"
            np.save(root / name, (np.clip(x, -1, 1) * 32767).astype(np.int16))
            rows.append((name, allowed))
        pd.DataFrame(rows, columns=["path", "allowed"]).to_csv(root / f"{split}.csv", index=False)
    return root


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory):
    return write_contract_dataset(
        tmp_path_factory.mktemp("tiny"), {"train": 16, "val": 8, "test": 8}
    )


@pytest.fixture(scope="session")
def smoke_data(tmp_path_factory):
    return write_contract_dataset(
        tmp_path_factory.mktemp("smoke"), {"train": 128, "val": 64, "test": 64}
    )


@pytest.fixture(autouse=True)
def _no_shared_mlflow(monkeypatch):
    """Tests never log to the shared MLflow server (Dockerfile.dev sets MLFLOW_TRACKING_URI);
    a test that wants tracking sets its own uri."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("IML_TRACKING", raising=False)


@pytest.fixture(autouse=True)
def _hf_offline(monkeypatch):
    """Tests never download pretrained weights: the env var (read by huggingface_hub at import,
    hence also set at the top of this file) and the already-imported constant."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    try:
        import huggingface_hub.constants as hf

        monkeypatch.setattr(hf, "HF_HUB_OFFLINE", True)
    except ImportError:  # pragma: no cover
        pass
