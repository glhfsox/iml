import os

import pytest
import torch

from iml.config import ExperimentConfig
from iml.data.dataset import DummyDataset, make_loaders
from iml.runs import find_runs, load_checkpoint, load_run
from iml.train import fit, predict_proba

EXPECTED_FILES = {
    "config.json",
    "metrics.json",
    "history.csv",
    "loss.png",
    "metrics.png",
    "best.pt",
    "last.pt",
    "env.json",
}


def _loaders(cfg, n=64):
    return make_loaders(
        {s: DummyDataset(cfg, n=n, seed=i) for i, s in enumerate(("train", "val", "test"))}, cfg
    )


def test_fit_writes_complete_run_folder(tmp_path):
    cfg = ExperimentConfig(
        model="simple_cnn",
        width=16,
        epochs=6,
        lr=3e-3,
        batch_size=8,
        num_workers=0,
        member="t",
        topic="smoke",
        mc_dropout_samples=3,
        dropout=0.1,
    )
    rec = fit(cfg, _loaders(cfg, n=128), root=tmp_path, device=torch.device("cpu"), verbose=False)
    assert EXPECTED_FILES <= {p.name for p in rec.path.iterdir()}
    assert rec.path.parent.parent.parent == tmp_path and rec.path.parent.name == "smoke"
    for split in ("train", "val", "test"):
        assert {"far", "frr", "eer", "macro_f1", "accuracy", "loss", "mc_std_mean"} <= set(
            rec.metrics[split]
        )
    assert (
        len(rec.history) == 6
        and rec.history["train_loss"].iloc[-1] < rec.history["train_loss"].iloc[0]
    )
    assert rec.metrics["val"]["macro_f1"] > 0.8  # dummy problem is easy
    thr = rec.metrics["decision_threshold"]
    assert 0.0 <= thr <= 1.0 and rec.metrics["test"]["threshold"] == thr
    assert rec.metrics["val"]["n_pos"] + rec.metrics["val"]["n_neg"] == rec.metrics["val"]["n"]

    again = load_run(rec.path)
    assert again.cfg == cfg and again.metrics["best_epoch"] == rec.metrics["best_epoch"]
    assert [r.name for r in find_runs(tmp_path)] == [rec.name]

    ckpt = load_checkpoint(rec.best_checkpoint)
    assert ckpt["config"] == cfg and "state_dict" in ckpt
    assert ckpt["metrics"]["decision_threshold"] == thr

    # a finished run is not retrained unless asked
    same = fit(cfg, _loaders(cfg, n=128), root=tmp_path, device=torch.device("cpu"), verbose=False)
    assert same.metrics["best_epoch"] == rec.metrics["best_epoch"]


def test_early_stopping_and_sgd_cosine(tmp_path):
    cfg = ExperimentConfig(
        width=4,
        epochs=10,
        early_stopping_patience=1,
        optimizer="sgd",
        scheduler="cosine",
        lr=0.0,
        batch_size=16,
        num_workers=0,
    )
    rec = fit(cfg, _loaders(cfg, n=32), root=tmp_path, device=torch.device("cpu"), verbose=False)
    assert rec.metrics["epochs_run"] == 2  # lr=0 -> no improvement after epoch 1


def test_mc_dropout_gives_spread():
    cfg = ExperimentConfig(width=4, dropout=0.5)
    from iml.models import build_model

    model = build_model(cfg)
    x = torch.randn(4, 1, 64, 50)
    p, s = predict_proba(model, x, mc_samples=8)
    p0, s0 = predict_proba(model, x, mc_samples=0)
    assert p.shape == (4,) and s.max() > 0 and s0.max() == 0


@pytest.mark.slow
def test_mlflow_tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlruns.db")
    monkeypatch.chdir(tmp_path)  # mlflow artifact root is relative to cwd
    cfg = ExperimentConfig(width=4, epochs=1, batch_size=16, num_workers=0, member="t", topic="mlf")
    rec = fit(
        cfg, _loaders(cfg, n=32), root=tmp_path / "runs", device=torch.device("cpu"), verbose=False
    )
    import mlflow

    runs = mlflow.search_runs(experiment_names=["t/mlf"])
    assert len(runs) == 1 and runs.iloc[0]["tags.mlflow.runName"] == rec.name
    assert runs.iloc[0]["metrics.val_macro_f1"] == pytest.approx(rec.metrics["val"]["macro_f1"])
    assert os.path.exists(rec.path / "metrics.json")


def test_class_weight_balanced(tmp_path):
    cfg = ExperimentConfig(width=4, epochs=1, batch_size=16, num_workers=0, class_weight="balanced")
    rec = fit(cfg, _loaders(cfg, n=32), root=tmp_path, device=torch.device("cpu"), verbose=False)
    assert "far" in rec.metrics["val"]
