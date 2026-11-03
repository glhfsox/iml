"""Regression: an early zero EER must not lock a loss-monitored checkpoint."""

import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import iml.train as training
from iml.config import ExperimentConfig
from iml.runs import classification_metrics, load_checkpoint
from iml.tracking import Tracker


def test_checkpoint_metric_validation_and_legacy_names():
    cfg = ExperimentConfig()
    legacy = cfg.to_dict()
    legacy.pop("checkpoint_metric")
    for key in ("member", "topic", "tags", "notes", "num_workers", "export_onnx"):
        legacy.pop(key)
    expected = hashlib.sha1(json.dumps(legacy, sort_keys=True).encode()).hexdigest()[:6]
    assert cfg.config_hash() == expected
    assert cfg.run_name() != cfg.with_overrides("checkpoint_metric=loss").run_name()
    with pytest.raises(ValueError, match="checkpoint_metric"):
        ExperimentConfig(checkpoint_metric="test_loss")


@pytest.mark.parametrize(
    ("metric", "patience", "losses", "best_epoch", "epochs_run"),
    [
        ("loss", 0, [0.4, 0.2, 0.05, 0.1], 3, 4),
        ("eer", 0, [0.4, 0.2, 0.05, 0.1], 1, 4),
        ("loss", 1, [0.4, 0.2, 0.05, 0.1], 3, 4),
        ("loss", 1, [0.4, 0.2, 0.3, 0.1], 2, 3),
    ],
)
def test_fit_retains_correct_weights_and_uses_same_early_stop_monitor(
    tmp_path, monkeypatch, metric, patience, losses, best_epoch, epochs_run
):
    monkeypatch.setattr(training, "build_model", lambda cfg: nn.Linear(1, 2))
    epoch_counter = 0

    def train_epoch(model, *args):
        nonlocal epoch_counter
        epoch_counter += 1
        with torch.no_grad():
            model.weight.fill_(epoch_counter)
        return 1 / epoch_counter

    def evaluate(model, loader, device, **kwargs):
        epoch = int(model.weight[0, 0].item())
        labels, scores = np.array([0, 1]), np.array([0.1, 0.9])
        out = classification_metrics(labels, scores >= kwargs.get("threshold", 0.5), scores)
        out.update(
            loss=losses[epoch - 1], _labels=labels, _scores=scores, _stds=np.zeros_like(scores)
        )
        return out

    monkeypatch.setattr(training, "train_one_epoch", train_epoch)
    monkeypatch.setattr(training, "evaluate", evaluate)
    cfg = ExperimentConfig(
        epochs=4,
        checkpoint_metric=metric,
        early_stopping_patience=patience,
        export_onnx=False,
        amp=False,
        num_workers=0,
    )
    loader = DataLoader(TensorDataset(torch.zeros(2, 1), torch.tensor([0, 1])), batch_size=2)
    record = training.fit(
        cfg,
        {s: loader for s in ("train", "val", "test")},
        root=tmp_path,
        device=torch.device("cpu"),
        tracker=Tracker(enabled=False),
        verbose=False,
    )
    assert record.metrics["best_epoch"] == best_epoch
    assert record.metrics["epochs_run"] == epochs_run
    assert record.metrics["select_metric"] == metric
    assert record.metrics["val"]["loss"] == losses[best_epoch - 1]
    best = load_checkpoint(record.path / "best.pt", torch.device("cpu"))
    last = load_checkpoint(record.path / "last.pt", torch.device("cpu"))
    assert torch.all(best["state_dict"]["weight"] == best_epoch)
    assert torch.all(last["state_dict"]["weight"] == epochs_run)
