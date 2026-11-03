"""Reproduce the ResNet-18 sweep and publish two figures, keeping run artifacts in runs/."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from iml.cli import check_run_data, data_fingerprint, train_main
from iml.config import ExperimentConfig
from iml.runs import run_dir

DATA_SHA1 = "10ac1fb033e3b1845bc52ea2e2229c2982ad11f0"
BASE = ExperimentConfig(
    member="petr",
    topic="resnet18_lr",
    model="resnet18",
    pretrained=True,
    optimizer="adamw",
    batch_size=16,
    dropout=0.2,
    weight_decay=0.05,
    lr=2e-4,
    epochs=12,
    early_stopping_patience=4,
    seed=42,
    export_onnx=False,
)


def read_run(cfg, fingerprint, data, train):
    folder = run_dir(cfg)
    check_run_data(folder, fingerprint, overwrite=False)
    if train:
        folder.mkdir(parents=True, exist_ok=True)
        config_path = folder / "config.json"
        cfg.to_json(config_path)
        train_main(["--config", str(config_path), "--data", str(data)])
    if not (folder / "metrics.json").exists():
        raise FileNotFoundError(f"Missing run {folder}; pass --train to train it")
    return cfg, json.loads((folder / "metrics.json").read_text()), folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/training_dataset_v2", type=Path)
    parser.add_argument(
        "--train", action="store_true", help="Train missing runs; reuse finished runs"
    )
    parser.add_argument("--out", type=Path, default=Path("runs/petr/resnet18_lr/report"))
    args = parser.parse_args()
    fingerprint = data_fingerprint(args.data)
    if fingerprint["sha1"] != DATA_SHA1:
        raise ValueError("Dataset differs from the report; use the original dataset snapshot")
    trials = [
        read_run(replace(BASE, lr=lr), fingerprint, args.data, args.train)
        for lr in (5e-5, 2e-4, 5e-4)
    ]
    # Choose LR among seed-42 runs only. Test metrics never participate in selection.
    selected = min(trials, key=lambda r: (r[1]["val"]["eer"], r[1]["val"]["loss"], r[0].lr))
    repeats = [
        read_run(replace(selected[0], seed=seed), fingerprint, args.data, args.train)
        for seed in (43, 44)
    ]
    args.out.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(selected[2] / "history.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.epoch, history.train_loss, "o-", label="Training")
    ax.plot(history.epoch, history.val_loss, "o-", label="Validation")
    ax.axvline(selected[1]["best_epoch"], color="gray", linestyle="--", label="Selected checkpoint")
    ax.set(
        xlabel="Epoch",
        ylabel="Cross-entropy loss",
        title="ResNet-18: selected run (LR 0.0002, seed 42)",
    )
    ax.set_xticks(history.epoch)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "loss.png", dpi=150)
    plt.close(fig)

    runs = trials + repeats
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        range(len(runs)),
        [100 * r[1]["val"]["eer"] for r in runs],
        color=["#789bb5", "#22648c", "#789bb5", "#78a389", "#78a389"],
    )
    ax.set_xticks(range(len(runs)), [f"LR {r[0].lr:g}\nseed {r[0].seed}" for r in runs])
    ax.bar_label(bars, fmt="%.2f%%", padding=4)
    ax.set(
        xlabel="Learning-rate trials (seed 42) and repeats of selected LR",
        ylabel="Validation EER (%) — lower is better",
        ylim=(0, 5),
        title="ResNet-18: checkpoint validation EER",
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.out / "validation_eer.png", dpi=150)
    plt.close(fig)

    print(f"Selected LR: {selected[0].lr:g}; seed: 42; epoch: {selected[1]['best_epoch']}")
    for cfg, metrics, _ in runs:
        print(f"LR {cfg.lr:g}, seed {cfg.seed}: val EER {metrics['val']['eer']:.2%}")
    test = selected[1]["test"]
    print(f"Test: accuracy {test['accuracy']:.2%}, FAR {test['far']:.2%}, FRR {test['frr']:.2%}")
    print(f"Figures: {args.out}")


if __name__ == "__main__":
    main()
