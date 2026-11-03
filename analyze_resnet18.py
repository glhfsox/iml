"""Compare ImageNet fine-tuning runs, audit predictions, and reproduce report figures."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iml.cli import check_run_data, data_fingerprint, expand_grid, load_splits, record_run_data
from iml.config import ExperimentConfig
from iml.runs import load_run, run_dir
from iml.tracking import Tracker, notebook_run
from iml.train import fit

DATA_SHA1 = "10ac1fb033e3b1845bc52ea2e2229c2982ad11f0"
# Shared template pipeline; this team experiment fine-tunes ImageNet weights only.
BASE = ExperimentConfig(
    model="resnet18",
    member="petr",
    topic="resnet18_review_grid",
    pretrained=True,
    init="he",  # Initializes the new classifier; pretrained backbone weights are retained.
    epochs=20,
    export_onnx=False,
)
GRID = {"lr": [1e-3, 3e-4], "dropout": [0.0, 0.2]}
QUALITY = ("accuracy", "far", "frr", "eer", "macro_f1")


def obtain(cfg, fp, data, train):
    if not cfg.pretrained:
        raise ValueError("This experiment supports ImageNet fine-tuning only")
    out = run_dir(cfg)
    check_run_data(out, fp, overwrite=False)
    if not (out / "metrics.json").exists():
        if not train:
            raise FileNotFoundError(f"Missing {out}; pass --train to train missing runs")
        result = fit(cfg, load_splits(cfg, data_dir=data), tracker=Tracker(enabled=False))
        record_run_data(out, fp)
    else:
        print(f"Cached: {out.name}", flush=True)
        result = load_run(out)
    return result


def audit_data(data):
    frames = {s: pd.read_csv(data / f"{s}.csv") for s in ("train", "val", "test")}
    sources, hashes, groups = {}, {}, {}
    for split, df in frames.items():
        sources[split] = {Path(p).stem.rsplit("_", 1)[0] for p in df.path}
        counter = Counter()
        for p, y in zip(df.path, df.allowed, strict=True):
            name = Path(p).stem
            speaker = name.split("_")[0]
            counter[("N" if speaker.startswith("N") else speaker, int(y))] += 1
            key = hashlib.sha256(np.load(data / p, allow_pickle=False).tobytes()).hexdigest()
            if key in hashes:
                previous_split, previous_label = hashes[key]
                if previous_split != split or previous_label != int(y):
                    raise ValueError("Duplicate audio across splits or with conflicting labels")
            hashes[key] = split, int(y)
        groups[split] = {f"{speaker}:allowed={label}": n for (speaker, label), n in counter.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if sources[a] & sources[b]:
            raise ValueError(f"Recording ID overlap between {a} and {b}")
    t_recordings = {
        s: len({Path(p).stem.rsplit("_", 1)[0] for p in df.path if Path(p).name.startswith("T_")})
        for s, df in frames.items()
    }
    return frames, {"groups": groups, "T_source_recordings": t_recordings}


def audit_predictions(record, frames):
    threshold = record.metrics["decision_threshold"]
    errors = {}
    for split in ("train", "val", "test"):
        saved = np.load(record.path / f"predictions_{split}.npz")
        labels, scores = saved["labels"], saved["scores"]
        if split != "train" and not np.array_equal(labels, frames[split].allowed.to_numpy()):
            raise ValueError(f"Prediction/manifest label mismatch: {record.name}/{split}")
        predicted = (scores >= threshold).astype(int)
        positive, negative = scores[labels == 1], scores[labels == 0]
        fp = int((negative >= threshold).sum())
        fn = int((positive < threshold).sum())
        tp, tn = len(positive) - fn, len(negative) - fp
        actual = {
            "accuracy": (tp + tn) / len(labels),
            "far": fp / len(negative),
            "frr": fn / len(positive),
            "macro_f1": (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)) / 2,
        }
        for key, value in actual.items():
            if not np.isclose(value, record.metrics[split][key], atol=1e-12):
                raise ValueError(f"Metric mismatch: {record.name}/{split}/{key}")
        # Independent empirical ROC sweep. Retain all exactly equidistant candidates:
        # the shared nearest-point estimate can choose either due to float rounding.
        thresholds = np.r_[np.inf, np.unique(scores)[::-1]]
        fps = (negative[:, None] >= thresholds).sum(axis=0)
        fns = (positive[:, None] < thresholds).sum(axis=0)
        distances = np.abs(fps * len(positive) - fns * len(negative))
        rates = (fps / len(negative) + fns / len(positive)) / 2
        candidates = rates[distances == distances.min()]
        if not np.any(np.isclose(candidates, record.metrics[split]["eer"], atol=1e-12)):
            raise ValueError(f"EER mismatch: {record.name}/{split}")
        if split != "train":
            errors[split] = dict(
                Counter(
                    "N" if Path(p).name.startswith("N") else Path(p).stem.split("_")[0]
                    for p in frames[split].loc[predicted != labels, "path"]
                )
            )
    return errors


def plot_grid(records, out):
    """Artem's three-panel layout: consistent run colors, bottom legend, checkpoint rings."""
    plt.rcParams.update({"font.size": 11})
    colors = ["#0072b2", "#e69f00", "#009e73", "#d55e00"]
    markers = ["o", "s", "^", "D"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    handles, labels = [], []
    for index, ((name, r), color, marker) in enumerate(zip(records, colors, markers, strict=True)):
        h = r.history
        label = (
            f"{name}: LR={r.cfg.lr:g}, dropout={r.cfg.dropout:g}; "
            f"best EER {100 * r.metrics['val']['eer']:.2f}% (epoch {r.metrics['best_epoch']})"
        )
        for ax, column, scale in zip(
            axes, ("train_loss", "val_loss", "val_eer"), (1, 1, 100), strict=True
        ):
            (line,) = ax.plot(
                h.epoch, h[column] * scale, color=color, marker=marker, markersize=4, linewidth=1.8
            )
            if column == "val_eer":
                handles.append(line)
                labels.append(label)
                best = h.loc[h.epoch == r.metrics["best_epoch"]].iloc[0]
                ax.scatter(
                    best.epoch,
                    best.val_eer * 100,
                    s=140 + 50 * index,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.6,
                    zorder=5,
                )
    loss_max = max(
        float(r.history[c].max()) for _, r in records for c in ("train_loss", "val_loss")
    )
    for i, ax in enumerate(axes):
        ax.set_title(("Training cross-entropy", "Validation cross-entropy", "Validation EER")[i])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("EER (%)" if i == 2 else "Cross-entropy loss (symlog)")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlim(0.5, 20.5)
        ax.grid(alpha=0.3)
        if i < 2:
            ax.set_yscale("symlog", linthresh=0.01)
            ax.set_ylim(0, loss_max * 1.6)
            ax.set_yticks([0, 0.01, 0.1, 1, 10], ["0", "0.01", "0.1", "1", "10"])
        else:
            top = max(float(r.history.val_eer.max() * 100) for _, r in records)
            ax.set_ylim(-2, max(5, top * 1.08))
    fig.suptitle(
        "ResNet-18: ImageNet fine-tuning | learning rate x dropout | 20 epochs", fontsize=15
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.07),
    )
    fig.text(
        0.5,
        0.02,
        "Rings: retained checkpoints. Overlapping traces have equal EER.\n"
        "Loss: shared symlog scale, linear below 0.01; large early spikes remain visible.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.2, 1, 0.93))
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)


def plot_scores(record, name, frames, out):
    """Show actual clip scores; y positions are labeled groups, not embedding coordinates."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    threshold = record.metrics["decision_threshold"]
    for ax, split in zip(axes, ("val", "test"), strict=True):
        p = np.load(record.path / f"predictions_{split}.npz")
        labels, scores = p["labels"], p["scores"]
        paths = frames[split].path
        masks = [
            labels == 1,
            np.array([Path(x).name.startswith("T_") for x in paths]),
            np.array([Path(x).name.startswith("N") for x in paths]),
        ]
        for row, mask, color in zip(
            range(3), masks, ("#0072b2", "#d55e00", "#666666"), strict=True
        ):
            values = scores[mask]
            y = row + np.linspace(-0.15, 0.15, len(values))
            wrong = (values >= threshold).astype(int) != labels[mask]
            ax.scatter(values[~wrong], y[~wrong], s=30, color=color, marker="o")
            ax.scatter(values[wrong], y[wrong], s=65, color=color, marker="x", linewidths=1.6)
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2)
        ax.set_title("Validation: 63 clips" if split == "val" else "Test: 63 clips")
        ax.set_xlabel("Allowed-class score (0–1)")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlim(-0.025, 1.025)
        ax.set_yticks([0, 1, 2], ["Allowed (28)", "Not allowed: T (7)", "Not allowed: N (28)"])
        ax.tick_params(labelleft=True)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(
        f"{name}, checkpoint epoch {record.metrics['best_epoch']}: validation vs test scores",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.02,
        f"Dashed line: fixed validation threshold {threshold:.6f}. Circles: correct; x: errors. "
        "\nAccept when score >= threshold; vertical offsets separate clips.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(out / "score_distributions.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/training_dataset_v2"))
    parser.add_argument(
        "--train", action="store_true", help="Train missing runs; keep completed runs"
    )
    parser.add_argument("--out", type=Path, default=Path("runs/petr/resnet18_review/report"))
    args = parser.parse_args()
    fp = data_fingerprint(args.data)
    if fp["sha1"] != DATA_SHA1:
        raise ValueError("Dataset differs from this report; use the recorded dataset snapshot")
    frames, data_audit = audit_data(args.data)
    configs = expand_grid(BASE, GRID)
    records = []
    for name, cfg in [(f"FT{i}", c) for i, c in enumerate(configs, 1)]:
        print(f"START {name}: {cfg.to_dict()}", flush=True)
        r = obtain(cfg, fp, args.data, args.train)
        expected = list(range(1, int(r.metrics["epochs_run"]) + 1))
        if r.history.epoch.tolist() != expected or len(expected) != 20:
            raise ValueError(f"Incomplete history: {name}")
        errors = audit_predictions(r, frames)
        records.append((name, r, errors))
    grid = [(name, r) for name, r, _ in records]
    selected_name, selected = min(
        grid, key=lambda x: (x[1].metrics["val"]["eer"], x[1].metrics["val"]["loss"], x[0])
    )
    if data_fingerprint(args.data)["sha1"] != fp["sha1"]:
        raise ValueError("Dataset changed during training; report generation refused")
    args.out.mkdir(parents=True, exist_ok=True)
    plot_grid(grid, args.out)
    plot_scores(selected, selected_name, frames, args.out)
    summary = {"dataset": fp, "data_audit": data_audit, "selected": selected_name, "runs": []}
    for name, r, errors in records:
        variant = f"{name}: lr={r.cfg.lr:g},dropout={r.cfg.dropout:g},pretrained={r.cfg.pretrained}"
        params = {"model": r.cfg.model, **r.cfg.diff_from_default()}
        metrics = {f"{s}_{k}": r.metrics[s][k] for s in ("val", "test") for k in QUALITY}
        receipt = r.path / "review_tracking.json"
        tracking = json.loads(receipt.read_text()) if receipt.exists() else None
        if tracking is None:
            with notebook_run(
                member=r.cfg.member,
                topic=r.cfg.topic,
                run_name=r.name,
                params=params,
                tags={"variant": variant},
            ) as run:
                run.log_metrics(metrics)
                tracking = {"active": run.active, "run_id": run.run_id}
            if tracking["active"]:
                receipt.write_text(json.dumps(tracking, indent=2) + "\n")
        summary["runs"].append(
            {
                "id": name,
                "path": str(r.path),
                "config": r.cfg.to_dict(),
                "metrics": r.metrics,
                "errors_by_group": errors,
                "tracking": tracking,
            }
        )
    (args.out / "review.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("REVIEW_SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
