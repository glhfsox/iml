"""Run the template ResNet-18 grid, audit predictions, and plot every epoch of every run."""

import argparse
import hashlib
import json
import math
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
# Same defaults and grid as notebooks/_template/architecture/experiments.ipynb.
BASE = ExperimentConfig(
    model="resnet18",
    member="petr",
    topic="resnet18_review_grid",
    epochs=20,
    export_onnx=False,
)
GRID = {"lr": [1e-3, 3e-4], "dropout": [0.0, 0.2], "pretrained": [True, False]}
RECHECK = ExperimentConfig(
    model="resnet18",
    member="petr",
    topic="resnet18_review_recheck",
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
QUALITY = ("accuracy", "far", "frr", "eer", "macro_f1")


def obtain(cfg, fp, data, train):
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
    plt.rcParams.update({"font.size": 11})
    colors = {1e-3: "#176ca4", 3e-4: "#bb5e16"}
    groups = [(True, 0.0), (True, 0.2), (False, 0.0), (False, 0.2)]
    for kind, filename in (("eer", "validation_eer.png"), ("loss", "loss.png")):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, squeeze=False)
        for ax, (pretrained, dropout) in zip(axes.flat, groups, strict=True):
            for name, r in records:
                if (r.cfg.pretrained, r.cfg.dropout) != (pretrained, dropout):
                    continue
                h, color = r.history, colors[r.cfg.lr]
                label = f"{name}: LR={r.cfg.lr:g}"
                if kind == "eer":
                    ax.plot(
                        h.epoch,
                        h.val_eer * 100,
                        "o-",
                        color=color,
                        markersize=3,
                        label=f"{label}; best epoch {r.metrics['best_epoch']}",
                    )
                    best = h.loc[h.epoch == r.metrics["best_epoch"]].iloc[0]
                    ax.scatter(
                        best.epoch,
                        best.val_eer * 100,
                        marker="*",
                        s=140,
                        color=color,
                        edgecolor="black",
                        linewidth=0.5,
                        zorder=4,
                    )
                else:
                    ax.plot(h.epoch, h.train_loss, "--", color=color, label=f"{name} train")
                    ax.plot(h.epoch, h.val_loss, "-", color=color, label=f"{name} validation")
            ax.set_title(
                f"{'ImageNet fine-tuning' if pretrained else 'From scratch'} | dropout={dropout:g}"
            )
            ax.set_xlabel("Epoch (one pass over training batches)")
            ax.set_ylabel("Validation EER (%)" if kind == "eer" else "Cross-entropy loss")
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.set_xlim(0.5, 20.5)
            ax.tick_params(labelbottom=True)
            ax.grid(alpha=0.25)
            if kind == "eer":
                top = max(float(r.history.val_eer.max() * 100) for _, r in records)
                ax.set_ylim(-1, max(10, math.ceil(top / 10) * 10) + 2)
            else:
                ax.set_ylim(bottom=0)
            ax.legend(fontsize=11, loc="upper right")
        metric_title = (
            "Validation EER at every epoch" if kind == "eer" else "Training and validation loss"
        )
        fig.suptitle(f"ResNet-18 | {metric_title}", fontsize=15, y=0.99)
        caption = (
            "G1–G8: eight fresh seed-42 runs, 20 epochs each; no early stopping.\n"
            "Stars mark retained checkpoints."
            if kind == "eer"
            else "Blue: LR=0.001; orange: LR=0.0003. Dashed: online training loss (training mode). "
            "\nSolid: end-of-epoch validation loss (evaluation mode); independent linear y scales."
        )
        fig.text(0.5, 0.015, caption, ha="center", fontsize=11)
        fig.tight_layout(rect=(0, 0.065, 1, 0.965))
        fig.savefig(out / filename, dpi=150)
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
    for name, cfg in [("R0", RECHECK)] + [(f"G{i}", c) for i, c in enumerate(configs, 1)]:
        print(f"START {name}: {cfg.to_dict()}", flush=True)
        r = obtain(cfg, fp, args.data, args.train)
        expected = list(range(1, int(r.metrics["epochs_run"]) + 1))
        if r.history.epoch.tolist() != expected or (name != "R0" and len(expected) != 20):
            raise ValueError(f"Incomplete history: {name}")
        errors = audit_predictions(r, frames)
        records.append((name, r, errors))
    grid = [(name, r) for name, r, _ in records if name != "R0"]
    selected_name, selected = min(
        grid, key=lambda x: (x[1].metrics["val"]["eer"], x[1].metrics["val"]["loss"], x[0])
    )
    if data_fingerprint(args.data)["sha1"] != fp["sha1"]:
        raise ValueError("Dataset changed during training; report generation refused")
    args.out.mkdir(parents=True, exist_ok=True)
    plot_grid(grid, args.out)
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
