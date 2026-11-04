"""Run folders: everything one experiment leaves on disk, and how to read it back.

runs/<member>/<topic>/<run_name>/
    config.json    full ExperimentConfig
    metrics.json   {"train": {...}, "val": {...}, "test": {...}, "best_epoch": n, ...}
    history.csv    one row per epoch (loss, val metrics, lr, time)
    loss.png       train/val loss vs epoch
    metrics.png    val macro-F1, FAR, FRR vs epoch
    best.pt        state_dict + config of the best-val epoch (what app/ loads)
    last.pt        state_dict of the final epoch (for resuming)

Also the metrics required by the project description that go into ``metrics.json``: FAR, FRR
(+ EER, accuracy, macro-F1): ``far``, ``frr``, ``eer``, ``classification_metrics`` (bottom).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from iml.config import ExperimentConfig  # noqa: E402

RUNS_ROOT = Path(__file__).resolve().parents[2] / "runs"


def run_dir(cfg: ExperimentConfig, root: str | Path = RUNS_ROOT) -> Path:
    return Path(root) / cfg.member / cfg.topic / cfg.run_name()


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def save_history(history: list[dict[str, Any]], directory: Path) -> pd.DataFrame:
    df = pd.DataFrame(history)
    df.to_csv(directory / "history.csv", index=False)
    return df


def plot_history(df: pd.DataFrame, directory: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["epoch"], df["train_loss"], label="train loss")
    if "val_loss" in df:
        ax.plot(df["epoch"], df["val_loss"], label="val loss")
    ax.set_xlabel("epoch"), ax.set_ylabel("cross-entropy"), ax.legend(), ax.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(directory / "loss.png", dpi=120), plt.close(fig)

    cols = [c for c in ("val_macro_f1", "val_far", "val_frr") if c in df]
    if cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        for c in cols:
            ax.plot(df["epoch"], df[c], label=c.removeprefix("val_"))
        ax.set_xlabel("epoch"), ax.set_ylim(0, 1), ax.legend(), ax.grid(alpha=0.3)
        ax.set_title("validation metrics")
        fig.tight_layout(), fig.savefig(directory / "metrics.png", dpi=120), plt.close(fig)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    cfg: ExperimentConfig,
    epoch: int,
    metrics: dict[str, Any] | None = None,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "epoch": epoch,
            "metrics": metrics or {},
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    ckpt["config"] = ExperimentConfig.from_dict(ckpt["config"])
    return ckpt


@dataclass
class RunRecord:
    """A finished run read back from disk; what compare.py and app/ work with."""

    path: Path
    cfg: ExperimentConfig
    metrics: dict[str, Any]
    history: pd.DataFrame

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def best_checkpoint(self) -> Path:
        return self.path / "best.pt"


def load_run(path: str | Path) -> RunRecord:
    path = Path(path)
    return RunRecord(
        path=path,
        cfg=ExperimentConfig.from_json(path / "config.json"),
        metrics=load_json(path / "metrics.json"),
        history=pd.read_csv(path / "history.csv"),
    )


def find_runs(root: str | Path = RUNS_ROOT) -> list[RunRecord]:
    """All finished runs (those with metrics.json) under ``root``, any depth."""
    return [load_run(p.parent) for p in sorted(Path(root).rglob("metrics.json"))]


# =================================================================================================
# metrics
# Metrics required by the project description: FAR, FRR (+ EER, accuracy, macro-F1).
#
# Convention: label 1 = allowed person (accept), label 0 = not allowed (reject).
# ``scores`` are P(class 1) in [0, 1]; ``y_pred`` are hard 0/1 decisions.
# =================================================================================================


def far(y_true, y_pred) -> float:
    """False Acceptance Ratio: wrongly accepted class-0 samples / all class-0 samples."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    neg = y_true == 0
    if neg.sum() == 0:
        return float("nan")
    return float((y_pred[neg] == 1).mean())


def frr(y_true, y_pred) -> float:
    """False Rejection Ratio: wrongly rejected class-1 samples / all class-1 samples."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    pos = y_true == 1
    if pos.sum() == 0:
        return float("nan")
    return float((y_pred[pos] == 0).mean())


def eer(y_true, scores) -> tuple[float, float]:
    """Equal Error Rate and the score threshold at which FAR == FRR.

    Sweeps all thresholds via the ROC curve (FAR = FPR, FRR = 1 - TPR) and returns the point
    where the two are closest. Needs both classes present.
    """
    y_true, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y_true, scores, drop_intermediate=False)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    if fpr[i] == 0.0 and fnr[i] == 0.0:
        # perfectly separable: put the threshold in the middle of the gap between the classes
        # instead of on the lowest positive score, so near-boundary samples are not rejected
        thr_i = (scores[y_true == 0].max() + scores[y_true == 1].min()) / 2
    else:
        thr_i = thr[i]
    return float((fpr[i] + fnr[i]) / 2), float(np.clip(thr_i, 0.0, 1.0))


def classification_metrics(y_true, y_pred, scores=None) -> dict[str, float]:
    """All metrics for one split as a flat dict (what goes into metrics.json)."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "far": far(y_true, y_pred),
        "frr": frr(y_true, y_pred),
        "n": int(len(np.asarray(y_true))),
        "n_pos": int((np.asarray(y_true) == 1).sum()),
        "n_neg": int((np.asarray(y_true) == 0).sum()),
    }
    if scores is not None:
        e, t = eer(y_true, scores)
        out["eer"] = e
        out["eer_threshold"] = t
    return out
