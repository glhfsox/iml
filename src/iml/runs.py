"""Run folders: everything one experiment leaves on disk, and how to read it back.

runs/<member>/<topic>/<run_name>/
    config.json    full ExperimentConfig
    metrics.json   {"train": {...}, "val": {...}, "test": {...}, "best_epoch": n, ...}
    history.csv    one row per epoch (loss, val metrics, lr, time)
    loss.png       train/val loss vs epoch
    metrics.png    val macro-F1, FAR, FRR vs epoch
    best.pt        state_dict + config of the best-val epoch (what app/ loads)
    last.pt        state_dict of the final epoch (for resuming)
    env.json       git commit, torch/python versions, device, timestamp
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from iml.config import ExperimentConfig  # noqa: E402
from iml.device import device_info  # noqa: E402

RUNS_ROOT = Path(__file__).resolve().parents[2] / "runs"


def run_dir(cfg: ExperimentConfig, root: str | Path = RUNS_ROOT) -> Path:
    return Path(root) / cfg.member / cfg.topic / cfg.run_name()


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write_env(path: str | Path, device: torch.device) -> None:
    save_json(
        {
            "git_commit": git_commit(),
            "time": datetime.now().isoformat(timespec="seconds"),
            **device_info(device),
        },
        path,
    )


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
