"""The one training loop everybody uses: ``fit(cfg, loaders) -> RunRecord``.

Features: optimizer/scheduler from config, AMP on the GPU (HIP/ROCm or CUDA), gradient
clipping, early stopping and checkpoint selection on validation EER, best/last checkpoints,
per-epoch history, a decision threshold chosen on the validation set (FAR == FRR point) and
FAR/FRR/EER on every split at that threshold, optional Monte-Carlo dropout, optional MLflow.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from iml.config import ExperimentConfig
from iml.device import GPU, get_device
from iml.metrics import classification_metrics, eer
from iml.models import build_model, count_params
from iml.runs import (
    RUNS_ROOT,
    RunRecord,
    load_checkpoint,
    load_run,
    plot_history,
    run_dir,
    save_checkpoint,
    save_history,
    save_json,
    write_env,
)
from iml.seed import seed_everything
from iml.tracking import Tracker


def build_optimizer(cfg: ExperimentConfig, model: nn.Module) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"optimizer must be sgd|adam|adamw, got {cfg.optimizer!r}")


def build_scheduler(cfg: ExperimentConfig, optimizer: torch.optim.Optimizer):
    if cfg.scheduler == "none":
        return None
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.1
        )
    raise ValueError(f"scheduler must be none|cosine|step, got {cfg.scheduler!r}")


def dataset_labels(loader: DataLoader) -> np.ndarray:
    """Labels of a loader's dataset without decoding audio when the dataset exposes them."""
    ds = loader.dataset
    if hasattr(ds, "labels"):
        return np.asarray(ds.labels)
    if hasattr(ds, "items"):
        return np.asarray([int(label) for _, label in ds.items])
    return np.asarray([int(y) for _, y in ds])


def class_weights(cfg: ExperimentConfig, loader: DataLoader) -> torch.Tensor | None:
    """Inverse-frequency cross-entropy weights (``class_weight="balanced"``) or None."""
    if cfg.class_weight == "none":
        return None
    if cfg.class_weight != "balanced":
        raise ValueError(f"class_weight must be none|balanced, got {cfg.class_weight!r}")
    labels = dataset_labels(loader)
    counts = np.bincount(labels, minlength=2).astype(float)
    if (counts == 0).any():
        raise ValueError(f"class_weight=balanced needs both classes in train, got counts {counts}")
    w = counts.sum() / (2 * counts)
    return torch.tensor(w, dtype=torch.float32)


def enable_mc_dropout(model: nn.Module) -> None:
    """eval() everything except Dropout layers, which stay stochastic (PDF point (g))."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout | nn.Dropout2d):
            m.train()


@torch.no_grad()
def predict_proba(
    model: nn.Module, x: torch.Tensor, mc_samples: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """P(class 1) and its MC-dropout std for a batch. ``mc_samples=0`` -> deterministic, std=0."""
    if mc_samples <= 0:
        model.eval()
        p = torch.softmax(model(x), dim=1)[:, 1]
        return p, torch.zeros_like(p)
    enable_mc_dropout(model)
    ps = torch.stack([torch.softmax(model(x), dim=1)[:, 1] for _ in range(mc_samples)])
    model.eval()
    return ps.mean(0), ps.std(0)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mc_samples: int = 0,
    threshold: float = 0.5,
    weight: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Loss + FAR/FRR/EER/F1/accuracy over a loader at ``threshold``; also raw scores/labels."""
    model.eval()
    losses, scores, stds, labels = [], [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(nn.functional.cross_entropy(logits, y, weight=weight, reduction="sum").item())
        p, s = predict_proba(model, x, mc_samples)
        scores.append(p.cpu()), stds.append(s.cpu()), labels.append(y.cpu())
    scores, stds, labels = (torch.cat(t).numpy() for t in (scores, stds, labels))
    out = classification_metrics(labels, (scores >= threshold).astype(int), scores)
    out["loss"] = float(sum(losses) / max(1, len(labels)))
    out["threshold"] = float(threshold)
    if mc_samples > 0:
        out["mc_std_mean"] = float(stds.mean())
    out["_scores"], out["_labels"], out["_stds"] = scores, labels, stds
    return out


def _public(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


def _selection_score(ev: dict[str, Any]) -> float:
    """Higher is better: negative EER (threshold-free, the spec's criterion); macro-F1 if EER
    is undefined (single-class validation set)."""
    e = ev.get("eer", float("nan"))
    return -e if math.isfinite(e) else ev["macro_f1"]


def train_one_epoch(model, loader, optimizer, device, scaler, cfg, weight) -> float:
    model.train()
    total, n = 0.0, 0
    use_amp = scaler is not None
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=GPU, enabled=use_amp):  # GPU = HIP or CUDA
            loss = nn.functional.cross_entropy(model(x), y, weight=weight)
        if use_amp:
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer), scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(1, n)


def fit(
    cfg: ExperimentConfig,
    loaders: dict[str, DataLoader],
    root: str | Path = RUNS_ROOT,
    device: torch.device | None = None,
    tracker: Tracker | None = None,
    verbose: bool = True,
    overwrite: bool = False,
) -> RunRecord:
    """Train according to ``cfg`` on ``loaders["train"]``, select on ``loaders["val"]``,
    report on every split present, and write the run folder. Returns the RunRecord.

    A finished run folder (``metrics.json`` present) is returned as is unless
    ``overwrite=True``, so re-executing a notebook cell never destroys a result.
    """
    if "train" not in loaders:
        raise ValueError("loaders must contain 'train'")
    select_split = "val" if "val" in loaders else "train"
    out = run_dir(cfg, root)
    if (out / "metrics.json").exists():
        if not overwrite:
            if verbose:
                print(f"run already finished, loading it: {out} (overwrite=True to retrain)")
            return load_run(out)
        (out / "metrics.json").unlink()  # a crash now cannot leave a half-old finished folder
    device = device or get_device()
    seed_everything(cfg.seed)
    out.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out / "config.json")
    write_env(out / "env.json", device)

    model = build_model(cfg).to(device)
    train_model = torch.compile(model) if cfg.torch_compile else model
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    scaler = torch.amp.GradScaler(GPU) if (cfg.amp and device.type == GPU) else None
    weight = class_weights(cfg, loaders["train"])
    weight = weight.to(device) if weight is not None else None
    tracker = tracker or Tracker()
    tracker.start(cfg, out.name)
    if verbose:
        print(
            f"run: {out}\nmodel: {cfg.model} ({count_params(model) / 1e6:.2f}M params) on {device}"
        )

    history: list[dict[str, Any]] = []
    best_score, best_epoch, bad_epochs = -math.inf, 0, 0
    epochs = tqdm(range(1, cfg.epochs + 1), disable=not verbose, unit="epoch")
    for epoch in epochs:
        t0 = time.time()
        train_loss = train_one_epoch(
            train_model, loaders["train"], optimizer, device, scaler, cfg, weight
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        ev = _public(evaluate(model, loaders[select_split], device, weight=weight))
        row.update({f"val_{k}": v for k, v in ev.items() if k not in ("n", "n_pos", "n_neg")})
        row["time_s"] = time.time() - t0
        if scheduler is not None:
            scheduler.step()
        history.append(row)
        tracker.log_metrics(row, step=epoch)
        epochs.set_postfix(
            loss=f"{train_loss:.3f}",
            val_eer=f"{ev['eer']:.3f}",
            far=f"{ev['far']:.2f}",
            frr=f"{ev['frr']:.2f}",
        )

        score = _selection_score(ev)
        if score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            save_checkpoint(out / "best.pt", model, cfg, epoch, ev)
        else:
            bad_epochs += 1
        if not math.isfinite(train_loss):
            print("non-finite training loss, stopping")
            break
        if cfg.early_stopping_patience and bad_epochs >= cfg.early_stopping_patience:
            if verbose:
                print(f"early stopping at epoch {epoch} (best {best_epoch})")
            break
    save_checkpoint(out / "last.pt", model, cfg, epoch)

    # Final report with the best checkpoint. The decision threshold is the FAR == FRR point
    # of the selection split (val), then applied unchanged to every split (PDF: FAR/FRR on
    # train/val/test, both equally important).
    model.load_state_dict(load_checkpoint(out / "best.pt", device)["state_dict"])
    # same MC-dropout setting as the final report, so the threshold matches the scores it is used on
    sel = evaluate(model, loaders[select_split], device, mc_samples=cfg.mc_dropout_samples)
    _, thr = eer(sel["_labels"], sel["_scores"])
    threshold = thr if math.isfinite(thr) else 0.5
    metrics: dict[str, Any] = {
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "params": count_params(model),
        "select_split": select_split,
        "select_metric": "eer" if math.isfinite(sel.get("eer", float("nan"))) else "macro_f1",
        "decision_threshold": threshold,
    }
    for split, loader in loaders.items():
        res = evaluate(
            model, loader, device, mc_samples=cfg.mc_dropout_samples, threshold=threshold
        )
        metrics[split] = _public(res)
        np.savez_compressed(
            out / f"predictions_{split}.npz",
            scores=res["_scores"],
            labels=res["_labels"],
            stds=res["_stds"],
        )
    save_checkpoint(
        out / "best.pt",
        model,
        cfg,
        best_epoch,
        {"decision_threshold": threshold, **metrics[select_split]},
    )
    save_json(metrics, out / "metrics.json")
    df = save_history(history, out)
    plot_history(df, out)
    final = {f"{s}_{k}": v for s in loaders for k, v in metrics[s].items()}
    final["decision_threshold"] = threshold
    tracker.log_metrics(
        final, step=epoch
    )  # same step as the last epoch so MLflow shows these as latest
    tracker.log_artifacts(out)
    tracker.end()
    if verbose:
        keys = ("accuracy", "macro_f1", "far", "frr", "eer")
        print(f"decision threshold (val EER point): {threshold:.3f}")
        print({s: {k: round(v, 3) for k, v in metrics[s].items() if k in keys} for s in loaders})
    return RunRecord(path=out, cfg=cfg, metrics=metrics, history=df)
