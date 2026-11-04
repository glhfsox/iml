"""The one training loop everybody uses: ``fit(cfg, loaders) -> RunRecord``.

Features: optimizer/scheduler from config, AMP on the GPU (HIP/ROCm or CUDA), gradient
clipping, early stopping and checkpoint selection on validation EER, best/last checkpoints,
per-epoch history, the loss of ``cfg.loss`` and two-phase fine-tuning (``cfg.finetune_from``,
see ``iml.imbalance``), a decision threshold chosen on the validation set (FAR == FRR point) and
FAR/FRR/EER on every split at that threshold, optional Monte-Carlo dropout, optional MLflow,
and a verified ``model.onnx`` next to ``best.pt`` (``cfg.export_onnx``).

The device and reproducibility helpers every run uses sit at the top: ``get_device`` /
``GPU``
"""

from __future__ import annotations

import math
import os
import random
import time
import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from iml.config import ExperimentConfig
from iml.data import make_loaders
from iml.export import export_onnx, export_verified
from iml.imbalance import (
    balanced_subset,
    criterion,
    dataset_labels,
    freeze_backbone,
    keep_frozen_bn_eval,
)
from iml.models import build_model, check_knobs, count_params
from iml.models.spec import describe_spec
from iml.runs import (
    RUNS_ROOT,
    RunRecord,
    classification_metrics,
    eer,
    load_checkpoint,
    load_run,
    plot_history,
    run_dir,
    save_checkpoint,
    save_history,
    save_json,
)
from iml.tracking import Tracker

# =================================================================================================
# device: pick the best available device: AMD GPU via HIP (ROCm) or NVIDIA via CUDA -> Apple MPS
# -> CPU.
#
# PyTorch exposes the HIP backend under the ``"cuda"`` device name (``torch.device("hip")``
# does not exist), so on the ROCm server ``get_device()`` returns ``cuda`` and the AMD card is
# used. ``torch.version.hip`` tells the two apart.
# =================================================================================================

GPU = "cuda"  # device string PyTorch uses for both HIP (ROCm) and CUDA
_ALIASES = {"hip": GPU, "rocm": GPU, "gpu": GPU}


def is_hip() -> bool:
    """True when this torch build runs on ROCm (AMD GPU through HIP)."""
    return getattr(torch.version, "hip", None) is not None


def get_device(prefer: str | None = None) -> torch.device:
    """Return ``torch.device``; ``prefer`` forces a backend ("hip"/"cuda", "mps", "cpu")."""
    if prefer is not None:
        return torch.device(_ALIASES.get(prefer.lower(), prefer))
    if torch.cuda.is_available():  # HIP on ROCm builds, CUDA on NVIDIA builds
        return torch.device(GPU)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =================================================================================================
# seed: reproducibility helpers
# =================================================================================================


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (all devices).

    ``deterministic=True`` additionally disables cuDNN autotuning; slower but bit-reproducible
    on the same hardware. Leave it off for sweeps where speed matters more.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


# =================================================================================================
# training loop
# =================================================================================================


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
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Loss + FAR/FRR/EER/F1/accuracy over a loader at ``threshold``; also raw scores/labels.

    The loss is ``loss_fn`` (default: cross-entropy with ``weight``) over the whole split at once.
    """
    model.eval()
    logits, scores, stds, labels = [], [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits.append(model(x).float())
        p, s = predict_proba(model, x, mc_samples)
        scores.append(p.cpu()), stds.append(s.cpu()), labels.append(y)
    loss_fn = loss_fn or partial(nn.functional.cross_entropy, weight=weight)
    loss = loss_fn(torch.cat(logits), torch.cat(labels)).item()
    scores, stds, labels = (torch.cat(t).cpu().numpy() for t in (scores, stds, labels))
    out = classification_metrics(labels, (scores >= threshold).astype(int), scores)
    out["loss"] = float(loss)
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


def train_one_epoch(model, loader, optimizer, device, scaler, cfg, loss_fn) -> float:
    model.train()
    if cfg.finetune_from:
        keep_frozen_bn_eval(model)
    total, n = 0.0, 0
    use_amp = scaler is not None
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=GPU, enabled=use_amp):  # GPU = HIP or CUDA
            loss = loss_fn(model(x), y)
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


def _export_run_onnx(out: Path, verbose: bool) -> float | None:
    """``best.pt`` -> ``model.onnx`` next to it, verified against PyTorch before it gets its name;
    a failure only warns and leaves no graph behind."""
    path = out / "model.onnx"
    try:
        diff = export_verified(path, lambda tmp: export_onnx(out, tmp), run=out)
    except Exception as e:  # the run itself is fine; the graph can be re-exported by hand
        warnings.warn(
            f"ONNX export of {out.name} failed ({e}); use `iml-export --run {out}`", stacklevel=2
        )
        return None
    if verbose:
        mb = path.stat().st_size / 1e6
        print(f"model.onnx written ({mb:.1f} MB, max |onnx - torch| {diff:.1e})")
    return diff


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
    check_knobs(cfg)  # fail before a run folder exists
    if cfg.model == "custom":
        describe_spec(cfg.arch)  # validates the block list
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
    (out / "model.onnx").unlink(missing_ok=True)  # never ship a previous run's graph
    cfg.to_json(out / "config.json")

    model = build_model(cfg, load_pretrained=not cfg.finetune_from).to(device)
    train_loader = loaders["train"]
    if cfg.finetune_from:  # two-phase fine-tuning, phase 2: head only, balanced train subset
        src = Path(cfg.finetune_from)
        src = src / "best.pt" if src.is_dir() else src
        model.load_state_dict(load_checkpoint(src, device)["state_dict"])
        freeze_backbone(model)
        balanced = {"train": balanced_subset(train_loader.dataset, cfg.seed)}
        train_loader = make_loaders(balanced, cfg, train_loader.num_workers)["train"]
    train_model = torch.compile(model) if cfg.torch_compile else model
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    scaler = torch.amp.GradScaler(GPU) if (cfg.amp and device.type == GPU) else None
    weight = class_weights(cfg, train_loader)
    weight = weight.to(device) if weight is not None else None
    loss_fn = criterion(cfg, weight)
    tracker = tracker or Tracker()
    tracker.start(cfg, out.name)
    if verbose:
        n_params = count_params(model, trainable_only=False) / 1e6
        print(f"run: {out}\nmodel: {cfg.model} ({n_params:.2f}M params) on {device}")

    history: list[dict[str, Any]] = []
    best_score, best_epoch, bad_epochs = -math.inf, 0, 0
    epochs = tqdm(range(1, cfg.epochs + 1), disable=not verbose, unit="epoch")
    for epoch in epochs:
        t0 = time.time()
        train_loss = train_one_epoch(
            train_model, train_loader, optimizer, device, scaler, cfg, loss_fn
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        ev = _public(evaluate(model, loaders[select_split], device, loss_fn=loss_fn))
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
        "params": count_params(model, trainable_only=False),
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
    df = save_history(history, out)
    plot_history(df, out)
    save_json(metrics, out / "metrics.json")  # the run is complete here; ONNX is an extra
    if cfg.export_onnx:
        metrics["onnx_max_abs_diff"] = _export_run_onnx(out, verbose)
        save_json(metrics, out / "metrics.json")
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
