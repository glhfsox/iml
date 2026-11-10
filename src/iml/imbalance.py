
"""Class-imbalance toolkit: measure it, balance a split, imbalance-aware losses, head-only tuning.

* analysis: ``dataset_labels``, ``imbalance_report`` (counts and majority-per-minority ratio);
* balancing: ``balanced_indices`` / ``balanced_subset`` (undersampling to the smallest class);
* rotating balancing: ``RotatingBalancedSampler`` (balanced epochs with rotating majority samples);
* losses (``cfg.loss``): ``ce`` (+ ``class_weight="balanced"`` = weighted cross-entropy),
  ``focal`` (``focal_loss``), ``soft_f1`` (``soft_f1_loss``, differentiable macro-F1);
  ``criterion(cfg, weight)`` returns the one ``fit`` trains with;
* two-phase fine-tuning: phase 1 is an ordinary ``fit`` on the imbalanced train split; phase 2 is
  ``fit`` with ``finetune_from=<phase-1 run folder>``: it loads that run's ``best.pt``, freezes
  everything except the classification head (``freeze_backbone``) and trains the head on a
  perfectly balanced subset of the train split (``balanced_subset``)::

      rec = fit(cfg, loaders)
      fit(cfg.with_overrides(f"finetune_from={rec.path}", "lr=1e-4", "epochs=5"), loaders)
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from iml.config import LOSSES, ExperimentConfig

__all__ = [
    # analysis
    "dataset_labels",
    "imbalance_report",
    # balancing
    "balanced_indices",
    "balanced_subset",
    "RotatingBalancedSampler",
    # losses
    "LOSSES",
    "focal_loss",
    "soft_f1_loss",
    "criterion",
    # two-phase fine-tuning
    "HEAD_NAMES",
    "freeze_backbone",
    "keep_frozen_bn_eval",
]


# =================================================================================================
# analysis
# =================================================================================================


def dataset_labels(data: Dataset | DataLoader) -> np.ndarray:
    """Labels of a dataset (or a loader's dataset) without decoding audio when it exposes them."""
    ds = data.dataset if isinstance(data, DataLoader) else data  # a Subset keeps its own labels
    if hasattr(ds, "labels"):
        return np.asarray(ds.labels)
    if hasattr(ds, "items"):
        return np.asarray([int(label) for _, label in ds.items])
    return np.asarray([int(y) for _, y in ds])


def imbalance_report(data: Dataset | DataLoader | Any) -> dict[str, Any]:
    """Class counts and ``ratio`` = majority samples per one minority sample (1.0 = balanced).

    ``data`` is a dataset, a loader or the labels themselves.
    """
    labels = dataset_labels(data) if isinstance(data, Dataset | DataLoader) else np.asarray(data)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) == 0:
        raise ValueError("no labels")
    lo, hi = int(counts.argmin()), int(counts.argmax())
    return {
        "counts": {int(c): int(n) for c, n in zip(classes, counts, strict=True)},
        "minority": int(classes[lo]),
        "majority": int(classes[hi]),
        "ratio": float(counts[hi] / counts[lo]),
    }


# =================================================================================================
# balancing: random undersampling and rotating balanced sampling
# =================================================================================================


def balanced_indices(labels, seed: int = 0) -> np.ndarray:
    """Sorted indices of a perfectly balanced random subset (every class = smallest class)."""
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise ValueError(f"balancing needs at least two classes, got {classes.tolist()}")
    rng = np.random.default_rng(seed)
    n = counts.min()
    picked = [rng.choice(np.flatnonzero(labels == c), n, replace=False) for c in classes]
    return np.sort(np.concatenate(picked))


def balanced_subset(dataset: Dataset, seed: int = 0) -> Subset:
    """``dataset`` undersampled to a perfectly balanced ``Subset`` (keeps ``.labels``, so
    ``dataset_labels`` and ``class_weights`` never decode audio)."""
    labels = dataset_labels(dataset)
    idx = balanced_indices(labels, seed)
    sub = Subset(dataset, idx.tolist())
    sub.labels = labels[idx]
    return sub


class RotatingBalancedSampler(Sampler[int]):
    """Balance two classes per epoch and rotate majority-class samples.

    Every epoch contains all minority-class samples and an equal
    number of majority-class samples.

    Majority samples are drawn from a shuffled pool without replacement.
    Once the pool is exhausted, a new shuffled cycle begins.

    Sampling is deterministic for a given seed.
    """

    def __init__(self, labels, seed: int = 42):
        self.labels = np.asarray(labels)
        classes, counts = np.unique(self.labels, return_counts=True)

        if len(classes) != 2:
            raise ValueError("Expected exactly two classes")

        self.minority = np.flatnonzero(
            self.labels == classes[np.argmin(counts)]
        )
        self.majority = np.flatnonzero(
            self.labels == classes[np.argmax(counts)]
        )

        self.n = len(self.minority)
        self.rng = np.random.default_rng(seed)

        self.pool = self.rng.permutation(self.majority)
        self.position = 0

    def __len__(self):
        return 2 * self.n

    def __iter__(self):
        selected = []
        used = set()

        while len(selected) < self.n:
            if self.position >= len(self.pool):
                # Start a new shuffled cycle.
                new_pool = self.rng.permutation(self.majority)

                # Samples already selected in this epoch go to the end.
                # This avoids duplicates within the same epoch.
                unused = [
                    int(i) for i in new_pool if int(i) not in used
                ]
                already_used = [
                    int(i) for i in new_pool if int(i) in used
                ]

                self.pool = np.asarray(
                    unused + already_used,
                    dtype=int,
                )
                self.position = 0

            candidate = int(self.pool[self.position])
            self.position += 1

            if candidate not in used:
                selected.append(candidate)
                used.add(candidate)

        indices = np.concatenate([
            self.minority,
            np.asarray(selected, dtype=int),
        ])

        self.rng.shuffle(indices)
        return iter(indices.tolist())


# =================================================================================================
# losses for imbalanced data; ``weight`` = per-class weights (``class_weight="balanced"``)
# =================================================================================================


def focal_loss(
    logits: torch.Tensor, y: torch.Tensor, weight: torch.Tensor | None = None, gamma: float = 2.0
) -> torch.Tensor:
    """Focal loss (Lin et al. 2017): cross-entropy scaled by ``(1 - p_true) ** gamma``, so the
    many easy (mostly majority-class) samples stop dominating the gradient. ``weight`` is the
    per-class alpha, normalised like weighted cross-entropy; ``gamma=0`` is plain CE."""
    ce = F.cross_entropy(logits.float(), y, reduction="none")
    # clamp: 0 ** gamma has an infinite gradient for gamma < 1 (confidently correct samples)
    loss = (-torch.expm1(-ce)).clamp_min(1e-12) ** gamma * ce
    if weight is None:
        return loss.mean()
    w = weight[y]
    return (loss * w).sum() / w.sum()


def soft_f1_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``1 - macro-F1`` over the batch, with probabilities in place of hard decisions so it is
    differentiable; every class counts equally whatever its size."""
    p = torch.softmax(logits.float(), dim=1)
    t = F.one_hot(y, p.shape[1]).float()
    tp, fp, fn = (p * t).sum(0), (p * (1 - t)).sum(0), ((1 - p) * t).sum(0)
    return 1 - (2 * tp / (2 * tp + fp + fn + 1e-8)).mean()


def criterion(
    cfg: ExperimentConfig, weight: torch.Tensor | None = None
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """The training loss of ``cfg`` as ``loss(logits, y) -> scalar``."""
    if cfg.loss == "ce":
        return partial(F.cross_entropy, weight=weight)
    if cfg.loss == "focal":
        return partial(focal_loss, weight=weight, gamma=cfg.focal_gamma)
    if cfg.loss == "soft_f1":
        return soft_f1_loss
    raise ValueError(f"loss must be one of {LOSSES}, got {cfg.loss!r}")


# =================================================================================================
# two-phase fine-tuning: freeze the feature extractor, train only the classification head
# =================================================================================================

HEAD_NAMES = ("fc", "head")  # resnet18 / TimmClassifier: ``fc``; CustomCNN: ``head``


def freeze_backbone(model: nn.Module) -> None:
    """``requires_grad=False`` on every parameter outside the classification head."""
    for name, p in model.named_parameters():
        p.requires_grad = name.split(".")[0] in HEAD_NAMES
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError(f"{type(model).__name__} has no head named one of {HEAD_NAMES}")


def keep_frozen_bn_eval(model: nn.Module) -> None:
    """Call after ``model.train()``: frozen BatchNorm layers keep their phase-1 running stats."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(
            p.requires_grad for p in m.parameters()
        ):
            m.eval()
