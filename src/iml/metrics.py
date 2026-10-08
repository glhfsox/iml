"""Metrics required by the project description: FAR, FRR (+ EER, accuracy, macro-F1).

Convention: label 1 = allowed person (accept), label 0 = not allowed (reject).
``scores`` are P(class 1) in [0, 1]; ``y_pred`` are hard 0/1 decisions.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve


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
