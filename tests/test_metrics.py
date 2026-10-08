import math

import numpy as np
import pytest

from iml.metrics import classification_metrics, eer, far, frr


def test_far_counts_only_class0():
    y_true = [0, 0, 0, 0, 1, 1]
    y_pred = [1, 0, 0, 1, 0, 0]  # 2 of 4 negatives accepted
    assert far(y_true, y_pred) == 0.5


def test_frr_counts_only_class1():
    y_true = [0, 0, 1, 1, 1, 1]
    y_pred = [1, 1, 0, 1, 1, 1]  # 1 of 4 positives rejected
    assert frr(y_true, y_pred) == 0.25


def test_far_frr_nan_without_that_class():
    assert math.isnan(far([1, 1], [1, 0]))
    assert math.isnan(frr([0, 0], [1, 0]))


def test_eer_perfect_separation_is_zero():
    y = [0, 0, 0, 1, 1, 1]
    s = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    e, t = eer(y, s)
    assert e == 0.0 and t == pytest.approx(0.5)  # midpoint of the gap between 0.3 and 0.7


def test_eer_random_scores_near_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    s = rng.random(2000)
    e, _ = eer(y, s)
    assert 0.4 < e < 0.6


def test_classification_metrics_keys():
    m = classification_metrics([0, 1, 0, 1], [0, 1, 1, 1], scores=[0.1, 0.9, 0.6, 0.8])
    assert m["far"] == 0.5 and m["frr"] == 0.0 and m["accuracy"] == 0.75 and m["n"] == 4
    assert m["n_pos"] == 2 and m["n_neg"] == 2
    assert {"macro_f1", "eer", "eer_threshold"} <= set(m)


def test_eer_threshold_is_finite_for_constant_scores():
    e, t = eer([0, 1, 0, 1], [0.3, 0.3, 0.3, 0.3])
    assert math.isfinite(t) and 0.0 <= t <= 1.0 and 0 <= e <= 1
