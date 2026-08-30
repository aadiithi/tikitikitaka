"""Metrics and calibration: verified against cases with known answers."""

import numpy as np
import pytest

from aigcdet.eval.metrics import binary_metrics, bootstrap_ci
from aigcdet.models.calibration import (
    TemperatureScaler,
    expected_calibration_error,
    pick_threshold,
)


def test_perfect_separation():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.01, 0.02, 0.03, 0.97, 0.98, 0.99])
    m = binary_metrics(scores, labels, threshold=0.5)
    assert m["auc"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["separation"] > 0.9


def test_random_scores_give_chance_auc():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 4000)
    scores = rng.random(4000)
    assert binary_metrics(scores, labels)["auc"] == pytest.approx(0.5, abs=0.05)


def test_inverted_scores_give_auc_below_half():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert binary_metrics(scores, labels)["auc"] < 0.5


def test_single_class_input_is_nan_not_a_crash():
    m = binary_metrics(np.array([0.3, 0.6]), np.array([1, 1]))
    assert np.isnan(m["auc"])
    assert m["n_real"] == 0


def test_threshold_changes_precision_recall_tradeoff():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.45, 0.55, 0.45, 0.6, 0.8, 0.9])
    low = binary_metrics(scores, labels, threshold=0.3)
    high = binary_metrics(scores, labels, threshold=0.7)
    assert low["recall"] >= high["recall"]
    assert high["precision"] >= low["precision"]


def test_tpr_at_fpr_budget_respects_the_budget():
    rng = np.random.default_rng(1)
    labels = np.concatenate([np.zeros(500), np.ones(500)]).astype(int)
    scores = np.concatenate([rng.normal(0.3, 0.15, 500), rng.normal(0.7, 0.15, 500)]).clip(0, 1)
    m = binary_metrics(scores, labels)
    picked = pick_threshold(scores, labels, max_fpr=0.05)
    assert picked["fpr"] <= 0.05 + 1e-9
    assert 0.0 <= m["tpr_at_5fpr"] <= 1.0


def test_temperature_scaling_preserves_ranking():
    rng = np.random.default_rng(2)
    logits = rng.normal(0, 4, 500)
    labels = (logits + rng.normal(0, 1, 500) > 0).astype(int)
    scaler = TemperatureScaler().fit(logits, labels)
    probs = scaler.transform(logits)
    # A monotone transform cannot change the ordering, therefore not the AUC.
    assert np.array_equal(np.argsort(logits), np.argsort(probs))


def test_temperature_scaling_improves_calibration_of_overconfident_scores():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, 2000)
    # Wildly over-confident logits: the right answer 75% of the time, at |logit| 8.
    logits = np.where(rng.random(2000) < 0.75, 1.0, -1.0) * np.where(labels == 1, 8.0, -8.0)
    before = expected_calibration_error(1 / (1 + np.exp(-logits)), labels)
    after = expected_calibration_error(TemperatureScaler().fit(logits, labels).transform(logits), labels)
    assert after < before


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(4)
    labels = np.concatenate([np.zeros(300), np.ones(300)]).astype(int)
    scores = np.concatenate([rng.normal(0.35, 0.2, 300), rng.normal(0.65, 0.2, 300)]).clip(0, 1)
    point = binary_metrics(scores, labels)["auc"]
    lo, hi = bootstrap_ci(scores, labels, n_boot=200)
    assert lo <= point <= hi
