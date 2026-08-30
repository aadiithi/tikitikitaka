"""Calibration and threshold selection.

The deliverable asks for a *confidence score*, not a label, which means the
number has to mean something. An uncalibrated network's 0.9 is not "90% of
images I score 0.9 are generated"; it is usually far more confident than it
deserves, and it gets worse under distribution shift - exactly the regime this
project is about.

Two pieces:

* `TemperatureScaler` - a single learned scalar dividing the logit. It cannot
  change the ranking (so AUC is untouched) but it fixes over-confidence, which
  is what makes the score usable for a real moderation threshold.

* `pick_threshold` - we do *not* use 0.5. On a platform, a false positive means
  telling a real photographer their work is synthetic; that is a much more
  expensive mistake than missing one generated image. We therefore choose the
  operating point that maximises recall subject to a false-positive-rate budget
  the operator sets, and we report the threshold alongside every number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TemperatureScaler:
    """Single-parameter logit scaling, fitted by minimising NLL on held-out data."""

    temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()

        def nll(t: float) -> float:
            z = logits / max(t, 1e-3)
            # numerically stable binary cross-entropy from logits
            return float(np.mean(np.logaddexp(0.0, z) - labels * z))

        grid = np.concatenate([np.linspace(0.05, 2.0, 40), np.linspace(2.0, 10.0, 33)])
        losses = [nll(float(t)) for t in grid]
        self.temperature = float(grid[int(np.argmin(losses))])
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        z = np.asarray(logits, dtype=np.float64) / max(self.temperature, 1e-3)
        return 1.0 / (1.0 + np.exp(-z))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE: average gap between confidence and accuracy across probability bins.

    Reported in the robustness table because a detector whose score decays
    gracefully under damage is far more useful than one that stays confident
    while becoming wrong.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi)
        if not m.any():
            continue
        ece += m.mean() * abs(labels[m].mean() - probs[m].mean())
    return float(ece)


def pick_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    max_fpr: float = 0.05,
) -> dict:
    """Lowest threshold whose false-positive rate stays within `max_fpr`.

    Returns the threshold plus the operating point it buys, so the README can
    state "at a 5% false-accusation budget we catch X% of generated images"
    instead of an uninterpretable accuracy figure.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=int).ravel()
    pos, neg = labels == 1, labels == 0

    if not pos.any() or not neg.any():
        return {"threshold": 0.5, "fpr": float("nan"), "tpr": float("nan"), "max_fpr": max_fpr}

    candidates = np.unique(np.concatenate([probs, [0.0, 1.0]]))
    best = {"threshold": 1.0, "fpr": 0.0, "tpr": 0.0, "max_fpr": max_fpr}
    for t in candidates:
        pred = probs >= t
        fpr = float(pred[neg].mean())
        if fpr <= max_fpr:
            tpr = float(pred[pos].mean())
            if tpr > best["tpr"]:
                best = {"threshold": float(t), "fpr": fpr, "tpr": tpr, "max_fpr": max_fpr}
    return best
