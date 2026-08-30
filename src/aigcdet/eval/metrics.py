"""Metrics, with the operating point treated as a first-class input.

Accuracy at 0.5 is the least informative number we could report and it is the
one most submissions lead with. We compute it, but we lead with:

* **AUC** - threshold-free ranking quality, the only number comparable across
  conditions when the score distribution shifts under damage (which it does).
* **TPR@5%FPR** - "what fraction of generated images do we catch if we are only
  willing to falsely accuse 5% of real photographers". This is the number an
  actual trust-and-safety team would sign off on.
* **ECE** - whether the confidence means anything.

All of them are computed at a *fixed* threshold carried over from validation,
never re-tuned per condition. Re-tuning per condition would let the model look
robust by quietly moving the goalposts each time the image got harder.
"""

from __future__ import annotations

import numpy as np

from ..models.calibration import expected_calibration_error

METRIC_COLUMNS = [
    "n",
    "n_real",
    "n_fake",
    "auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "tpr_at_5fpr",
    "fpr",
    "ece",
    "mean_score_real",
    "mean_score_fake",
    "separation",
]


def binary_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    fpr_budget: float = 0.05,
) -> dict:
    """All headline metrics for one (scores, labels) pair."""
    from sklearn.metrics import roc_auc_score, roc_curve

    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=int).ravel()
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())

    pred = scores >= threshold
    tp = int((pred & pos).sum())
    fp = int((pred & neg).sum())
    fn = int((~pred & pos).sum())
    tn = int((~pred & neg).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    if n_pos and n_neg:
        auc = float(roc_auc_score(labels, scores))
        fpr_curve, tpr_curve, _ = roc_curve(labels, scores)
        allowed = fpr_curve <= fpr_budget
        tpr_at_budget = float(tpr_curve[allowed].max()) if allowed.any() else 0.0
        ece = expected_calibration_error(scores, labels)
    else:
        auc = tpr_at_budget = ece = float("nan")

    return {
        "n": int(len(labels)),
        "n_real": n_neg,
        "n_fake": n_pos,
        "auc": auc,
        "accuracy": float((tp + tn) / max(len(labels), 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tpr_at_5fpr": tpr_at_budget,
        "fpr": float(fp / n_neg) if n_neg else float("nan"),
        "ece": ece,
        "mean_score_real": float(scores[neg].mean()) if n_neg else float("nan"),
        "mean_score_fake": float(scores[pos].mean()) if n_pos else float("nan"),
        # How far apart the two score distributions sit. Watching this collapse
        # under damage is more diagnostic than watching accuracy fall, because
        # it distinguishes "the model got confused" from "the model got shy".
        "separation": (
            float(scores[pos].mean() - scores[neg].mean()) if (n_pos and n_neg) else float("nan")
        ),
    }


def bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    metric: str = "auc",
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 1337,
) -> tuple[float, float]:
    """Percentile bootstrap CI, so we do not over-read a 1-point difference.

    With a 3,000-image test set an AUC gap below roughly 0.01 is noise; the
    write-up quotes these intervals rather than bare point estimates.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel()
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) < 2:
            continue
        vals.append(binary_metrics(scores[idx], labels[idx])[metric])
    if not vals:
        return (float("nan"), float("nan"))
    return (
        float(np.percentile(vals, 100 * alpha / 2)),
        float(np.percentile(vals, 100 * (1 - alpha / 2))),
    )
