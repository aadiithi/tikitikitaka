"""The robustness harness: deliverable #4, built on day one rather than day four.

For every condition in the grid we take the *same* test images, apply that one
corruption, re-encode them through the frozen backbone, score them with a fixed
threshold, and write one row of metrics. Because the image set and the
threshold are held constant, the only thing varying between rows is the damage,
which is what makes the table readable as a causal statement.

The harness scores an arbitrary number of models in the same pass over each
condition. That matters practically: encoding is the expensive step, so
evaluating the clean-trained and augmentation-trained models together costs
almost exactly what evaluating one of them costs, and guarantees they saw
byte-identical corrupted images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..aug.transforms import Damage, list_conditions
from ..data.normalize import CANONICAL_SPEC, canonicalize
from ..utils.io import ensure_dir, load_image
from ..utils.logging import get_logger
from .metrics import METRIC_COLUMNS, binary_metrics

log = get_logger("eval.robustness")


def run_robustness_grid(
    backbone,
    models: Mapping[str, object],
    paths: Sequence[str],
    labels: Sequence[int],
    conditions: Sequence[Damage] | None = None,
    thresholds: Mapping[str, float] | None = None,
    batch_size: int = 32,
    canonical: bool = True,
    with_fourier: bool = False,
    save_scores_to: str | Path | None = None,
) -> pd.DataFrame:
    """Evaluate every model on every condition.

    `models` maps a display name to anything with `.score_features(np.ndarray)`,
    which both `Detector` and a bare head wrapper satisfy.

    Returns a tidy DataFrame - one row per (model, condition) - ready for
    `results/robustness.csv`, and optionally dumps every raw score so error
    analysis does not require a second pass over the images.
    """
    from tqdm.auto import tqdm

    conditions = list(conditions) if conditions is not None else list_conditions()
    thresholds = thresholds or {}
    labels = np.asarray(labels, dtype=int)
    paths = [str(p) for p in paths]

    rows, score_records = [], []

    for cond in conditions:
        feats, kept_labels, kept_paths = [], [], []
        for start in tqdm(
            range(0, len(paths), batch_size), desc=f"{cond.name:<18s}", unit="batch", leave=False
        ):
            chunk = paths[start : start + batch_size]
            chunk_labels = labels[start : start + batch_size]
            images, ok_labels, ok_paths = [], [], []
            for p, y in zip(chunk, chunk_labels):
                try:
                    im = load_image(p)
                    if canonical:
                        im = canonicalize(im, CANONICAL_SPEC)
                    # Damage is applied AFTER canonicalisation, in that order on
                    # purpose: canonicalisation represents the platform's own
                    # normalisation, and the damage is what happened to the file
                    # before it ever reached us.
                    images.append(cond(im))
                    ok_labels.append(int(y))
                    ok_paths.append(p)
                except Exception as exc:
                    log.warning("skip %s under %s (%s)", p, cond.name, exc)
            if not images:
                continue
            f = backbone.encode(images, batch_size=batch_size)
            if with_fourier:
                from ..features.fourier import fourier_features

                ff = np.stack([fourier_features(im) for im in images]).astype(np.float32)
                f = np.concatenate([f, ff], axis=1)
            feats.append(f)
            kept_labels.extend(ok_labels)
            kept_paths.extend(ok_paths)

        if not feats:
            log.error("condition %s produced no usable images", cond.name)
            continue

        X = np.concatenate(feats, axis=0)
        y = np.asarray(kept_labels, dtype=int)

        for model_name, model in models.items():
            scores = np.asarray(model.score_features(X)).ravel()
            thr = float(thresholds.get(model_name, getattr(model, "threshold", 0.5)))
            m = binary_metrics(scores, y, threshold=thr)
            rows.append(
                {
                    "model": model_name,
                    "condition": cond.name,
                    "family": cond.family,
                    "severity": cond.severity,
                    "held_out": bool(cond.held_out),
                    "threshold": thr,
                    **{k: m[k] for k in METRIC_COLUMNS},
                }
            )
            if save_scores_to is not None:
                for p, s, yy in zip(kept_paths, scores, y):
                    score_records.append(
                        {
                            "model": model_name,
                            "condition": cond.name,
                            "image_path": p,
                            "score": float(s),
                            "label": int(yy),
                        }
                    )

        log.info(
            "%-18s | %s",
            cond.name,
            "  ".join(
                f"{r['model']}: AUC {r['auc']:.4f} acc {r['accuracy']:.3f}"
                for r in rows
                if r["condition"] == cond.name
            ),
        )

    df = pd.DataFrame(rows)

    if save_scores_to is not None and score_records:
        p = Path(save_scores_to)
        ensure_dir(p.parent)
        pd.DataFrame(score_records).to_parquet(p) if p.suffix == ".parquet" else pd.DataFrame(
            score_records
        ).to_csv(p, index=False)
        log.info("raw per-image scores -> %s", p)

    return df


def robustness_summary(df: pd.DataFrame, metric: str = "auc") -> pd.DataFrame:
    """Condition x model pivot - the compact table the deliverable asks for."""
    return df.pivot_table(index=["family", "condition"], columns="model", values=metric).round(4)


def degradation_table(df: pd.DataFrame, metric: str = "auc") -> pd.DataFrame:
    """Per model: clean score, mean damaged score, and the drop between them.

    The `retention` column (damaged / clean) is the single number this whole
    project optimises, and the one the headline claim is built from.
    """
    out = []
    for model, grp in df.groupby("model"):
        clean_rows = grp[grp["condition"] == "clean"]
        clean = float(clean_rows[metric].iloc[0]) if len(clean_rows) else float("nan")

        spec = grp[(grp["condition"] != "clean") & (~grp["held_out"]) & (grp["family"] != "compound")]
        held = grp[grp["held_out"]]
        comp = grp[grp["family"] == "compound"]

        row = {
            "model": model,
            f"clean_{metric}": clean,
            f"spec_transforms_{metric}": float(spec[metric].mean()) if len(spec) else float("nan"),
            f"held_out_{metric}": float(held[metric].mean()) if len(held) else float("nan"),
            f"compound_{metric}": float(comp[metric].mean()) if len(comp) else float("nan"),
            f"worst_{metric}": float(grp[metric].min()),
            "worst_condition": str(grp.loc[grp[metric].idxmin(), "condition"]),
        }
        row["drop"] = clean - row[f"spec_transforms_{metric}"]
        row["retention"] = (
            row[f"spec_transforms_{metric}"] / clean if clean and not np.isnan(clean) else float("nan")
        )
        out.append(row)
    return pd.DataFrame(out).round(4)
