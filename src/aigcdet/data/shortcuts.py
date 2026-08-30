"""The metadata-only shortcut probe.

Before any pixel is examined, check whether the label can already be guessed
from container-level metadata alone: image dimensions, aspect ratio, file
size, format, colour mode, EXIF presence, and JPEG quantisation history. If a
plain classifier trained on nothing but this can separate the classes well
above chance, any pixel-based accuracy reported later is at least partly an
artefact of how the two classes happen to have been saved -- not of anything
the model learned to see (TECHNICAL_DESIGN.md §2.3). This is the single most
common way a hackathon detector quietly reports a fake 0.99 AUC, and it is
also the cheapest check to run: no GPU, no backbone, seconds per thousand
images.

`canonicalize` (in `data.normalize`) is the fix once this probe finds a
problem; this module is only the diagnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..utils.logging import get_logger

log = get_logger("data.shortcuts")

NUMERIC_COLS = ["width", "height", "aspect_ratio", "file_size_bytes", "n_channels"]
CATEGORICAL_COLS = ["format", "mode", "has_exif", "jpeg_quant_bucket"]

# AUC verdict thresholds. Below 0.65 is treated as "no strong shortcut found";
# a metadata-only classifier will rarely land exactly at 0.50 even on a clean
# dataset, so this deliberately isn't a razor's-edge cutoff.
_WARNING_AUC = 0.65
_SEVERE_AUC = 0.85


def _extract_metadata_row(path: str | Path) -> dict:
    """Read everything used by the probe from a file, without decoding pixels
    beyond what PIL needs for `.size`/`.mode`/`.getexif()` (no pixel array is
    ever built here)."""
    p = Path(path)
    row = {
        "width": np.nan,
        "height": np.nan,
        "aspect_ratio": np.nan,
        "file_size_bytes": p.stat().st_size if p.exists() else np.nan,
        "n_channels": np.nan,
        "format": "unknown",
        "mode": "unknown",
        "has_exif": "false",
        "jpeg_quant_bucket": "na",
    }
    try:
        with Image.open(p) as im:
            row["width"], row["height"] = im.size
            row["aspect_ratio"] = im.width / max(im.height, 1)
            row["format"] = (im.format or "unknown").lower()
            row["mode"] = im.mode
            row["n_channels"] = len(im.getbands())
            try:
                exif = im.getexif()
                row["has_exif"] = str(bool(exif) and len(exif) > 0).lower()
            except Exception:
                pass
            quant = getattr(im, "quantization", None)
            if quant:
                table = quant.get(0) if 0 in quant else next(iter(quant.values()), [])
                if table:
                    # Bucket rather than use the raw table: we want "roughly
                    # this compression history", not to treat every distinct
                    # table as its own category and overfit the probe itself.
                    row["jpeg_quant_bucket"] = str(int(sum(table)) // 500 * 500)
    except Exception as exc:
        log.debug("shortcut probe: could not read metadata from %s (%s)", p, exc)
    return row


def _verdict(auc: float) -> str:
    if auc >= _SEVERE_AUC:
        return "severe"
    if auc >= _WARNING_AUC:
        return "warning"
    return "ok"


def metadata_leak_probe(
    image_paths: Sequence[str],
    labels: Sequence[int],
    seed: int = 1337,
    n_splits: int = 5,
) -> dict:
    """Cross-validated AUC of a classifier trained on file metadata alone.

    Returns a dict with:
        n            - number of images probed
        auc          - out-of-fold AUC of the metadata-only classifier
        verdict      - "ok" (<0.65) / "warning" (>=0.65) / "severe" (>=0.85)
        top_features - up to 5 metadata features with the largest absolute
                       logistic-regression coefficient, for a one-line
                       diagnosis of *what* is leaking (e.g. "width" or
                       "format_png" dominating means a resolution/format
                       shortcut, not a real detection signal)

    Uses out-of-fold predictions from StratifiedKFold rather than a single
    train/test split: probe samples are often only a few thousand images, and
    a single split's AUC is noisy enough to give a false sense of security in
    either direction.
    """
    if len(image_paths) != len(labels):
        raise ValueError("image_paths and labels must be the same length")

    labels_arr = np.asarray(labels)
    if len(np.unique(labels_arr)) < 2:
        log.warning("metadata_leak_probe: only one class present, cannot compute an AUC")
        return {"n": int(len(labels_arr)), "auc": 0.5, "verdict": "insufficient_data", "top_features": {}}

    meta = pd.DataFrame([_extract_metadata_row(p) for p in image_paths])

    min_class_count = int(np.bincount(labels_arr).min())
    n_splits_eff = min(n_splits, min_class_count)
    if n_splits_eff < 2:
        raise ValueError(
            f"metadata_leak_probe needs at least 2 examples of the minority class "
            f"for cross-validation; got {min_class_count}. Pass a larger sample."
        )
    if n_splits_eff < n_splits:
        log.info(
            "metadata_leak_probe: reducing n_splits from %d to %d (minority class "
            "has only %d examples)", n_splits, n_splits_eff, min_class_count,
        )

    pipe = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), NUMERIC_COLS),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    cv = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
    oof_proba = cross_val_predict(pipe, meta, labels_arr, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(labels_arr, oof_proba))

    # Refit once on all data purely to report which fields drove the score --
    # diagnostic only; the AUC above already comes from held-out folds.
    top_features: dict[str, float] = {}
    try:
        pipe.fit(meta, labels_arr)
        feature_names = pipe.named_steps["prep"].get_feature_names_out()
        coefs = pipe.named_steps["clf"].coef_[0]
        ranked = sorted(zip(feature_names, coefs), key=lambda kv: -abs(kv[1]))[:5]
        top_features = {name: round(float(c), 3) for name, c in ranked}
    except Exception as exc:
        log.debug("metadata_leak_probe: could not extract top features (%s)", exc)

    verdict = _verdict(auc)
    if verdict == "severe":
        log.warning(
            "metadata-only AUC=%.3f (SEVERE) -- file metadata alone predicts the "
            "label well above chance. Top signal: %s. Canonicalise both classes "
            "identically before trusting any pixel-based accuracy number.",
            auc, top_features,
        )
    elif verdict == "warning":
        log.warning(
            "metadata-only AUC=%.3f (warning) -- some container-level shortcut "
            "signal present. Top signal: %s. Worth investigating before relying "
            "on clean-set accuracy.",
            auc, top_features,
        )
    else:
        log.info("metadata-only AUC=%.3f (ok) -- no strong container-level shortcut found", auc)

    return {"n": int(len(labels_arr)), "auc": auc, "verdict": verdict, "top_features": top_features}
