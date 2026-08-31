"""Turn a manifest of image paths into a cached array of backbone embeddings.

This is deliberately "the one expensive pass": everything after this module
reads a `.npz` file, never an image, because the backbone is frozen and its
output for a given (image, damage) pair never changes.

Two extraction modes, controlled by `n_views`:

* `n_views=1`, no policy - one row per image, undamaged (after
  canonicalisation). This is the "clean" feature bank.
* `n_views>1` with a `policy` - each image is damaged `n_views` independent
  times (calling `policy(image)` fresh each time) and each damaged copy
  becomes its own row, all carrying the source image's label. This is the
  "augmented" feature bank, and it's what buys robustness per
  TECHNICAL_DESIGN.md §3.

`source_index` on the resulting `FeatureBundle` records which original image
each row came from, so damaged copies of one photo can never end up split
across train and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..data.normalize import CANONICAL_SPEC, canonicalize
from ..utils.io import ensure_dir, load_image
from ..utils.logging import get_logger

log = get_logger("features.extract")


@dataclass
class FeatureBundle:
    """Everything downstream code needs, in one self-describing array set."""

    features: np.ndarray            # (n_rows, dim), float32
    labels: np.ndarray              # (n_rows,), int
    source_index: np.ndarray        # (n_rows,), int - which original image each row came from
    paths: list                     # (n_source_images,), the ORIGINAL image path per source index
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        ensure_dir(p.parent)
        np.savez_compressed(
            p,
            features=self.features,
            labels=self.labels,
            source_index=self.source_index,
            paths=np.array(self.paths, dtype=object),
            meta=np.array([self.meta], dtype=object),
        )
        log.info(
            "saved feature bundle -> %s (%d rows, dim=%d, %d source images)",
            p, len(self.features), self.features.shape[1] if len(self.features) else 0,
            len(self.paths),
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> "FeatureBundle":
        with np.load(path, allow_pickle=True) as data:
            return cls(
                features=data["features"],
                labels=data["labels"],
                source_index=data["source_index"],
                paths=list(data["paths"]),
                meta=dict(data["meta"][0]),
            )


def _damaged_views(image, n_views: int, policy) -> list:
    """Produce `n_views` independently-damaged copies of one canonicalised
    image. Each call to `policy(image)` samples a fresh random damage chain
    (TrainDamagePolicy is stateful only in its RNG), so `n_views` copies of
    the same image are not duplicates of each other."""
    return [policy(image) for _ in range(n_views)]


def extract_features(
    backbone,
    paths: Sequence[str],
    labels: Sequence[int],
    n_views: int = 1,
    policy=None,
    canonical: bool = True,
    batch_size: int = 64,
    with_fourier: bool = False,
    progress: bool = True,
) -> FeatureBundle:
    """Encode a list of images (optionally with damaged views) into a FeatureBundle.

    `n_views > 1` requires `policy` (a callable `image -> damaged image`, e.g.
    `TrainDamagePolicy`) -- otherwise every "view" would be an identical
    duplicate of the clean image, which silently inflates the effective
    dataset size without adding any information. This is checked explicitly
    rather than left to produce a confusing downstream result.
    """
    if n_views > 1 and policy is None:
        raise ValueError(
            "n_views > 1 requires a damage policy; otherwise every view would "
            "be an identical duplicate of the clean image"
        )

    paths = [str(p) for p in paths]
    labels = np.asarray(labels, dtype=int)
    if len(paths) != len(labels):
        raise ValueError(f"paths ({len(paths)}) and labels ({len(labels)}) must be the same length")

    iterator = range(len(paths))
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="loading+damaging", unit="img")

    all_images, row_labels, row_source_idx = [], [], []
    n_failed = 0
    for i in iterator:
        try:
            img = load_image(paths[i])
            if canonical:
                img = canonicalize(img, CANONICAL_SPEC)
            views = _damaged_views(img, n_views, policy) if n_views > 1 else [img]
        except Exception as exc:
            log.warning("skipping unreadable image %s (%s)", paths[i], exc)
            n_failed += 1
            continue
        for v in views:
            all_images.append(v)
            row_labels.append(int(labels[i]))
            row_source_idx.append(i)

    if n_failed:
        log.warning("%d/%d images could not be read and were skipped", n_failed, len(paths))
    if not all_images:
        raise ValueError("no images could be read; nothing to extract features from")

    feats = []
    fe_iterator = range(0, len(all_images), batch_size)
    if progress:
        from tqdm.auto import tqdm

        fe_iterator = tqdm(fe_iterator, desc="encoding", unit="batch")
    for start in fe_iterator:
        chunk = all_images[start : start + batch_size]
        f = backbone.encode(chunk, batch_size=batch_size)
        if with_fourier:
            from .fourier import fourier_features

            ff = np.stack([fourier_features(im) for im in chunk]).astype(np.float32)
            f = np.concatenate([f, ff], axis=1)
        feats.append(f)

    features = np.concatenate(feats, axis=0).astype(np.float32)
    meta = {
        "backbone": backbone.name,
        "n_views": n_views,
        "canonical": canonical,
        "with_fourier": with_fourier,
        "n_source_images": len(paths),
        "n_failed": n_failed,
        "policy": policy.describe() if policy is not None and hasattr(policy, "describe") else None,
    }

    return FeatureBundle(
        features=features,
        labels=np.asarray(row_labels, dtype=int),
        source_index=np.asarray(row_source_idx, dtype=int),
        paths=paths,
        meta=meta,
    )


def extract_features_for_manifest(
    backbone,
    manifest: pd.DataFrame,
    split: str | None = None,
    n_views: int = 1,
    policy=None,
    canonical: bool = True,
    batch_size: int = 64,
    with_fourier: bool = False,
    progress: bool = True,
) -> FeatureBundle:
    """Convenience wrapper: filter a manifest DataFrame to one split (or use
    every row if `split` is None), then call `extract_features`. This is the
    function `scripts/extract_features.py` calls directly."""
    df = manifest
    if split is not None:
        if "split" not in df.columns:
            raise ValueError("manifest has no 'split' column; cannot filter by split")
        df = df[df["split"] == split]
        if df.empty:
            raise ValueError(f"no rows with split == {split!r} in this manifest")

    bundle = extract_features(
        backbone,
        df["image_path"].tolist(),
        df["label"].tolist(),
        n_views=n_views,
        policy=policy,
        canonical=canonical,
        batch_size=batch_size,
        with_fourier=with_fourier,
        progress=progress,
    )
    bundle.meta["split"] = split
    return bundle
