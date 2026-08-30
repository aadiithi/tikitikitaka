"""Build a manifest from image folders, and split it without leaking generators.

A manifest is a CSV with one row per image: `image_path`, `label` (0 = real,
1 = AI-generated), `source` (which dataset it came from), `family` (which
generator produced a fake, or a pseudo-family for reals), and `split`.

The function that matters most here is `split_by_family`. A random per-image
train/test split lets the same generator's fingerprint appear on both sides,
which inflates the reported score without teaching the model to generalise
(Ojha et al., CVPR 2023) -- a detector that has partly memorised "this is what
SDXL's upsampler looks like" from training will of course recognise SDXL
images in test too. We therefore hold out *entire* generator families, never
individual images from a family that's otherwise in training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..utils.io import iter_image_files
from ..utils.logging import get_logger

log = get_logger("data.manifest")

# Matched against a folder name exactly (case-insensitive), not as a
# substring -- so a folder called "unrealistic_renders" is not misread as
# a "real" folder. Extend these sets if a dataset uses different naming.
REAL_KEYWORDS = {"real", "authentic", "camera", "photo", "photos", "pristine"}
FAKE_KEYWORDS = {"fake", "generated", "synthetic", "ai", "aigc", "gen"}


def _classify_and_family(rel_parts: Sequence[str]) -> tuple[int | None, str]:
    """Read label + generator family out of an image's path components.

    `rel_parts` excludes the labelled root itself (e.g. for
    `data/synthetic/fake/sdxl_like/img_0.jpg` relative to `data/synthetic`,
    this is `("fake", "sdxl_like", "img_0.jpg")`). The label comes from the
    first folder matching REAL_KEYWORDS/FAKE_KEYWORDS; the family is the next
    folder down, if any, else "unknown" (or "real" for an unlabelled real).
    """
    label: int | None = None
    label_idx: int | None = None
    for i, part in enumerate(rel_parts[:-1]):  # last element is the filename
        low = part.lower()
        if low in REAL_KEYWORDS:
            label, label_idx = 0, i
            break
        if low in FAKE_KEYWORDS:
            label, label_idx = 1, i
            break
    if label is None:
        return None, "unknown"

    remaining = rel_parts[label_idx + 1 : -1]
    if remaining:
        family = remaining[0]
    else:
        family = "real" if label == 0 else "unknown"
    return label, family


def _balance_and_limit(
    df: pd.DataFrame, limit_per_class: int | None, seed: int
) -> pd.DataFrame:
    if limit_per_class is None:
        return df
    parts = []
    for _label, group in df.groupby("label"):
        n = min(limit_per_class, len(group))
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def build_manifest_from_labelled_root(
    root: str | Path,
    source: str = "auto",
    limit_per_class: int | None = None,
    seed: int = 1337,
) -> pd.DataFrame:
    """Auto-detect real/fake images anywhere under `root` and build a manifest.

    Looks for a folder anywhere in the tree named one of REAL_KEYWORDS
    (label 0) or FAKE_KEYWORDS (label 1). A folder one level below that,
    if present, becomes `family` -- this is where generator names like
    "sdxl_like" or "glide_like" live and it's what `split_by_family` groups
    on. Raises if nothing matches, so a typo'd folder name fails loudly
    rather than silently producing an empty or tiny manifest.
    """
    root = Path(root)
    resolved_source = root.name if source == "auto" else source

    rows: list[dict] = []
    skipped = 0
    for path in iter_image_files(root):
        rel_parts = path.relative_to(root).parts
        label, family = _classify_and_family(rel_parts)
        if label is None:
            skipped += 1
            continue
        rows.append(
            {
                "image_path": str(path),
                "label": label,
                "source": resolved_source,
                "family": family,
            }
        )

    if skipped:
        log.warning(
            "%d file(s) under %s matched neither a real- nor a fake-labelled "
            "folder and were skipped. If that count looks wrong, check "
            "REAL_KEYWORDS/FAKE_KEYWORDS in data/manifest.py against your "
            "actual folder names.",
            skipped, root,
        )
    if not rows:
        raise ValueError(
            f"no labelled images found under {root!s}. Expected a folder named "
            f"one of {sorted(REAL_KEYWORDS)} and one named one of "
            f"{sorted(FAKE_KEYWORDS)} somewhere in the directory tree."
        )

    df = pd.DataFrame(rows)
    df = _balance_and_limit(df, limit_per_class, seed)
    log.info(
        "built manifest from %s: %d real, %d fake, %d generator families",
        root, int((df.label == 0).sum()), int((df.label == 1).sum()),
        df.loc[df.label == 1, "family"].nunique(),
    )
    return df.sort_values("image_path").reset_index(drop=True)


def build_manifest_from_dirs(
    real_dirs: Iterable[str | Path],
    fake_dirs: Iterable[str | Path],
    source: str = "auto",
    limit_per_class: int | None = None,
    seed: int = 1337,
) -> pd.DataFrame:
    """Build a manifest from explicit real/fake directories.

    Use this when directory names don't follow the auto-detectable real/fake
    convention -- the canonical case is the official demo set, e.g.
    `--real_dir data/coco/val2017 --fake_dir data/dalle_advanced`, where
    neither folder name contains the word "real" or "fake". A subfolder one
    level inside a given dir becomes its `family`; images directly inside the
    given dir get `family = "real"` (for real dirs) or the directory's own
    name (for fake dirs, so at least each `--fake_dir` counts as one family).
    """
    real_dirs = [Path(p) for p in real_dirs]
    fake_dirs = [Path(p) for p in fake_dirs]
    if not real_dirs and not fake_dirs:
        raise ValueError("pass at least one of real_dirs / fake_dirs")

    rows: list[dict] = []
    for label, dirs in ((0, real_dirs), (1, fake_dirs)):
        for d in dirs:
            if not d.is_dir():
                raise FileNotFoundError(f"not a directory: {d}")
            resolved_source = d.name if source == "auto" else source
            n_before = len(rows)
            for path in iter_image_files(d):
                rel = path.relative_to(d).parts[:-1]
                if rel:
                    family = rel[0]
                elif label == 0:
                    family = "real"
                else:
                    family = d.name
                rows.append(
                    {
                        "image_path": str(path),
                        "label": label,
                        "source": resolved_source,
                        "family": family,
                    }
                )
            if len(rows) == n_before:
                log.warning("no images found under %s", d)

    if not rows:
        raise ValueError("no images found in the given real_dirs/fake_dirs")

    df = pd.DataFrame(rows)
    df = _balance_and_limit(df, limit_per_class, seed)
    return df.sort_values("image_path").reset_index(drop=True)


def _group_holdout(
    df_slice: pd.DataFrame, key_col: str, target_frac: float, rng: np.random.Generator
) -> set:
    """Pick whole groups (by `key_col`) covering ~target_frac of `df_slice`'s
    rows, for a group-disjoint holdout. Returns an empty set if there's only
    one group (nothing can be held out without leaving one side empty)."""
    counts = df_slice.groupby(key_col).size().to_dict()
    groups = list(counts)
    if len(groups) <= 1 or target_frac <= 0:
        return set()

    order = groups.copy()
    rng.shuffle(order)
    target = target_frac * len(df_slice)
    running = 0
    chosen: set = set()
    for g in order:
        if running >= target:
            break
        chosen.add(g)
        running += counts[g]
    if not chosen:
        # target_frac was smaller than even the smallest group -- hold out
        # the smallest group anyway so test is never accidentally empty.
        chosen = {min(groups, key=lambda g: counts[g])}
    return chosen


def _row_stratified_holdout(
    df_slice: pd.DataFrame, test_frac: float, rng: np.random.Generator
) -> pd.Index:
    if test_frac <= 0 or len(df_slice) == 0:
        return pd.Index([])
    n_test = min(len(df_slice), max(1, round(test_frac * len(df_slice))))
    return pd.Index(rng.choice(df_slice.index.to_numpy(), size=n_test, replace=False))


def split_by_family(
    df: pd.DataFrame,
    holdout_families: Sequence[str] | None = None,
    test_frac: float = 0.2,
    seed: int = 1337,
) -> pd.DataFrame:
    """Assign `split` in {"train", "test"} without leaking a family across it.

    Fakes are split by generator `family`: a family is entirely train or
    entirely test, never both. This is the honest generalisation test --
    performance on a held-out generator family, not on held-out images from
    generators the model has already partly memorised.

    Reals are split by `source` when more than one source is present (e.g.
    training reals from one dataset, held-out reals from another, matching
    the val_unseen design in TECHNICAL_DESIGN.md §2.2). If only one real
    source exists, falls back to a per-row stratified split for reals --
    there's no generator identity to leak on the real side, so this is a
    reasonable relaxation, but it should be named as a limitation in the
    README if it's the path actually taken (see the log line this emits).

    Raises rather than silently degrading the fake side: if there is only
    one fake generator family and no `holdout_families` override, there is
    nothing to hold out and the resulting "test" split would tell you nothing
    about generalisation to unseen generators -- callers should either supply
    more generator diversity or explicitly acknowledge a random split.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)

    fake = df[df["label"] == 1]
    if fake.empty:
        raise ValueError("split_by_family: no fake (label=1) rows to split")
    families = fake["family"].unique().tolist()

    if holdout_families is not None:
        requested = set(holdout_families)
        unknown = requested - set(families)
        if unknown:
            log.warning(
                "holdout_families %s not present in the data and will be ignored",
                sorted(unknown),
            )
        test_families = requested & set(families)
        if not test_families:
            raise ValueError(
                f"none of holdout_families={list(holdout_families)} are present; "
                f"available families are {sorted(families)}"
            )
    elif len(families) > 1:
        test_families = _group_holdout(fake, "family", test_frac, rng)
    else:
        log.warning(
            "split_by_family: only one fake generator family ('%s') is present. "
            "Falling back to a per-row random split for fakes -- this cannot "
            "measure generalisation to an unseen generator and should be "
            "reported as a limitation, not as a generator-disjoint result.",
            families[0],
        )
        test_families = set()

    if test_families:
        fake_split = pd.Series(
            np.where(fake["family"].isin(test_families), "test", "train"),
            index=fake.index,
        )
    else:
        test_idx = _row_stratified_holdout(fake, test_frac, rng)
        fake_split = pd.Series("train", index=fake.index)
        fake_split.loc[test_idx] = "test"

    real = df[df["label"] == 0]
    if real.empty:
        real_split = pd.Series(dtype=object)
    elif real["source"].nunique() > 1:
        held_sources = _group_holdout(real, "source", test_frac, rng)
        real_split = pd.Series(
            np.where(real["source"].isin(held_sources), "test", "train"), index=real.index
        )
    else:
        test_idx = _row_stratified_holdout(real, test_frac, rng)
        real_split = pd.Series("train", index=real.index)
        real_split.loc[test_idx] = "test"

    df.loc[fake.index, "split"] = fake_split
    df.loc[real.index, "split"] = real_split

    log.info(
        "split_by_family: %d/%d fake images -> test (families held out: %s); "
        "%d/%d real images -> test",
        int((fake_split == "test").sum()), len(fake), sorted(test_families) or "none (per-row split)",
        int((real_split == "test").sum()) if len(real) else 0, len(real),
    )
    return df


def summarize_manifest(df: pd.DataFrame) -> str:
    """Human-readable summary for logging and the README's data section."""
    lines = ["manifest summary", "-" * 40, f"total images: {len(df)}"]

    if "split" in df.columns:
        table = df.groupby(["split", "label"]).size().unstack(fill_value=0)
        lines.append(table.to_string())
    else:
        lines.append(df.groupby("label").size().to_string())

    if "family" in df.columns:
        n_fam = df.loc[df["label"] == 1, "family"].nunique()
        lines.append(f"distinct fake generator families: {n_fam}")

    if "source" in df.columns:
        lines.append(f"sources: {sorted(df['source'].unique().tolist())}")

    return "\n".join(lines)
