"""Duplicate and near-duplicate checks against the official held-out demo set.

The problem statement's demo set (COCO val2017 + WildFake DALL-E-Advanced) must
never be trained on. It is easy to violate this by accident: public datasets
draw from overlapping sources, and the same real photo -- or a lightly
re-encoded copy of it -- can appear in more than one place on the internet.

Two checks, run in order:

1. `drop_exact_duplicates` - SHA-256 over the raw file bytes. Catches byte-
   identical copies (the same file downloaded twice, mirrored under a
   different name).
2. `drop_demo_set_leaks` - perceptual hash (pHash) comparison against the demo
   set. Catches near-duplicates: the same image re-compressed, resized, or
   re-saved in a different format, which SHA-256 would treat as unrelated
   files but which is the exact same content a judge's demo-set score would
   be evaluating. TECHNICAL_DESIGN.md sets the threshold at Hamming distance
   <= 5 on a 64-bit pHash; that default is kept here.

Any training image that survives both checks is safe to train on with respect
to *this* leakage risk. It does not by itself prove there's no shortcut in the
data -- see `data.shortcuts` for the separate metadata-only-leak probe.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..utils.io import iter_image_files, load_image
from ..utils.logging import get_logger

log = get_logger("data.dedup")

# Hamming distance on a 64-bit pHash. TECHNICAL_DESIGN.md §2.3: "Drop any
# training image within Hamming distance <= 5 of a demo image."
DEFAULT_PHASH_THRESHOLD = 5


def sha256_of_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hash raw file bytes, not decoded pixels -- this is meant to catch
    literal duplicate files, independent of what an image library makes of
    them."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def drop_exact_duplicates(
    df: pd.DataFrame, path_col: str = "image_path"
) -> tuple[pd.DataFrame, int]:
    """Drop rows whose file is byte-identical to an earlier row's file.

    Returns (deduplicated_df, n_dropped). Keeps the first occurrence of each
    hash; which copy is "first" depends only on the input row order, so callers
    that care about which specific duplicate survives should sort first.
    """
    if df.empty:
        return df, 0
    hashes = df[path_col].apply(sha256_of_file)
    dup_mask = hashes.duplicated(keep="first")
    n_dropped = int(dup_mask.sum())
    if n_dropped:
        log.warning("%d exact (SHA-256) duplicate file(s) found and dropped", n_dropped)
    else:
        log.info("no exact duplicate files found")
    return df[~dup_mask].reset_index(drop=True), n_dropped


def _compute_phashes(paths: Sequence[str]) -> dict:
    """Best-effort pHash for each path; unreadable files are skipped with a
    debug-level log rather than aborting the whole batch."""
    import imagehash

    out = {}
    for p in paths:
        try:
            out[p] = imagehash.phash(load_image(p))
        except Exception as exc:
            log.debug("dedup: could not hash %s (%s)", p, exc)
    return out


def find_demo_set_leaks(
    candidate_paths: Sequence[str],
    demo_dir: str | Path,
    threshold: int = DEFAULT_PHASH_THRESHOLD,
) -> pd.DataFrame:
    """For each candidate image, find its minimum pHash distance to the demo set.

    `demo_dir` should point at a local copy of the official demo images (both
    the COCO val2017 half and the DALL-E-Advanced half can be checked in one
    call by pointing this at a parent directory containing both, or by calling
    this twice and concatenating the reports).

    Returns a DataFrame with columns: image_path, min_hamming_distance,
    is_leak. This is a diagnostic report; `drop_demo_set_leaks` is the
    function that actually filters a manifest using it.
    """
    demo_paths = [str(p) for p in iter_image_files(demo_dir)]
    if not demo_paths:
        raise ValueError(f"no images found under demo_dir={demo_dir!s}")

    log.info("hashing %d demo-set images for the leak check", len(demo_paths))
    demo_hashes = list(_compute_phashes(demo_paths).values())
    if not demo_hashes:
        raise ValueError(f"could not read any images under demo_dir={demo_dir!s}")

    candidate_hashes = _compute_phashes(candidate_paths)
    rows = []
    for path, h in candidate_hashes.items():
        min_dist = min(h - dh for dh in demo_hashes)
        rows.append(
            {"image_path": path, "min_hamming_distance": int(min_dist), "is_leak": min_dist <= threshold}
        )

    unreadable = set(candidate_paths) - set(candidate_hashes)
    if unreadable:
        log.warning(
            "%d candidate image(s) could not be read and were skipped by the leak "
            "check (not flagged as leaks, but also not verified clean)",
            len(unreadable),
        )

    report = pd.DataFrame(rows)
    n_leaks = int(report["is_leak"].sum()) if not report.empty else 0
    log.info(
        "%d/%d checked images are near-duplicates of the demo set (threshold=%d)",
        n_leaks, len(report), threshold,
    )
    return report


def drop_demo_set_leaks(
    df: pd.DataFrame,
    demo_dir: str | Path,
    threshold: int = DEFAULT_PHASH_THRESHOLD,
    path_col: str = "image_path",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter a manifest to remove near-duplicates of the demo set.

    Returns (cleaned_df, leak_report_df). The report is worth keeping and
    logging in full: "we checked and found N near-duplicates, here they are"
    is exactly the evidence the README's data section and the Q&A answer
    ("how do you know the demo set was never trained on?") both need.
    """
    report = find_demo_set_leaks(df[path_col].tolist(), demo_dir, threshold)
    leak_paths = set(report.loc[report["is_leak"], "image_path"])
    cleaned = df[~df[path_col].isin(leak_paths)].reset_index(drop=True)
    return cleaned, report
