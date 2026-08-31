#!/usr/bin/env python3
"""Step 1b - drop training images that leak into the official demo set.

    python scripts/check_demo_leakage.py \
        --manifest data/manifest.csv \
        --demo_dir data/demo_set \
        --out data/manifest_clean.csv

`--demo_dir` should contain a local copy of the official demo images (COCO
val2017 + WildFake DALL-E-Advanced). Run this AFTER `build_manifest.py` and
BEFORE `extract_features.py` -- the whole point is to never extract features
from, or train on, an image that turns out to be a near-duplicate of what the
demo set will be scored on.

Writes a cleaned manifest plus a JSON leak report (`<out>.leak_report.json`)
recording exactly how many exact and near-duplicates were found and dropped.
Paste that JSON into the README's data section -- it is the evidence for the
"we verified the demo set was never trained on" claim required by the brief.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from aigcdet.data.dedup import DEFAULT_PHASH_THRESHOLD, drop_demo_set_leaks, drop_exact_duplicates
from aigcdet.utils.io import read_manifest, write_json, write_manifest
from aigcdet.utils.logging import get_logger

log = get_logger("check_demo_leakage")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--demo_dir", type=Path, required=True,
                    help="local copy of the official demo set (COCO val2017 + DALL-E-Advanced)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=int, default=DEFAULT_PHASH_THRESHOLD,
                    help="pHash Hamming-distance leak threshold (default: %(default)s, per TECHNICAL_DESIGN.md)")
    ap.add_argument("--skip_exact", action="store_true", help="skip the SHA-256 exact-duplicate pass")
    args = ap.parse_args()

    df = read_manifest(args.manifest)
    log.info("checking %d manifest rows against demo set at %s", len(df), args.demo_dir)

    n_exact = 0
    if not args.skip_exact:
        df, n_exact = drop_exact_duplicates(df)

    df, report = drop_demo_set_leaks(df, args.demo_dir, threshold=args.threshold)
    n_near = int(report["is_leak"].sum()) if not report.empty else 0

    write_manifest(df, args.out)
    summary = {
        "manifest_in": str(args.manifest),
        "manifest_out": str(args.out),
        "demo_dir": str(args.demo_dir),
        "threshold": args.threshold,
        "rows_before": int(len(read_manifest(args.manifest))),
        "rows_after": int(len(df)),
        "exact_duplicates_dropped": n_exact,
        "near_duplicates_dropped": n_near,
    }
    write_json(summary, args.out.with_suffix(".leak_report.json"))

    log.info(
        "clean manifest -> %s (dropped %d exact + %d near-duplicate of %d checked)",
        args.out, n_exact, n_near, summary["rows_before"],
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
