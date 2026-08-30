#!/usr/bin/env python3
"""Step 1 - turn image folders into a manifest with an honest split.

    # auto-detect real/fake from folder names anywhere under the root
    python scripts/build_manifest.py --root data/SID_Set --out data/manifest.csv

    # or name the directories explicitly
    python scripts/build_manifest.py \
        --real_dir data/coco/val2017 --fake_dir data/dalle_advanced \
        --out data/manifest_demo.csv --no_split

Also runs the metadata shortcut probe and refuses to continue quietly if the
dataset is separable from container metadata alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from aigcdet.data.manifest import (
    build_manifest_from_dirs,
    build_manifest_from_labelled_root,
    split_by_family,
    summarize_manifest,
)
from aigcdet.data.shortcuts import metadata_leak_probe
from aigcdet.utils.io import write_json, write_manifest
from aigcdet.utils.logging import get_logger
from aigcdet.utils.seed import set_seed

log = get_logger("build_manifest")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, help="dataset root with real/fake folder names inside")
    ap.add_argument("--real_dir", type=Path, nargs="*", default=[])
    ap.add_argument("--fake_dir", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--source", type=str, default="auto")
    ap.add_argument("--limit_per_class", type=int, default=None,
                    help="subsample to at most N images per class, balanced across families")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--holdout_families", type=str, nargs="*", default=None,
                    help="generator families to reserve for test; default picks automatically")
    ap.add_argument("--no_split", action="store_true",
                    help="do not assign a split column (use for the held-out demo set)")
    ap.add_argument("--skip_probe", action="store_true")
    ap.add_argument("--probe_sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)

    if args.root:
        df = build_manifest_from_labelled_root(
            args.root, source=args.source, limit_per_class=args.limit_per_class, seed=args.seed
        )
    elif args.real_dir or args.fake_dir:
        df = build_manifest_from_dirs(
            args.real_dir, args.fake_dir, source=args.source,
            limit_per_class=args.limit_per_class, seed=args.seed,
        )
    else:
        ap.error("pass either --root or (--real_dir and --fake_dir)")

    if not args.no_split:
        df = split_by_family(
            df, holdout_families=args.holdout_families, test_frac=args.test_frac, seed=args.seed
        )
    else:
        df["split"] = "demo"

    write_manifest(df, args.out)
    log.info("manifest -> %s\n%s", args.out, summarize_manifest(df))

    report = {"n_images": int(len(df)), "manifest": str(args.out)}
    if not args.skip_probe:
        sample = df.sample(n=min(args.probe_sample, len(df)), random_state=args.seed)
        probe = metadata_leak_probe(sample["image_path"].tolist(), sample["label"].tolist())
        report["metadata_leak_probe"] = probe
        log.info("metadata-only AUC = %.4f -> %s", probe["auc"], probe["verdict"])
        if probe["auc"] > 0.90:
            log.warning(
                "This dataset is nearly separable from file metadata alone. Canonicalisation "
                "(on by default during feature extraction) removes most of this, but say so "
                "explicitly in the write-up rather than reporting the raw accuracy."
            )

    write_json(report, Path(args.out).with_suffix(".report.json"))
    print(json.dumps(report, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
