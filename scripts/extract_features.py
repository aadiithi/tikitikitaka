#!/usr/bin/env python3
"""Step 2 - the one expensive pass. Encode images into cached feature banks.

    # clean bank (the control)
    python scripts/extract_features.py --manifest data/manifest.csv --split train \
        --out features/train_clean.npz --n_views 1

    # augmented bank (the treatment) - 4 independently damaged copies per image
    python scripts/extract_features.py --manifest data/manifest.csv --split train \
        --out features/train_aug.npz --n_views 4 --augment

Run both from the same manifest so the clean-vs-augmented comparison is a fair
test of one variable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from aigcdet.aug.transforms import TrainDamagePolicy
from aigcdet.features.backbone import load_backbone
from aigcdet.features.extract import extract_features_for_manifest
from aigcdet.utils.io import read_manifest
from aigcdet.utils.logging import get_logger
from aigcdet.utils.seed import set_seed

log = get_logger("extract_features")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--split", type=str, default=None, help="train / test / demo; omit for all rows")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backbone", type=str, default="clip-vit-l14")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_views", type=int, default=1,
                    help="damaged copies per image; >1 requires --augment")
    ap.add_argument("--augment", action="store_true", help="apply the training damage policy")
    ap.add_argument("--no_canonical", action="store_true",
                    help="skip size/format canonicalisation (use only for the shortcut ablation)")
    ap.add_argument("--with_fourier", action="store_true", help="append spectral features (ablation)")
    ap.add_argument("--p_clean", type=float, default=0.25)
    ap.add_argument("--p_chain", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    manifest = read_manifest(args.manifest)

    policy = None
    if args.augment:
        policy = TrainDamagePolicy(p_clean=args.p_clean, p_chain=args.p_chain, seed=args.seed)
        log.info("damage policy: %s", policy.describe())
    elif args.n_views > 1:
        raise SystemExit("--n_views > 1 without --augment would produce identical duplicate rows")

    backbone = load_backbone(args.backbone, device=args.device)
    log.info("backbone %s (dim=%d) on %s", backbone.name, backbone.dim, backbone.device)

    bundle = extract_features_for_manifest(
        backbone,
        manifest,
        split=args.split,
        n_views=args.n_views,
        policy=policy,
        canonical=not args.no_canonical,
        batch_size=args.batch_size,
        with_fourier=args.with_fourier,
    )
    bundle.meta["manifest"] = str(args.manifest)
    bundle.meta["seed"] = args.seed
    bundle.save(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
