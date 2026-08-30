#!/usr/bin/env python3
"""Step 3 - train the head. Seconds, not hours, because the features are cached.

    python scripts/train_head.py --features features/train_clean.npz \
        --out checkpoints/detector_clean.pt

    python scripts/train_head.py --features features/train_aug.npz \
        --out checkpoints/detector_robust.pt

Writes a self-describing checkpoint (weights + standardisation + temperature +
operating threshold + provenance) and a metrics JSON beside it.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import _bootstrap  # noqa: F401

from aigcdet.features.extract import FeatureBundle
from aigcdet.models.head import HeadConfig, save_checkpoint, train_head
from aigcdet.utils.io import write_json
from aigcdet.utils.logging import get_logger
from aigcdet.utils.seed import set_seed

log = get_logger("train_head")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--max_fpr", type=float, default=0.05,
                    help="false-positive budget used to pick the operating threshold")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    bundle = FeatureBundle.load(args.features)
    log.info(
        "features %s from %s | labels: %d real / %d fake",
        bundle.features.shape, bundle.meta.get("backbone", "?"),
        int((bundle.labels == 0).sum()), int((bundle.labels == 1).sum()),
    )

    cfg = HeadConfig(
        input_dim=bundle.features.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    result = train_head(
        bundle.features, bundle.labels, bundle.source_index, cfg=cfg, device=args.device
    )

    save_checkpoint(
        args.out,
        result["model"],
        backbone_name=bundle.meta.get("backbone", "clip-vit-l14"),
        temperature=result["temperature"],
        threshold=result["threshold"],
        extra={
            "features_file": str(args.features),
            "feature_meta": bundle.meta,
            "val_auc": result["val_auc"],
            "val_ece": result["val_ece"],
            "n_trainable_params": result["n_trainable_params"],
            "with_fourier": bool(bundle.meta.get("with_fourier", False)),
        },
    )
    write_json(
        {
            "features": str(args.features),
            "checkpoint": str(args.out),
            "config": asdict(cfg),
            "val_auc": result["val_auc"],
            "val_ece": result["val_ece"],
            "best_epoch": result["best_epoch"],
            "temperature": result["temperature"],
            "threshold": result["threshold"],
            "n_trainable_params": result["n_trainable_params"],
            "history": result["history"],
        },
        args.out.with_suffix(".metrics.json"),
    )
    log.info(
        "done | trainable params %s | val AUC %.4f",
        f"{result['n_trainable_params']:,}", result["val_auc"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
