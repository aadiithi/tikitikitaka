#!/usr/bin/env python3
"""Step 4 - the robustness table (deliverable #4) and its figures.

    python scripts/evaluate_robustness.py \
        --manifest data/manifest.csv --split test \
        --checkpoints clean=checkpoints/detector_clean.pt \
                      robust=checkpoints/detector_robust.pt \
        --out results/

Evaluates every named checkpoint on every corruption condition, using the same
images and each model's own fixed validation-chosen threshold, then writes:

    results/robustness.csv        one row per (model, condition)
    results/scores.csv            per-image scores, for error analysis
    results/summary.csv           clean / spec / unseen / compound retention
    results/robustness_auc.png    headline dumbbell figure
    results/degradation_auc.png   degradation curves
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from aigcdet.aug.transforms import COMPOUND_GRID, EVAL_GRID, HELD_OUT_GRID
from aigcdet.eval.report import write_report
from aigcdet.eval.robustness import degradation_table, run_robustness_grid
from aigcdet.features.backbone import load_backbone
from aigcdet.models.head import load_checkpoint
from aigcdet.models.calibration import TemperatureScaler
from aigcdet.utils.io import ensure_dir, read_manifest
from aigcdet.utils.logging import get_logger
from aigcdet.utils.seed import set_seed

log = get_logger("evaluate_robustness")


class _ScoredHead:
    """Adapter: a checkpoint's head + calibration, scoring pre-computed features.

    The grid encodes each corrupted image once and then scores it with every
    model, so this deliberately does not own a backbone.
    """

    def __init__(self, ckpt_path: str, device: str | None = None):
        ck = load_checkpoint(ckpt_path, device=device)
        self.head = ck["model"]
        self.scaler = TemperatureScaler(ck["temperature"])
        self.threshold = float(ck["threshold"].get("threshold", 0.5))
        self.backbone_name = ck["backbone"]
        self.with_fourier = bool(ck["extra"].get("with_fourier", False))
        self.input_dim = ck["config"].input_dim

    def score_features(self, features):
        return self.scaler.transform(self.head.logits(features))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--checkpoints", type=str, nargs="+", required=True,
                    help="name=path pairs, e.g. clean=ckpt/a.pt robust=ckpt/b.pt")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--backbone", type=str, default=None,
                    help="override; defaults to the backbone recorded in the first checkpoint")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="evaluate on at most N test images")
    ap.add_argument("--skip_held_out", action="store_true")
    ap.add_argument("--skip_compound", action="store_true")
    ap.add_argument("--metric", type=str, default="auc")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = ensure_dir(args.out)

    models = {}
    for spec in args.checkpoints:
        if "=" not in spec:
            ap.error(f"--checkpoints entries must look like name=path (got {spec!r})")
        name, path = spec.split("=", 1)
        models[name] = _ScoredHead(path, device=args.device)
        log.info("loaded %-10s from %s (threshold %.3f)", name, path, models[name].threshold)

    first = next(iter(models.values()))
    backbone = load_backbone(args.backbone or first.backbone_name, device=args.device)

    df_manifest = read_manifest(args.manifest)
    sub = df_manifest if args.split in (None, "all") else df_manifest[df_manifest["split"] == args.split]
    if args.limit:
        sub = sub.sample(n=min(args.limit, len(sub)), random_state=args.seed)
    if len(sub) == 0:
        raise SystemExit(f"no rows in manifest for split={args.split!r}")
    log.info("evaluating on %d images (split=%s)", len(sub), args.split)

    conditions = list(EVAL_GRID)
    if not args.skip_held_out:
        conditions += HELD_OUT_GRID
    if not args.skip_compound:
        conditions += COMPOUND_GRID

    results = run_robustness_grid(
        backbone,
        models,
        sub["image_path"].tolist(),
        sub["label"].tolist(),
        conditions=conditions,
        thresholds={k: m.threshold for k, m in models.items()},
        batch_size=args.batch_size,
        with_fourier=first.with_fourier,
        save_scores_to=out_dir / "scores.csv",
    )
    results.to_csv(out_dir / "robustness.csv", index=False)
    log.info("robustness table -> %s", out_dir / "robustness.csv")

    summary = degradation_table(results, metric=args.metric)
    print("\n=== Robustness summary ===")
    print(summary.to_string(index=False))

    write_report(out_dir / "robustness.csv", out_dir, metric=args.metric,
                 scores_csv=out_dir / "scores.csv")

    # The headline sentence, generated rather than written by hand.
    if {"clean", "robust"} <= set(summary["model"]):
        c = summary.set_index("model")
        line = (
            f"Training on damaged images changed mean {args.metric.upper()} on the specified "
            f"transforms from {c.loc['clean', f'spec_transforms_{args.metric}']:.4f} to "
            f"{c.loc['robust', f'spec_transforms_{args.metric}']:.4f}, and on corruptions never "
            f"seen in training from {c.loc['clean', f'held_out_{args.metric}']:.4f} to "
            f"{c.loc['robust', f'held_out_{args.metric}']:.4f}, "
            f"at a cost of {c.loc['clean', f'clean_{args.metric}'] - c.loc['robust', f'clean_{args.metric}']:+.4f} "
            f"on clean images."
        )
        print("\n" + line)
        (out_dir / "headline.txt").write_text(line + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
