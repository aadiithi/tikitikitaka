#!/usr/bin/env python3
"""Regenerate every figure and table from results/robustness.csv.

    python scripts/make_report.py --results results/robustness.csv --out results/

Kept separate from the evaluation so that restyling a chart never requires
re-running an hour of GPU work, and so that no number in the README is ever
typed by a human.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from aigcdet.eval.report import write_report
from aigcdet.utils.logging import get_logger

log = get_logger("make_report")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results/robustness.csv"))
    ap.add_argument("--scores", type=Path, default=Path("results/scores.csv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--metric", type=str, default="auc")
    args = ap.parse_args()

    r = write_report(args.results, args.out, metric=args.metric, scores_csv=args.scores)
    print(r["summary"].to_string(index=False))
    log.info("figures: %s", ", ".join(str(v) for v in r["figures"].values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
