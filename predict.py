#!/usr/bin/env python3
"""Required deliverable: image directory in, confidence scores out.

    python predict.py --image_dir path/to/images --output predictions.json

Writes a JSON array of records:

    [
      {"image_path": "path/to/images/a.jpg", "pred": 0.9312},
      {"image_path": "path/to/images/b.png", "pred": 0.0417}
    ]

`pred` is P(image is AI-generated), calibrated, in [0, 1].

Contract guarantees, because a batch scorer that dies on one bad file is
useless in production:

* Every image found under `--image_dir` gets exactly one record.
* An unreadable file gets `pred: 0.5` and an `error` field rather than crashing
  the run or being silently dropped.
* Output order matches the sorted walk of the directory, so two runs diff
  cleanly.
* Exit code is 0 on success, 2 on a configuration problem (no checkpoint, empty
  directory), 1 on an unexpected error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aigcdet.models.detector import Detector           # noqa: E402
from aigcdet.utils.io import ensure_dir, iter_image_files  # noqa: E402
from aigcdet.utils.logging import get_logger           # noqa: E402

log = get_logger("predict")

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "detector_robust.pt"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score images for the likelihood that they are AI-generated.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image_dir", required=True, type=Path,
                   help="directory of images to score (searched recursively)")
    p.add_argument("--output", "-o", type=Path, default=Path("predictions.json"),
                   help="path to write the JSON results to")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                   help="trained head checkpoint (.pt)")
    p.add_argument("--backbone", type=str, default=None,
                   help="override the backbone recorded in the checkpoint "
                        "(e.g. 'dummy' for an offline smoke test)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda", "mps"],
                   help="force a device; default auto-detects CUDA/MPS")
    p.add_argument("--no_recursive", action="store_true",
                   help="only score images directly inside --image_dir")
    p.add_argument("--include_label", action="store_true",
                   help="also emit the thresholded 0/1 decision alongside the score")
    p.add_argument("--quiet", action="store_true", help="suppress the progress bar")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    if not args.image_dir.exists():
        log.error("image_dir does not exist: %s", args.image_dir)
        return 2

    paths = [str(p) for p in iter_image_files(args.image_dir, recursive=not args.no_recursive)]
    if not paths:
        log.error("no images found under %s (looked for jpg/png/bmp/webp/tif)", args.image_dir)
        return 2
    log.info("found %d images under %s", len(paths), args.image_dir)

    if not args.checkpoint.exists():
        log.error(
            "checkpoint not found: %s\n"
            "Train one with `python scripts/train_head.py`, or download the released "
            "checkpoint as described in the README (Setup -> Pretrained weights).",
            args.checkpoint,
        )
        return 2

    detector = Detector(args.checkpoint, backbone_name=args.backbone, device=args.device)
    preds = detector.predict_paths(paths, batch_size=args.batch_size, progress=not args.quiet)

    records = [p.as_record(include_label=args.include_label) for p in preds]
    ensure_dir(args.output.parent if args.output.parent.as_posix() else Path("."))
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
        fh.write("\n")

    n_flagged = sum(1 for p in preds if p.pred >= detector.threshold)
    n_failed = sum(1 for p in preds if p.error)
    log.info(
        "wrote %d predictions -> %s | %d flagged at threshold %.3f | %d unreadable | %.1fs",
        len(records), args.output, n_flagged, detector.threshold, n_failed, time.time() - t0,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warning("interrupted")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001 - top-level guard, we re-raise the detail
        log.exception("prediction failed: %s", exc)
        sys.exit(1)
