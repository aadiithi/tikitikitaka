#!/usr/bin/env python3
"""Generate a tiny procedural dataset so the pipeline can be run with no downloads.

    python scripts/make_synthetic_dataset.py --out data/synthetic --n 60

This is **not** a research dataset and no reported result may come from it. It
exists for exactly two purposes:

* CI and `make smoke` - so a fresh clone can prove the whole chain works
  (manifest -> features -> train -> robustness -> predict.py) in under a minute
  on a laptop with no network.
* Reviewer convenience - anyone can verify our code runs before deciding
  whether to spend an hour downloading SID_Set.

The "authentic" class is built from broadband noise textures (camera-like: energy
at every spatial frequency). The "generated" class is built from smooth
low-frequency gradients with a faint periodic grid, which imitates the
upsampling checkerboard real generators leave behind. A detector will find this
trivially easy - that is the point; it makes a broken pipeline obvious.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _authentic(rng: np.random.Generator, size: int) -> Image.Image:
    """Broadband texture with grain - a stand-in for sensor noise."""
    base = rng.normal(0.5, 0.18, (size, size, 3))
    for scale in (2, 4, 8, 16):
        coarse = rng.normal(0.0, 0.12, (scale, scale, 3))
        up = np.asarray(
            Image.fromarray(((coarse - coarse.min()) / (np.ptp(coarse) + 1e-8) * 255)
                            .astype(np.uint8)).resize((size, size), Image.BICUBIC),
            dtype=np.float32,
        ) / 255.0
        base += (up - up.mean()) * (0.35 / scale ** 0.5)
    base += rng.normal(0.0, 0.05, base.shape)  # fine grain at every frequency
    return Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8), "RGB")


def _generated(rng: np.random.Generator, size: int) -> Image.Image:
    """Smooth gradients plus a faint periodic grid - upsampler-like artefacts."""
    y, x = np.mgrid[0:size, 0:size] / float(size)
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        a, b = rng.uniform(0.5, 2.5), rng.uniform(0.5, 2.5)
        phase = rng.uniform(0, 2 * np.pi)
        img[..., c] = 0.5 + 0.28 * np.sin(2 * np.pi * (a * x + b * y) + phase)
    grid = 0.03 * np.sin(2 * np.pi * x * size / 8.0) * np.sin(2 * np.pi * y * size / 8.0)
    img += grid[..., None]
    img += rng.normal(0.0, 0.008, img.shape)  # far less high-frequency energy
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--n", type=int, default=60, help="images per class")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # Two pseudo-"generator families" on the fake side so the family-disjoint
    # split has something to hold out, exercising that code path too.
    layout = {
        ("real", "real_photo"): _authentic,
        ("fake/glide_like", "glide"): _generated,
        ("fake/sdxl_like", "sdxl"): _generated,
    }
    total = 0
    for (rel, _family), fn in layout.items():
        d = args.out / rel
        d.mkdir(parents=True, exist_ok=True)
        count = args.n if rel == "real" else args.n // 2
        for i in range(count):
            fn(rng, args.size).save(d / f"{Path(rel).name}_{i:04d}.jpg", quality=95)
            total += 1
        print(f"{d}: {count} images")
    print(f"\n{total} images written under {args.out}")
    print(f"Next:  python scripts/build_manifest.py --root {args.out} --out data/manifest_synth.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
