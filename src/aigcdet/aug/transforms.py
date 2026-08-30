"""The damage model: what happens to an image between "generated" and "seen".

This module is the centre of the project, so it is worth being explicit about
the design.

An image posted to a platform is never the tensor the generator produced. It is
re-encoded, resized to a thumbnail, screenshotted, cropped to a 4:5 feed, run
through a colour filter, and re-uploaded by the next person who reposts it.
Detectors that quietly learn generator-specific high-frequency fingerprints
score brilliantly on pristine files and collapse on that pipeline, because the
fingerprint lives exactly in the frequencies that JPEG throws away first.

We therefore keep three separate transform families, and never mix them:

* `EVAL_GRID`      - the transforms and severities named in the problem
                     statement. These are our *reported* robustness axis.
* `TrainDamagePolicy` - what we actually train on. Continuous severity ranges
                     that *bracket* the eval grid rather than reproducing it,
                     so the head never memorises "JPEG exactly at quality 70".
* `HELD_OUT_GRID`  - damage types the model is never trained on (WebP, screen
                     re-capture, sharpening, small rotations). Scores here are
                     the honest generalisation number, and the one we lead with
                     in the write-up.

`COMPOUND_GRID` chains two operations to simulate a second repost, which is
where most detectors actually die.

Every transform takes and returns a PIL RGB image, so they compose freely and
can be applied either at feature-extraction time or inside a DataLoader.
"""

from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# --------------------------------------------------------------------------
# Primitive operations
# --------------------------------------------------------------------------


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """Re-encode as JPEG at a given quality and decode back.

    The single most destructive real-world operation, and the one that removes
    the high-frequency generator fingerprints naive detectors rely on.
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), subsampling=2)
    buf.seek(0)
    with Image.open(buf) as out:
        return out.convert("RGB").copy()


def webp_compress(img: Image.Image, quality: int) -> Image.Image:
    """WebP re-encode. Held out from training: different artefact structure
    from JPEG (no 8x8 block grid), so it tests whether the model learned
    "JPEG blocks" or something about the image itself."""
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as out:
        return out.convert("RGB").copy()


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def sharpen(img: Image.Image, amount: float) -> Image.Image:
    """Unsharp mask. Held out: phone galleries and messaging apps sharpen on
    export, and sharpening *amplifies* high-frequency content rather than
    removing it, so it probes the opposite failure mode from blur."""
    return ImageEnhance.Sharpness(img).enhance(1.0 + float(amount))


def downscale_upscale(img: Image.Image, factor: float) -> Image.Image:
    """Shrink then restore original size - the thumbnail round-trip.

    Information is destroyed on the way down and interpolated back on the way
    up, which is precisely what a platform's thumbnail pipeline does.
    """
    w, h = img.size
    nw, nh = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
    small = img.resize((nw, nh), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    """Additive Gaussian noise, sigma in [0, 1] units of pixel intensity."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    noise = np.random.normal(0.0, float(sigma), arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((out * 255.0).round().astype(np.uint8), mode="RGB")


def color_jitter(img: Image.Image, strength: float, rng: random.Random | None = None) -> Image.Image:
    """Jitter brightness, contrast and saturation by +/- `strength` (fraction).

    Deterministic when `rng` is None: we walk each channel to a fixed corner of
    the jitter box so that the eval grid is reproducible. Training uses a real
    RNG so the model sees the whole box.
    """
    if rng is None:
        factors = (1.0 + strength, 1.0 - strength, 1.0 + strength)
    else:
        factors = tuple(1.0 + rng.uniform(-strength, strength) for _ in range(3))
    out = ImageEnhance.Brightness(img).enhance(factors[0])
    out = ImageEnhance.Contrast(out).enhance(factors[1])
    out = ImageEnhance.Color(out).enhance(factors[2])
    return out


def center_crop(img: Image.Image, keep: float) -> Image.Image:
    """Keep the central `keep` fraction of each side, then restore size.

    Restoring the original size matters: otherwise crop and rescale become the
    same experiment, and we would not be able to tell which one hurt us.
    """
    w, h = img.size
    nw, nh = max(1, int(round(w * keep))), max(1, int(round(h * keep)))
    left, top = (w - nw) // 2, (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.BICUBIC)


def random_crop(img: Image.Image, keep: float, rng: random.Random) -> Image.Image:
    w, h = img.size
    nw, nh = max(1, int(round(w * keep))), max(1, int(round(h * keep)))
    left = rng.randint(0, max(0, w - nw))
    top = rng.randint(0, max(0, h - nh))
    return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.BICUBIC)


def rotate_small(img: Image.Image, degrees: float) -> Image.Image:
    """Small rotation with resampling. Held out: it resamples every pixel on a
    non-integer grid, which destroys pixel-aligned periodic artefacts (e.g. the
    checkerboard left by transposed-convolution upsamplers) without visibly
    changing the picture."""
    return img.rotate(float(degrees), resample=Image.BICUBIC, expand=False)


def screen_recapture(img: Image.Image, scale: float = 0.75) -> Image.Image:
    """Crude 'someone screenshotted it' simulation: downscale, slight blur,
    mild contrast lift, JPEG at a phone-screenshot quality. Held out."""
    out = downscale_upscale(img, scale)
    out = gaussian_blur(out, 0.4)
    out = ImageEnhance.Contrast(out).enhance(1.08)
    return jpeg_compress(out, 82)


# --------------------------------------------------------------------------
# Condition registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Damage:
    """One named, reproducible corruption condition.

    `name` is what appears in the robustness table, so it doubles as the
    column key in `results/robustness.csv`.
    """

    name: str
    family: str
    severity: float
    fn: Callable[[Image.Image], Image.Image] = field(repr=False)
    held_out: bool = False

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.fn(img)


def _d(name, family, severity, fn, held_out=False) -> Damage:
    return Damage(name=name, family=family, severity=severity, fn=fn, held_out=held_out)


# The grid named in the problem statement. `clean` is included so that every
# table has its own baseline row and nobody has to join two files.
EVAL_GRID: List[Damage] = [
    _d("clean", "clean", 0.0, lambda im: im),
    # JPEG compression
    _d("jpeg_q90", "jpeg", 90, lambda im: jpeg_compress(im, 90)),
    _d("jpeg_q70", "jpeg", 70, lambda im: jpeg_compress(im, 70)),
    _d("jpeg_q50", "jpeg", 50, lambda im: jpeg_compress(im, 50)),
    _d("jpeg_q30", "jpeg", 30, lambda im: jpeg_compress(im, 30)),
    # Gaussian blur
    _d("blur_s0.5", "blur", 0.5, lambda im: gaussian_blur(im, 0.5)),
    _d("blur_s1.0", "blur", 1.0, lambda im: gaussian_blur(im, 1.0)),
    _d("blur_s2.0", "blur", 2.0, lambda im: gaussian_blur(im, 2.0)),
    # Rescale (down then back up)
    _d("rescale_0.50", "rescale", 0.50, lambda im: downscale_upscale(im, 0.50)),
    _d("rescale_0.25", "rescale", 0.25, lambda im: downscale_upscale(im, 0.25)),
    # Additive noise
    _d("noise_s0.02", "noise", 0.02, lambda im: gaussian_noise(im, 0.02)),
    _d("noise_s0.05", "noise", 0.05, lambda im: gaussian_noise(im, 0.05)),
    _d("noise_s0.10", "noise", 0.10, lambda im: gaussian_noise(im, 0.10)),
    # Colour adjustment
    _d("color_20pct", "color", 0.20, lambda im: color_jitter(im, 0.20)),
    # Cropping
    _d("crop_80pct", "crop", 0.80, lambda im: center_crop(im, 0.80)),
]

# Never used in training. This is the number we lead with, because it is the
# only one that answers "does this survive damage you did not anticipate".
HELD_OUT_GRID: List[Damage] = [
    _d("webp_q80", "webp", 80, lambda im: webp_compress(im, 80), held_out=True),
    _d("webp_q50", "webp", 50, lambda im: webp_compress(im, 50), held_out=True),
    _d("sharpen_1.0", "sharpen", 1.0, lambda im: sharpen(im, 1.0), held_out=True),
    _d("rotate_2deg", "rotate", 2.0, lambda im: rotate_small(im, 2.0), held_out=True),
    _d("screen_recapture", "recapture", 1.0, screen_recapture, held_out=True),
]

# Two operations in sequence: the second repost. Nobody sees a single-corruption
# image in the wild.
COMPOUND_GRID: List[Damage] = [
    _d(
        "rescale0.5+jpeg50",
        "compound",
        1.0,
        lambda im: jpeg_compress(downscale_upscale(im, 0.5), 50),
    ),
    _d(
        "crop80+jpeg30",
        "compound",
        2.0,
        lambda im: jpeg_compress(center_crop(im, 0.80), 30),
    ),
    _d(
        "blur1.0+jpeg70",
        "compound",
        1.5,
        lambda im: jpeg_compress(gaussian_blur(im, 1.0), 70),
    ),
    _d(
        "recapture+jpeg50",
        "compound",
        3.0,
        lambda im: jpeg_compress(screen_recapture(im), 50),
        held_out=True,
    ),
]

_REGISTRY: Dict[str, Damage] = {d.name: d for d in EVAL_GRID + HELD_OUT_GRID + COMPOUND_GRID}


def get_transform(name: str) -> Damage:
    if name not in _REGISTRY:
        raise KeyError(f"unknown condition '{name}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_conditions(
    include_eval: bool = True, include_held_out: bool = True, include_compound: bool = True
) -> List[Damage]:
    out: List[Damage] = []
    if include_eval:
        out += EVAL_GRID
    if include_held_out:
        out += HELD_OUT_GRID
    if include_compound:
        out += COMPOUND_GRID
    return out


def apply_damage(img: Image.Image, condition: str | Damage) -> Image.Image:
    d = condition if isinstance(condition, Damage) else get_transform(condition)
    return d(img)


# --------------------------------------------------------------------------
# Training-time policy
# --------------------------------------------------------------------------


@dataclass
class TrainDamagePolicy:
    """Sample a random damage chain for a training image.

    Two deliberate choices, both of which we defend in the write-up:

    * Severity ranges are *continuous and wider* than the eval grid. Training at
      JPEG q in [25, 95] rather than at {30, 50, 70, 90} means the head cannot
      overfit to four specific quantisation tables, and it means our reported
      eval numbers are not "we trained on the test set" in disguise.

    * With probability `p_chain` we apply two operations. Single-corruption
      training produces a model that handles any one insult and falls over on
      the realistic combination of a resize followed by a re-encode.

    `p_clean` keeps a fraction of images pristine, so the model does not lose
    accuracy on undamaged uploads - that regression is a real cost of
    corruption training and we measure it rather than hide it.
    """

    p_clean: float = 0.25
    p_chain: float = 0.35
    jpeg_quality: tuple = (25, 95)
    blur_sigma: tuple = (0.3, 2.5)
    rescale_factor: tuple = (0.20, 0.85)
    noise_sigma: tuple = (0.01, 0.12)
    color_strength: tuple = (0.05, 0.30)
    crop_keep: tuple = (0.65, 0.95)
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # -- individual samplers -------------------------------------------------
    def _sample_op(self) -> Callable[[Image.Image], Image.Image]:
        rng = self._rng
        kind = rng.choice(["jpeg", "blur", "rescale", "noise", "color", "crop"])
        if kind == "jpeg":
            q = rng.randint(*self.jpeg_quality)
            return lambda im: jpeg_compress(im, q)
        if kind == "blur":
            s = rng.uniform(*self.blur_sigma)
            return lambda im: gaussian_blur(im, s)
        if kind == "rescale":
            f = rng.uniform(*self.rescale_factor)
            return lambda im: downscale_upscale(im, f)
        if kind == "noise":
            s = rng.uniform(*self.noise_sigma)
            return lambda im: gaussian_noise(im, s)
        if kind == "color":
            s = rng.uniform(*self.color_strength)
            return lambda im: color_jitter(im, s, rng)
        k = rng.uniform(*self.crop_keep)
        return lambda im: random_crop(im, k, rng)

    def __call__(self, img: Image.Image) -> Image.Image:
        rng = self._rng
        if rng.random() < self.p_clean:
            return img
        out = self._sample_op()(img)
        if rng.random() < self.p_chain:
            out = self._sample_op()(out)
        return out

    def describe(self) -> dict:
        """Machine-readable policy description, written into every run's
        metadata so a reviewer can tell exactly what the model was trained on."""
        return {
            "p_clean": self.p_clean,
            "p_chain": self.p_chain,
            "jpeg_quality": list(self.jpeg_quality),
            "blur_sigma": list(self.blur_sigma),
            "rescale_factor": list(self.rescale_factor),
            "noise_sigma": list(self.noise_sigma),
            "color_strength": list(self.color_strength),
            "crop_keep": list(self.crop_keep),
            "never_trained_on": sorted({d.family for d in HELD_OUT_GRID}),
        }
