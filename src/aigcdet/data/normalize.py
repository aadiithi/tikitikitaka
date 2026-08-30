"""Canonicalisation - remove the container-level shortcuts before pixels matter.

If real images and fake images happen to differ systematically in resolution,
aspect ratio, or colour mode -- which is extremely common when they come from
different sources, e.g. camera photos vs. 1024x1024 diffusion output -- a
model can learn "big square image => fake" and score brilliantly for reasons
that have nothing to do with detecting synthesis. `canonicalize` forces every
image through an identical resize -> centre-crop -> mode pipeline so that
signal is no longer available, regardless of what the pixels look like.

This is deliberately a *lossy*, deterministic transform, not a "best effort"
resize: the whole point is that two images with different native resolutions
must come out pixel-dimension-identical on the other side. `extract_features`
and `Detector.embed` both route every image through this before it reaches
the backbone; `--no_canonical` exists only to run the shortcut ablation named
in the design doc, and should never be the default at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class CanonicalSpec:
    """The one shape every image is forced into before feature extraction."""

    square: int = 224
    resample: int = Image.LANCZOS
    # Re-encode quality used only when a canonicalised image needs to be
    # serialised (e.g. for the metadata-shortcut ablation, or for saving
    # a canonicalised preview) -- not applied to in-memory tensors.
    jpeg_quality: int = 95


CANONICAL_SPEC = CanonicalSpec()


def canonicalize(image: Image.Image, spec: CanonicalSpec = CANONICAL_SPEC) -> Image.Image:
    """Resize, centre-crop, and strip metadata so every image is comparable.

    Steps, in order, and why each one exists:

    1. Convert to RGB. Some real photos are stored as grayscale or CMYK,
       some fakes are saved with an alpha channel from a transparent-PNG
       pipeline -- either would be a channel-count shortcut if left alone.
    2. Resize the shorter side to `spec.square`, preserving aspect ratio,
       then centre-crop to an exact `spec.square` x `spec.square`. A resize
       to a fixed size discards native resolution as a signal entirely
       (rather than e.g. padding, which would make the padding shortcut).
    3. Rebuild the image from raw pixel data into a fresh `Image` object.
       PIL images loaded from disk can carry EXIF orientation tags and other
       metadata in `.info`; simply calling `.convert("RGB")` does not clear
       this. Rebuilding from `getdata()` guarantees nothing but pixels
       survives, matching the EXIF-stripping step already applied when
       images are ingested (see `utils.io.load_image`) -- this function is
       the second, defence-in-depth pass that runs on every image regardless
       of how it was loaded.

    Deterministic given the same input and spec: two images of different
    original size and format that depict the same scene are indistinguishable
    to anything downstream based on their container alone.
    """
    img = image.convert("RGB")
    width, height = img.size
    short_side = min(width, height)
    if short_side <= 0:
        raise ValueError(f"cannot canonicalise a zero-size image ({width}x{height})")

    scale = spec.square / short_side
    new_width = max(spec.square, round(width * scale))
    new_height = max(spec.square, round(height * scale))
    resized = img.resize((new_width, new_height), spec.resample)

    left = (new_width - spec.square) // 2
    top = (new_height - spec.square) // 2
    cropped = resized.crop((left, top, left + spec.square, top + spec.square))

    clean = Image.new("RGB", cropped.size)
    clean.putdata(list(cropped.getdata()))
    return clean
