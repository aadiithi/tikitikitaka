"""Where is the score coming from? Occlusion sensitivity.

We use occlusion rather than Grad-CAM on purpose. Grad-CAM needs a convolutional
feature map and gradients threaded through a specific architecture; occlusion
needs only the thing we already have - a function from image to score. That
makes the explanation valid for *whatever* backbone is loaded, including the
dummy one, and makes it honest: we are literally measuring "if this patch were
not here, how much would the verdict change", with no assumption about how the
model works internally.

The cost is compute - one forward pass per patch - which is acceptable for a
demo (a 8x8 grid is 64 embeddings, about a second on a GPU) and is why this
runs on demand in the UI rather than on every prediction.

Read the maps with appropriate caution, and say so in the write-up: occlusion
shows what the *current* model is sensitive to, which is evidence about the
model, not proof about the image.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def occlusion_saliency(
    detector,
    image: Image.Image,
    grid: int = 8,
    mode: str = "blur",
    batch_size: int = 32,
) -> np.ndarray:
    """(grid, grid) map of score change when each patch is occluded.

    Positive values mean "removing this region lowered the AI-generated score",
    i.e. the region was pushing the verdict toward *generated*.

    `mode="blur"` replaces the patch with a blurred copy of itself rather than
    grey. A grey box is itself an unnatural artefact that the detector may react
    to, which contaminates the explanation; a local blur removes high-frequency
    evidence while leaving colour and layout intact.
    """
    from ..data.normalize import CANONICAL_SPEC, canonicalize

    base = canonicalize(image, CANONICAL_SPEC) if detector.canonical else image.convert("RGB")
    w, h = base.size
    base_score = float(detector.score_images([base], batch_size=1)[0])

    blurred = base.filter(ImageFilter.GaussianBlur(radius=max(w, h) / (grid * 3.0)))
    grey = Image.new("RGB", base.size, (127, 127, 127))
    filler = blurred if mode == "blur" else grey

    variants, coords = [], []
    pw, ph = w / grid, h / grid
    for gy in range(grid):
        for gx in range(grid):
            box = (int(gx * pw), int(gy * ph), int((gx + 1) * pw), int((gy + 1) * ph))
            v = base.copy()
            v.paste(filler.crop(box), box)
            variants.append(v)
            coords.append((gy, gx))

    scores = detector.score_images(variants, batch_size=batch_size)
    smap = np.zeros((grid, grid), dtype=np.float32)
    for (gy, gx), s in zip(coords, scores):
        smap[gy, gx] = base_score - float(s)
    return smap


def overlay_heatmap(
    image: Image.Image,
    smap: np.ndarray,
    alpha: float = 0.5,
    size: int | None = None,
) -> Image.Image:
    """Render a saliency map over the image as a diverging blue<->red overlay.

    Diverging is the correct encoding here: the quantity has a meaningful zero
    (this patch changed nothing) and a sign. Red = pushed toward "AI-generated",
    blue = pushed toward "authentic", grey in the middle.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps

    img = image.convert("RGB")
    if size:
        img = img.resize((size, size), Image.BICUBIC)
    w, h = img.size

    m = np.asarray(smap, dtype=np.float32)
    scale = float(np.abs(m).max()) or 1.0
    normed = (m / scale + 1.0) / 2.0  # -> [0, 1] with 0.5 at "no effect"

    cmap = colormaps["coolwarm"]
    rgba = (cmap(normed)[..., :3] * 255).astype(np.uint8)
    heat = Image.fromarray(rgba, mode="RGB").resize((w, h), Image.BICUBIC)
    return Image.blend(img, heat, alpha=alpha)
