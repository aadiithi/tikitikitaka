"""aigcdet - robust AI-generated image detection.

A frozen-backbone detector: a pretrained CLIP vision encoder produces a fixed
embedding per image, and a small trained head maps that embedding to a
calibrated "probability this image is AI-generated" score.

The design bet of this project is that *robustness* comes from three places,
none of which is a bigger model:

1. `aug.transforms`  - training on realistically damaged images, sampled from a
   wider distribution than the one we evaluate on.
2. `data.normalize`  - removing the dataset shortcuts (resolution, file size,
   JPEG history) that make lab accuracy lie.
3. `eval.robustness` - measuring per-transform, per-severity, so degradation is
   visible instead of averaged away.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
