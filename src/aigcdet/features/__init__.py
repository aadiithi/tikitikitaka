"""aigcdet.features - frozen image backbones and cached embeddings.

`backbone.load_backbone(name)` returns a frozen encoder ("dummy" for tests/CI,
"clip-vit-l14"/"clip-vit-b16" for real runs). `extract.extract_features(...)`
runs images (optionally with damaged views) through a backbone once and caches
the result as a `FeatureBundle`, so every subsequent experiment reads an array
from disk instead of re-encoding images. `fourier.fourier_features(...)` is an
optional 28-d spectral feature, concatenated onto the backbone embedding when
`--with_fourier` is set.
"""

from __future__ import annotations

__all__ = ["backbone", "extract", "fourier"]
