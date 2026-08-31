"""Fourier-domain features: a cheap, interpretable signal alongside CLIP.

RESEARCH_BRIEF.md (Frank et al., ICML 2020): GAN/diffusion upsampling leaves
periodic artefacts visible as regular peaks in the frequency domain, and a
simple classifier on DCT/FFT features works well on clean images. This module
is the numeric counterpart to the FFT spectrum *figure* used for
explainability -- same underlying computation, packaged as a fixed-length
feature vector instead of a plot.

Deliberately hand-crafted and fixed-length rather than learned: `Detector`
concatenates this directly onto the frozen CLIP embedding
(`backbone.dim + 28`), so it has to be cheap, deterministic, and the same
length for every image regardless of the image's original size.

28 dimensions total:
  - 24 radial bins of the log-magnitude spectrum (low to high frequency),
    averaged over concentric rings and over colour channels.
  - 4 global summary stats: mean, std, max, and the high-frequency energy
    ratio (energy outside the innermost quarter-radius vs. total energy).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

N_RADIAL_BINS = 24
N_SUMMARY_STATS = 4
FOURIER_FEATURE_DIM = N_RADIAL_BINS + N_SUMMARY_STATS


def _radial_profile(magnitude: np.ndarray, n_bins: int) -> np.ndarray:
    """Average a 2D magnitude spectrum over concentric rings around the DC
    component, producing a fixed-length profile regardless of image size."""
    h, w = magnitude.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = r.max()
    if r_max <= 0:
        return np.zeros(n_bins, dtype=np.float32)

    bin_idx = np.minimum((r / r_max * n_bins).astype(int), n_bins - 1)
    profile = np.zeros(n_bins, dtype=np.float64)
    counts = np.bincount(bin_idx.ravel(), minlength=n_bins).astype(np.float64)
    sums = np.bincount(bin_idx.ravel(), weights=magnitude.ravel(), minlength=n_bins)
    nonzero = counts > 0
    profile[nonzero] = sums[nonzero] / counts[nonzero]
    return profile.astype(np.float32)


def fourier_features(image: Image.Image) -> np.ndarray:
    """Compute the 28-d Fourier feature vector for one PIL image."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)

    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(spectrum))

    radial = _radial_profile(magnitude, N_RADIAL_BINS)

    total_energy = float(magnitude.sum()) + 1e-8
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    inner_r = min(h, w) / 8.0  # innermost quarter-radius
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    high_freq_energy = float(magnitude[r > inner_r].sum())
    high_freq_ratio = high_freq_energy / total_energy

    summary = np.array(
        [magnitude.mean(), magnitude.std(), magnitude.max(), high_freq_ratio],
        dtype=np.float32,
    )

    return np.concatenate([radial, summary]).astype(np.float32)
