"""Frozen image backbones: a real one, and a fast fake one for tests.

Every backbone this module returns satisfies one contract:

    .name    - str, identifies the backbone in checkpoints and logs
    .dim     - int, output feature dimension
    .device  - str, where it runs
    .encode(images: Sequence[PIL.Image], batch_size: int) -> np.ndarray
             shape (len(images), .dim), L2-normalised rows, float32

`Detector`, `extract_features`, and the robustness harness all call only
`.encode` and read `.dim`/`.name`/`.device` -- nothing else about a backbone's
internals leaks into the rest of the codebase. That's deliberate: it's what
lets `"dummy"` stand in for `"clip-vit-l14"` in every test and in `make smoke`,
with no network and no GPU, while still exercising every line of code the real
backbone would run through.

Why frozen: TECHNICAL_DESIGN.md's whole speed argument -- feature extraction
runs once, in minutes, and every subsequent experiment (different head,
different augmentation mix, different threshold) rereads a `.npz` from disk
instead of touching a GPU again.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

from ..utils.logging import get_logger

log = get_logger("features.backbone")

# name -> (open_clip model tag, pretrained tag, output dim)
# ViT-L/14 is the design doc's default: "a meaningfully stronger feature space
# [than B/16] and it costs nothing but inference time" because it's frozen.
_CLIP_REGISTRY = {
    "clip-vit-l14": ("ViT-L-14", "openai", 768),
    "clip-vit-b16": ("ViT-B-16", "openai", 512),
}


class DummyBackbone:
    """A fast, deterministic stand-in for CLIP. No network, no GPU, no torch
    hub cache -- exists purely so tests and `make smoke` can exercise the real
    extraction/training/inference code paths in milliseconds.

    Deterministic given the same image: the "embedding" is a low-dimensional
    hash of a downsampled version of the image's pixels, so two calls on the
    same image produce identical features (needed for the checkpoint
    round-trip test) and two different images very likely produce different
    features (needed for the "head learns a separable problem" test to have
    something to separate).
    """

    def __init__(self, dim: int = 32, device: str = "cpu"):
        self.name = "dummy"
        self.dim = dim
        self.device = device
        # Fixed random projection, seeded once, so "training" the projection
        # weights isn't a step anyone can forget to do consistently.
        self._proj = np.random.default_rng(0).normal(size=(3, dim)).astype(np.float32)

    def _embed_one(self, img: Image.Image) -> np.ndarray:
        small = img.convert("RGB").resize((16, 16), Image.BILINEAR)
        arr = np.asarray(small, dtype=np.float32) / 255.0  # (16, 16, 3)
        channel_means = arr.reshape(-1, 3).mean(axis=0)  # (3,) - cheap, stable summary
        vec = channel_means @ self._proj  # (dim,)
        # Add a touch of spatial detail so two images with the same average
        # colour don't collide: fold in coarse block variance too.
        block_var = arr.reshape(4, 4, 4, 4, 3).var(axis=(1, 3)).reshape(-1)  # (16*3,)
        vec = vec + 0.05 * (block_var[: self.dim % max(len(block_var), 1) or self.dim].sum())
        norm = np.linalg.norm(vec)
        return (vec / norm if norm > 1e-8 else vec).astype(np.float32)

    def encode(self, images: Sequence[Image.Image], batch_size: int = 32) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._embed_one(im) for im in images]).astype(np.float32)


class ClipBackbone:
    """Frozen OpenAI CLIP vision encoder via `open_clip`.

    Loaded once and never updated (`requires_grad_(False)`, `.eval()`) --
    per TECHNICAL_DESIGN.md §1, freezing is what preserves cross-generator
    generalisation and is what makes feature caching valid at all: if the
    backbone never changes, an image's embedding never changes, so it can be
    computed once and reused for every subsequent experiment.
    """

    def __init__(self, name: str, device: str | None = None):
        import torch

        if name not in _CLIP_REGISTRY:
            raise ValueError(f"unknown CLIP backbone '{name}'; known: {sorted(_CLIP_REGISTRY)}")
        model_tag, pretrained_tag, dim = _CLIP_REGISTRY[name]

        self.name = name
        self.dim = dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_tag, pretrained=pretrained_tag
        )
        model.eval().requires_grad_(False)
        self._model = model.to(self.device)
        self._preprocess = preprocess
        self._torch = torch

        log.info(
            "loaded frozen %s (%s/%s, dim=%d) on %s",
            self.name, model_tag, pretrained_tag, self.dim, self.device,
        )

    def _encode_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self._torch
        batch = torch.stack([self._preprocess(im) for im in images]).to(self.device)
        with torch.no_grad():
            feats = self._model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feats.float().cpu().numpy()

    def encode(self, images: Sequence[Image.Image], batch_size: int = 32) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for start in range(0, len(images), batch_size):
            out.append(self._encode_batch(images[start : start + batch_size]))
        return np.concatenate(out, axis=0).astype(np.float32)


def load_backbone(name: str, device: str | None = None):
    """Factory: `"dummy"` for tests/CI, `"clip-vit-l14"` / `"clip-vit-b16"` for
    real runs. Raising on an unknown name here, rather than deep inside
    `ClipBackbone`, keeps the error message right next to the one place every
    caller actually specifies the name."""
    if name == "dummy":
        return DummyBackbone(device=device or "cpu")
    if name in _CLIP_REGISTRY:
        return ClipBackbone(name, device=device)
    raise ValueError(
        f"unknown backbone '{name}'. Known: 'dummy', {sorted(_CLIP_REGISTRY)}"
    )
