"""`Detector` - the one object the demo, the eval harness and predict.py share.

Keeping inference in a single class means the number in the Gradio demo, the
number in the robustness table and the number in `predictions.json` cannot
drift apart. There is exactly one code path from an image file to a score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from ..data.normalize import CANONICAL_SPEC, canonicalize
from ..features.backbone import load_backbone
from ..models.calibration import TemperatureScaler
from ..models.head import load_checkpoint
from ..utils.io import load_image
from ..utils.logging import get_logger

log = get_logger("models.detector")


@dataclass
class Prediction:
    image_path: str
    pred: float          # P(AI-generated), calibrated, in [0, 1]
    label: int           # thresholded decision at the operating point
    error: str | None = None

    def as_record(self, include_label: bool = False) -> dict:
        """The submission format is exactly {image_path, pred}; the label is
        opt-in so we never break the required contract by accident."""
        rec = {"image_path": self.image_path, "pred": round(float(self.pred), 6)}
        if include_label:
            rec["label"] = int(self.label)
        if self.error:
            rec["error"] = self.error
        return rec


class Detector:
    """Frozen backbone + trained head + calibration, behind one `.score()`."""

    def __init__(
        self,
        checkpoint: str | Path,
        backbone_name: str | None = None,
        device: str | None = None,
        canonical: bool = True,
        with_fourier: bool | None = None,
    ):
        ck = load_checkpoint(checkpoint, device=device)
        self.head = ck["model"]
        self.temperature = ck["temperature"]
        self.threshold = float(ck["threshold"].get("threshold", 0.5))
        self.extra = ck["extra"]
        self.canonical = canonical
        self.with_fourier = (
            self.extra.get("with_fourier", False) if with_fourier is None else with_fourier
        )
        self.backbone_name = backbone_name or ck["backbone"]
        self.backbone = load_backbone(self.backbone_name, device=device)
        self._scaler = TemperatureScaler(self.temperature)

        expected = ck["config"].input_dim
        got = self.backbone.dim + (24 + 4 if self.with_fourier else 0)
        if expected != got:
            raise ValueError(
                f"checkpoint expects {expected}-d features but backbone "
                f"'{self.backbone_name}' produces {got}-d. The checkpoint was trained "
                f"with backbone '{ck['backbone']}' - pass that one."
            )
        log.info(
            "detector ready | backbone=%s | threshold=%.3f | temperature=%.3f",
            self.backbone_name, self.threshold, self.temperature,
        )

    # -- core -------------------------------------------------------------
    def embed(self, images: Sequence[Image.Image], batch_size: int = 32) -> np.ndarray:
        prepped = [canonicalize(im, CANONICAL_SPEC) if self.canonical else im for im in images]
        feats = self.backbone.encode(prepped, batch_size=batch_size)
        if self.with_fourier:
            from ..features.fourier import fourier_features

            ff = np.stack([fourier_features(im) for im in prepped]).astype(np.float32)
            feats = np.concatenate([feats, ff], axis=1)
        return feats

    def score_images(self, images: Sequence[Image.Image], batch_size: int = 32) -> np.ndarray:
        """P(AI-generated) for already-loaded PIL images."""
        if not images:
            return np.zeros(0, dtype=np.float32)
        logits = self.head.logits(self.embed(images, batch_size=batch_size))
        return self._scaler.transform(logits).astype(np.float32)

    def score_features(self, features: np.ndarray) -> np.ndarray:
        return self._scaler.transform(self.head.logits(features)).astype(np.float32)

    def predict_paths(
        self, paths: Iterable[str | Path], batch_size: int = 32, progress: bool = True
    ) -> list[Prediction]:
        """Score files on disk. Never raises on a bad file - a single corrupt
        image must not take down a batch job over a whole upload directory."""
        from tqdm.auto import tqdm

        paths = [str(p) for p in paths]
        results: list[Prediction] = []
        iterator = range(0, len(paths), batch_size)
        if progress:
            iterator = tqdm(iterator, desc="scoring", unit="batch")

        for start in iterator:
            chunk = paths[start : start + batch_size]
            images, ok = [], []
            for p in chunk:
                try:
                    images.append(load_image(p))
                    ok.append(p)
                except Exception as exc:
                    # Neutral score: refusing to guess is more honest than 0 or 1,
                    # and the error field tells the operator what happened.
                    results.append(Prediction(p, 0.5, 0, error=f"unreadable: {type(exc).__name__}"))
            if not images:
                continue
            scores = self.score_images(images, batch_size=batch_size)
            for p, s in zip(ok, scores):
                results.append(Prediction(p, float(s), int(s >= self.threshold)))

        order = {p: i for i, p in enumerate(paths)}
        results.sort(key=lambda r: order[r.image_path])
        return results
