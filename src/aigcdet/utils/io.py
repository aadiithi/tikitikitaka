"""File / manifest / image IO helpers.

A "manifest" here is a CSV with one row per image and at least the columns
`image_path` and `label` (0 = authentic, 1 = AI-generated). Optional columns
`source` (dataset name) and `family` (generator family, e.g. "sdxl", "dalle3",
"coco") drive the generator-family split used for the generalisation test.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}

MANIFEST_COLUMNS = ["image_path", "label", "source", "family", "split"]


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(obj: Any, path: str | os.PathLike, indent: int = 2) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False)
        fh.write("\n")
    return p


def is_image_file(path: str | os.PathLike) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def iter_image_files(root: str | os.PathLike, recursive: bool = True) -> Iterator[Path]:
    """Yield image files under `root` in a stable, sorted order.

    Sorted order matters: `predict.py` output should be diffable between runs,
    and reviewers comparing two JSON files should not see spurious reordering.
    """
    root = Path(root)
    if root.is_file():
        if is_image_file(root):
            yield root
        return
    pattern = "**/*" if recursive else "*"
    for p in sorted(root.glob(pattern)):
        if p.is_file() and is_image_file(p):
            yield p


def load_image(path: str | os.PathLike):
    """Open an image as RGB.

    Deliberately strips EXIF and any alpha channel. Both are shortcut risks:
    real photos carry camera EXIF that generated images do not, and a detector
    that learns "has EXIF => real" scores brilliantly in the lab and fails the
    moment anything is re-encoded by a social platform.
    """
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.load()
    return im


def read_manifest(path: str | os.PathLike):
    import pandas as pd

    df = pd.read_csv(path)
    missing = {"image_path", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"manifest {path} is missing required column(s): {sorted(missing)}")
    for col in ("source", "family", "split"):
        if col not in df.columns:
            df[col] = "unknown"
    df["label"] = df["label"].astype(int)
    return df


def write_manifest(df, path: str | os.PathLike):
    p = Path(path)
    ensure_dir(p.parent)
    cols = [c for c in MANIFEST_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    df[cols + extra].to_csv(p, index=False)
    return p


def chunked(items: Iterable, size: int) -> Iterator[list]:
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
