import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture
def rgb_image() -> Image.Image:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (128, 160, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def image_dir(tmp_path, rgb_image):
    """A directory with three readable images and one file that is not an image."""
    d = tmp_path / "images"
    d.mkdir()
    for i in range(3):
        rgb_image.save(d / f"img_{i}.jpg", quality=90)
    (d / "notes.txt").write_text("not an image")
    return d
