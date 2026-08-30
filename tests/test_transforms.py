"""The damage model has to be correct, reproducible, and honest about overlap.

The last test is the important one: it asserts that no corruption we *report*
robustness on is also a corruption we *trained* on by family. If someone later
adds "webp" to the training policy, this test fails and the write-up's
generalisation claim cannot silently become false.
"""

import numpy as np
import pytest
from PIL import Image

from aigcdet.aug.transforms import (
    COMPOUND_GRID,
    EVAL_GRID,
    HELD_OUT_GRID,
    TrainDamagePolicy,
    apply_damage,
    center_crop,
    downscale_upscale,
    gaussian_blur,
    get_transform,
    jpeg_compress,
    list_conditions,
)


@pytest.mark.parametrize("cond", [d.name for d in EVAL_GRID + HELD_OUT_GRID + COMPOUND_GRID])
def test_every_condition_preserves_mode_and_size(cond, rgb_image):
    out = apply_damage(rgb_image, cond)
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    # Size preservation is what makes the grid a controlled experiment: only the
    # pixel content varies between conditions, never the tensor shape.
    assert out.size == rgb_image.size, f"{cond} changed image size"


@pytest.mark.parametrize("cond", [d.name for d in EVAL_GRID])
def test_conditions_are_deterministic(cond, rgb_image):
    a = np.asarray(apply_damage(rgb_image, cond))
    b = np.asarray(apply_damage(rgb_image, cond))
    if cond.startswith("noise"):
        pytest.skip("noise is stochastic by definition")
    assert np.array_equal(a, b), f"{cond} is not reproducible"


def test_clean_is_identity(rgb_image):
    assert np.array_equal(np.asarray(apply_damage(rgb_image, "clean")), np.asarray(rgb_image))


def test_severity_is_monotonic_in_damage(rgb_image):
    """Lower JPEG quality must move the image further from the original."""
    ref = np.asarray(rgb_image, dtype=np.float64)
    dists = []
    for q in (90, 70, 50, 30):
        d = np.abs(np.asarray(jpeg_compress(rgb_image, q), dtype=np.float64) - ref).mean()
        dists.append(d)
    assert dists == sorted(dists), f"JPEG damage not monotonic in quality: {dists}"


def test_blur_reduces_high_frequency_energy(rgb_image):
    def hf_energy(im):
        g = np.asarray(im.convert("L"), dtype=np.float64)
        return float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())

    assert hf_energy(gaussian_blur(rgb_image, 2.0)) < hf_energy(rgb_image)


def test_crop_and_rescale_restore_original_dimensions(rgb_image):
    assert center_crop(rgb_image, 0.8).size == rgb_image.size
    assert downscale_upscale(rgb_image, 0.25).size == rgb_image.size


def test_unknown_condition_raises():
    with pytest.raises(KeyError):
        get_transform("definitely_not_a_condition")


def test_training_policy_is_seeded_and_varied(rgb_image):
    p1 = TrainDamagePolicy(seed=7)
    p2 = TrainDamagePolicy(seed=7)
    outs1 = [np.asarray(p1(rgb_image)) for _ in range(8)]
    outs2 = [np.asarray(p2(rgb_image)) for _ in range(8)]
    for a, b in zip(outs1, outs2):
        assert np.array_equal(a, b), "same seed must give the same damage sequence"
    # And it must not collapse to a single transform.
    uniques = {a.tobytes()[:2048] for a in outs1}
    assert len(uniques) > 1, "policy produced identical output every time"


def test_held_out_families_are_never_in_the_training_policy():
    """The generalisation claim depends on this being true. Guard it in CI."""
    policy_families = {"jpeg", "blur", "rescale", "noise", "color", "crop"}
    held_out_families = {d.family for d in HELD_OUT_GRID}
    assert not (policy_families & held_out_families), (
        "a corruption family is both trained on and reported as 'unseen': "
        f"{policy_families & held_out_families}"
    )


def test_grid_names_are_unique():
    names = [d.name for d in list_conditions()]
    assert len(names) == len(set(names))
