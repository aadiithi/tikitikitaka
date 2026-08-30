"""End-to-end: synthetic data -> manifest -> features -> head -> predict.py.

Runs entirely offline on the dummy backbone, in a few seconds. This is the test
that catches the interface drift that unit tests miss - a checkpoint that
`train_head.py` writes but `Detector` cannot load is the failure that ends
demos, and it is exactly what this test would catch.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aigcdet.aug.transforms import TrainDamagePolicy
from aigcdet.data.manifest import build_manifest_from_labelled_root, split_by_family
from aigcdet.data.normalize import CANONICAL_SPEC, canonicalize
from aigcdet.features.backbone import load_backbone
from aigcdet.features.extract import FeatureBundle, extract_features
from aigcdet.models.detector import Detector
from aigcdet.models.head import HeadConfig, load_checkpoint, save_checkpoint, train_head

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "make_synthetic_dataset.py"),
         "--out", str(d), "--n", "24", "--size", "96"],
        check=True, capture_output=True,
    )
    return d


def test_canonicalize_removes_size_and_format_differences(rgb_image):
    from PIL import Image

    a = canonicalize(rgb_image)
    b = canonicalize(Image.new("RGB", (900, 300), (10, 200, 40)))
    assert a.size == b.size == (CANONICAL_SPEC.square, CANONICAL_SPEC.square)
    assert a.mode == b.mode == "RGB"


def test_manifest_and_family_split(dataset):
    df = build_manifest_from_labelled_root(dataset)
    assert len(df) > 0
    assert set(df["label"]) == {0, 1}
    df = split_by_family(df, test_frac=0.3)
    assert set(df["split"]) <= {"train", "test"}
    # A held-out generator family must not appear on both sides of the split.
    fake = df[df["label"] == 1]
    per_family_splits = fake.groupby("family")["split"].nunique()
    assert (per_family_splits == 1).all(), "a generator family leaked across the split"


def test_augmented_extraction_produces_n_views_and_traceable_rows(dataset):
    df = build_manifest_from_labelled_root(dataset)
    paths = df["image_path"].tolist()[:8]
    labels = df["label"].tolist()[:8]
    bb = load_backbone("dummy")

    clean = extract_features(bb, paths, labels, n_views=1, progress=False)
    aug = extract_features(
        bb, paths, labels, n_views=3, policy=TrainDamagePolicy(seed=0), progress=False
    )

    assert clean.features.shape == (8, bb.dim)
    assert aug.features.shape == (24, bb.dim)
    # Every augmented row traces back to a real source image.
    assert set(aug.source_index.tolist()) == set(range(8))
    assert np.allclose(np.linalg.norm(clean.features, axis=1), 1.0, atol=1e-4)


def test_feature_bundle_roundtrip(tmp_path, dataset):
    df = build_manifest_from_labelled_root(dataset)
    bb = load_backbone("dummy")
    b = extract_features(bb, df["image_path"].tolist()[:6], df["label"].tolist()[:6], progress=False)
    p = b.save(tmp_path / "f.npz")
    loaded = FeatureBundle.load(p)
    assert np.array_equal(b.features, loaded.features)
    assert np.array_equal(b.labels, loaded.labels)
    assert loaded.meta["backbone"] == bb.name


def test_head_learns_a_separable_problem():
    rng = np.random.default_rng(0)
    n, d = 400, 32
    X = np.concatenate([rng.normal(-1, 1, (n, d)), rng.normal(1, 1, (n, d))]).astype(np.float32)
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    res = train_head(X, y, cfg=HeadConfig(input_dim=d, epochs=30, patience=30), verbose=False)
    assert res["val_auc"] > 0.9
    assert res["n_trainable_params"] < 500_000, "the head is supposed to stay small"


def test_grouped_split_keeps_views_of_one_image_together():
    """Views of the same photo must never straddle train and validation."""
    rng = np.random.default_rng(1)
    src = np.repeat(np.arange(50), 4)          # 50 images x 4 views
    X = rng.normal(size=(200, 16)).astype(np.float32)
    y = (src % 2).astype(int)
    res = train_head(X, y, source_index=src, cfg=HeadConfig(input_dim=16, epochs=2), verbose=False)
    assert res is not None  # the assertion of interest is inside _grouped_split
    from aigcdet.models.head import _grouped_split

    tr, va = _grouped_split(src, 0.2, 1337)
    assert not (set(src[tr]) & set(src[va])), "source images appear in both splits"


def test_full_chain_and_predict_json_contract(tmp_path, dataset):
    """Train a real checkpoint on the dummy backbone, then run predict.py on it."""
    df = split_by_family(build_manifest_from_labelled_root(dataset), test_frac=0.3)
    bb = load_backbone("dummy")
    train = df[df["split"] == "train"]

    bundle = extract_features(
        bb, train["image_path"].tolist(), train["label"].tolist(),
        n_views=2, policy=TrainDamagePolicy(seed=0), progress=False,
    )
    res = train_head(
        bundle.features, bundle.labels, bundle.source_index,
        cfg=HeadConfig(input_dim=bundle.features.shape[1], epochs=40, patience=15), verbose=False,
    )
    ckpt = tmp_path / "detector.pt"
    save_checkpoint(ckpt, res["model"], "dummy", res["temperature"], res["threshold"],
                    extra={"with_fourier": False})

    # Loading must reproduce the same scores - checkpoint round-trip.
    reloaded = load_checkpoint(ckpt)
    assert np.allclose(
        reloaded["model"].logits(bundle.features[:5]), res["model"].logits(bundle.features[:5]),
        atol=1e-5,
    )

    det = Detector(ckpt, backbone_name="dummy")
    from PIL import Image as PILImage

    scores = det.score_images(
        [PILImage.open(p).convert("RGB") for p in train["image_path"].tolist()[:4]]
    )
    assert scores.shape == (4,)
    assert ((scores >= 0) & (scores <= 1)).all(), "scores must be probabilities"

    # And the required CLI contract.
    out_json = tmp_path / "predictions.json"
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "predict.py"),
         "--image_dir", str(dataset / "real"), "--output", str(out_json),
         "--checkpoint", str(ckpt), "--backbone", "dummy", "--quiet"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    records = json.loads(out_json.read_text())
    assert isinstance(records, list) and len(records) > 0
    for rec in records:
        assert set(rec) >= {"image_path", "pred"}
        assert isinstance(rec["image_path"], str)
        assert isinstance(rec["pred"], (int, float))
        assert 0.0 <= rec["pred"] <= 1.0
    # One record per image, no duplicates, stable order.
    paths = [r_["image_path"] for r_ in records]
    assert len(paths) == len(set(paths))
    assert paths == sorted(paths)


def test_predict_handles_a_corrupt_file_without_crashing(tmp_path, dataset):
    """A single unreadable file must not take down a batch job."""
    df = build_manifest_from_labelled_root(dataset)
    bb = load_backbone("dummy")
    b = extract_features(bb, df["image_path"].tolist()[:8], df["label"].tolist()[:8], progress=False)
    res = train_head(b.features, b.labels, b.source_index,
                     cfg=HeadConfig(input_dim=b.features.shape[1], epochs=5), verbose=False)
    ckpt = tmp_path / "d.pt"
    save_checkpoint(ckpt, res["model"], "dummy", res["temperature"], res["threshold"])

    bad_dir = tmp_path / "mixed"
    bad_dir.mkdir()
    import shutil

    shutil.copy(df["image_path"].iloc[0], bad_dir / "good.jpg")
    (bad_dir / "broken.jpg").write_bytes(b"this is not a JPEG")

    det = Detector(ckpt, backbone_name="dummy")
    preds = det.predict_paths([str(p) for p in sorted(bad_dir.iterdir())], progress=False)
    assert len(preds) == 2
    broken = [p for p in preds if p.image_path.endswith("broken.jpg")][0]
    assert broken.error is not None
    assert broken.pred == pytest.approx(0.5)
