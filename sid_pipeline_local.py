"""
sid_pipeline_local.py

One local CPU script that:
 1. Checks your machine can handle streaming + local processing
 2. Streams SID_Set from HuggingFace (never downloads the full 124GB)
 3. Cleans + standardizes each image as it arrives (resize, strip EXIF, dedupe)
 4. Exports to real/ and fake/<generator>/ folders
 5. Builds the manifest CSV (generator-disjoint split, metadata-shortcut probe)
 6. Optionally runs the demo-set leak check

Usage:
    pip install datasets pillow pandas numpy scikit-learn imagehash psutil tqdm
    python sid_pipeline_local.py --n_train 20000 --n_test 4000
"""

import os
import sys
import shutil
import hashlib
import argparse
from pathlib import Path
from collections import Counter
from typing import Optional

import psutil
from PIL import Image, ImageFile
from tqdm import tqdm
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

TARGET_SIZE = 256
JPEG_QUALITY = 95


# ============================================================
# 1. CAPABILITY CHECK
# ============================================================
def check_machine():
    print("=" * 60)
    print("MACHINE CAPABILITY CHECK")
    print("=" * 60)

    ok = True

    mem = psutil.virtual_memory()
    avail_gb = mem.available / (1024 ** 3)
    print(f"RAM available: {avail_gb:.1f} GB (total {mem.total / (1024**3):.1f} GB, "
          f"{mem.percent}% used)")
    if avail_gb < 2:
        print("  ⚠ Under 2GB free — close other apps before running. Streaming is "
              "designed to use very little RAM, but your system is already tight.")
        ok = False
    else:
        print("  ✓ Enough headroom for streaming (this script holds ~1 image in "
              "memory at a time, not the dataset)")

    disk = shutil.disk_usage(".")
    free_gb = disk.free / (1024 ** 3)
    print(f"Disk free (current drive): {free_gb:.1f} GB")
    needed_gb_estimate = 2
    if free_gb < needed_gb_estimate:
        print(f"  ⚠ Less than {needed_gb_estimate}GB free — that's tight even for "
              f"resized output.")
        ok = False
    else:
        print(f"  ✓ Plenty of disk for standardized output (resized JPEGs are "
              f"small — a few hundred bytes to a few KB each)")

    try:
        import socket
        socket.setdefaulttimeout(5)
        socket.gethostbyname("huggingface.co")
        print("  ✓ Can resolve huggingface.co")
    except Exception:
        print("  ✗ Cannot resolve huggingface.co — check your internet connection")
        ok = False

    print()
    if ok:
        print("Verdict: your machine can run this. Streaming avoids downloading")
        print("the full 124GB — you'll only pull the images you actually keep.\n")
    else:
        print("Verdict: proceed with caution, see warnings above. You can still")
        print("run this, just watch RAM/disk while it runs.\n")
    return ok


# ============================================================
# Helpers: cleaning + standardizing a single image
# ============================================================
def standardize_and_save(img: Image.Image, dst_path: Path) -> Optional[str]:
    """Resize, strip EXIF, convert to RGB, save as JPEG. Returns sha256 or None on failure."""
    try:
        img = img.convert("RGB")
        w, h = img.size
        if min(w, h) > TARGET_SIZE:
            scale = TARGET_SIZE / min(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, "JPEG", quality=JPEG_QUALITY, exif=b"")
        return hashlib.sha256(dst_path.read_bytes()).hexdigest()
    except Exception as e:
        print(f"  [SKIP] {dst_path.name}: {e}")
        return None


# ============================================================
# 2+3. STREAM + CLEAN + STANDARDIZE + EXPORT
# ============================================================
def stream_and_export(export_root: Path, n_train: int, n_test: int,
                       label_col: str, generator_col: str):
    from datasets import load_dataset
        
    if export_root.exists():
        print(f"Clearing previous export at {export_root}...")
        shutil.rmtree(export_root)
    n_target = n_train + n_test
    print(f"Streaming SID_Set from HuggingFace (target: {n_target} images total)...")
    print("This pulls examples one at a time — the full dataset is never downloaded.\n")

    ds = load_dataset("saberzl/SID_Set", split="train", streaming=True)

    # --- inspect first row before committing to field names ---
    first = next(iter(ds))
    print("First row keys:", list(first.keys()))
    if label_col not in first:
        print(f"⚠ '{label_col}' not found in row keys — inspect the printout above "
              f"and re-run with --label_col set correctly.")
        sys.exit(1)
    print(f"Sample label value: {first[label_col]!r}\n")

    (export_root / "real").mkdir(parents=True, exist_ok=True)

    counts = Counter()
    seen_hashes = set()
    idx = 0
    label_counts_seen = Counter()

    ds = load_dataset("saberzl/SID_Set", split="train", streaming=True)  # restart iterator

    pbar = tqdm(total=n_target, desc="Streaming + cleaning")
    rows_seen = 0
    target_n = n_target // 2  # roughly balanced real/fake

    for row in ds:
        rows_seen += 1
        if rows_seen % 200 == 0:
            print(f"  ...scanned {rows_seen} rows so far, kept real={counts['real']}, "
                  f"fake={counts['fake']} (target {target_n} each)")

        raw_label = row[label_col]
        label_counts_seen[raw_label] += 1

        # SID_Set uses 0 = real, 1 = fake (confirmed from your test run's printed
        # label values). Adjust here if a future dataset version differs.
        if raw_label == 0:
            label = "real"
        elif raw_label == 1:
            label = "fake"
        else:
            continue  # e.g. "tampered" or an unexpected value — skip

        if counts[label] >= target_n:
            if all(counts[l] >= target_n for l in ("real", "fake")):
                break
            continue

        img = row["image"]
        if not isinstance(img, Image.Image):
            continue

        if label == "real":
            out_dir = export_root / "real"
        else:
            generator = row.get(generator_col, "unknown_generator")
            out_dir = export_root / "fake" / str(generator)

        dst = out_dir / f"{idx:07d}.jpg"
        sha = standardize_and_save(img, dst)
        if sha is None:
            continue
        if sha in seen_hashes:
            dst.unlink(missing_ok=True)  # exact duplicate — drop it
            continue
        seen_hashes.add(sha)

        counts[label] += 1
        idx += 1
        pbar.update(1)

    pbar.close()
    print(f"\nExported: real={counts['real']}, fake={counts['fake']}")
    print(f"Raw label values seen (for sanity check): {dict(label_counts_seen)}")
    return counts


# ============================================================
# 4. BUILD MANIFEST (calls your repo's own script)
# ============================================================
def build_manifest(repo_root: Path, export_root: Path, out_csv: Path, test_frac: float):
    import subprocess
    cmd = [
        sys.executable, str(repo_root / "scripts" / "build_manifest.py"),
        "--root", str(export_root),
        "--out", str(out_csv),
        "--test_frac", str(test_frac),
    ]
    print("\nRunning:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=repo_root)
    if result.returncode != 0:
        print("⚠ build_manifest.py failed — check the repo is cloned locally at "
              f"{repo_root} and scripts/build_manifest.py exists.")
        sys.exit(1)


def check_demo_leakage(repo_root: Path, manifest_csv: Path, demo_dir: Path, out_csv: Path):
    import subprocess
    if not demo_dir.is_dir() or not any(demo_dir.iterdir()):
        print(f"\n⚠ DEMO_DIR ({demo_dir}) not found or empty — leak check SKIPPED.")
        print("  This must run before the real training run. See explanation below.")
        return manifest_csv

    cmd = [
        sys.executable, str(repo_root / "scripts" / "check_demo_leakage.py"),
        "--manifest", str(manifest_csv),
        "--demo_dir", str(demo_dir),
        "--out", str(out_csv),
    ]
    print("\nRunning:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=repo_root)
    if result.returncode != 0:
        print("⚠ check_demo_leakage.py failed — see error above.")
        return manifest_csv
    return out_csv


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".", help="Path to your local tikitikitaka clone")
    ap.add_argument("--export_root", default="sid_export")
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_test", type=int, default=4000)
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--generator_col", default="generator")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--demo_dir", default="demo_set",
                     help="Local folder with COCO val2017 + WildFake DALL-E-Advanced samples")
    ap.add_argument("--skip_check", action="store_true", help="Skip the machine capability check")
    args = ap.parse_args()

    if not args.skip_check:
        check_machine()
        input("Press Enter to continue, or Ctrl+C to stop and free up resources first...")

    repo_root = Path(args.repo_root).resolve()
    export_root = Path(args.export_root).resolve()

    if not (repo_root / "scripts" / "build_manifest.py").exists():
        print(f"⚠ Can't find scripts/build_manifest.py under {repo_root}")
        print("  Make sure --repo_root points at your local clone of tikitikitaka.")
        sys.exit(1)

    stream_and_export(export_root, args.n_train, args.n_test,
                       args.label_col, args.generator_col)

    manifest_csv = repo_root / "data" / "manifest_sidset.csv"
    (repo_root / "data").mkdir(exist_ok=True)
    build_manifest(repo_root, export_root, manifest_csv, args.test_frac)

    clean_csv = repo_root / "data" / "manifest_sidset_clean.csv"
    final_manifest = check_demo_leakage(repo_root, manifest_csv, Path(args.demo_dir), clean_csv)

    print("\n" + "=" * 60)
    print(f"DONE. Final manifest: {final_manifest}")
    print("=" * 60)


if __name__ == "__main__":
    main()