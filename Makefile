.PHONY: help install test smoke clean lint

help:
	@echo "make install  - install the package and dependencies"
	@echo "make test     - run the unit tests (offline, seconds)"
	@echo "make smoke    - full pipeline on procedural data, no downloads, no GPU"
	@echo "make clean    - remove generated artefacts"

install:
	pip install -e ".[clip,demo,dev]"

test:
	pytest -q

# Proves the whole chain works on a machine with no network and no GPU.
# This is what CI runs, and what a reviewer should run first.
smoke:
	python scripts/make_synthetic_dataset.py --out data/synthetic --n 40 --size 128
	python scripts/build_manifest.py --root data/synthetic --out data/manifest_synth.csv --test_frac 0.3
	python scripts/extract_features.py --manifest data/manifest_synth.csv --split train \
		--out features/synth_clean.npz --backbone dummy
	python scripts/extract_features.py --manifest data/manifest_synth.csv --split train \
		--out features/synth_aug.npz --backbone dummy --augment --n_views 3
	python scripts/train_head.py --features features/synth_clean.npz \
		--out checkpoints/synth_clean.pt --epochs 40
	python scripts/train_head.py --features features/synth_aug.npz \
		--out checkpoints/synth_robust.pt --epochs 40
	python scripts/evaluate_robustness.py --manifest data/manifest_synth.csv --split test \
		--checkpoints clean=checkpoints/synth_clean.pt robust=checkpoints/synth_robust.pt \
		--out results_smoke/ --backbone dummy
	python predict.py --image_dir data/synthetic/real --output results_smoke/predictions.json \
		--checkpoint checkpoints/synth_robust.pt --backbone dummy --quiet
	@echo "\nSmoke test passed. See results_smoke/."

clean:
	rm -rf features checkpoints/*.pt results_smoke data/synthetic data/manifest_synth*
	find . -name __pycache__ -type d -exec rm -rf {} +
