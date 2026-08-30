# Robust AIGC Image Detection

**A detector built for images that have already been through the internet.**

Frozen CLIP vision encoder → small calibrated head, trained on deliberately
damaged images and evaluated per-corruption rather than on average.

<!-- TODO after your first full run: replace with results/robustness_auc.png -->
<!-- ![Robustness](results/robustness_auc.png) -->

---

## 1. The problem, as we understand it

Detecting AI-generated images on a clean benchmark is close to solved. Detecting
them on a *platform* is not, and the gap between those two statements is the
entire project.

An image that reaches a moderation queue has never been the file the generator
produced. It has been re-encoded as JPEG two or three times, resized to a
thumbnail and back, cropped to a feed's aspect ratio, screenshotted, colour-
filtered, and reposted. Most published detectors reach very high accuracy by
learning generator-specific high-frequency fingerprints — and those fingerprints
live in exactly the frequency bands JPEG discards first. The detector does not
degrade gracefully; it falls off a cliff, and it does so *while remaining
confident*, which is worse than failing loudly.

So we did not optimise for peak accuracy. We optimised for the number that
survives, and we built the measurement apparatus first so we could watch it.

### The three things we actually claim

1. **Training on damage transfers to damage we never trained on.** We train on
   continuous, randomised corruption ranges that *bracket* the evaluation grid
   rather than reproducing it, and we hold out four corruption families
   entirely (WebP, sharpening, small rotation, screen re-capture). The held-out
   number is the one we lead with. A CI test (`test_transforms.py`) fails the
   build if a held-out family ever leaks into the training policy.

2. **Most reported AIGC accuracy is partly a dataset artefact, and we measure
   how much.** `scripts/build_manifest.py` trains a classifier on **file
   metadata alone** — dimensions, file size, format, JPEG quantisation tables,
   EXIF presence — and reports its AUC. When real photos are 640×480 JPEGs with
   EXIF and generated images are 1024×1024 PNGs, a five-line script beats a
   neural network and no result from that split means anything. We canonicalise
   every image to identical size, format and compression history before the
   backbone sees it, and we report the probe's verdict before and after.

3. **A score has to mean something to be usable.** The head is
   temperature-calibrated on held-out data, and the operating threshold is
   chosen against a **false-positive budget**, not set to 0.5 — because telling
   a real photographer their work is synthetic is a materially different, and
   more expensive, error than missing one generated image.

## 2. Architecture

```
image ──▶ canonicalise ──▶ [ frozen CLIP ViT-L/14 ] ──▶ 768-d ──▶ [ MLP head ] ──▶ logit
          224², JPEG q92     304M params, never             embedding   ~200k params      │
          EXIF stripped      updated, shared, cacheable                   trainable        ▼
                                                                              temperature scaling
                                                                                     │
                                                                                     ▼
                                                                    P(AI-generated) ∈ [0,1]
                                                                    + threshold @ 5% FPR
```

**Why frozen.** Three reasons, in the order they mattered to us:

- *Iteration.* Features are computed once and cached. Every experiment after
  that — clean vs augmented, head size, threshold, ablations — is seconds on a
  CPU rather than an hour on a free GPU we can be disconnected from at any
  moment. This is what let us run the comparison properly instead of once.
- *Generalisation.* Fine-tuning a full ViT on a few thousand images from a
  handful of generators is an efficient way to memorise those generators. A
  frozen general-purpose representation cannot drift toward generator
  fingerprints, because it is never allowed to move.
- *Deployability.* The backbone is a fixed, shareable artefact. A platform
  computes an embedding once per upload and can retrain or re-threshold the head
  daily as new generators appear, without recomputing anything. The expensive
  part is amortised across every downstream use of the embedding.

The vision tower is 304M parameters — comfortably under the 2B cap — and the
trainable part is roughly 200k.

## 3. Results

> Regenerate everything in this section with
> `python scripts/make_report.py`. **Never type a number here by hand** — the
> tables and figures are generated from `results/robustness.csv` so that the
> write-up cannot disagree with the run.

### Headline

<!-- Paste the contents of results/headline.txt here after your run. -->
_Run `scripts/evaluate_robustness.py` and paste `results/headline.txt` here._

### Clean vs transformed (deliverable #4)

<!-- Paste results/robustness_auc.md here. -->
| condition | clean-trained | augmentation-trained |
|---|---|---|
| _generated by `scripts/make_report.py`_ | | |

![Per-corruption AUC](results/robustness_auc.png)
![Degradation curves](results/degradation_auc.png)

### Dataset shortcut audit

| stage | metadata-only AUC | verdict |
|---|---|---|
| raw dataset | _from `data/manifest.report.json`_ | |
| after canonicalisation | _re-run the probe on canonicalised copies_ | |

### Error analysis (deliverable #5)

See [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md), generated from
`results/scores.csv`.

## 4. Setup

Python 3.9+. A GPU makes feature extraction ~20× faster but nothing here
requires one.

```bash
git clone https://github.com/YOUR_USERNAME/aigc-robust-detector.git
cd aigc-robust-detector
pip install -e ".[clip,demo,dev]"
```

### Verify the install without downloading anything

```bash
make test     # 60+ unit tests, offline, ~10 seconds
make smoke    # the entire pipeline on procedurally generated images, ~1 minute
```

`make smoke` builds a small procedural dataset, extracts features with a stub
backbone, trains both heads, runs the full robustness grid and executes
`predict.py`. It requires no network, no GPU and no dataset. If it passes, the
code is sound and the only remaining variable is the data.

### Pretrained weights

The trained checkpoint is attached to the
[latest GitHub Release](../../releases/latest) as `detector_robust.pt`. Place it
at `checkpoints/detector_robust.pt`.

## 5. Reproducing our results

The full run is one Colab notebook:
[`notebooks/01_end_to_end_colab.ipynb`](notebooks/01_end_to_end_colab.ipynb) —
open it in Colab, select a T4 GPU, set `QUICK = True` for a 15-minute smoke run,
then `QUICK = False` for the real thing.

Or, from a shell:

```bash
# 1. manifest, with a generator-family-disjoint split and the shortcut audit
python scripts/build_manifest.py --root data/SID_Set --out data/manifest.csv \
    --limit_per_class 10000 --test_frac 0.25

# 2. features - the only expensive step. Two banks, one variable changed.
python scripts/extract_features.py --manifest data/manifest.csv --split train \
    --out features/train_clean.npz
python scripts/extract_features.py --manifest data/manifest.csv --split train \
    --out features/train_aug.npz --augment --n_views 4

# 3. two heads, seconds each
python scripts/train_head.py --features features/train_clean.npz \
    --out checkpoints/detector_clean.pt
python scripts/train_head.py --features features/train_aug.npz \
    --out checkpoints/detector_robust.pt

# 4. the robustness grid, both models on identical corrupted images
python scripts/evaluate_robustness.py --manifest data/manifest.csv --split test \
    --checkpoints clean=checkpoints/detector_clean.pt robust=checkpoints/detector_robust.pt \
    --out results/

# 5. error analysis
python scripts/error_analysis.py --scores results/scores.csv \
    --manifest data/manifest.csv --model robust --out docs/ERROR_ANALYSIS.md
```

Every step is seeded (`--seed`, default 1337) and every checkpoint records the
feature file, damage policy and metrics it came from.

## 6. The required inference script

```bash
python predict.py --image_dir path/to/images --output predictions.json
```

```json
[
  {"image_path": "path/to/images/a.jpg", "pred": 0.9312},
  {"image_path": "path/to/images/b.png", "pred": 0.0417}
]
```

`pred` is P(AI-generated), calibrated, in [0, 1]. The script guarantees exactly
one record per image found, emits `pred: 0.5` plus an `error` field for
unreadable files rather than crashing the batch, and writes records in sorted
path order so two runs diff cleanly.

## 7. Live demo

```bash
python app/demo.py --checkpoint checkpoints/detector_robust.pt
```

Three tabs: score an image with an occlusion map; **apply damage with sliders
and watch the score hold**; and score a whole folder in the `predict.py` output
format. The middle tab is the demo — the other two support it.

## 8. Repository layout

```
predict.py                    required deliverable: dir in, JSON out
app/demo.py                   Gradio demo
notebooks/                    one-click Colab pipeline
src/aigcdet/
  aug/transforms.py           the damage model: eval grid, held-out grid, training policy
  data/normalize.py           canonicalisation - kills the resolution/format shortcut
  data/shortcuts.py           the metadata-only leak probe
  data/manifest.py            manifests + generator-family-disjoint splits
  features/backbone.py        frozen CLIP loader (+ offline stub for CI)
  features/extract.py         cached feature banks, clean and augmented
  features/fourier.py         spectral features - used as an ablation, not shipped
  models/head.py              the small trained head, grouped validation split
  models/calibration.py       temperature scaling + FPR-budget threshold
  models/detector.py          the single inference path everything shares
  eval/metrics.py             AUC, TPR@5%FPR, ECE, separation
  eval/robustness.py          the grid harness
  eval/error_analysis.py      most-confident errors + contact sheets
  eval/report.py              every figure and table, generated from the CSV
  explain/saliency.py         occlusion sensitivity
scripts/                      one CLI per pipeline stage
tests/                        61 tests, all offline
```

## 9. Limitations

Stated plainly, because a prototype that oversells itself is worse than one that
doesn't.

- **Generator coverage.** We train on the generator families in our datasets and
  hold some out to measure the gap, but a genuinely novel architecture released
  after our training data was collected is out of distribution and we should be
  expected to do worse on it. Our held-out-family number is an estimate of that
  penalty, not a guarantee.
- **The frozen backbone is a ceiling as well as a floor.** If CLIP's
  representation does not encode a particular synthesis artefact, no head on top
  of it can recover that artefact. Fine-tuning would raise the ceiling at the
  cost of the generalisation and iteration-speed properties we chose it for.
- **Corruption training costs clean accuracy.** It is a real trade-off, it shows
  up in our own numbers, and we report the size of it rather than only the
  improvement.
- **Adversarial robustness is out of scope.** We defend against *incidental*
  degradation — compression, resizing, reposting. An adversary optimising
  perturbations specifically against this detector would defeat it, and nothing
  in this repository claims otherwise.
- **Partially-edited images are not modelled.** Our labels are binary
  (authentic / fully synthetic). An inpainted region in an otherwise real
  photograph is a localisation problem and we deliberately excluded that class
  rather than pretending a binary label covers it.
- **Dataset provenance.** Public AIGC datasets contain mislabelled and
  lightly-edited "real" images. We use a small amount of label smoothing to
  avoid over-claiming on hard 0/1 targets, but we have not audited the labels
  ourselves.
- **This is a triage aid, not an arbiter.** A score near the threshold means
  *uncertain*, and the correct action is human review. It should never be the
  sole basis for removing content or accusing a creator.

### What we would do with more time

1. **Train the head on features from several backbones** (CLIP, DINOv2, a
   convolutional model) and average. Backbone diversity is the cheapest known
   defence against a single representation's blind spots, and the frozen-feature
   design makes it nearly free — extraction is the only added cost.
2. **Active-learning loop on the errors.** The most confidently wrong images are
   the highest-value labelling targets; feeding a few hundred of them back would
   likely beat any architecture change we could make in the same time.
3. **Per-generator reporting.** Our family-disjoint split gives one aggregate
   generalisation number. A per-family breakdown would tell an operator which
   generators they are actually blind to, which is what they need to know.
4. **Real platform corruption instead of our simulation.** Round-tripping images
   through actual upload/download on a real platform would replace our
   approximation of the damage pipeline with the genuine one.
5. **Watermark and provenance signals (C2PA) as a separate channel.** Detection
   should be the fallback for content with no provenance, not the primary
   mechanism.

## 10. Team

| Member | Contribution |
|---|---|
| _Name_ | DATA — dataset acquisition, manifest construction, family-disjoint split, shortcut audit |
| _Name_ | MODEL — backbone integration, feature pipeline, head training and calibration |
| _Name_ | EVAL — robustness harness, metrics, error analysis, figures |
| _Name_ | PRODUCT — `predict.py`, Gradio demo, README, video |

_Replace with real names before submitting._

## 11. Credits

- **Datasets** — [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set),
  [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images),
  [WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake/summary),
  [COCO val2017](https://cocodataset.org/). Each remains under its own licence;
  none is redistributed here.
- **Models** — OpenAI CLIP via
  [open_clip](https://github.com/mlfoundations/open_clip).
- **Libraries** — PyTorch, scikit-learn, Pillow, pandas, matplotlib, Gradio.

MIT licensed — see [LICENSE](LICENSE).
