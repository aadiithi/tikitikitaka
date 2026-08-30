# Devpost write-up (paste into the submission form)

> Fill the bracketed numbers from `results/summary.csv` and
> `results/headline.txt` before submitting. Everything else is ready to paste.

---

## Inspiration

Detecting AI-generated images on a clean benchmark is close to solved. Detecting
them on a platform is not — and almost every published accuracy figure is
measured in the first setting and quoted as if it applied to the second.

The reason is mechanical. An image that reaches a moderation queue has been
re-encoded as JPEG two or three times, resized to a thumbnail and back, cropped
to a feed's aspect ratio, screenshotted, colour-filtered, and reposted. Most
detectors reach their headline number by learning generator-specific
high-frequency fingerprints, and those fingerprints live in exactly the
frequency bands JPEG discards first. The detector does not degrade gracefully —
it falls off a cliff, and it stays confident while doing it.

We decided to treat that as the actual problem rather than a footnote.

## What it does

Given an image, it returns a calibrated probability that the image is
AI-generated, and it keeps returning approximately the same probability after
the image has been compressed, blurred, resized, cropped, colour-shifted or
screenshotted.

- `predict.py --image_dir <dir> --output predictions.json` scores a folder and
  writes the required `{image_path, pred}` JSON.
- A Gradio demo lets you damage an image with sliders and watch the score hold.
- A full robustness table reports performance per corruption and per severity,
  including four corruption types the model was **never trained on**.

## How we built it

**Architecture.** A frozen CLIP ViT-L/14 vision encoder (304M parameters, under
the 2B cap) produces one embedding per image; a ~200k-parameter MLP head maps
that embedding to a logit, which temperature scaling turns into a calibrated
probability.

We chose *not* to fine-tune, for three reasons:

1. **Iteration.** Features are extracted once and cached, so every subsequent
   experiment is seconds on a CPU rather than an hour on a free GPU that can
   disconnect. This is what let us run the clean-vs-augmented comparison
   properly rather than once.
2. **Generalisation.** Fine-tuning a full ViT on images from a handful of
   generators is an efficient way to memorise those generators. A frozen
   representation cannot drift toward generator fingerprints because it is never
   allowed to move.
3. **Deployability.** The backbone is a fixed, shareable artefact. A platform
   embeds each upload once and can retrain or re-threshold the head daily as new
   generators appear, without recomputing anything.

**Robustness training.** We train on randomly damaged images, with severity
ranges that are continuous and *wider* than the evaluation grid — JPEG quality
sampled from [25, 95] rather than at {30, 50, 70, 90} — so the head cannot
overfit to the specific corruptions it is scored on. 35% of training images get
a chain of two corruptions, because nobody in the wild sees a single-corruption
image. 25% stay pristine, so clean accuracy is protected.

**Three things we did that we have not often seen in this space:**

*Held-out corruption families.* WebP, sharpening, small rotation and simulated
screen re-capture are never used in training and are reported separately. That
number — [X] → [Y] AUC — is the honest generalisation claim, and a CI test
fails the build if one of those families ever leaks into the training policy.

*A dataset shortcut audit.* Before training anything, we train a classifier on
**file metadata alone** — dimensions, file size, format, JPEG quantisation
tables, EXIF presence — and measure its AUC. Public AIGC datasets frequently
pair 640×480 JPEGs-with-EXIF against 1024×1024 PNGs, and on those a five-line
script beats a neural network. Our probe scored [X] on the raw data. We then
canonicalise every image — identical resolution, identical format, one shared
round of JPEG — before the backbone sees it, and re-measure. Reporting both is
what makes the rest of our numbers mean anything.

*Generator-family-disjoint splits.* We hold out entire generator families rather
than splitting randomly, because a random split measures how well we recognise
*these* generators, and by the time anything ships there is a new one.

**Calibration and the operating point.** The threshold is not 0.5. It is chosen
so that no more than 5% of authentic images are flagged, because telling a real
photographer their work is synthetic is a visible accusation, whereas missing
one generated image on a platform with layered defences is recoverable. We
report TPR at that budget alongside AUC.

## Challenges we ran into

- **Our first strong result was a dataset artefact.** The metadata probe was the
  cell that caught it; canonicalisation is what fixed it. This is why the audit
  is now step zero of the pipeline rather than an afterthought.
- **Corruption training costs clean accuracy.** It is a genuine trade-off, it
  appears in our numbers, and we report its size rather than only the
  improvement.
- **Grouped validation.** With four damaged copies of each photo in the training
  set, a naive row-level validation split leaks the same photograph across both
  sides and inflates every number. We split on the source image instead.

## Accomplishments we're proud of

- The full pipeline runs offline, with no GPU and no dataset, in about a minute
  (`make smoke`) — so a reviewer can verify the code works before spending an
  hour on downloads.
- Every figure and table in the README is generated from `results/robustness.csv`
  by `scripts/make_report.py`. No number is typed by hand, so the write-up
  cannot drift from the run.
- 61 offline tests, including one that guards the generalisation claim itself.

## What we learned

Robustness is a measurement problem before it is a modelling problem. We got
more out of building the per-corruption evaluation harness on day one — and out
of the metadata probe that told us our data was lying to us — than we got from
any modelling decision that followed.

## What's next

Multi-backbone ensembling (nearly free with frozen features), an active-learning
loop on the most confidently wrong images, per-generator reporting so an operator
knows which generators they are blind to, corruption sampled from real platform
round-trips instead of our simulation, and C2PA provenance as a separate channel
so detection is the fallback rather than the primary mechanism.

---

## Built with

**Development tools:** Google Colab (T4 GPU), VS Code, Jupyter, Git/GitHub, Make

**Models:** OpenAI CLIP ViT-L/14 (frozen vision encoder, via `open_clip`);
custom 2-layer MLP classification head (~200k trainable parameters)

**Libraries and frameworks:** PyTorch · open_clip_torch · Hugging Face
`transformers` and `datasets` · scikit-learn · NumPy · pandas · Pillow ·
matplotlib · tqdm · Gradio · pytest

**Datasets:** SID_Set (HuggingFace) · CIFAKE (Kaggle) · WildFake (ModelScope) ·
COCO val2017 + DALL·E Advanced (organisers' demo set, never trained on)

**Links:** GitHub repository · YouTube demo video
