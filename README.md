# Robust AIGC Image Detection

Tells you whether an image is AI-generated, and keeps working after the image
has been compressed, blurred or resized.

Frozen CLIP ViT-B/16 plus a small trained head.
Repo: https://github.com/aadiithi/tikitikitaka

---

## What it does

Point it at a folder of images. You get back a JSON file with a score from 0 to
1 for each one, where 1 means almost certainly AI-generated.

```json
[
  {"image_path": "images/a.jpg", "pred": 0.9312},
  {"image_path": "images/b.png", "pred": 0.0417}
]
```

## The result

We trained two identical models. The only difference is that one saw damaged
images during training and one didn't. Both were then scored on the same 400
test images under 9 conditions.

**Across the 8 damaged conditions:**

| | Clean-trained | Damage-trained |
|---|---|---|
| Mean accuracy | 94.1% | **97.4%** |
| Mean AUC | 0.9910 | **0.9955** |
| Total mistakes (out of 3,200) | 189 | **85** |

**On undamaged images**, the damage-trained model is also better: 98.5% vs
97.0%. It didn't cost us anything on clean inputs.

Full per-condition table below.

## Per-condition results

400 test images per row, 200 real and 200 fake. Both models scored on exactly
the same corrupted images.

| Condition | Clean-trained AUC | Clean-trained acc | Robust AUC | Robust acc |
|---|---|---|---|---|
| No damage | 0.9966 | 97.0% | 0.9981 | **98.5%** |
| JPEG quality 90 | 0.9915 | 96.3% | 0.9969 | **98.5%** |
| JPEG quality 70 | 0.9915 | 91.0% | 0.9960 | **97.5%** |
| JPEG quality 50 | 0.9906 | 92.2% | 0.9955 | **96.8%** |
| JPEG quality 30 | 0.9808 | 94.0% | 0.9915 | **96.0%** |
| Blur σ 0.5 | 0.9975 | 97.5% | 0.9976 | **98.3%** |
| Blur σ 1.0 | 0.9970 | 95.8% | 0.9976 | **98.0%** |
| Blur σ 2.0 | 0.9837 | 90.5% | 0.9921 | **96.3%** |
| Rescale 0.5x | 0.9954 | 95.5% | 0.9965 | **97.5%** |

The robust model wins on every single row, including the undamaged one.

## What didn't finish

We had no GPU in this session, so everything ran on CPU. Each condition took
about 6 minutes to score, and we stopped the grid after 9 conditions rather
than wait out the full 24.

So we have **no results** for: rescale 0.25x, the three noise levels, colour
jitter, crop, the five held-out corruption types (WebP, sharpening, rotation,
screen re-capture) or the four two-step corruptions. Those conditions are all
implemented and tested, they just didn't run in time.

Because the script was interrupted before it wrote its output file, the numbers
in the table above come from the console log in the notebook, not from
`results/robustness.csv`.

## Install

```bash
git clone https://github.com/aadiithi/tikitikitaka.git
cd tikitikitaka
pip install -r requirements.txt
pip install -e .
```

Check it works, no downloads and no GPU needed:

```bash
make smoke      # about 1 minute
```

## Score some images

```bash
python predict.py \
  --image_dir path/to/images \
  --output predictions.json \
  --checkpoint checkpoints/sid_robust.pt \
  --backbone clip-vit-b16
```

`--checkpoint` is not optional. `predict.py` looks for a different filename by
default and will exit if you leave it out.

On CPU it does about 1 image per second, so 100 images takes a minute and a half
and a few thousand takes hours. Use a GPU runtime if you have one.

## Run the whole pipeline

Open `notebooks/TRIAL_2_DOWNSIZED.ipynb` in Colab. **Turn on the T4 GPU first**
(Runtime → Change runtime type). Our run didn't have one and it cost us about
three hours.

| Stage | What it does | Our CPU time |
|---|---|---|
| 1. Setup, unzip data | Clone repo, extract images, fix the manifest paths | 5 min |
| 2. Subsample | Pick 1,500 train and 400 test images | instant |
| 3. Clean features | Encode each image once | 33 min |
| 4. Damaged features | Encode 2 damaged copies of each training image | 43 min |
| 5. Train both heads | One on clean features, one on damaged | 4 sec each |
| 6. Robustness grid | Score both models under each condition | 6 min per condition |
| 7. Error analysis | Find and show the worst mistakes | 4 min |

Copy `features/`, `checkpoints/` and `results/` to Drive after every stage.
Colab deletes everything when the runtime restarts, and we've already lost a
finished run that way.

## Things that will trip you up

**The backbone needs to be on the GPU.** After loading it, print `bb.device`.
If it says `cpu`, either the runtime has no GPU or you passed `device="cpu"`.
That one argument is what turned a 20-minute run into a 3-hour one for us.

**Train both heads.** One cell for the clean features, one for the damaged
features. Forgetting the second one means there's no experiment to report.

**Offset `source_index` when you merge the feature chunks.** Each chunk numbers
its images from 0, so concatenating five chunks makes them all claim indices
0 to 299. We didn't do this and it broke the validation split for the robust
model (details in `docs/ERROR_ANALYSIS.md`). Fix:

```python
offset, src = 0, []
for b in bundles:
    src.append(b.source_index + offset)
    offset += int(b.source_index.max()) + 1
```

**`predict.py` on a big folder is slow on CPU.** We pointed it at all 12,000
real images and gave up 15 minutes in at 9% done.

## Demo

```bash
python app/demo.py --checkpoint checkpoints/sid_robust.pt
```

Upload an image, drag the damage slider, watch the score hold.

## Where things are

```
predict.py                        the required deliverable
notebooks/TRIAL_2_DOWNSIZED.ipynb the full run
src/aigcdet/aug/transforms.py     the damage model, read this first
src/aigcdet/models/head.py        the trained part
src/aigcdet/data/normalize.py     the 224x224 canonicalisation
scripts/                          one script per pipeline stage
docs/                             technical design, error analysis, write-up
```

## What we can't claim

**Generalising to new generators.** Our dataset has no generator labels at all,
so we split train and test randomly by image. We can say it works on images it
hasn't seen. We can't say it works on a generator it hasn't seen.

**Generalising to corruption types we didn't train on.** That was the whole
point of the held-out grid, and the grid didn't finish. All 8 damaged
conditions we do have results for are types the model saw during training.

**The absolute accuracy numbers.** The raw dataset has a shortcut in it: you can
tell real from fake 98.5% of the time from the file header alone (image
dimensions, file size), before looking at any pixels. We strip that out by
resizing everything to 224x224 and deleting metadata before the model sees it,
but we never re-ran the check afterwards to prove nothing's left. The gap
between our two models is the trustworthy part, since both see identical data.

**Anything from a small difference.** 400 test images means 1 percentage point
is 4 images. The aggregate (189 mistakes vs 85 across all conditions) is solid.
A single row differing by 2 points is not.

More detail in `docs/TECHNICAL_DESIGN.md` and `docs/ERROR_ANALYSIS.md`.
