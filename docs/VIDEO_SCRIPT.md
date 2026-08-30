# Demo video — shot list and script

**Target: 2:30–3:00.** Under three minutes is a constraint, not a suggestion —
judges watch a lot of these.

**Method that saves an hour:** screen-record all seven shots silently first,
with no talking and no mistakes to edit around. Then record the narration in one
take over the finished footage. Trying to demo and narrate simultaneously is how
teams end up on take nine at 2am.

Have everything pre-loaded before you hit record: the Gradio demo running, a
terminal in the repo, `results/robustness_auc.png` open, and the two example
images ready.

---

## Shot 1 — the problem (0:00–0:25)

*Screen: a real photo and an AI-generated photo side by side, then the same two
after heavy JPEG compression.*

> "Telling AI-generated images apart from real ones is close to solved on clean
> data. On a platform it isn't — because by the time an image reaches a
> moderation queue it's been compressed, resized, cropped and reposted. Most
> detectors learn generator fingerprints that live in exactly the detail JPEG
> throws away first. They don't degrade gracefully. They fall off a cliff, and
> they stay confident while doing it."

## Shot 2 — what we built (0:25–0:50)

*Screen: the architecture diagram from the README.*

> "We built a detector for images that have already been through the internet. A
> frozen CLIP encoder produces one embedding per image; a small calibrated head
> turns it into a probability. We don't fine-tune — a frozen representation
> can't drift toward the generators in our training set, and caching the
> embeddings meant we could run the experiment that matters many times instead
> of once."

## Shot 3 — it works on a clean image (0:50–1:10)

*Screen: Gradio tab 1. Drop in an AI image → score. Drop in a real photo → score.
Show the occlusion map on one of them.*

> "Here's a generated image: [score]. Here's a real photograph: [score]. The
> heatmap shows which regions pushed the verdict — red pushed toward
> 'generated'."

## Shot 4 — **the point of the whole project** (1:10–1:50)

*Screen: Gradio tab 2. Same image. Drag JPEG quality to 30, blur to 1.0, rescale
to 0.25. Show the damaged image next to the before/after table.*

> "Now the part that matters. Same image — JPEG quality 30, blurred, resized to
> a quarter and back. That's a normal repost. The baseline model we trained only
> on pristine images goes from [X] to [Y] and flips its verdict. Ours moves from
> [X] to [Y] and holds."

**Spend the most time here. If you cut anything, cut somewhere else.**

## Shot 5 — the required script (1:50–2:05)

*Screen: terminal.*

```bash
python predict.py --image_dir demo_images/ --output predictions.json
```

*Then `cat predictions.json | head -20`.*

> "The submission script: a folder of images in, a JSON confidence score per
> image out. One record per image, and an unreadable file gets a neutral score
> and an error field instead of taking down the batch."

## Shot 6 — the evidence (2:05–2:35)

*Screen: `results/robustness_auc.png`, then scroll to the held-out rows.*

> "We measured every corruption at every severity, not an average. Blue is
> trained on clean images only; orange is trained on damaged ones. Across the
> specified transforms, [X] to [Y]. And these rows at the bottom are corruptions
> we never trained on at all — WebP, sharpening, rotation, screen re-capture —
> where we go from [X] to [Y]. That's the number we'd want to be judged on,
> because it's the only one that says anything about damage we didn't
> anticipate."

## Shot 7 — one honest failure (2:35–3:00)

*Screen: `results/false_positives.png`.*

> "Here's where we fail. [Describe the actual pattern you found — for example:
> heavily stylised illustrations get called generated, because our real-image
> training data is almost entirely photographs.] We tune the threshold against a
> five-percent false-positive budget, because telling a real photographer their
> work is fake is a much more expensive mistake than missing one generated
> image. This is a triage tool for human reviewers, not an arbiter."

---

## Checklist before you upload

- [ ] Under 3 minutes
- [ ] No copyrighted music; no third-party logos or trademarks on screen
- [ ] Every number spoken matches `results/summary.csv`
- [ ] Uploaded to YouTube, **visibility set to Public** (not Unlisted)
- [ ] Opened in an incognito window to confirm it actually plays
- [ ] Link pasted into the Devpost description
