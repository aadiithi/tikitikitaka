# Submission checklist

Work down this list. Anything unticked at T-2 hours gets cut in favour of the
items marked **required** — those are worth more than any accuracy gain.

## Required deliverables

- [ ] **Public GitHub repo** — settings → visibility → Public. Verify in an
      incognito window; a private repo is an automatic zero.
- [ ] **`predict.py`** works from a fresh clone: `python predict.py --image_dir
      <dir> --output predictions.json` produces `[{"image_path", "pred"}, ...]`
- [ ] **README** with: overview · setup · reproduce steps · limitations and what
      you'd improve · team contributions
- [ ] **Robustness table** — clean vs each transform (`docs/ROBUSTNESS.md` +
      `results/robustness_auc.png`)
- [ ] **Error analysis note** — representative FPs and FNs, trade-offs
      (`docs/ERROR_ANALYSIS.md`)
- [ ] **Demo video** on YouTube, **Public** (not Unlisted), under 3 minutes,
      linked in the Devpost description
- [ ] **Devpost write-up** — problem framing · dev tools · models/APIs ·
      libraries · datasets (`docs/DEVPOST.md` is ready to paste)

## Before you call it done

- [ ] **Clean-clone test.** Someone who did *not* write the code clones the repo
      into a fresh directory, follows the README setup, and runs `make test` and
      `predict.py`. This catches the "works on my machine because of a file I
      never committed" failure, which is the single most common way a good
      submission loses points.
- [ ] Checkpoint is reachable — attached to a GitHub Release, or the README says
      exactly how to train one.
- [ ] `results/robustness.csv`, `results/summary.csv` and the figures are
      committed. A reviewer should see your results without running anything.
- [ ] Every number in the README and the video matches `results/summary.csv`.
      Regenerate with `python scripts/make_report.py` rather than editing text.
- [ ] The demo set (COCO val2017 + DALL·E Advanced) was scored **once**, at the
      end, and never trained on. Say so explicitly.
- [ ] Team names filled into the README contributions table.
- [ ] `git log` has commits from more than one person if you are a team.
- [ ] Repo has no datasets, no `.pt` files over ~100MB, no API keys.

## Judge questions you should be able to answer cold

Each of these has a real answer in this codebase. Know them well enough to say
them without notes — the pitch is 10% but the Q&A is where a team either shows
it understands its own project or doesn't.

1. **Why CLIP, and why frozen?** Iteration speed (features cached once, every
   experiment is seconds), generalisation (a frozen representation can't drift
   toward the generators in your training set), deployability (embed once per
   upload, retrain the head daily).
2. **How do you know you're not just detecting resolution or file format?** The
   metadata-only probe in `scripts/build_manifest.py`, plus canonicalisation.
   Quote the before/after AUC.
3. **How do you know the robustness isn't just memorising the test transforms?**
   Training severities are continuous ranges that bracket the eval grid, and
   four corruption families are held out entirely. A CI test fails the build if
   one leaks.
4. **What does corruption training cost you?** Clean AUC drops by [X]. Say the
   number.
5. **What's your worst failure case?** Point at `results/false_positives.png`
   and describe the actual pattern you found.
6. **Why isn't your threshold 0.5?** Because a false positive is an accusation
   against a real creator. Ours is set at a 5% false-positive budget; quote the
   TPR you get there.
7. **What would you do with more time?** Multi-backbone ensembling — nearly free
   with frozen features — and an active-learning loop on the confident errors.
8. **Who would use this and how?** A first-pass triage signal in a moderation
   queue that routes uncertain scores to human review, not an automated
   takedown mechanism.

## Timeline discipline

- Freeze the model **early**. Whatever exists at the freeze deadline is final.
- The README, the video, the robustness table and the error note are worth more
  than another point of AUC. If something has to give, it is never one of those.
- Submit with hours to spare, not minutes. Then use the remaining time to
  rehearse the eight answers above out loud.
