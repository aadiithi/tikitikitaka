# Error Analysis

_This file is generated. Run:_

```bash
python scripts/error_analysis.py --scores results/scores.csv \
    --manifest data/manifest.csv --model robust --out docs/ERROR_ANALYSIS.md
```

_after `scripts/evaluate_robustness.py`, then review the two contact sheets it
produces (`results/false_positives.png`, `results/false_negatives.png`) as a
team and write the closing "What we would change" section by hand. That
paragraph is the part judges actually read._
