"""Error analysis: deliverable #5.

Two things a judge wants to see and most submissions do not provide:

1. The *most confidently wrong* predictions, not a random sample of mistakes. A
   model that is wrong at 0.51 is uncertain; a model that is wrong at 0.99 has
   learned something false, and that is the interesting failure.

2. Which slices those failures cluster in - generator family, condition,
   image characteristics - so the note says "we fail on heavily stylised
   illustrations, because our real-image training data contains almost none"
   rather than "some images were misclassified".

`write_error_contact_sheet` renders the worst cases into a single labelled PNG
that drops straight into the README and the demo video.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..utils.io import ensure_dir, load_image
from ..utils.logging import get_logger

log = get_logger("eval.error_analysis")


def collect_errors(
    scores: Sequence[float],
    labels: Sequence[int],
    paths: Sequence[str],
    threshold: float = 0.5,
    top_k: int = 30,
    families: Sequence[str] | None = None,
    conditions: Sequence[str] | None = None,
) -> dict:
    """Rank errors by confidence and slice them.

    Returns `false_positives` / `false_negatives` DataFrames sorted worst-first,
    plus per-slice error rates.
    """
    df = pd.DataFrame(
        {
            "image_path": list(paths),
            "score": np.asarray(scores, dtype=float),
            "label": np.asarray(labels, dtype=int),
        }
    )
    if families is not None:
        df["family"] = list(families)
    if conditions is not None:
        df["condition"] = list(conditions)

    df["pred"] = (df["score"] >= threshold).astype(int)
    df["correct"] = df["pred"] == df["label"]
    # How wrong, in probability units. This is the ranking key.
    df["confidence_error"] = np.where(df["label"] == 1, 1.0 - df["score"], df["score"])

    fp = df[(df["label"] == 0) & (df["pred"] == 1)].sort_values("score", ascending=False)
    fn = df[(df["label"] == 1) & (df["pred"] == 0)].sort_values("score", ascending=True)

    slices = {}
    for col in ("family", "condition"):
        if col in df.columns:
            g = df.groupby(col).agg(
                n=("correct", "size"),
                error_rate=("correct", lambda s: 1.0 - float(s.mean())),
                mean_score=("score", "mean"),
            )
            slices[col] = g.sort_values("error_rate", ascending=False).round(4)

    log.info(
        "errors: %d false positives, %d false negatives out of %d (threshold %.3f)",
        len(fp), len(fn), len(df), threshold,
    )

    return {
        "all": df,
        "false_positives": fp.head(top_k).reset_index(drop=True),
        "false_negatives": fn.head(top_k).reset_index(drop=True),
        "n_false_positives": int(len(fp)),
        "n_false_negatives": int(len(fn)),
        "slices": slices,
        "threshold": float(threshold),
    }


def write_error_contact_sheet(
    rows: pd.DataFrame,
    out_path: str | Path,
    title: str = "Most confident errors",
    n_cols: int = 6,
    thumb: int = 180,
) -> Path | None:
    """Render up to `len(rows)` images into one labelled grid PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if rows is None or len(rows) == 0:
        log.info("no rows for contact sheet '%s' - skipping", title)
        return None

    n = len(rows)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.3, n_rows * 2.7))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for ax, (_, r) in zip(axes, rows.iterrows()):
        try:
            im = load_image(r["image_path"]).resize((thumb, thumb))
            ax.imshow(im)
        except Exception:
            ax.text(0.5, 0.5, "unreadable", ha="center", va="center", fontsize=8)
        caption = f"score {r['score']:.3f}\ntrue={'AI' if r['label'] == 1 else 'real'}"
        for extra in ("family", "condition"):
            if extra in r and pd.notna(r[extra]):
                caption += f"\n{r[extra]}"
        ax.set_title(caption, fontsize=7)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = Path(out_path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("contact sheet -> %s", p)
    return p


def summarize_for_readme(errors: dict, max_rows: int = 8) -> str:
    """Markdown block for ERROR_ANALYSIS.md, generated from the actual numbers
    so the document can never disagree with the results CSV."""
    lines = [
        f"- Operating threshold: **{errors['threshold']:.3f}**",
        f"- False positives (real images called AI): **{errors['n_false_positives']}**",
        f"- False negatives (AI images called real): **{errors['n_false_negatives']}**",
        "",
    ]
    for name, key in (("Most confident false positives", "false_positives"),
                      ("Most confident false negatives", "false_negatives")):
        rows = errors[key].head(max_rows)
        if not len(rows):
            continue
        lines += [f"### {name}", "", "| score | true label | image |", "|---|---|---|"]
        for _, r in rows.iterrows():
            lines.append(
                f"| {r['score']:.3f} | {'AI' if r['label'] == 1 else 'real'} | "
                f"`{Path(r['image_path']).name}` |"
            )
        lines.append("")
    for col, g in errors.get("slices", {}).items():
        lines += [f"### Error rate by {col}", "", g.head(12).to_markdown(), ""]
    return "\n".join(lines)
