"""Charts and markdown tables, generated from `results/robustness.csv`.

Every figure in the README and the demo video is produced by this module from
the results CSV. Nothing is typed by hand, so a number in the write-up can never
disagree with a number in the results - which is the failure mode that costs
teams credibility in the Q&A.

Design notes (they are deliberate, not decoration):

* Two series only - the clean-trained baseline and the augmentation-trained
  model - assigned fixed categorical slots, never cycled.
* The headline figure is a **dumbbell plot**: one row per corruption, a dot for
  each model, a connecting line whose length *is* the improvement. Magnitude
  comparison across many categories, which a grouped bar chart at 24 categories
  cannot do legibly.
* Degradation curves are **small multiples** with one shared y-axis, never a
  dual axis.
* Both series are direct-labelled on the headline chart in addition to the
  legend, so identity never rests on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.io import ensure_dir
from ..utils.logging import get_logger

log = get_logger("eval.report")

# Reference categorical palette, light mode. Slot 1 = baseline, slot 2 = robust.
SERIES = {
    "baseline": "#2a78d6",   # blue
    "robust": "#eb6834",     # orange
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"

# Families whose severity parameter runs the "wrong" way - a higher JPEG quality
# or a larger keep-fraction is *less* damage. Their axes are reversed so that in
# every panel of the small multiples, right means worse.
INVERTED_SEVERITY = {"jpeg", "webp", "rescale", "crop"}
AXIS_LABEL = {
    "jpeg": "JPEG quality (lower = worse)",
    "webp": "WebP quality (lower = worse)",
    "blur": "Gaussian sigma",
    "noise": "noise sigma",
    "rescale": "downscale factor (lower = worse)",
    "crop": "fraction kept (lower = worse)",
    "color": "jitter strength",
}


def _style(ax) -> None:
    """Recessive axes and grid: the data is the only thing with contrast."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def _series_color(model_name: str, index: int) -> str:
    """Colour follows the entity, not its position in a filtered list."""
    name = model_name.lower()
    if any(k in name for k in ("aug", "robust", "damage")):
        return SERIES["robust"]
    if any(k in name for k in ("clean", "base", "vanilla")):
        return SERIES["baseline"]
    return [SERIES["baseline"], SERIES["robust"]][index % 2]


def plot_robustness_dumbbell(
    df: pd.DataFrame, out_path: str | Path, metric: str = "auc", title: str | None = None
) -> Path:
    """Headline figure: per-condition metric for each model, gap drawn as a line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(df["model"].unique())[:2]
    pivot = df.pivot_table(index=["family", "condition", "held_out"], columns="model", values=metric)
    pivot = pivot.reset_index()

    # Clean first, then spec transforms grouped by family, then held-out, then compound.
    order_key = pivot["condition"].eq("clean").map({True: 0, False: 1})
    tier = np.where(pivot["family"] == "compound", 3, np.where(pivot["held_out"], 2, 1))
    pivot = pivot.assign(_o=order_key * 1, _t=np.where(pivot["condition"] == "clean", 0, tier))
    pivot = pivot.sort_values(["_t", "family", "condition"]).reset_index(drop=True)

    y = np.arange(len(pivot))[::-1]
    fig_h = max(4.5, 0.30 * len(pivot) + 1.8)
    fig, ax = plt.subplots(figsize=(9.5, fig_h), facecolor=SURFACE)
    _style(ax)

    if len(models) == 2:
        a, b = pivot[models[0]].to_numpy(), pivot[models[1]].to_numpy()
        ax.hlines(y, np.minimum(a, b), np.maximum(a, b), color=INK_MUTED, linewidth=2, alpha=0.45,
                  zorder=1)

    for i, m in enumerate(models):
        ax.scatter(
            pivot[m], y, s=64, color=_series_color(m, i), label=m,
            zorder=3, edgecolors=SURFACE, linewidths=2,  # 2px surface ring on overlap
        )

    labels = [
        f"{r.condition}" + ("  (unseen)" if r.held_out else "")
        for r in pivot.itertuples()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=INK_PRIMARY)
    for tick, r in zip(ax.get_yticklabels(), pivot.itertuples()):
        if r.condition == "clean":
            tick.set_fontweight("bold")
        elif r.held_out or r.family == "compound":
            tick.set_color(INK_SECONDARY)

    # Zoom to the data range rather than forcing 0-1: the differences this chart
    # exists to show are a few points of AUC, and a fixed full-range axis hides
    # them. The axis is labelled, so there is no misreading risk.
    vals = pivot[models].to_numpy()
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    pad = max(0.02, (hi - lo) * 0.15)
    ax.set_xlim(max(0.0, lo - pad), min(1.005, hi + pad))
    ax.set_xlabel(metric.upper(), fontsize=9, color=INK_SECONDARY)
    ax.set_title(
        title or f"Robustness: {metric.upper()} per corruption\n"
        "bold = clean baseline · (unseen) = corruption never trained on",
        fontsize=11, color=INK_PRIMARY, loc="left", pad=14,
    )

    # Direct labels on the top row in addition to the legend, so the two series
    # are identifiable without reading colour.
    if len(pivot):
        for i, m in enumerate(models):
            ax.annotate(
                m, (pivot[m].iloc[0], y[0]), textcoords="offset points", xytext=(0, 12 + 10 * i),
                ha="center", fontsize=8, color=_series_color(m, i), fontweight="bold",
            )
    handles, labels_ = ax.get_legend_handles_labels()
    leg = fig.legend(
        handles, labels_, frameon=False, fontsize=9, loc="lower center",
        ncol=len(models), bbox_to_anchor=(0.5, -0.015),
    )
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    fig.tight_layout()
    p = Path(out_path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    log.info("figure -> %s", p)
    return p


def plot_degradation_curves(
    df: pd.DataFrame, out_path: str | Path, metric: str = "auc"
) -> Path:
    """Small multiples: metric vs severity, one panel per corruption family."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = [f for f in df["family"].unique() if f not in ("clean", "compound")]
    fams = sorted(fams, key=lambda f: -df[df["family"] == f]["severity"].nunique())
    fams = [f for f in fams if df[df["family"] == f]["severity"].nunique() >= 2]
    if not fams:
        fams = sorted(set(df["family"]) - {"clean"})

    models = list(df["model"].unique())[:2]
    n = len(fams)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.1 * n_cols, 2.7 * n_rows), facecolor=SURFACE, squeeze=False,
        sharey=True,
    )
    axes_flat = axes.ravel()

    clean = df[df["condition"] == "clean"].set_index("model")[metric].to_dict()

    for ax, fam in zip(axes_flat, fams):
        _style(ax)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        sub = df[df["family"] == fam].sort_values("severity")
        for i, m in enumerate(models):
            s = sub[sub["model"] == m]
            ax.plot(
                s["severity"], s[metric], marker="o", markersize=6, linewidth=2,
                color=_series_color(m, i), label=m, markeredgecolor=SURFACE, markeredgewidth=1.5,
            )
            if m in clean:
                ax.axhline(clean[m], color=_series_color(m, i), linewidth=1, alpha=0.30, linestyle=":")
        ax.set_title(fam, fontsize=10, color=INK_PRIMARY, loc="left")
        ax.set_xlabel(AXIS_LABEL.get(fam, "severity"), fontsize=8, color=INK_MUTED)
        # For quality-like parameters a *higher* number means *less* damage, so
        # the axis is reversed and every panel reads "worse to the right".
        if fam in INVERTED_SEVERITY:
            ax.invert_xaxis()

    for ax in axes_flat[len(fams):]:
        ax.axis("off")
    axes_flat[0].set_ylabel(metric.upper(), fontsize=9, color=INK_SECONDARY)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center",
                     ncol=len(models), bbox_to_anchor=(0.5, -0.02))
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    fig.suptitle(
        f"{metric.upper()} as corruption gets worse (dotted line = each model's clean score)",
        fontsize=11, color=INK_PRIMARY, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    p = Path(out_path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    log.info("figure -> %s", p)
    return p


def plot_score_distributions(
    scores_df: pd.DataFrame, out_path: str | Path, model: str, conditions: list[str] | None = None
) -> Path:
    """Where the two classes' scores sit, clean vs damaged.

    This is the figure that explains *how* a model fails: the distributions
    either overlap (confused) or collapse toward the middle (uncertain), and
    the fix is different in each case.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = scores_df[scores_df["model"] == model]
    conditions = conditions or ["clean", "jpeg_q30", "rescale_0.25"]
    conditions = [c for c in conditions if c in set(sub["condition"])]
    if not conditions:
        conditions = list(sub["condition"].unique())[:3]

    fig, axes = plt.subplots(
        1, len(conditions), figsize=(3.4 * len(conditions), 2.9), facecolor=SURFACE, squeeze=False,
        sharey=True,
    )
    bins = np.linspace(0, 1, 31)
    for ax, cond in zip(axes[0], conditions):
        _style(ax)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        s = sub[sub["condition"] == cond]
        ax.hist(s[s["label"] == 0]["score"], bins=bins, color=SERIES["baseline"], alpha=0.75,
                label="authentic", edgecolor=SURFACE, linewidth=0.5)
        ax.hist(s[s["label"] == 1]["score"], bins=bins, color=SERIES["robust"], alpha=0.75,
                label="AI-generated", edgecolor=SURFACE, linewidth=0.5)
        ax.set_title(cond, fontsize=10, color=INK_PRIMARY, loc="left")
        ax.set_xlabel("score", fontsize=8, color=INK_MUTED)
    axes[0][0].set_ylabel("images", fontsize=9, color=INK_SECONDARY)
    handles, labels = axes[0][0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=2,
                     bbox_to_anchor=(0.5, -0.06))
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    fig.suptitle(f"Score distributions - {model}", fontsize=11, color=INK_PRIMARY, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))

    p = Path(out_path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    log.info("figure -> %s", p)
    return p


def markdown_table(df: pd.DataFrame, metric: str = "auc") -> str:
    """The compact clean-vs-transformed table the deliverable asks for."""
    from .robustness import robustness_summary

    piv = robustness_summary(df, metric=metric)
    return piv.to_markdown()


def write_report(
    results_csv: str | Path,
    out_dir: str | Path = "results",
    metric: str = "auc",
    scores_csv: str | Path | None = None,
) -> dict:
    """Regenerate every figure and table from the results CSV."""
    from .robustness import degradation_table

    out_dir = ensure_dir(out_dir)
    df = pd.read_csv(results_csv)
    figs = {
        "dumbbell": plot_robustness_dumbbell(df, out_dir / f"robustness_{metric}.png", metric),
        "curves": plot_degradation_curves(df, out_dir / f"degradation_{metric}.png", metric),
    }
    if scores_csv and Path(scores_csv).exists():
        sdf = pd.read_csv(scores_csv)
        model = sorted(sdf["model"].unique())[-1]
        figs["distributions"] = plot_score_distributions(
            sdf, out_dir / "score_distributions.png", model=model
        )

    summary = degradation_table(df, metric=metric)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / f"robustness_{metric}.md").write_text(markdown_table(df, metric), encoding="utf-8")

    return {"figures": figs, "summary": summary, "results": df}
