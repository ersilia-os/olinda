"""Figures for a validation report, drawn through :mod:`olinda.style`.

Every function takes the axis to draw into, so the caller owns the figure — :func:`render` builds it,
writes a PNG and a vector PDF, and closes it. Closing matters: matplotlib keeps figures alive until
told otherwise, and a multi-column report opens dozens.

Nothing here computes metrics. Values come in already computed from :mod:`olinda.metrics`, so a
number shown on a figure and the same number in ``metrics.json`` cannot disagree. Metrics are
stated in the **title** rather than in a boxed annotation, and the column name is left out of it —
the page already says which column a figure belongs to, and repeating it wastes the widest line
on the panel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from olinda.style import (
    REPORT_DPI,
    density,
    figure,
    import_stylia,
    limits,
    reference,
    rgb,
    subsample,
)

MAX_SCATTER = (
    400_000  # hexbin bins server-side, so this cap is about read time, not legibility
)

# Every figure's human title and one-line caption, in one table. The plot draws the title and the
# report page prints the caption, so a figure cannot end up described two different ways — which
# is what happened while the page derived its captions from the file name.
FIGURES = {
    "correlation": (
        "Agreement with the teacher",
        "Teacher value against olinda's prediction, one hexagon per group of compounds.",
    ),
    "residuals": (
        "Residual distribution",
        "How far predictions land from the teacher, and in which direction.",
    ),
    "residual_structure": (
        "Residual structure",
        "Residual against prediction — a fan or a slope means the error depends on the answer.",
    ),
    "calibration": (
        "Calibration",
        "Mean teacher value against mean prediction, in equal-population bins.",
    ),
    "residual_qq": (
        "Residual QQ",
        "Residual quantiles against a normal reference; curvature means heavy tails.",
    ),
    "roc": ("ROC", "True positives against false positives across every threshold."),
    "precision_recall": (
        "Precision–recall",
        "Precision against recall, with the hit rate as the honest baseline.",
    ),
    "enrichment": (
        "Enrichment",
        "How many more actives the top of the ranking holds than chance would give.",
    ),
    "score_by_class": (
        "Score by true label",
        "The score distribution for actives and inactives — the overlap is what AUROC summarises.",
    ),
    "soft_calibration": (
        "S · surrogate correction",
        "The isotonic map S applies to its own raw output, read from the graph.",
    ),
    "hard_calibration": (
        "H → H_S correction",
        "The isotonic map carrying H onto S's scale, read from the graph.",
    ),
    "score_distributions": (
        "Score distributions",
        "What the model predicts against what the teacher says, as distributions rather than pairs.",
    ),
}

# Footprint on the 3 cm reference grid, (rows, cols). Square data spaces stay square; a
# distribution or a log-x sweep earns the extra width.
CELLS = {
    "residuals": (2, 3),
    "score_by_class": (2, 3),
    "enrichment": (2, 4),
    "score_distributions": (2, 4),
}


def caption(name: str) -> tuple[str, str]:
    """``(title, caption)`` for a figure, falling back to a readable name for unknown ones.

    Figures are written as ``<task>_<figure>`` and both halves contain underscores, so the figure is
    identified by the *longest* registry key the name ends with — matching the shortest would let
    ``calibration`` win over ``soft_calibration``.
    """
    for key in sorted(FIGURES, key=len, reverse=True):
        if name == key or name.endswith(f"_{key}"):
            return FIGURES[key]
    return name.replace("_", " "), ""


def _stylia():
    """The configured stylia module, or ``None`` when it is not installed."""
    return import_stylia()


def render(
    draw, out_dir: str | Path, name: str, *, cells: tuple[int, int] | None = None
) -> dict | None:
    """Draw one figure and write ``png/<name>.png`` and ``pdf/<name>.pdf`` under *out_dir*.

    Parameters
    ----------
    draw : callable
        Receives ``(ax, stylia)`` and draws the figure. Returning ``False`` declines it.
    cells : tuple of int, optional
        Footprint on the 3 cm grid as ``(rows, cols)``; defaults to the square ``(2, 2)``.

    Returns
    -------
    dict or None
        ``{"name", "png", "pdf", "title", "caption"}``, or ``None`` if stylia is missing or the
        draw call declined.
    """
    import matplotlib.pyplot as plt

    st = _stylia()
    if st is None:
        return None

    fig, ax = figure(st, cells or (2, 2))
    try:
        if draw(ax, st) is False:  # a plot can decline (no positives, empty input, ...)
            return None
        out_dir = Path(out_dir)
        png, pdf = out_dir / "png" / f"{name}.png", out_dir / "pdf" / f"{name}.pdf"
        png.parent.mkdir(parents=True, exist_ok=True)
        pdf.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        fig.savefig(png, dpi=REPORT_DPI, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        title, text = caption(name)
        rows, cols = cells or (2, 2)
        return {
            "name": name,
            "png": f"png/{png.name}",
            "pdf": f"pdf/{pdf.name}",
            "title": title,
            "caption": text,
            "aspect": round(
                cols / rows, 3
            ),  # the page sizes each image box to this, so nothing letterboxes
        }
    finally:
        plt.close(fig)


# ── soft labels: how close is the student to the teacher ─────────────────────


def correlation(ax, st, y, p, *, metrics: dict, task: str) -> None:
    """Predicted against observed as a density surface, with the y=x line."""
    ys, ps = subsample([np.asarray(y), np.asarray(p)], MAX_SCATTER, 0)
    density(ax, st, ys, ps, role="teacher")
    lo, hi = limits(ax, ys, ps)
    reference(ax, st, "diagonal", lo=lo, hi=hi)
    st.label(
        ax,
        xlabel="Teacher value",
        ylabel="olinda prediction",
        title=f"R² = {metrics['r2']:.3f} · ρ = {metrics['spearman']:.3f}",
    )


def residual_hist(ax, st, y, p) -> None:
    """Distribution of prediction − truth. A centred, narrow peak is the goal."""
    r = np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    ax.hist(r, bins=60, color=rgb(st, "model"), edgecolor="white", linewidth=0.4)
    reference(ax, st, "vertical", value=0.0)
    st.label(
        ax,
        xlabel="Prediction − teacher",
        ylabel="Compounds",
        title=f"bias = {r.mean():+.4f} · sd = {r.std():.4f}",
    )


def residuals_vs_pred(ax, st, y, p) -> None:
    """Residual against predicted value — a fan or a slope means the error depends on the answer."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    ps, rs = subsample([p, p - y], MAX_SCATTER, 0)
    density(ax, st, ps, rs, role="model")
    reference(ax, st, "horizontal", value=0.0)
    st.label(
        ax,
        xlabel="olinda prediction",
        ylabel="Prediction − teacher",
        title="Residual structure",
    )


def calibration_bins(ax, st, y, p, n_bins: int = 20) -> None:
    """Mean observed value within each predicted-value bin, against the bin's mean prediction.

    This is the plot that shows a systematic offset: points below y=x mean the model reads high.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return False
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    xs = np.array(
        [p[idx == b].mean() for b in range(len(edges) - 1) if (idx == b).any()]
    )
    ys = np.array(
        [y[idx == b].mean() for b in range(len(edges) - 1) if (idx == b).any()]
    )
    lo, hi = limits(ax, xs, ys)
    reference(ax, st, "diagonal", lo=lo, hi=hi)
    ax.plot(xs, ys, marker="o", color=rgb(st, "model"))
    st.label(
        ax,
        xlabel="Mean prediction in bin",
        ylabel="Mean teacher value in bin",
        title="Calibration",
    )


def qq_residuals(ax, st, y, p) -> None:
    """Residual quantiles against a normal reference — curvature means heavy tails."""
    from olinda.metrics import average_ranks

    r = np.sort(np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64))
    if len(r) < 3 or r.std() == 0:
        return False
    q = (average_ranks(r) + 0.5) / len(r)
    # Normal quantiles without scipy: Acklam-style via the error function is overkill here, and
    # numpy has no ppf — use the Beasley-Springer-Moro-free identity sqrt(2)*erfinv(2q-1) with an
    # erfinv approximation (Winitzki), which is accurate to ~2e-3 in the body and fine for a QQ plot.
    x = 2.0 * q - 1.0
    a = 0.147
    ln = np.log(np.clip(1 - x * x, 1e-300, None))
    t = 2.0 / (np.pi * a) + ln / 2.0
    theoretical = np.sqrt(2.0) * np.sign(x) * np.sqrt(np.sqrt(t * t - ln / a) - t)
    lo, hi = float(theoretical.min()), float(theoretical.max())
    ax.plot(
        [lo, hi],
        [lo * r.std() + r.mean(), hi * r.std() + r.mean()],
        linestyle="--",
        color=rgb(st, "neutral"),
        linewidth=1,
        zorder=0,
    )
    ax.scatter(theoretical, r, s=8, color=rgb(st, "model"), edgecolors="none")
    st.label(
        ax, xlabel="Normal quantile", ylabel="Residual quantile", title="Residual QQ"
    )


# ── hard labels: does the ranking find the actives ───────────────────────────


def roc(ax, st, y, s, *, metrics: dict) -> None:
    from olinda.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y, s)
    if not np.isfinite(fpr).all():
        return False
    reference(ax, st, "diagonal", lo=0.0, hi=1.0)
    ax.fill_between(fpr, tpr, color=rgb(st, "model"), alpha=0.16, linewidth=0)
    ax.plot(fpr, tpr, color=rgb(st, "model"), linewidth=1.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    st.label(
        ax,
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"AUROC = {metrics['auroc']:.3f}",
    )


def precision_recall(ax, st, y, s, *, metrics: dict) -> None:
    """PR curve with the chance line at the hit rate — the honest baseline on imbalanced sets."""
    from olinda.metrics import pr_curve

    recall, precision = pr_curve(y, s)
    if not np.isfinite(recall).all():
        return False
    reference(ax, st, "horizontal", value=metrics["hit_rate"])
    ax.fill_between(recall, precision, color=rgb(st, "model"), alpha=0.16, linewidth=0)
    ax.plot(recall, precision, color=rgb(st, "model"), linewidth=1.6)
    ax.set_xlim(0, 1)
    st.label(
        ax,
        xlabel="Recall",
        ylabel="Precision",
        title=f"AP = {metrics['average_precision']:.3f} · chance = {metrics['hit_rate']:.3f}",
    )


def enrichment(ax, st, y, s) -> None:
    """Enrichment factor across top-k fractions, against the 1.0 chance line."""
    from olinda.metrics import enrichment_factor

    fractions = np.geomspace(0.001, 1.0, 40)
    ef = np.array([enrichment_factor(y, s, f) for f in fractions])
    if not np.isfinite(ef).any():
        return False
    reference(ax, st, "horizontal", value=1.0)
    ax.plot(fractions * 100, ef, color=rgb(st, "model"), linewidth=1.6)
    ax.set_xscale("log")
    st.label(
        ax,
        xlabel="Top % of the ranking",
        ylabel="Enrichment over chance",
        title="Enrichment",
    )


def score_by_class(ax, st, y, s) -> None:
    """Predicted score split by true label — the overlap is what AUROC summarises in one number."""
    y = np.asarray(y).ravel().astype(int)
    s = np.asarray(s, dtype=np.float64).ravel()
    if not ((y == 1).any() and (y == 0).any()):
        return False
    bins = np.linspace(float(s.min()), float(s.max()), 50)
    ax.hist(
        s[y == 0],
        bins=bins,
        color=rgb(st, "inactive"),
        label=f"inactive (n={int((y == 0).sum()):,})",
        density=True,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.hist(
        s[y == 1],
        bins=bins,
        color=rgb(st, "active"),
        label=f"active (n={int((y == 1).sum()):,})",
        density=True,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.legend(fontsize=6, loc="upper right")
    st.label(
        ax, xlabel="olinda prediction", ylabel="Density", title="Score by true label"
    )


# ── the model's own internals ────────────────────────────────────────────────


def calibration_map(
    ax, st, curve: dict, *, title: str, xlabel: str, role: str = "model"
) -> None:
    """One isotonic stage as recovered from the graph: what the model does to its own raw score."""
    if curve is None or curve["n_anchors"] < 2:
        return False
    ax.plot(curve["x"], curve["y"], color=rgb(st, role), linewidth=1.6)
    st.label(
        ax,
        xlabel=xlabel,
        ylabel="Teacher scale",
        title=f"{title} · {curve['n_anchors']} anchors",
    )


def score_distributions(ax, st, y, p) -> None:
    """Teacher values and model predictions as overlaid distributions.

    The correlation plot pairs them row by row; this asks the separate question of whether the model
    reproduces the *shape* of the teacher's output. A regressor fitted on a skewed target is usually
    under-dispersed — it hedges toward the mean — which shows up here as a narrower histogram sitting
    inside the teacher's, while the correlation plot can still look healthy.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    lo = float(min(y.min(), p.min()))
    hi = float(max(y.max(), p.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return False
    bins = np.linspace(lo, hi, 60)
    ax.hist(
        y,
        bins=bins,
        color=rgb(st, "teacher"),
        alpha=0.55,
        label="Teacher",
        edgecolor="none",
    )
    ax.hist(
        p,
        bins=bins,
        color=rgb(st, "model"),
        alpha=0.55,
        label="olinda",
        edgecolor="none",
    )
    ax.legend()
    st.label(
        ax,
        xlabel="Value",
        ylabel="Compounds",
        title=f"spread {p.std():.3f} vs teacher {y.std():.3f}",
    )
