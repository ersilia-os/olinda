"""Figures for a validation report, drawn in the Ersilia house style via stylia.

Every function takes the axis to draw into, so the caller owns the figure — :func:`render` builds it,
writes a PNG and a vector PDF, and closes it. Closing matters: matplotlib keeps figures alive until
told otherwise, and a multi-column report opens dozens.

Nothing here computes metrics. Values come in already computed from :mod:`olinda.metrics`, so a
number shown on a figure and the same number in ``metrics.json`` cannot disagree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Screen resolution rather than stylia's 600 dpi: the PNGs are viewed in a browser at a few hundred
# pixels wide, where 600 dpi costs encoding time and disk for no visible gain. The PDF stays vector.
REPORT_DPI = 200
MAX_SCATTER = 40_000  # dense scatters beyond this are slow to draw and unreadable anyway


def _stylia():
  """The configured stylia module, or ``None`` when it is not installed."""
  try:
    import stylia
  except ImportError:
    return None
  # Format: slide | Style: ersilia — change with stylia.set_format() / stylia.set_style()
  stylia.set_format("slide")
  stylia.set_style("ersilia")
  return stylia


def render(draw, out_dir: str | Path, name: str, *, square: bool = True) -> dict | None:
  """Draw one figure and write ``png/<name>.png`` and ``pdf/<name>.pdf`` under *out_dir*.

  Parameters
  ----------
  draw : callable
      Receives ``(ax, stylia)`` and draws the figure.
  square : bool
      True for plots whose data space is square (correlation, ROC, calibration); False for wide
      ones (histograms, enrichment curves).

  Returns
  -------
  dict or None
      ``{"name", "png", "pdf"}``, or ``None`` if stylia is missing or the draw call declined.
  """
  import matplotlib.pyplot as plt

  st = _stylia()
  if st is None:
    return None

  fig, axs = st.create_figure(1, 1, width=0.5, height=0.5) if square else st.create_figure(1, 1)
  try:
    if draw(axs.next(), st) is False:  # a plot can decline (no positives, empty input, ...)
      return None
    out_dir = Path(out_dir)
    png, pdf = out_dir / "png" / f"{name}.png", out_dir / "pdf" / f"{name}.pdf"
    png.parent.mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(png, dpi=REPORT_DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return {"name": name, "png": f"png/{png.name}", "pdf": f"pdf/{pdf.name}"}
  finally:
    plt.close(fig)


# ── soft labels: how close is the student to the teacher ─────────────────────


def correlation(ax, st, y, p, *, metrics: dict, task: str) -> None:
  """Predicted vs observed, coloured by local density, with the y=x line."""
  from olinda.train.plots import density_scatter, subsample

  ys, ps = subsample([np.asarray(y), np.asarray(p)], MAX_SCATTER, 0)
  density_scatter(
    ax,
    st,
    ys,
    ps,
    xlabel="Teacher value",
    ylabel="olinda prediction",
    title=f"{task}  (R²={metrics['r2']:.3f}, ρ={metrics['spearman']:.3f})",
  )
  lo = float(min(ys.min(), ps.min()))
  hi = float(max(ys.max(), ps.max()))
  ax.plot([lo, hi], [lo, hi], linestyle="--", color=st.NamedColors().gray, zorder=0)


def residual_hist(ax, st, y, p) -> None:
  """Distribution of prediction − truth. A centred, narrow peak is the goal."""
  r = np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)
  ax.hist(r, bins=60, color=st.NamedColors().blue)
  ax.axvline(0.0, linestyle="--", color=st.NamedColors().gray)
  st.label(
    ax,
    xlabel="Prediction − teacher",
    ylabel="Compounds",
    title=f"Residuals  (bias={r.mean():+.4f}, sd={r.std():.4f})",
  )


def residuals_vs_pred(ax, st, y, p) -> None:
  """Residual against predicted value — a fan or a slope means the error depends on the answer."""
  from olinda.train.plots import density_scatter, subsample

  y = np.asarray(y, dtype=np.float64)
  p = np.asarray(p, dtype=np.float64)
  ps, rs = subsample([p, p - y], MAX_SCATTER, 0)
  density_scatter(
    ax, st, ps, rs, xlabel="olinda prediction", ylabel="Prediction − teacher", title="Residual structure"
  )
  ax.axhline(0.0, linestyle="--", color=st.NamedColors().gray, zorder=0)


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
  xs = np.array([p[idx == b].mean() for b in range(len(edges) - 1) if (idx == b).any()])
  ys = np.array([y[idx == b].mean() for b in range(len(edges) - 1) if (idx == b).any()])
  nc = st.NamedColors()
  lo, hi = float(min(xs.min(), ys.min())), float(max(xs.max(), ys.max()))
  ax.plot([lo, hi], [lo, hi], linestyle="--", color=nc.gray, zorder=0)
  ax.plot(xs, ys, marker="o", color=nc.plum)
  st.label(ax, xlabel="Mean prediction in bin", ylabel="Mean teacher value in bin", title="Calibration")


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
  nc = st.NamedColors()
  lo, hi = float(theoretical.min()), float(theoretical.max())
  ax.plot([lo, hi], [lo * r.std() + r.mean(), hi * r.std() + r.mean()], linestyle="--", color=nc.gray)
  ax.scatter(theoretical, r, color=nc.blue)
  st.label(ax, xlabel="Normal quantile", ylabel="Residual quantile", title="Residual QQ")


# ── hard labels: does the ranking find the actives ───────────────────────────


def roc(ax, st, y, s, *, metrics: dict) -> None:
  from olinda.metrics import roc_curve

  fpr, tpr, _ = roc_curve(y, s)
  if not np.isfinite(fpr).all():
    return False
  nc = st.NamedColors()
  ax.plot([0, 1], [0, 1], linestyle="--", color=nc.gray, zorder=0)
  ax.plot(fpr, tpr, color=nc.plum)
  st.label(
    ax, xlabel="False positive rate", ylabel="True positive rate", title=f"ROC (AUC={metrics['auroc']:.3f})"
  )


def precision_recall(ax, st, y, s, *, metrics: dict) -> None:
  """PR curve with the chance line at the hit rate — the honest baseline on imbalanced sets."""
  from olinda.metrics import pr_curve

  recall, precision = pr_curve(y, s)
  if not np.isfinite(recall).all():
    return False
  nc = st.NamedColors()
  ax.axhline(metrics["hit_rate"], linestyle="--", color=nc.gray, zorder=0)
  ax.plot(recall, precision, color=nc.purple)
  st.label(
    ax,
    xlabel="Recall",
    ylabel="Precision",
    title=f"Precision–recall (AP={metrics['average_precision']:.3f}, chance={metrics['hit_rate']:.3f})",
  )


def enrichment(ax, st, y, s) -> None:
  """Enrichment factor across top-k fractions, against the 1.0 chance line."""
  from olinda.metrics import enrichment_factor

  fractions = np.geomspace(0.001, 1.0, 40)
  ef = np.array([enrichment_factor(y, s, f) for f in fractions])
  if not np.isfinite(ef).any():
    return False
  nc = st.NamedColors()
  ax.axhline(1.0, linestyle="--", color=nc.gray, zorder=0)
  ax.plot(fractions * 100, ef, color=nc.mint)
  ax.set_xscale("log")
  st.label(ax, xlabel="Top % of the ranking", ylabel="Enrichment over chance", title="Enrichment")


def score_by_class(ax, st, y, s) -> None:
  """Predicted score split by true label — the overlap is what AUROC summarises in one number."""
  y = np.asarray(y).ravel().astype(int)
  s = np.asarray(s, dtype=np.float64).ravel()
  if not ((y == 1).any() and (y == 0).any()):
    return False
  nc = st.NamedColors()
  bins = np.linspace(float(s.min()), float(s.max()), 50)
  ax.hist(s[y == 0], bins=bins, color=nc.get("gray", lighten=0.3), label="inactive", density=True)
  ax.hist(s[y == 1], bins=bins, color=nc.plum, label="active", density=True, alpha=0.75)
  ax.legend()
  st.label(ax, xlabel="olinda prediction", ylabel="Density", title="Score by true label")


# ── the model's own internals ────────────────────────────────────────────────


def calibration_map(ax, st, curve: dict, *, title: str, xlabel: str) -> None:
  """One isotonic stage as recovered from the graph: what the model does to its own raw score."""
  if curve is None or curve["n_anchors"] < 2:
    return False
  nc = st.NamedColors()
  ax.plot(curve["x"], curve["y"], color=nc.purple)
  st.label(ax, xlabel=xlabel, ylabel="Teacher scale", title=f"{title}  ({curve['n_anchors']} anchors)")
