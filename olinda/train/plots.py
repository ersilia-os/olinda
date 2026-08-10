"""Training diagnostic plots (stylia). Imported lazily so CLI startup stays fast."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from olinda.helpers import logger


def _plot_true_pred(ax, stylia, y: np.ndarray, p: np.ndarray, title: str) -> None:
  """Density-coloured true-vs-pred scatter with a y=x reference line (draws into ``ax``)."""
  nc = stylia.NamedColors()
  lo = float(min(y.min(), p.min()))
  hi = float(max(y.max(), p.max()))
  # colour points by local density (2D histogram lookup), densest drawn last
  grid = 200
  hist, xe, ye = np.histogram2d(y, p, bins=grid)
  xi = np.clip(np.digitize(y, xe) - 1, 0, grid - 1)
  yi = np.clip(np.digitize(p, ye) - 1, 0, grid - 1)
  dens = hist[xi, yi]
  cm = stylia.FadingColormap("plum")
  cm.fit(dens)
  order = np.argsort(dens)
  ax.scatter(y[order], p[order], c=cm.transform(dens[order]))
  ax.plot([lo, hi], [lo, hi], color=nc.gray, linestyle="--")
  stylia.label(ax, xlabel="True (validation)", ylabel="Predicted", title=title)


def save_true_vs_pred(
  y_true: np.ndarray,
  y_pred: np.ndarray,
  out_path: str | Path,
  *,
  title: str | None = None,
  max_points: int = 40000,
  seed: int = 0,
) -> Path | None:
  """Save a validation true-vs-pred scatter to ``out_path``; return the path (or ``None`` if skipped).

  Points are coloured by local density with a y=x reference line and an R²/RMSE title. stylia/matplotlib
  are imported here (not at module import) so ``olinda`` CLI startup stays fast; if stylia is missing the
  plot is skipped with a warning rather than failing the run.

  Parameters
  ----------
  y_true, y_pred : np.ndarray
      Validation targets and predictions.
  out_path : str or Path
      Destination PNG path.
  title : str, optional
      Plot title; defaults to ``"Validation (R²=…, RMSE=…)"``.
  max_points : int
      Subsample cap for the scatter (dense 1M-point scatters are slow and unreadable).
  """
  try:
    import stylia
  except ImportError:
    logger.warning("stylia not installed — skipping true-vs-pred plot (pip install stylia)")
    return None

  # Format: slide | Style: ersilia — change with stylia.set_format() / stylia.set_style()
  stylia.set_format("slide")
  stylia.set_style("ersilia")

  y = np.asarray(y_true, dtype=np.float64).ravel()
  p = np.asarray(y_pred, dtype=np.float64).ravel()
  err = p - y
  rmse = float(np.sqrt((err**2).mean()))
  ss_tot = float(((y - y.mean()) ** 2).sum())
  r2 = 1.0 - float((err**2).sum()) / ss_tot if ss_tot else float("nan")

  if len(y) > max_points:
    idx = np.random.default_rng(seed).choice(len(y), size=max_points, replace=False)
    y, p = y[idx], p[idx]

  out_path = Path(out_path)
  fig, axs = stylia.create_figure(1, 1, width=0.5, height=0.5)
  _plot_true_pred(axs.next(), stylia, y, p, title or f"Validation  (R²={r2:.3f}, RMSE={rmse:.4f})")
  stylia.save_figure(str(out_path))
  return out_path


def _density_scatter(
  ax, stylia, x: np.ndarray, y: np.ndarray, *, xlabel: str, ylabel: str, title: str
) -> None:
  """Scatter of ``x`` vs ``y`` coloured by local 2D-histogram density (densest drawn last)."""
  grid = 200
  hist, xe, ye = np.histogram2d(x, y, bins=grid)
  xi = np.clip(np.digitize(x, xe) - 1, 0, grid - 1)
  yi = np.clip(np.digitize(y, ye) - 1, 0, grid - 1)
  dens = hist[xi, yi]
  cm = stylia.FadingColormap("plum")
  cm.fit(dens)
  order = np.argsort(dens)
  ax.scatter(x[order], y[order], c=cm.transform(dens[order]))
  stylia.label(ax, xlabel=xlabel, ylabel=ylabel, title=title)


def _subsample(arrays: list[np.ndarray], max_points: int, seed: int) -> list[np.ndarray]:
  """Jointly subsample a list of equal-length arrays down to ``max_points`` rows."""
  n = len(arrays[0])
  if n <= max_points:
    return arrays
  idx = np.random.default_rng(seed).choice(n, size=max_points, replace=False)
  return [a[idx] for a in arrays]


def save_ground_truth_plots(
  g: np.ndarray,
  soft: np.ndarray,
  calibrator,
  out_dir: str | Path,
  *,
  direction: str,
  pearson_after: float,
  max_points: int = 40000,
  seed: int = 0,
) -> list[Path]:
  """Save the three ``learn-hard`` calibration diagnostics; return the written PNG paths.

  All three describe how the hard model ``G`` relates to the teacher's soft labels over the reference
  library. stylia/matplotlib are imported here (not at module import) so ``olinda`` CLI startup stays
  fast; if stylia is missing every plot is skipped with a warning rather than failing the run.

  Parameters
  ----------
  g, soft : np.ndarray
      ``G``'s hard score and the teacher's soft label, paired row-for-row over the reference library
      (already masked to finite pairs by the caller).
  calibrator : IsotonicCalibrator
      The fitted ``G`` → soft-scale map, used to draw the isotonic curve and the calibrated scatter.
  out_dir : str or Path
      Directory to write the PNGs into (created if absent).
  direction : str
      ``"increasing"`` or ``"decreasing"`` — the learned monotonic direction (for the title).
  pearson_after : float
      Pearson(calibrator(G), soft) — for the calibrated-vs-soft title.
  max_points : int
      Subsample cap for the scatter panels (dense 1M-point scatters are slow and unreadable).
  """
  try:
    import stylia
  except ImportError:
    logger.warning("stylia not installed — skipping learn-hard plots (pip install stylia)")
    return []

  # Format: slide | Style: ersilia — change with stylia.set_format() / stylia.set_style()
  stylia.set_format("slide")
  stylia.set_style("ersilia")

  g = np.asarray(g, dtype=np.float64).ravel()
  soft = np.asarray(soft, dtype=np.float64).ravel()
  nc = stylia.NamedColors()
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  written: list[Path] = []

  gs, ss = _subsample([g, soft], max_points, seed)

  # 1) Calibration map: G vs soft density with the fitted isotonic curve overlaid ----------
  p1 = out_dir / "calibration_map.png"
  fig, axs = stylia.create_figure(1, 1, width=0.5, height=0.5)
  ax = axs.next()
  _density_scatter(
    ax, stylia, gs, ss, xlabel="G score", ylabel="Soft label", title=f"Calibration map ({direction})"
  )
  x_curve = np.linspace(float(g.min()), float(g.max()), 200)
  ax.plot(x_curve, calibrator.transform(x_curve), color=nc.blue)
  stylia.save_figure(str(p1))
  written.append(p1)

  # 2) Score distributions: G scores and soft labels over the reference --------------------
  p2 = out_dir / "score_distributions.png"
  fig, axs = stylia.create_figure(1, 2)
  ax = axs.next()
  ax.hist(g, bins=60, color=nc.plum)
  stylia.label(ax, xlabel="G score", ylabel="Reference compounds", title="G over reference")
  ax = axs.next()
  ax.hist(soft, bins=60, color=nc.mint)
  stylia.label(ax, xlabel="Soft label", ylabel="Reference compounds", title="Soft labels over reference")
  stylia.save_figure(str(p2))
  written.append(p2)

  # 3) Calibrated vs soft: calibrator(G) against the soft labels, with a y=x line ----------
  p3 = out_dir / "calibrated_vs_soft.png"
  cal = np.asarray(calibrator.transform(gs), dtype=np.float64)
  fig, axs = stylia.create_figure(1, 1, width=0.5, height=0.5)
  ax = axs.next()
  _density_scatter(
    ax,
    stylia,
    cal,
    ss,
    xlabel="Calibrated G",
    ylabel="Soft label",
    title=f"Calibrated vs soft  (r={pearson_after:.3f})",
  )
  lo = float(min(cal.min(), ss.min()))
  hi = float(max(cal.max(), ss.max()))
  ax.plot([lo, hi], [lo, hi], color=nc.gray, linestyle="--")
  stylia.save_figure(str(p3))
  written.append(p3)

  return written
