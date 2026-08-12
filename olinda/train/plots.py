"""Training diagnostic plots (stylia). Imported lazily so CLI startup stays fast.

Styling lives in :mod:`olinda.style`, shared with the validation report, so a training figure and
a report figure use the same colours for the same things. These are written at stylia's own
600 dpi — unlike the report PNGs, they are looked at one at a time rather than tiled in a page.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from olinda.helpers import logger
from olinda.style import density, figure, import_stylia, limits, reference, subsample


def _close(fig) -> None:
  """Release a figure once it is written.

  ``stylia.save_figure`` only calls ``savefig``, so pyplot keeps every figure — and its data —
  alive until told otherwise. One per column from ``learn-soft`` plus three from ``learn-hard``
  adds up over a multi-column run, which is also what raises matplotlib's "More than 20 figures"
  warning.
  """
  import matplotlib.pyplot as plt

  plt.close(fig)


def density_scatter(
  ax,
  stylia,
  x: np.ndarray,
  y: np.ndarray,
  *,
  xlabel: str,
  ylabel: str,
  title: str,
  role: str = "teacher",
  diagonal: bool = False,
) -> None:
  """Density surface of ``x`` against ``y``, optionally with a y=x reference line.

  With *diagonal* set, both axes are pinned to one shared range so the line really is at 45° —
  otherwise matplotlib scales the two independently and "above the line" stops meaning what it
  looks like.
  """
  density(ax, stylia, x, y, role=role)
  if diagonal:
    lo, hi = limits(ax, x, y)
    reference(ax, stylia, "diagonal", lo=lo, hi=hi)
  stylia.label(ax, xlabel=xlabel, ylabel=ylabel, title=title)


def save_true_vs_pred(
  y_true: np.ndarray,
  y_pred: np.ndarray,
  out_path: str | Path,
  *,
  title: str | None = None,
  max_points: int = 400000,
  seed: int = 0,
) -> Path | None:
  """Save a validation true-vs-pred density plot to ``out_path``; return it (``None`` if skipped).

  stylia/matplotlib are imported here (not at module import) so ``olinda`` CLI startup stays fast;
  if stylia is missing the plot is skipped with a warning rather than failing the run.

  Parameters
  ----------
  y_true, y_pred : np.ndarray
      Validation targets and predictions.
  out_path : str or Path
      Destination PNG path.
  title : str, optional
      Plot title; defaults to ``"R² = … · RMSE = …"``.
  max_points : int
      Read cap for very large validation sets.
  """
  stylia = import_stylia()
  if stylia is None:
    logger.warning("stylia not installed — skipping true-vs-pred plot (pip install stylia)")
    return None

  y = np.asarray(y_true, dtype=np.float64).ravel()
  p = np.asarray(y_pred, dtype=np.float64).ravel()
  err = p - y
  rmse = float(np.sqrt((err**2).mean()))
  ss_tot = float(((y - y.mean()) ** 2).sum())
  r2 = 1.0 - float((err**2).sum()) / ss_tot if ss_tot else float("nan")

  y, p = subsample([y, p], max_points, seed)

  out_path = Path(out_path)
  fig, ax = figure(stylia)
  density_scatter(
    ax,
    stylia,
    y,
    p,
    xlabel="True (validation)",
    ylabel="Predicted",
    title=title or f"R² = {r2:.3f} · RMSE = {rmse:.4f}",
    diagonal=True,
  )
  stylia.save_figure(str(out_path))
  _close(fig)
  return out_path
