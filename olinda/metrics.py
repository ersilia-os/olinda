"""Regression metrics shared by training, distillation, and robustness evaluation.

Numpy-only, no SciPy: the same six numbers are reported by ``olinda learn-soft``
(``val_metrics.json``), by :mod:`olinda.pipeline`, and by :mod:`olinda.robustness`, so they stay
comparable across every code path that scores a student model.

``top_decile_rmse`` (error on the sparse high-value tail) and ``spearman`` (rank correlation) are
robust to skew, so weighted-vs-unweighted runs can be compared beyond bulk-dominated R².
"""

from __future__ import annotations

import numpy as np


def _pearsonr(y, p) -> float:
  y = np.asarray(y, dtype=np.float64)
  p = np.asarray(p, dtype=np.float64)
  y = y - y.mean()
  p = p - p.mean()
  denom = np.sqrt((y * y).sum()) * np.sqrt((p * p).sum())
  if denom == 0:
    return float("nan")
  return float((y * p).sum() / denom)


def _spearmanr(y, p) -> float:
  y = np.asarray(y, dtype=np.float64)
  p = np.asarray(p, dtype=np.float64)
  ry = np.argsort(np.argsort(y))
  rp = np.argsort(np.argsort(p))
  return _pearsonr(ry, rp)


def _r2(y, p) -> float:
  y = np.asarray(y, dtype=np.float64)
  p = np.asarray(p, dtype=np.float64)
  ss_res = ((y - p) ** 2).sum()
  ss_tot = ((y - y.mean()) ** 2).sum()
  if ss_tot == 0:
    return float("nan")
  return float(1.0 - ss_res / ss_tot)


def _mae(y, p) -> float:
  y = np.asarray(y, dtype=np.float64)
  p = np.asarray(p, dtype=np.float64)
  return float(np.mean(np.abs(y - p)))


def _rmse(y, p) -> float:
  y = np.asarray(y, dtype=np.float64)
  p = np.asarray(p, dtype=np.float64)
  return float(np.sqrt(np.mean((y - p) ** 2)))


def regression_metrics(y_true, y_pred) -> dict:
  """Score a regression prediction: MAE / RMSE / R² / Pearson / Spearman / top-decile-true RMSE.

  Degenerate inputs yield ``nan`` rather than raising: a constant ``y_true`` gives ``nan`` for R²,
  Pearson, and Spearman, since those are undefined without variance.

  Parameters
  ----------
  y_true : array_like
      Observed values; flattened to 1-D.
  y_pred : array_like
      Predicted values, same length as ``y_true``.

  Returns
  -------
  dict
      ``{"n", "mae", "rmse", "r2", "pearson", "spearman", "top_decile_rmse"}``, all plain Python
      floats (``n`` an int) so the result is directly JSON-serializable.
  """
  y = np.asarray(y_true, dtype=np.float64).ravel()
  p = np.asarray(y_pred, dtype=np.float64).ravel()
  tail = y >= np.quantile(y, 0.9)
  err = p - y
  return {
    "n": int(len(y)),
    "mae": _mae(y, p),
    "rmse": _rmse(y, p),
    "r2": _r2(y, p),
    "pearson": _pearsonr(y, p),
    "spearman": _spearmanr(y, p),
    "top_decile_rmse": float(np.sqrt((err[tail] ** 2).mean())) if tail.any() else float("nan"),
  }
