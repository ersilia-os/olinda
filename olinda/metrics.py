"""Regression metrics shared by training, distillation, and robustness evaluation.

Numpy-only, no SciPy: the same six numbers are reported by ``olinda learn-soft``
(``val_metrics.json``) for every column, and by :mod:`olinda.robustness`, so they stay
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


def average_ranks(x) -> np.ndarray:
  """Ranks of ``x`` with tied values sharing their average rank.

  This is what Spearman is defined on. The cheaper ``argsort(argsort(x))`` gives *ordinal* ranks,
  which break ties by array position and so manufacture agreement: on ``y = [0]*8 + [1]*2`` against
  ``p = 0..9`` it reports a rank correlation of 1.000 where the true value is 0.696. Teacher columns
  of probabilities routinely carry a mass of identical values, so that error is not hypothetical.

  Vectorised — one sort plus a group-mean — so it costs about 100 ms on the 1.35M-row reference
  library. Matches ``scipy.stats.rankdata`` exactly (0-based).
  """
  x = np.asarray(x)
  n = len(x)
  if n == 0:
    return np.empty(0, dtype=np.float64)
  order = np.argsort(x, kind="stable")
  ordered = x[order]
  is_new = np.empty(n, dtype=bool)
  is_new[0] = True
  np.not_equal(ordered[1:], ordered[:-1], out=is_new[1:])
  group = np.cumsum(is_new) - 1
  counts = np.bincount(group)
  starts = np.cumsum(counts) - counts
  ranks = np.empty(n, dtype=np.float64)
  ranks[order] = (starts + (counts - 1) / 2.0)[group]
  return ranks


def _spearmanr(y, p) -> float:
  y = np.asarray(y, dtype=np.float64)
  p = np.asarray(p, dtype=np.float64)
  return _pearsonr(average_ranks(y), average_ranks(p))


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


def json_safe(obj):
  """Replace non-finite floats with ``None`` so the result is valid JSON for any consumer.

  ``NaN`` and ``Infinity`` are Python-specific extensions that strict JSON parsers reject, and these
  metrics get embedded in ``model.onnx`` — where they may be read by tooling in any language.
  """
  if isinstance(obj, dict):
    return {k: json_safe(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [json_safe(v) for v in obj]
  if isinstance(obj, float) and not np.isfinite(obj):
    return None
  if isinstance(obj, np.floating):
    return None if not np.isfinite(obj) else float(obj)
  if isinstance(obj, np.integer):
    return int(obj)
  return obj


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
