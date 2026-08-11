"""Value-stratified train/val split of the reference library, as row indices.

Sorts the reference compounds by their teacher value and takes equally-spaced (by rank) samples as
the validation set, so validation covers the value distribution uniformly rather than by luck. Only
indices are produced: every column of a run shares one copy of the descriptor matrix, so
materialising the features per column would duplicate gigabytes for no benefit.
"""

from __future__ import annotations

import numpy as np


def split_reference_to_indices(
  y, val_frac: float = 0.1, seed: int = 42, limit: int | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
  """Value-stratified train/val split of the reference library, returned as row indices.

  Rows are sorted by value, validation is taken at equally-spaced ranks, and both sides are then
  shuffled. Nothing is read or written — the caller gathers these rows from the shared library.

  Parameters
  ----------
  y : array-like
      Teacher value per reference row. Non-finite entries are excluded from both splits.
  val_frac : float, optional
      Validation fraction, taken equally-spaced by value rank.
  seed : int, optional
      Shuffle seed.
  limit : int, optional
      Consider only the first N reference rows (development subsampling).

  Returns
  -------
  (np.ndarray, np.ndarray, dict)
      Shuffled ``train_idx`` and ``val_idx`` (int64, indices into the reference library), and an
      ``info`` dict with ``n_total``, ``n_used``, ``n_dropped``, ``vmin`` and ``vmax``.
  """
  y = np.asarray(y, dtype=np.float32)
  n_total = len(y)
  if limit is not None and limit < n_total:
    n_total = int(limit)
    y = y[:n_total]

  valid = np.where(np.isfinite(y))[0]
  m = len(valid)
  if m < 2:
    raise ValueError("need at least 2 finite reference-calcs values to split")

  order = valid[np.argsort(y[valid], kind="stable")]
  n_val = max(1, min(int(round(m * val_frac)), m - 1))
  val_pos = np.unique(np.round(np.linspace(0, m - 1, n_val)).astype(np.int64))
  is_val = np.zeros(m, dtype=bool)
  is_val[val_pos] = True

  rng = np.random.default_rng(seed)
  train_idx = rng.permutation(order[~is_val])
  val_idx = rng.permutation(order[is_val])
  info = {
    "n_total": int(n_total),
    "n_used": int(m),
    "n_dropped": int(n_total - m),
    "vmin": float(y[valid].min()),
    "vmax": float(y[valid].max()),
  }
  return train_idx.astype(np.int64), val_idx.astype(np.int64), info
