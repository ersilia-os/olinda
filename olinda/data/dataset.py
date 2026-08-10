from __future__ import annotations

from contextlib import suppress

import numpy as np
import xgboost as xgb

from olinda.helpers import logger


def detect_imbalance_from_y(
  y_all: np.ndarray,
  n_bins: int = 20,
  cv_threshold: float = 0.5,
) -> tuple[bool, float, dict]:
  """Detect whether a regression target is imbalanced (operates on a y vector).

  Uses equal-width bins over the [1st, 99th] percentile range and measures the coefficient of
  variation (CV) of bin counts. Returns ``(is_imbalanced, cv_score, details_dict)``.
  """
  y_all = np.asarray(y_all, dtype=np.float64)
  n = len(y_all)

  lo, hi = np.percentile(y_all, [1, 99])
  if hi - lo < 1e-12:
    return False, 0.0, {"reason": "near-constant target", "n": n}

  bin_edges = np.linspace(lo, hi, n_bins + 1)
  bin_edges[0] = -np.inf
  bin_edges[-1] = np.inf

  bin_indices = np.digitize(y_all, bin_edges[1:-1])
  bin_counts = np.bincount(bin_indices, minlength=n_bins).astype(np.float64)

  non_empty = bin_counts[bin_counts > 0]
  mean_count = non_empty.mean()
  std_count = non_empty.std()
  cv = float(std_count / mean_count) if mean_count > 0 else 0.0

  empty_frac = float(np.sum(bin_counts == 0)) / n_bins
  imbalance_ratio = float(non_empty.max() / non_empty.min()) if len(non_empty) > 1 else 1.0

  is_imbalanced = cv > cv_threshold or empty_frac > 0.3

  details = {
    "cv": round(cv, 4),
    "cv_threshold": cv_threshold,
    "empty_frac": round(empty_frac, 4),
    "imbalance_ratio": round(imbalance_ratio, 2),
    "n": n,
    "n_bins": n_bins,
    "y_min": float(y_all.min()),
    "y_max": float(y_all.max()),
    "y_mean": float(y_all.mean()),
    "y_std": float(y_all.std()),
  }

  return is_imbalanced, cv, details


def regression_weights_from_y(
  y_all: np.ndarray,
  n_bins: int = 20,
  max_weight: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
  """Inverse-density sample weights over equal-width bins (operates on a y vector).

  Inverse-density sample weights over equal-width bins on the [1st, 99th] percentile range: an
  average-density bin gets weight ~1, sparse bins are boosted (capped at *max_weight*), dense bins
  are floored at 1. Returns (bin_edges, bin_weights).
  """
  y_all = np.asarray(y_all, dtype=np.float64)

  lo, hi = np.percentile(y_all, [1, 99])
  if hi - lo < 1e-12:
    edges = np.array([-np.inf, np.inf])
    return edges, np.array([1.0], dtype=np.float32)

  bin_edges = np.linspace(lo, hi, n_bins + 1)
  bin_edges[0] = -np.inf
  bin_edges[-1] = np.inf

  bin_indices = np.digitize(y_all, bin_edges[1:-1])
  bin_counts = np.bincount(bin_indices, minlength=n_bins).astype(np.float64)

  total = float(len(y_all))
  with np.errstate(divide="ignore", invalid="ignore"):
    bin_weights = np.where(bin_counts > 0, total / (n_bins * bin_counts), 1.0)
  bin_weights = np.clip(bin_weights, 1.0, max_weight)

  logger.debug(
    f"Regression reweighting: {n_bins} equal-width bins, "
    f"weight range [{bin_weights.min():.4f}, {bin_weights.max():.4f}], "
    f"max_weight cap={max_weight}"
  )
  return bin_edges, bin_weights.astype(np.float32)


def apply_bin_weights(
  y: np.ndarray,
  bin_edges: np.ndarray,
  bin_weights: np.ndarray,
) -> np.ndarray:
  """Map each y value to its bin weight (works for both the coarse bin table and the fine KDE grid)."""
  bin_indices = np.digitize(y, bin_edges[1:-1])
  return bin_weights[np.clip(bin_indices, 0, len(bin_weights) - 1)]


def density_weights_from_y(
  y_all: np.ndarray,
  alpha: float = 1.0,
  max_weight: float = 10.0,
  n_grid: int = 512,
  bandwidth: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Smooth kernel-density inverse-weights (DenseWeight-style, bins-free), tail-robust.

  Unlike :func:`regression_weights_from_y`'s coarse equal-width bins, this evaluates a Gaussian-smoothed
  density of ``y`` *pointwise* over the full ``[min, max]`` range, so rare values in the extreme tails
  (which a coarse bin would lump in with the dense flank of a nearby mode) get their own low density →
  high weight. Handles skewed / bimodal / multimodal shapes uniformly. Reference: Steininger et al.,
  "Density-based weighting for imbalanced regression", *Machine Learning* 2021.

  Implemented on a fine grid (histogram → Gaussian convolution → per-cell weight) so it is numpy-only and
  O(n) for large ``y``. Returns ``(edges, weights)`` in the SAME layout as :func:`regression_weights_from_y`
  (``edges`` length ``n_grid + 1`` with ``±inf`` ends, ``weights`` length ``n_grid``), so it flows through
  :func:`apply_bin_weights` / :class:`H5DataIter` unchanged.

  Parameters
  ----------
  alpha : float
      Weighting intensity. 0 → all weights 1 (off); 1 → full inverse-density. Values in between soften it.
  max_weight : float
      Cap on the up-weight for the sparsest regions (floor is always 1).
  n_grid : int
      Density-grid resolution (fine, unlike the coarse ``n_bins`` of the binned scheme).
  bandwidth : float, optional
      Gaussian kernel bandwidth in y-units. Defaults to a robust Silverman estimate.
  """
  y_all = np.asarray(y_all, dtype=np.float64)
  lo, hi = float(y_all.min()), float(y_all.max())
  if hi - lo < 1e-12:
    return np.array([-np.inf, np.inf]), np.array([1.0], dtype=np.float32)

  # Robust Silverman bandwidth: min(std, IQR/1.349) guards against heavy tails / outliers inflating std.
  std = float(y_all.std())
  q75, q25 = np.percentile(y_all, [75, 25])
  iqr = float(q75 - q25)
  scale = min(std, iqr / 1.349) if iqr > 0 else std
  if bandwidth is None:
    bandwidth = max(0.9 * scale * len(y_all) ** (-1 / 5), (hi - lo) / n_grid)

  edges = np.linspace(lo, hi, n_grid + 1)
  counts, _ = np.histogram(y_all, bins=edges)

  dx = (hi - lo) / n_grid
  half = max(int(3 * bandwidth / dx), 1)
  kx = np.arange(-half, half + 1) * dx
  kernel = np.exp(-0.5 * (kx / bandwidth) ** 2)
  kernel /= kernel.sum()
  density = np.convolve(counts.astype(np.float64), kernel, mode="same")
  density = np.maximum(density, density.max() * 1e-6)  # guard empty grid cells

  dens_at_y = density[np.clip(np.digitize(y_all, edges[1:-1]), 0, n_grid - 1)]
  typical = float(dens_at_y.mean())  # density a typical sample sees → this density gets weight 1
  with np.errstate(divide="ignore", invalid="ignore"):
    weights = np.clip((typical / density) ** float(alpha), 1.0, max_weight)

  edges[0] = -np.inf
  edges[-1] = np.inf
  logger.debug(
    f"KDE reweighting: {n_grid} grid cells, bandwidth={bandwidth:.4g}, alpha={alpha}, "
    f"weight range [{weights.min():.4f}, {weights.max():.4f}], max_weight cap={max_weight}"
  )
  return edges, weights.astype(np.float32)


def choose_weighting_strategy(
  y_all: np.ndarray,
  n_bins: int = 20,
  cv_threshold: float = 0.5,
  min_n_for_kde: int = 1000,
  min_unique_for_kde: int = 20,
) -> tuple[str, dict]:
  """Pick the reweighting strategy from the target's shape: ``"none"`` | ``"bins"`` | ``"kde"``.

  - Balanced target (uniform / evenly-covered) → ``"none"`` (no weighting; also avoids KDE boundary
    up-weighting on uniform data).
  - Imbalanced but discrete / small sample (few unique values or tiny ``n``) → ``"bins"`` (KDE smoothing is
    meaningless on a spiky/discrete target; coarse bins are stable).
  - Imbalanced continuous (skewed / bimodal / multimodal / heavy-tailed) → ``"kde"`` (tail-robust).
  """
  y_all = np.asarray(y_all, dtype=np.float64)
  n = len(y_all)
  n_unique = int(np.unique(y_all).size)
  is_imbalanced, cv, details = detect_imbalance_from_y(y_all, n_bins=n_bins, cv_threshold=cv_threshold)

  info = {"cv": round(float(cv), 4), "n": n, "n_unique": n_unique}
  if not is_imbalanced:
    return "none", {"reason": f"balanced target (cv={cv:.2f} ≤ {cv_threshold})", **info}
  if n < min_n_for_kde or n_unique < min_unique_for_kde:
    return "bins", {"reason": f"imbalanced but discrete/small (n={n}, unique={n_unique})", **info}
  return "kde", {"reason": f"imbalanced continuous (cv={cv:.2f}, unique={n_unique})", **info}


def resolve_regression_weights(
  y_all: np.ndarray,
  mode: str = "auto",
  *,
  alpha: float = 1.0,
  n_bins: int = 20,
  max_weight: float = 10.0,
  n_grid: int = 512,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
  """Resolve target reweighting to ``(edges, weights, info)``.

  ``mode`` is ``auto`` | ``on`` | ``off`` | ``kde`` | ``bins`` | ``none``:

  - ``auto`` (default): weight only when the target is imbalanced (via :func:`choose_weighting_strategy`) —
    ``none`` for balanced targets, ``kde``/``bins`` otherwise.
  - ``on``: force weighting even on a balanced target (picks ``kde`` for continuous, ``bins`` for discrete).
  - ``off`` / ``none``: never weight.
  - ``kde`` / ``bins``: force that specific strategy.

  Returns ``(None, None, info)`` when no weighting applies. ``edges``/``weights`` (when present) plug
  straight into :func:`apply_bin_weights`.
  """
  if mode in ("off", "none"):
    strategy, info = "none", {"reason": f"reweighting disabled (mode={mode})"}
  elif mode == "auto":
    strategy, info = choose_weighting_strategy(y_all, n_bins=n_bins)
  elif mode == "on":
    strategy, info = choose_weighting_strategy(y_all, n_bins=n_bins)
    if strategy == "none":  # forced on: balanced target, but pick a strategy by continuity anyway
      n_rows, n_uniq = info.get("n", len(y_all)), info.get("n_unique", 0)
      strategy = "kde" if (n_rows >= 1000 and n_uniq >= 20) else "bins"
      info = {**info, "reason": f"forced on despite balanced target (cv={info.get('cv')})"}
  elif mode in ("kde", "bins"):
    strategy, info = mode, {"reason": f"forced by mode={mode}"}
  else:
    raise ValueError(f"unknown weighting mode: {mode!r}")

  if strategy == "none":
    return None, None, {"strategy": "none", **info}
  if strategy == "bins":
    edges, weights = regression_weights_from_y(y_all, n_bins=n_bins, max_weight=max_weight)
  else:
    edges, weights = density_weights_from_y(y_all, alpha=alpha, max_weight=max_weight, n_grid=n_grid)
  info = {
    "strategy": strategy,
    "max_weight": max_weight,
    "weight_range": [float(weights.min()), float(weights.max())],
    **({"alpha": alpha, "n_grid": n_grid} if strategy == "kde" else {"n_bins": n_bins}),
    **info,
  }
  return edges, weights, info


class IndexDataIter(xgb.DataIter):
  """Streams rows of an in-RAM :class:`~olinda.data.matrix.ReferenceMatrix` selected by index.

  The multi-column counterpart of :class:`H5DataIter`: instead of reading a per-column HDF5 split
  contiguously, it gathers this column's rows from the one resident copy of the reference library.
  Only a single batch is float32 at a time, so peak memory is the library (uint8) plus one batch.
  """

  def __init__(
    self,
    matrix,
    y: np.ndarray,
    idx: np.ndarray,
    batch_rows: int = 65536,
    bin_edges: np.ndarray | None = None,
    bin_weights: np.ndarray | None = None,
  ) -> None:
    super().__init__()
    self.matrix = matrix
    self.idx = np.asarray(idx)
    self.y = np.asarray(y, dtype=np.float32)
    self.batch_rows = int(batch_rows)
    self.bin_edges = bin_edges
    self.bin_weights = bin_weights
    self._n = int(len(self.idx))
    self._pos = 0

  @property
  def n_rows(self) -> int:
    return self._n

  @property
  def n_cols(self) -> int:
    return int(self.matrix.n_cols)

  def reset(self) -> None:
    self._pos = 0

  def next(self, input_data) -> bool:  # type: ignore[override]
    if self._pos >= self._n:
      return False
    j = min(self._pos + self.batch_rows, self._n)
    rows = self.idx[self._pos : j]
    x = self.matrix.gather(rows)
    y = self.y[rows]
    kwargs = {"data": x, "label": y}
    if self.bin_edges is not None and self.bin_weights is not None:
      kwargs["weight"] = apply_bin_weights(y, self.bin_edges, self.bin_weights)
    input_data(**kwargs)
    self._pos = j
    return True

  def close(self) -> None:
    return None


class H5DataIter(xgb.DataIter):
  """Streams an HDF5 split (`x` float32 (m, dim), `y` float32 (m,)) into XGBoost sequentially.

  The rows are already shuffled on disk (see ``split_reference_to_h5``), so a plain sequential scan
  feeds XGBoost well-mixed data with fast contiguous reads — no reshuffle needed. Used to build a
  ``QuantileDMatrix`` without loading the whole matrix into RAM.
  """

  def __init__(
    self,
    h5_path: str,
    batch_rows: int = 65536,
    bin_edges: np.ndarray | None = None,
    bin_weights: np.ndarray | None = None,
  ) -> None:
    super().__init__()
    import h5py

    self.h5_path = str(h5_path)
    self.batch_rows = int(batch_rows)
    self.bin_edges = bin_edges
    self.bin_weights = bin_weights
    self._f = h5py.File(self.h5_path, "r")
    self._x = self._f["x"]
    self._y = self._f["y"]
    self._n = int(self._x.shape[0])
    self._dim = int(self._x.shape[1])
    self._pos = 0

  @property
  def n_rows(self) -> int:
    return self._n

  @property
  def n_cols(self) -> int:
    return self._dim

  def reset(self) -> None:
    self._pos = 0

  def next(self, input_data) -> bool:  # type: ignore[override]
    if self._pos >= self._n:
      return False
    j = min(self._pos + self.batch_rows, self._n)
    x = np.asarray(self._x[self._pos : j], dtype=np.float32)  # contiguous sequential read
    y = np.asarray(self._y[self._pos : j], dtype=np.float32).reshape(-1)
    kwargs = {"data": x, "label": y}
    if self.bin_edges is not None and self.bin_weights is not None:
      kwargs["weight"] = apply_bin_weights(y, self.bin_edges, self.bin_weights)
    input_data(**kwargs)
    self._pos = j
    return True

  def close(self) -> None:
    with suppress(Exception):
      self._f.close()
