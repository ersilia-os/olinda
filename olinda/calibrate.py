"""Post-hoc monotonic calibration for regression predictions.

Fits an isotonic regression from raw XGBoost predictions to teacher soft labels
on the validation set. At inference, maps raw predictions through the learned
monotonic function so outputs stay within the teacher's range.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from olinda.helpers import logger


class IsotonicCalibrator:
  """Monotonic piecewise-linear map learned via isotonic regression (PAVA).

  After fitting on (raw_pred, y_true) pairs from the validation set, calling
  ``transform(raw_pred)`` returns calibrated values that:

  - preserve rank ordering
  - stay within the teacher's observed [min, max] range
  - minimize squared error against teacher soft labels
  """

  def __init__(self) -> None:
    self._x: np.ndarray | None = None  # sorted anchor x values
    self._y: np.ndarray | None = None  # corresponding isotonic y values

  @property
  def is_fitted(self) -> bool:
    return self._x is not None

  def fit(self, raw: np.ndarray, target: np.ndarray) -> "IsotonicCalibrator":
    """Fit isotonic regression: raw predictions → teacher soft labels."""
    raw = np.asarray(raw, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if len(raw) != len(target):
      raise ValueError("raw and target must have the same length")

    # Sort by raw prediction
    order = np.argsort(raw)
    x_sorted = raw[order]
    y_sorted = target[order]

    # Pool Adjacent Violators Algorithm (PAVA) — increasing
    n = len(y_sorted)
    y_iso = y_sorted.copy()
    blocks = list(range(n))  # block[i] = start index of block containing i
    weights = np.ones(n, dtype=np.float64)

    i = 0
    while i < n - 1:
      if y_iso[i] > y_iso[i + 1]:
        # Merge blocks: pool i and i+1
        # Find the block starting at i
        j = i + 1
        # Weighted average
        w_sum = weights[i] + weights[j]
        y_iso[i] = (weights[i] * y_iso[i] + weights[j] * y_iso[j]) / w_sum
        y_iso[j] = y_iso[i]
        weights[i] = w_sum
        weights[j] = 0

        # Propagate merge: collapse j into i
        # Now back up to check previous blocks
        while i > 0 and y_iso[i - 1] > y_iso[i]:
          i -= 1
          w_sum = weights[i] + weights[i + 1]
          y_iso[i] = (weights[i] * y_iso[i] + weights[i + 1] * y_iso[i + 1]) / w_sum
          weights[i] = w_sum
          weights[i + 1] = 0
          # Fill forward
          for k in range(i + 1, n):
            if weights[k] == 0:
              y_iso[k] = y_iso[i]
            else:
              break

        # Fill forward from i
        for k in range(i + 1, n):
          if weights[k] == 0:
            y_iso[k] = y_iso[i]
          else:
            break
        i = k
      else:
        i += 1

    # Deduplicate: keep unique x anchors with their isotonic y
    # For duplicate x values, take the mean y
    ux, inv = np.unique(x_sorted, return_inverse=True)
    uy = np.zeros_like(ux)
    counts = np.zeros_like(ux)
    np.add.at(uy, inv, y_iso)
    np.add.at(counts, inv, 1)
    uy /= counts

    self._x = ux
    self._y = uy

    logger.info(
      f"Isotonic calibrator fitted: {len(ux)} anchors, "
      f"output range [{uy.min():.6f}, {uy.max():.6f}]"
    )
    return self

  def transform(self, raw: np.ndarray) -> np.ndarray:
    """Apply the fitted monotonic calibration map."""
    if not self.is_fitted:
      raise RuntimeError("calibrator not fitted")
    raw = np.asarray(raw, dtype=np.float64).ravel()
    return np.interp(raw, self._x, self._y).astype(np.float32)

  def save(self, path: str | Path) -> None:
    """Save calibrator anchors to JSON."""
    if not self.is_fitted:
      raise RuntimeError("calibrator not fitted")
    path = Path(path)
    data = {
      "type": "isotonic",
      "x": self._x.tolist(),
      "y": self._y.tolist(),
    }
    with open(path, "w") as fp:
      json.dump(data, fp)
    logger.info(f"Calibrator saved to {path}")

  @classmethod
  def load(cls, path: str | Path) -> "IsotonicCalibrator":
    """Load calibrator from JSON."""
    path = Path(path)
    with open(path, "r") as fp:
      data = json.load(fp)
    cal = cls()
    cal._x = np.asarray(data["x"], dtype=np.float64)
    cal._y = np.asarray(data["y"], dtype=np.float64)
    return cal
