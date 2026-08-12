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


def _pava_increasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted Pool Adjacent Violators Algorithm (non-decreasing fit).

    Given target values ``y`` already sorted by their x coordinate (with weights
    ``w``), return the least-squares non-decreasing fit. Uses the standard
    O(n) block-stack formulation: append each point as a singleton block, then
    merge it back into earlier blocks while it violates monotonicity, replacing
    merged blocks with their weighted mean.

    Parameters
    ----------
    y : np.ndarray
        Target values sorted by ascending x.
    w : np.ndarray
        Per-point weights (same length as ``y``).

    Returns
    -------
    np.ndarray
        Non-decreasing fitted values, one per input position.
    """
    n = len(y)
    # Block stacks: value (weighted mean), weight, and number of points.
    bv = np.empty(n, dtype=np.float64)
    bw = np.empty(n, dtype=np.float64)
    bs = np.empty(n, dtype=np.int64)
    top = -1
    for i in range(n):
        top += 1
        bv[top] = y[i]
        bw[top] = w[i]
        bs[top] = 1
        # Merge backwards while the previous block is higher than this one.
        while top > 0 and bv[top - 1] > bv[top]:
            merged_w = bw[top - 1] + bw[top]
            bv[top - 1] = (bv[top - 1] * bw[top - 1] + bv[top] * bw[top]) / merged_w
            bw[top - 1] = merged_w
            bs[top - 1] += bs[top]
            top -= 1

    # Expand pooled block values back to per-position fitted values.
    out = np.empty(n, dtype=np.float64)
    pos = 0
    for k in range(top + 1):
        out[pos : pos + bs[k]] = bv[k]
        pos += bs[k]
    return out


class IsotonicCalibrator:
    """Monotonic piecewise-linear map learned via isotonic regression (PAVA).

    After fitting on (raw_pred, y_true) pairs from the validation set, calling
    ``transform(raw_pred)`` returns calibrated values that:

    - preserve rank ordering
    - stay within the teacher's observed [min, max] range
    - minimize squared error against teacher soft labels
    """

    def __init__(self) -> None:
        self._x: np.ndarray | None = (
            None  # sorted anchor x values (in oriented space, i.e. sign*raw)
        )
        self._y: np.ndarray | None = None  # corresponding isotonic y values
        self._sign: int = 1  # +1 increasing, -1 decreasing (fit on -raw)
        self.rank_correlation: float | None = (
            None  # set when the direction was detected, not given
        )

    @property
    def is_fitted(self) -> bool:
        return self._x is not None

    def fit(
        self, raw: np.ndarray, target: np.ndarray, increasing: bool | str = True
    ) -> "IsotonicCalibrator":
        """Fit an isotonic map raw → target.

        Parameters
        ----------
        increasing : bool or "auto"
            ``True`` (default) fits a non-decreasing map (back-compat). ``False`` fits non-increasing. ``"auto"``
            picks the direction from the sign of the rank (Spearman) correlation between ``raw`` and ``target`` —
            so a low raw value mapping to a high target (inverse relationship) is handled correctly.
        """
        raw = np.asarray(raw, dtype=np.float64).ravel()
        target = np.asarray(target, dtype=np.float64).ravel()
        if len(raw) != len(target):
            raise ValueError("raw and target must have the same length")
        if len(raw) == 0:
            raise ValueError("cannot fit calibrator on empty input")

        # Orientation: a decreasing map is a non-decreasing PAVA fit on -raw. "auto" detects it from the sign of
        # the Spearman correlation (rank-based, robust to the eventual monotone nonlinearity).
        if increasing == "auto":
            # Kept so callers can report the magnitude without paying for a second ranking pass.
            self.rank_correlation = _spearman_sign(raw, target)
            self._sign = -1 if self.rank_correlation < 0 else 1
        else:
            self._sign = 1 if increasing else -1
        raw = self._sign * raw

        # Collapse points that share a raw value: the calibration map is a function
        # of the raw prediction, so tied x must map to a single value. Pool them by
        # mean target with weight = count, then run weighted PAVA over unique x.
        ux, inv = np.unique(raw, return_inverse=True)
        sums = np.zeros_like(ux)
        counts = np.zeros_like(ux)
        np.add.at(sums, inv, target)
        np.add.at(counts, inv, 1)
        grouped = sums / counts

        # Least-squares non-decreasing fit via Pool Adjacent Violators.
        uy = _pava_increasing(grouped, counts.astype(np.float64))

        self._x = ux
        self._y = uy

        logger.debug(
            f"Isotonic calibrator fitted: {len(ux)} anchors, output range [{uy.min():.6f}, {uy.max():.6f}]"
        )
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Apply the fitted monotonic calibration map."""
        if not self.is_fitted:
            raise RuntimeError("calibrator not fitted")
        raw = np.asarray(raw, dtype=np.float64).ravel()
        return np.interp(self._sign * raw, self._x, self._y).astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Save calibrator anchors to JSON."""
        if not self.is_fitted:
            raise RuntimeError("calibrator not fitted")
        path = Path(path)
        data = {
            "type": "isotonic",
            "sign": int(self._sign),
            "x": self._x.tolist(),
            "y": self._y.tolist(),
        }
        with open(path, "w") as fp:
            json.dump(data, fp)
        logger.debug(f"Calibrator saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "IsotonicCalibrator":
        """Load calibrator from JSON."""
        path = Path(path)
        with open(path, "r") as fp:
            data = json.load(fp)
        cal = cls()
        cal._x = np.asarray(data["x"], dtype=np.float64)
        cal._y = np.asarray(data["y"], dtype=np.float64)
        cal._sign = int(data.get("sign", 1))  # older files were increasing-only
        return cal


def _spearman_sign(a: np.ndarray, b: np.ndarray) -> float:
    """Rank (Spearman) correlation between ``a`` and ``b`` (0 if degenerate).

    Named for its use — :meth:`IsotonicCalibrator.fit` only needs the sign to orient the map — but the
    full correlation is returned, since ``learn-hard`` also reports its magnitude. Ties share their
    average rank (see :func:`olinda.metrics.average_ranks`); breaking them by position would inflate the
    correlation, and here that would be deciding the calibration's *direction* on an inflated number.
    """
    from olinda.metrics import average_ranks

    ra = average_ranks(a)
    rb = average_ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum()))
    if denom == 0.0:
        return 0.0
    return float((ra * rb).sum() / denom)
