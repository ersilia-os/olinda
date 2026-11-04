"""Regression-only pieces of the hard head: label scaling and the check that its ONNX export is usable."""

from __future__ import annotations

import numpy as np


def _label_scale(y: np.ndarray) -> float:
    """What to divide continuous labels by so the head trains on values of order one.

    Dividing by a positive constant changes nothing the pipeline keeps: the isotonic map onto the teacher's
    scale is monotone and is fitted to the scaled head's own output. What it does change is numerics. The
    exported Tweedie graph carries an offset of about the labels' mean, and float32 resolves a spread of
    order one only up to an offset of a few hundred; labels in the millions were rejected for exactly
    that. Squared-error and pseudo-Huber heads also stop depending on the labels' unit.
    """
    scale = float(np.mean(np.abs(y)))
    return scale if np.isfinite(scale) and scale > 0 else 1.0


def _regressor_margin(model, x) -> np.ndarray:
    """A hard regressor's raw score: the booster's margin, before any link function."""
    return np.asarray(
        model.booster_.inplace_predict(
            np.asarray(x, dtype=np.float32),
            iteration_range=(0, int(model.best_iteration_) + 1),
            predict_type="margin",
        ),
        dtype=np.float64,
    ).ravel()


def _check_regressor_export(hard_model, artifact, X) -> None:
    """Refuse a regression head whose saved ONNX does not follow the booster it was exported from.

    ``H`` is shipped as its ONNX, so the library is scored through the ONNX (see :func:`_score_reference`)
    and the isotonic map is fitted to exactly what ships. That makes the *level* of the exported score
    irrelevant, and it has to be: for a Tweedie objective, which lazy-qsar picks for skewed non-negative
    labels, the export is the margin plus a constant of its own (the base score, added without the log
    link). What must hold is that the ONNX is the booster's margin up to that constant, because a
    converter that reorders compounds would ship a head that predicts something else. An offset is
    allowed; a difference that varies from compound to compound is not.
    """
    rows = np.asarray(X[: min(len(X), 500)], dtype=np.float32)
    margin = _regressor_margin(hard_model, rows)
    onnx = np.asarray(artifact.run(rows), dtype=np.float64).ravel()
    spread = max(1e-9, float(np.ptp(margin)))
    drift = float(np.ptp(onnx - margin)) / spread
    if not drift < 1e-3:
        raise RuntimeError(
            f"the exported hard-label regressor does not follow its booster (the two differ by {drift:.1e} "
            f"of the score's range from one compound to the next; objective "
            f"{getattr(hard_model, 'params_', {}).get('objective')!r}). Not shipping it."
        )
