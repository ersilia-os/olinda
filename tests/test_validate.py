"""Tests for validation metrics, JSON sanitization, and OOF calibration."""
from __future__ import annotations

import json

import numpy as np
import pytest

from olinda.validate import (
  _concordance_index,
  _json_safe,
  _mae,
  _oof_calibrated,
  _pearsonr,
  _r2,
  _rmse,
  _spearmanr,
)


# ── Metric correctness on known inputs ───────────────────────────────────────


def test_mae_rmse_exact():
  y = np.array([0.0, 0.0])
  p = np.array([3.0, 4.0])
  assert _mae(y, p) == pytest.approx(3.5)
  assert _rmse(y, p) == pytest.approx(np.sqrt((9 + 16) / 2))


def test_perfect_fit_metrics():
  y = np.array([1.0, 2.0, 3.0, 4.0])
  assert _mae(y, y) == 0.0
  assert _rmse(y, y) == 0.0
  assert _r2(y, y) == pytest.approx(1.0)
  assert _pearsonr(y, y) == pytest.approx(1.0)
  assert _spearmanr(y, y) == pytest.approx(1.0)
  assert _concordance_index(y, y) == pytest.approx(1.0)


def test_spearman_is_rank_based():
  y = np.array([1.0, 2.0, 3.0, 4.0])
  p = np.array([10.0, 20.0, 30.0, 40.0])  # monotone but different scale
  assert _spearmanr(y, p) == pytest.approx(1.0)
  assert _spearmanr(y, -p) == pytest.approx(-1.0)


def test_degenerate_inputs_return_nan():
  const = np.array([5.0, 5.0, 5.0])
  assert np.isnan(_r2(const, const))  # ss_tot == 0
  assert np.isnan(_pearsonr(const, np.array([1.0, 2.0, 3.0])))


# ── JSON sanitization (A3) ───────────────────────────────────────────────────


def test_json_safe_replaces_non_finite():
  obj = {
    "a": 1.0,
    "b": float("nan"),
    "c": float("inf"),
    "d": [1.0, float("-inf"), {"e": float("nan")}],
    "f": "text",
  }
  safe = _json_safe(obj)
  assert safe["a"] == 1.0
  assert safe["b"] is None
  assert safe["c"] is None
  assert safe["d"] == [1.0, None, {"e": None}]
  assert safe["f"] == "text"
  # Must now serialize under strict JSON (no NaN/Infinity tokens).
  json.dumps(safe, allow_nan=False)


# ── Out-of-fold calibration (A1) ─────────────────────────────────────────────


def test_oof_returns_none_for_tiny_input():
  p = np.arange(5, dtype=np.float32)
  y = np.arange(5, dtype=np.float32)
  assert _oof_calibrated(p, y, k=5) is None


def test_oof_shape_and_finiteness():
  rng = np.random.default_rng(0)
  p = rng.standard_normal(200).astype(np.float32)
  y = (p * 0.8 + rng.standard_normal(200).astype(np.float32) * 0.3).astype(np.float32)
  out = _oof_calibrated(p, y, k=5, seed=0)
  assert out is not None
  assert out.shape == p.shape
  assert np.all(np.isfinite(out))


def test_oof_is_not_in_sample_optimistic():
  """OOF MAE should not beat the in-sample calibrated MAE (no leakage)."""
  from olinda.calibrate import IsotonicCalibrator

  rng = np.random.default_rng(1)
  p = rng.standard_normal(400).astype(np.float32)
  y = (p * 0.6 + rng.standard_normal(400).astype(np.float32) * 0.6).astype(np.float32)

  in_sample = IsotonicCalibrator().fit(p, y).transform(p)
  oof = _oof_calibrated(p, y, k=5, seed=0)

  # In-sample fitting can only look as good or better than honest OOF.
  assert _mae(y, oof) >= _mae(y, in_sample) - 1e-6
