"""Tests for the isotonic calibrator (olinda.calibrate)."""
from __future__ import annotations

import numpy as np
import pytest

from olinda.calibrate import IsotonicCalibrator


def _fit(raw, target):
  return IsotonicCalibrator().fit(
    np.asarray(raw, dtype=np.float32), np.asarray(target, dtype=np.float32)
  )


# ── Mathematical invariants ──────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(5))
def test_transform_is_monotone(seed):
  """A correct isotonic map is non-decreasing in the raw prediction."""
  rng = np.random.default_rng(seed)
  raw = rng.standard_normal(1000)
  target = raw * 0.7 + rng.standard_normal(1000) * 0.5
  cal = _fit(raw, target)

  grid = np.linspace(raw.min() - 1, raw.max() + 1, 500).astype(np.float32)
  out = cal.transform(grid)
  assert np.all(np.diff(out) >= -1e-6), "calibration map must be non-decreasing"


@pytest.mark.parametrize("seed", range(5))
def test_output_within_target_range(seed):
  """Calibrated values stay within the teacher's observed range."""
  rng = np.random.default_rng(seed)
  raw = rng.standard_normal(1000)
  target = raw * 0.7 + rng.standard_normal(1000) * 0.5
  cal = _fit(raw, target)

  # Includes out-of-range queries, which must clip to the anchor endpoints.
  grid = np.linspace(raw.min() - 5, raw.max() + 5, 500).astype(np.float32)
  out = cal.transform(grid)
  assert out.min() >= target.min() - 1e-5
  assert out.max() <= target.max() + 1e-5


# ── Hand-computed reference cases ────────────────────────────────────────────


def test_strictly_decreasing_pools_to_mean():
  """A perfectly anti-correlated target collapses to its global mean."""
  cal = _fit([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
  out = cal.transform(np.array([1, 2, 3, 4, 5], dtype=np.float32))
  assert np.allclose(out, 3.0, atol=1e-5)


def test_single_violation_pools_pair():
  """[1,3,2] -> isotonic [1, 2.5, 2.5]."""
  cal = _fit([1, 2, 3], [1, 3, 2])
  out = cal.transform(np.array([1, 2, 3], dtype=np.float32))
  assert np.allclose(out, [1.0, 2.5, 2.5], atol=1e-5)


def test_already_monotone_is_identity():
  cal = _fit([1, 2, 3, 4], [1, 2, 3, 4])
  out = cal.transform(np.array([1, 2, 3, 4], dtype=np.float32))
  assert np.allclose(out, [1, 2, 3, 4], atol=1e-5)


def test_tied_raw_values_pool_targets():
  """Duplicate raw values must map to a single (mean) calibrated value."""
  cal = _fit([1, 1, 2], [2, 4, 10])
  assert np.isclose(cal.transform(np.array([1.0], np.float32))[0], 3.0, atol=1e-5)
  assert np.isclose(cal.transform(np.array([2.0], np.float32))[0], 10.0, atol=1e-5)


# ── Edge cases and error handling ────────────────────────────────────────────


def test_constant_raw_returns_single_anchor():
  cal = _fit([5, 5, 5, 5], [1, 2, 3, 4])
  out = cal.transform(np.array([0.0, 5.0, 10.0], dtype=np.float32))
  assert np.allclose(out, 2.5, atol=1e-5)  # mean of targets


def test_empty_input_raises():
  with pytest.raises(ValueError):
    IsotonicCalibrator().fit(np.array([]), np.array([]))


def test_length_mismatch_raises():
  with pytest.raises(ValueError):
    IsotonicCalibrator().fit(np.array([1.0, 2.0]), np.array([1.0]))


def test_transform_before_fit_raises():
  with pytest.raises(RuntimeError):
    IsotonicCalibrator().transform(np.array([1.0]))


def test_save_load_round_trip(tmp_path):
  rng = np.random.default_rng(0)
  raw = rng.standard_normal(200)
  target = raw * 0.5 + rng.standard_normal(200) * 0.3
  cal = _fit(raw, target)

  path = tmp_path / "calibrator.json"
  cal.save(path)
  loaded = IsotonicCalibrator.load(path)

  grid = np.linspace(raw.min(), raw.max(), 100).astype(np.float32)
  assert np.allclose(cal.transform(grid), loaded.transform(grid), atol=1e-6)


# ── Optional cross-check against scikit-learn (skipped if unavailable) ────────


@pytest.mark.parametrize("seed", range(3))
def test_matches_sklearn_when_available(seed):
  """Where scikit-learn is installed, the fit must match its reference PAVA."""
  sk = pytest.importorskip("sklearn.isotonic")
  rng = np.random.default_rng(seed)
  # Fit both on identical float32 inputs: the calibrator casts to float32, and
  # fitting the reference on float64 would shift step locations near tied values.
  raw = rng.standard_normal(2000).astype(np.float32)
  target = (raw * 0.7 + rng.standard_normal(2000).astype(np.float32) * 0.5).astype(np.float32)

  cal = _fit(raw, target)
  ref = sk.IsotonicRegression(increasing=True, out_of_bounds="clip").fit(raw, target)

  grid = np.r_[raw, np.linspace(raw.min() - 1, raw.max() + 1, 100)].astype(np.float32)
  assert np.max(np.abs(cal.transform(grid) - ref.predict(grid))) < 1e-4
