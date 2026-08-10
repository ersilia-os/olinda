"""Tests for the shared regression metrics used by learn-soft, pipeline, and robustness."""

from __future__ import annotations

import json

import numpy as np
import pytest

from olinda.metrics import (
  _mae,
  _pearsonr,
  _r2,
  _rmse,
  _spearmanr,
  regression_metrics,
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


def test_spearman_is_rank_based():
  y = np.array([1.0, 2.0, 3.0, 4.0])
  p = np.array([10.0, 20.0, 30.0, 40.0])  # monotone but different scale
  assert _spearmanr(y, p) == pytest.approx(1.0)
  assert _spearmanr(y, -p) == pytest.approx(-1.0)


def test_degenerate_inputs_return_nan():
  const = np.array([5.0, 5.0, 5.0])
  assert np.isnan(_r2(const, const))  # ss_tot == 0
  assert np.isnan(_pearsonr(const, np.array([1.0, 2.0, 3.0])))


# ── The public dict written to val_metrics.json ──────────────────────────────


def test_regression_metrics_keys_and_perfect_fit():
  y = np.arange(50, dtype=np.float64)
  out = regression_metrics(y, y)
  assert set(out) == {"n", "mae", "rmse", "r2", "pearson", "spearman", "top_decile_rmse"}
  assert out["n"] == 50
  assert out["mae"] == 0.0
  assert out["rmse"] == 0.0
  assert out["r2"] == pytest.approx(1.0)
  assert out["top_decile_rmse"] == 0.0


def test_regression_metrics_agrees_with_helpers():
  rng = np.random.default_rng(0)
  y = rng.standard_normal(200)
  p = y * 0.7 + rng.standard_normal(200) * 0.4
  out = regression_metrics(y, p)
  assert out["mae"] == pytest.approx(_mae(y, p))
  assert out["rmse"] == pytest.approx(_rmse(y, p))
  assert out["r2"] == pytest.approx(_r2(y, p))
  assert out["pearson"] == pytest.approx(_pearsonr(y, p))
  assert out["spearman"] == pytest.approx(_spearmanr(y, p))


def test_regression_metrics_accepts_2d_and_is_json_serializable():
  """learn-soft passes raw h5 columns, which can arrive as (n, 1)."""
  y = np.arange(20, dtype=np.float32).reshape(-1, 1)
  p = (np.arange(20, dtype=np.float32) + 0.5).reshape(-1, 1)
  out = regression_metrics(y, p)
  assert out["n"] == 20
  assert out["mae"] == pytest.approx(0.5)
  json.dumps(out)  # plain Python floats, directly serializable


def test_top_decile_rmse_targets_the_high_tail():
  """Error confined to the top decile must show up there, and dilute in the global RMSE."""
  y = np.arange(100, dtype=np.float64)
  p = y.copy()
  p[90:] += 2.0  # miss only the highest-valued tenth
  out = regression_metrics(y, p)
  assert out["top_decile_rmse"] == pytest.approx(2.0)
  assert out["rmse"] == pytest.approx(np.sqrt(0.4))  # same error spread over all 100 rows
