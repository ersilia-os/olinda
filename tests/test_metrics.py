"""Tests for the shared regression metrics used by learn-soft, pipeline, and robustness."""

from __future__ import annotations

import json

import numpy as np
import pytest

from olinda.metrics import (
  _mae,
  average_precision,
  binary_metrics,
  enrichment_factor,
  json_safe,
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


# ── Binary metrics ───────────────────────────────────────────────────────────
#
# Checked against scikit-learn where one exists. sklearn is only a transitive dependency (via
# lazy-qsar's [fit] extra), so it is an oracle for the tests and never imported by olinda itself.


def test_auroc_matches_sklearn():
  from sklearn.metrics import roc_auc_score

  rng = np.random.default_rng(0)
  y = (rng.random(500) < 0.2).astype(int)
  s = np.clip(rng.normal(0.3 + 0.4 * y, 0.2), 0, 1)
  assert binary_metrics(y, s)["auroc"] == pytest.approx(roc_auc_score(y, s), abs=1e-9)


def test_average_precision_matches_sklearn():
  from sklearn.metrics import average_precision_score

  rng = np.random.default_rng(1)
  y = (rng.random(400) < 0.1).astype(int)
  s = np.clip(rng.normal(0.4 + 0.3 * y, 0.25), 0, 1)
  assert average_precision(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-9)


def test_tied_scores_do_not_look_like_separation():
  """A model that scores every compound identically separates nothing — AUROC must be 0.5.

  Stepping through the sorted array row by row would instead credit it for the order the labels
  happened to arrive in, which is the classic tie-handling bug.
  """
  y = np.array([1, 0, 1, 0, 1, 0])
  s = np.full(6, 0.7)
  assert binary_metrics(y, s)["auroc"] == pytest.approx(0.5)


def test_perfect_and_inverted_rankings():
  y = np.array([0, 0, 0, 1, 1, 1])
  assert binary_metrics(y, np.arange(6.0))["auroc"] == pytest.approx(1.0)
  assert binary_metrics(y, -np.arange(6.0))["auroc"] == pytest.approx(0.0)


def test_one_class_is_nan_not_an_exception():
  """A validation slice can legitimately contain no actives; AUROC is undefined, not an error."""
  out = binary_metrics(np.zeros(10, dtype=int), np.linspace(0, 1, 10))
  assert np.isnan(out["auroc"]) and np.isnan(out["average_precision"])
  assert out["n_positive"] == 0
  json.dumps(json_safe(out))  # still serializable, with the nans nulled


def test_enrichment_is_relative_to_chance():
  """1.0 is chance; the ceiling is 1/hit_rate, so a perfect ranking hits exactly that."""
  y = np.r_[np.ones(10, dtype=int), np.zeros(90, dtype=int)]
  perfect = np.r_[np.ones(10), np.zeros(90)]
  assert enrichment_factor(y, perfect, 0.10) == pytest.approx(1.0 / 0.10)  # every active in the top 10%
  rng = np.random.default_rng(3)
  chance = np.mean([enrichment_factor(y, rng.random(100), 0.10) for _ in range(400)])
  assert chance == pytest.approx(1.0, abs=0.15)


def test_binary_metrics_is_json_serializable():
  rng = np.random.default_rng(4)
  y = (rng.random(200) < 0.3).astype(int)
  out = binary_metrics(y, rng.random(200))
  assert set(out) == {"n", "n_positive", "hit_rate", "auroc", "average_precision", "enrichment"}
  assert set(out["enrichment"]) == {"top_0.01", "top_0.05", "top_0.1"}
  json.dumps(out)
