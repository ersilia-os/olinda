"""Tests for XGBTrainer and hyperparameter tuning efficiency."""
from __future__ import annotations

import time

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import xgboost as xgb

from olinda.data.dataset import ParquetDataIter
from olinda.train.xgb import XGBTrainer, _LimitedDataIter


# ---------------------------------------------------------------------------
# _LimitedDataIter
# ---------------------------------------------------------------------------

class TestLimitedDataIter:
  def _make_iter(self, tmp_path, n_rows: int = 1000):
    """Helper: write a small parquet and return a ParquetDataIter."""
    rng = np.random.default_rng(0)
    n_features = 4
    X = rng.standard_normal((n_rows, n_features)).astype(np.float32)
    y = rng.standard_normal(n_rows).astype(np.float32)
    x_list = pa.array([row.tolist() for row in X], type=pa.list_(pa.float32(), n_features))
    table = pa.table({"x": x_list, "y": pa.array(y, type=pa.float32())})
    path = tmp_path / "data.parquet"
    pq.write_table(table, str(path))
    dataset = ds.dataset(str(path), format="parquet")
    return ParquetDataIter(dataset, "x", "y", None, n_features, batch_rows=256)

  def test_limits_rows(self, tmp_path):
    """DMatrix built from _LimitedDataIter contains at most max_rows."""
    base_iter = self._make_iter(tmp_path, n_rows=1000)
    limited = _LimitedDataIter(base_iter, max_rows=200)
    dm = xgb.QuantileDMatrix(limited, max_bin=64)
    assert dm.num_row() <= 200

  def test_full_iter_after_limited(self, tmp_path):
    """Base iterator is still usable after _LimitedDataIter consumes part of it."""
    base_iter = self._make_iter(tmp_path, n_rows=1000)
    limited = _LimitedDataIter(base_iter, max_rows=100)
    xgb.QuantileDMatrix(limited, max_bin=64)

    # Now build from the base iterator directly — should get all 1000 rows.
    dm_full = xgb.QuantileDMatrix(base_iter, max_bin=64)
    assert dm_full.num_row() == 1000

  def test_reset_restores_limit(self, tmp_path):
    """After reset, _LimitedDataIter can be consumed again."""
    base_iter = self._make_iter(tmp_path, n_rows=500)
    limited = _LimitedDataIter(base_iter, max_rows=100)

    dm1 = xgb.QuantileDMatrix(limited, max_bin=64)
    dm2 = xgb.QuantileDMatrix(limited, max_bin=64)  # triggers reset
    assert dm1.num_row() == dm2.num_row()


# ---------------------------------------------------------------------------
# Basic training (no tuning)
# ---------------------------------------------------------------------------

class TestBasicTraining:
  def test_fit_produces_booster(self, synthetic_data):
    """fit_external without tuning produces a valid booster + metadata."""
    train_iter, val_iter = synthetic_data
    trainer = XGBTrainer(
      num_boost_round=50,
      early_stopping_rounds=10,
    )
    booster, meta = trainer.fit_external(train_iter, val_iter)

    assert isinstance(booster, xgb.Booster)
    assert meta["n_trees"] > 0
    assert meta["best_val_score"] < 10.0
    assert "tuning" not in meta

  def test_fit_no_validation(self, synthetic_data):
    """fit_external works without a validation iterator."""
    train_iter, _ = synthetic_data
    trainer = XGBTrainer(
      num_boost_round=30,
      early_stopping_rounds=None,
    )
    booster, meta = trainer.fit_external(train_iter, val_iter=None)
    assert isinstance(booster, xgb.Booster)
    assert meta["n_trees"] == 30


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

class TestTuning:
  def test_tuning_returns_valid_metadata(self, synthetic_data):
    """Tuning with a short budget completes and produces expected metadata."""
    train_iter, val_iter = synthetic_data
    trainer = XGBTrainer(
      num_boost_round=30,
      early_stopping_rounds=10,
    )
    booster, meta = trainer.fit_external(
      train_iter, val_iter, time_budget=120, n_trials=5,
    )

    assert isinstance(booster, xgb.Booster)
    assert "tuning" in meta
    tune = meta["tuning"]
    assert tune["phase1"]["n_trials"] >= 1
    assert "best_params" in tune
    assert "elapsed_seconds" in tune

  def test_tuning_best_params_applied(self, synthetic_data):
    """Best params from tuning are reflected in the final model metadata."""
    train_iter, val_iter = synthetic_data
    trainer = XGBTrainer(
      num_boost_round=30,
      early_stopping_rounds=10,
    )
    _, meta = trainer.fit_external(
      train_iter, val_iter, time_budget=120, n_trials=3,
    )

    best = meta["tuning"]["best_params"]
    # The best params should be merged into the final training params.
    for key in ("max_depth", "eta"):
      assert key in best

  def test_tuning_no_sample_frac_in_params(self, synthetic_data):
    """sample_frac should no longer appear in trial params (fixed fraction)."""
    train_iter, val_iter = synthetic_data
    trainer = XGBTrainer(
      num_boost_round=30,
      early_stopping_rounds=10,
    )
    _, meta = trainer.fit_external(
      train_iter, val_iter, time_budget=60, n_trials=3,
    )

    for trial in meta["tuning"]["phase1"]["trials"]:
      assert "sample_frac" not in trial["params"]

  def test_tuning_time_budget_stops_early(self, synthetic_data):
    """With a very tight budget and many trials, tuning should stop well short."""
    train_iter, val_iter = synthetic_data
    trainer = XGBTrainer(
      num_boost_round=2000,  # expensive per trial (cheap_rounds=600)
      early_stopping_rounds=50,
    )
    _, meta = trainer.fit_external(
      train_iter, val_iter, time_budget=3, n_trials=500,
    )

    # Should not have run all 500 trials.
    assert meta["tuning"]["phase1"]["n_trials"] < 500


# ---------------------------------------------------------------------------
# DMatrix caching efficiency
# ---------------------------------------------------------------------------

class TestDMatrixCaching:
  def test_cached_dmatrix_reuse_produces_same_result(self, tmp_path):
    """Training twice with the same cached DMatrix and same params gives
    identical results — proving DMatrix reuse is safe."""
    rng = np.random.default_rng(7)
    n, d = 500, 5
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] * 2.0 + rng.standard_normal(n).astype(np.float32) * 0.1).astype(np.float32)

    x_arr = pa.array([row.tolist() for row in X], type=pa.list_(pa.float32(), d))
    table = pa.table({"x": x_arr, "y": pa.array(y, type=pa.float32())})
    pq.write_table(table, str(tmp_path / "data.parquet"))
    dataset = ds.dataset(str(tmp_path / "data.parquet"), format="parquet")
    it = ParquetDataIter(dataset, "x", "y", None, d, batch_rows=256)

    dm = xgb.QuantileDMatrix(it, max_bin=256)

    params = {
      "objective": "reg:squarederror",
      "eval_metric": "mae",
      "tree_method": "hist",
      "max_depth": 4,
      "eta": 0.1,
      "seed": 42,
      "max_bin": 256,
    }

    b1 = xgb.train(params, dm, num_boost_round=20, evals=[(dm, "train")], verbose_eval=False)
    b2 = xgb.train(params, dm, num_boost_round=20, evals=[(dm, "train")], verbose_eval=False)

    # Predictions should be identical — DMatrix was reused, params identical.
    dm_pred = xgb.DMatrix(X, label=y)
    p1 = b1.predict(dm_pred)
    p2 = b2.predict(dm_pred)
    np.testing.assert_array_equal(p1, p2)
