"""Shared fixtures for olinda tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from olinda.data.dataset import ParquetDataIter


N_FEATURES = 10
_COEF = np.array([0.5, -0.3, 0.8, 0.0, 0.0, 1.2, -0.7, 0.0, 0.4, -0.1], dtype=np.float32)


def _make_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  X = rng.standard_normal((n, N_FEATURES)).astype(np.float32)
  y = (X @ _COEF + rng.standard_normal(n).astype(np.float32) * 0.5).astype(np.float32)
  return X, y


def _write_parquet(X: np.ndarray, y: np.ndarray, path: Path) -> ds.Dataset:
  x_list = pa.array([row.tolist() for row in X], type=pa.list_(pa.float32(), X.shape[1]))
  table = pa.table({"x": x_list, "y": pa.array(y, type=pa.float32())})
  pq.write_table(table, str(path))
  return ds.dataset(str(path), format="parquet")


@pytest.fixture()
def synthetic_data(tmp_path):
  """Create small synthetic train/val parquet datasets and return DataIters."""
  X_train, y_train = _make_data(2000, seed=42)
  X_val, y_val = _make_data(500, seed=123)

  train_ds = _write_parquet(X_train, y_train, tmp_path / "train.parquet")
  val_ds = _write_parquet(X_val, y_val, tmp_path / "val.parquet")

  train_iter = ParquetDataIter(train_ds, "x", "y", None, N_FEATURES, batch_rows=512)
  val_iter = ParquetDataIter(val_ds, "x", "y", None, N_FEATURES, batch_rows=512)

  return train_iter, val_iter
