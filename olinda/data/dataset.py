from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import xgboost as xgb

from olinda.helpers import logger


class ParquetDistillDataset:
  """Loader for a packed distillation dataset (meta.json + train/ + val/)."""

  def __init__(self, root: str | Path) -> None:
    self.root = Path(root)
    meta_path = self.root / "meta.json"
    if not meta_path.exists():
      raise ValueError(f"missing meta.json in {self.root}")
    with open(meta_path, "r") as fp:
      self.meta = json.load(fp)

    self.x_col: str = self.meta["x_col"]
    self.y_col: str = self.meta["y_col"]
    self.w_col: str | None = self.meta.get("w_col") or "w"
    self.x_dim: int = int(self.meta["x_dim"])

    self.train_dir = self.root / "train"
    self.val_dir = self.root / "val"

    self.train = ds.dataset(str(self.train_dir), format="parquet")
    self.val = ds.dataset(str(self.val_dir), format="parquet")

  def count(self) -> tuple[int, int]:
    return int(self.train.count_rows()), int(self.val.count_rows())


def _as_array(col: pa.Array | pa.ChunkedArray) -> pa.Array:
  if isinstance(col, pa.ChunkedArray):
    return col.combine_chunks()
  return col


def _x_to_2d(x_col: pa.Array | pa.ChunkedArray, x_dim: int) -> np.ndarray:
  arr = _as_array(x_col)
  t = arr.type

  if pa.types.is_fixed_size_list(t):
    if t.list_size != x_dim:
      raise ValueError(f"x_dim mismatch: parquet={t.list_size} expected={x_dim}")
    values = arr.values
    v = np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float32)
    return v.reshape(len(arr), x_dim)

  if pa.types.is_list(t) or pa.types.is_large_list(t):
    py = arr.to_pylist()
    X = np.asarray(py, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != x_dim:
      raise ValueError(f"x shape mismatch from list: got {X.shape}, expected (*,{x_dim})")
    return X

  raise ValueError(f"unsupported x column type: {t}")


class ParquetDataIter(xgb.DataIter):
  """Streams Parquet shards into XGBoost QuantileDMatrix without full RAM load."""

  def __init__(
    self,
    dataset: ds.Dataset,
    x_col: str,
    y_col: str,
    w_col: str | None,
    x_dim: int,
    batch_rows: int = 65536,
    shuffle_row_groups: bool = True,
    seed: int = 42,
  ) -> None:
    super().__init__()
    self.dataset = dataset
    self.x_col = x_col
    self.y_col = y_col
    self.w_col = w_col
    self.x_dim = int(x_dim)
    self.batch_rows = int(batch_rows)
    self.shuffle_row_groups = bool(shuffle_row_groups)
    self.seed = int(seed)

    self._batches: list | None = None
    self._iter = None

  def reset(self) -> None:
    cols = [self.x_col, self.y_col]
    if self.w_col:
      cols.append(self.w_col)

    scanner = self.dataset.scanner(columns=cols, batch_size=self.batch_rows)
    batches = list(scanner.to_batches())

    if self.shuffle_row_groups and len(batches) > 1:
      rng = np.random.default_rng(self.seed)
      order = rng.permutation(len(batches))
      batches = [batches[i] for i in order]

    self._batches = batches
    self._iter = iter(self._batches)

  def next(self, input_data) -> bool:  # type: ignore[override]
    if self._iter is None:
      return False
    try:
      b = next(self._iter)
    except StopIteration:
      return False

    t = pa.Table.from_batches([b])

    X = _x_to_2d(t.column(self.x_col), self.x_dim)
    y = np.asarray(
      _as_array(t.column(self.y_col)).to_numpy(zero_copy_only=False),
      dtype=np.float32,
    ).reshape(-1)

    if self.w_col and self.w_col in t.column_names:
      w = np.asarray(
        _as_array(t.column(self.w_col)).to_numpy(zero_copy_only=False),
        dtype=np.float32,
      ).reshape(-1)
    else:
      w = None

    input_data(data=X, label=y, weight=w)
    return True
