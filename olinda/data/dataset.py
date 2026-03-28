from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
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


def _scan_y(dataset: ds.Dataset, y_col: str) -> np.ndarray:
  """Read all y values from the dataset into a 1-D float64 array."""
  chunks = []
  for batch in dataset.scanner(columns=[y_col], batch_size=100_000).to_batches():
    chunks.append(batch.column(y_col).to_numpy(zero_copy_only=False))
  return np.concatenate(chunks).astype(np.float64)


def detect_regression_imbalance(
  dataset: ds.Dataset,
  y_col: str,
  n_bins: int = 20,
  cv_threshold: float = 0.5,
) -> tuple[bool, float, dict]:
  """Detect whether the regression target is imbalanced.

  Uses equal-width bins over the [1st, 99th] percentile range and measures
  the coefficient of variation (CV) of bin counts.  A high CV means some
  value ranges are heavily underrepresented.

  Returns (is_imbalanced, cv_score, details_dict).
  """
  y_all = _scan_y(dataset, y_col)
  n = len(y_all)

  lo, hi = np.percentile(y_all, [1, 99])
  if hi - lo < 1e-12:
    return False, 0.0, {"reason": "near-constant target", "n": n}

  bin_edges = np.linspace(lo, hi, n_bins + 1)
  bin_edges[0] = -np.inf
  bin_edges[-1] = np.inf

  bin_indices = np.digitize(y_all, bin_edges[1:-1])
  bin_counts = np.bincount(bin_indices, minlength=n_bins).astype(np.float64)

  non_empty = bin_counts[bin_counts > 0]
  mean_count = non_empty.mean()
  std_count = non_empty.std()
  cv = float(std_count / mean_count) if mean_count > 0 else 0.0

  empty_frac = float(np.sum(bin_counts == 0)) / n_bins
  imbalance_ratio = float(non_empty.max() / non_empty.min()) if len(non_empty) > 1 else 1.0

  is_imbalanced = cv > cv_threshold or empty_frac > 0.3

  details = {
    "cv": round(cv, 4),
    "cv_threshold": cv_threshold,
    "empty_frac": round(empty_frac, 4),
    "imbalance_ratio": round(imbalance_ratio, 2),
    "n": n,
    "n_bins": n_bins,
    "y_min": float(y_all.min()),
    "y_max": float(y_all.max()),
    "y_mean": float(y_all.mean()),
    "y_std": float(y_all.std()),
  }

  return is_imbalanced, cv, details


def compute_regression_weights(
  dataset: ds.Dataset,
  y_col: str,
  n_bins: int = 20,
  max_weight: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
  """Compute inverse-frequency sample weights using equal-width bins.

  Bins are equal-width over the [1st, 99th] percentile range so that
  sparse value regions genuinely receive higher weight.  Weights are
  capped at *max_weight* to avoid over-boosting extreme outliers.

  Returns (bin_edges, bin_weights).
  """
  y_all = _scan_y(dataset, y_col)

  lo, hi = np.percentile(y_all, [1, 99])
  if hi - lo < 1e-12:
    edges = np.array([-np.inf, np.inf])
    return edges, np.array([1.0], dtype=np.float32)

  bin_edges = np.linspace(lo, hi, n_bins + 1)
  bin_edges[0] = -np.inf
  bin_edges[-1] = np.inf

  bin_indices = np.digitize(y_all, bin_edges[1:-1])
  bin_counts = np.bincount(bin_indices, minlength=n_bins).astype(np.float64)

  total = float(len(y_all))
  with np.errstate(divide="ignore", invalid="ignore"):
    bin_weights = np.where(bin_counts > 0, total / (n_bins * bin_counts), 1.0)
  bin_weights = np.clip(bin_weights, 1.0, max_weight)

  logger.info(
    f"Regression reweighting: {n_bins} equal-width bins, "
    f"weight range [{bin_weights.min():.4f}, {bin_weights.max():.4f}], "
    f"max_weight cap={max_weight}"
  )
  return bin_edges, bin_weights.astype(np.float32)


def apply_bin_weights(
  y: np.ndarray,
  bin_edges: np.ndarray,
  bin_weights: np.ndarray,
) -> np.ndarray:
  """Map each y value to its bin weight."""
  bin_indices = np.digitize(y, bin_edges[1:-1])
  return bin_weights[np.clip(bin_indices, 0, len(bin_weights) - 1)]


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
    bin_edges: np.ndarray | None = None,
    bin_weights: np.ndarray | None = None,
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
    self.bin_edges = bin_edges
    self.bin_weights = bin_weights

    self._iter = None

  def reset(self) -> None:
    cols = [self.x_col, self.y_col]
    if self.w_col:
      cols.append(self.w_col)

    if self.shuffle_row_groups:
      # Shuffle at file (fragment) level — only metadata in memory
      fragments = list(self.dataset.get_fragments())
      if len(fragments) > 1:
        rng = np.random.default_rng(self.seed)
        rng.shuffle(fragments)

      def _lazy():
        for frag in fragments:
          pf = pq.ParquetFile(frag.path)
          for batch in pf.iter_batches(batch_size=self.batch_rows, columns=cols):
            yield batch

      self._iter = _lazy()
    else:
      scanner = self.dataset.scanner(columns=cols, batch_size=self.batch_rows)
      self._iter = scanner.to_batches()

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

    if self.bin_edges is not None and self.bin_weights is not None:
      bw = apply_bin_weights(y, self.bin_edges, self.bin_weights)
      w = bw if w is None else w * bw

    input_data(data=X, label=y, weight=w)
    return True
