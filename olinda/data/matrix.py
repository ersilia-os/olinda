"""One resident copy of the reference descriptors, shared by every column of a run.

Multi-column runs train K independent students over the *same* feature matrix, so materialising a
per-column split would write ~11 GB of near-identical float32 per column. Instead the library is read
once into RAM as uint8 (~2.8 GB for 1.35M x 2048) and each column addresses it through its own index
arrays.

Reading scattered rows straight from HDF5 is not a viable alternative — measured on the real library,
a 65,536-row gather takes ~49 s from disk versus ~0.05 s from RAM, a ~1000x difference. The whole
library loads in one sequential read in well under a second, so the resident copy is both faster and
simpler. uint8 -> float32 conversion happens per batch, never for the whole matrix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_BATCH_ROWS = 65536


class ReferenceMatrix:
  """The reference-library descriptors held in RAM, addressed by row index.

  Parameters
  ----------
  x : np.ndarray
      The descriptor matrix, ``(n, dim)``, normally uint8 as stored in ``erl0_morgan.h5``.

  Attributes
  ----------
  n_rows, n_cols : int
      Shape of the underlying matrix.
  """

  def __init__(self, x: np.ndarray) -> None:
    self.x = x
    self.n_rows = int(x.shape[0])
    self.n_cols = int(x.shape[1])

  @classmethod
  def load(cls, descriptors_h5: str | Path, dataset: str = "data") -> ReferenceMatrix:
    """Read the whole descriptor dataset into RAM in one sequential pass."""
    import h5py

    with h5py.File(str(descriptors_h5), "r") as f:
      return cls(f[dataset][:])

  def gather(self, idx: np.ndarray, dtype=np.float32, step: int = 8192) -> np.ndarray:
    """Return the rows at ``idx`` as a new array of ``dtype``.

    Filled in sub-chunks straight into the output. The obvious ``np.asarray(self.x[idx], dtype)``
    allocates the whole uint8 selection first and then the converted copy, so peak memory is both at
    once — measured 0.67 GB for a 0.54 GB float32 result, and 1.21 GB for a float64 one.
    """
    idx = np.asarray(idx)
    out = np.empty((len(idx), self.n_cols), dtype=dtype)
    for start in range(0, len(idx), step):
      out[start : start + step] = self.x[idx[start : start + step]]
    return out

  def nbytes(self) -> int:
    return int(self.x.nbytes)

  def assert_matches(self, reference: dict) -> None:
    """Refuse to proceed if this library is not the one a run's indices were computed against.

    Splits are stored as positional row indices, so a library that has been regenerated or swapped
    since ``prepare`` pairs each row's features with a different molecule's label. Training would
    complete and the metrics would look plausible; the model would be meaningless.
    """
    want_rows, want_dim = reference.get("n_rows"), reference.get("dim")
    if want_rows is not None and self.n_rows != want_rows:
      raise ValueError(
        f"reference library has {self.n_rows:,} rows but this run was prepared against "
        f"{want_rows:,} — the library changed. Re-run `olinda prepare`."
      )
    if want_dim is not None and self.n_cols != want_dim:
      raise ValueError(
        f"reference library has {self.n_cols}-d fingerprints but this run was prepared against "
        f"{want_dim}-d — the library changed. Re-run `olinda prepare`."
      )


def index_sequence(matrix: ReferenceMatrix, idx: np.ndarray, batch_rows: int = _BATCH_ROWS):
  """Return an ``lgb.Sequence`` over ``matrix`` restricted to ``idx``.

  LightGBM asks a Sequence for a contiguous slice when it streams rows into the Dataset, and for a
  *single* row — a bare int — when it samples to build the feature bins. The two answers have
  different shapes: 2-D for a slice, 1-D for a row. Built lazily so importing this module does not
  pull lightgbm.
  """
  import lightgbm as lgb

  class _IndexSeq(lgb.Sequence):
    def __init__(self) -> None:
      self.matrix = matrix
      self.idx = np.asarray(idx)
      self.batch_size = int(batch_rows)

    def __getitem__(self, item):
      # LightGBM samples in double for bin construction, so float64 here (not float32).
      if np.isscalar(item) or isinstance(item, np.integer):
        return np.asarray(self.matrix.x[self.idx[item]], dtype=np.float64)
      return self.matrix.gather(self.idx[item], dtype=np.float64)

    def __len__(self) -> int:
      return int(len(self.idx))

  return _IndexSeq()
