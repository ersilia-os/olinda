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
      The descriptor matrix, ``(n, dim)``, normally uint8 as stored in ``erl0_morgan.h5``. May be a
      row *prefix* of the library rather than all of it — see :meth:`load`.
  n_file_rows : int, optional
      How many rows the source library actually has, when *x* is only a prefix of it. Defaults to
      ``len(x)``, i.e. "this is the whole thing".

  Attributes
  ----------
  n_rows, n_cols : int
      Shape of the matrix held in RAM — what every consumer should size its work from.
  n_file_rows : int
      Rows in the library on disk. Only :meth:`assert_matches` cares: identity is a property of the
      file, not of how much of it was read.
  """

  def __init__(self, x: np.ndarray, n_file_rows: int | None = None) -> None:
    self.x = x
    self.n_rows = int(x.shape[0])
    self.n_cols = int(x.shape[1])
    self.n_file_rows = self.n_rows if n_file_rows is None else int(n_file_rows)

  @classmethod
  def load(cls, descriptors_h5: str | Path, dataset: str = "data", limit: int | None = None):
    """Read the descriptor dataset into RAM in one sequential pass.

    Parameters
    ----------
    limit : int, optional
      Read only the first *limit* rows. This is how ``--max-samples`` reaches every step that sweeps
      the library: the split indices it produces are all ``< limit`` (the flag truncates the head —
      see :func:`olinda.data.split.split_reference_to_indices`), so bounding the resident matrix
      bounds the hard-label scoring, the calibration and T too, without any of
      them needing to know a limit exists. ``None`` reads everything, which is what a real run does.
    """
    import h5py

    with h5py.File(str(descriptors_h5), "r") as f:
      ds = f[dataset]
      # ds[:None] is the whole dataset, so the unlimited path is exactly what it always was.
      return cls(ds[:limit], n_file_rows=int(ds.shape[0]))

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

    Checked against the row count of the **file**, not of the resident matrix: reading a prefix is a
    deliberate choice by the caller (``--max-samples``), while a file of the wrong length is a
    changed library. Both would otherwise look identical here.
    """
    want_rows, want_dim = reference.get("n_rows"), reference.get("dim")
    if want_rows is not None and self.n_file_rows != want_rows:
      raise ValueError(
        f"reference library has {self.n_file_rows:,} rows but this run was prepared against "
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
