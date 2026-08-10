"""Value-stratified train/val split of the reference library into shuffled HDF5 datasets.

Sorts the reference compounds by their teacher value, takes equally-spaced (by rank) samples as a
validation set — so val uniformly covers the value distribution — shuffles train and val
independently, and writes ``train.h5`` / ``val.h5`` (datasets ``x`` float32 ``(m, dim)`` and ``y``
float32 ``(m,)``) to the working directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from olinda.console import LiveBar, echo, rule, set_active_color, summary_panel

_SPLIT_COLOR = "green"


def _write_split_h5(
  X, y, idx, out_path: Path, batch_rows: int, label: str, attrs: dict | None = None
) -> None:
  """Write rows ``idx`` (in shuffled order) from the in-memory matrix ``X`` to ``out_path``.

  X is already resident in RAM, so gathering shuffled rows is a fast in-memory fancy index and
  every H5 write is sequential — far faster than scattered fancy-index reads straight from disk.
  """
  import h5py

  m = len(idx)
  dim = X.shape[1]
  with h5py.File(out_path, "w") as out:
    xout = out.create_dataset("x", shape=(m, dim), dtype="float32")
    yout = out.create_dataset("y", shape=(m,), dtype="float32")
    for k, v in (attrs or {}).items():
      out.attrs[k] = v
    with LiveBar(f"writing {label}.h5", m, color=_SPLIT_COLOR) as bar:
      for j in range(0, m, batch_rows):
        sl = idx[j : j + batch_rows]
        xout[j : j + len(sl)] = X[sl]
        yout[j : j + len(sl)] = y[sl]
        bar.update(min(j + len(sl), m))


def split_reference_to_indices(
  y, val_frac: float = 0.1, seed: int = 42, limit: int | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
  """Value-stratified train/val split of the reference library, returned as row indices.

  The split is the same one :func:`split_reference_to_h5` performs — rows sorted by value with
  equally-spaced-by-rank validation, then shuffled — but nothing is read or written. Multi-column runs
  need one of these per column while sharing a single copy of the descriptor matrix, so materialising
  the features per column (~11 GB each) is avoided entirely.

  Parameters
  ----------
  y : array-like
      Teacher value per reference row. Non-finite entries are excluded from both splits.
  val_frac : float, optional
      Validation fraction, taken equally-spaced by value rank.
  seed : int, optional
      Shuffle seed.
  limit : int, optional
      Consider only the first N reference rows (development subsampling).

  Returns
  -------
  (np.ndarray, np.ndarray, dict)
      Shuffled ``train_idx`` and ``val_idx`` (int64, indices into the reference library), and an
      ``info`` dict with ``n_total``, ``n_used``, ``n_dropped``, ``vmin`` and ``vmax``.
  """
  y = np.asarray(y, dtype=np.float32)
  n_total = len(y)
  if limit is not None and limit < n_total:
    n_total = int(limit)
    y = y[:n_total]

  valid = np.where(np.isfinite(y))[0]
  m = len(valid)
  if m < 2:
    raise ValueError("need at least 2 finite reference-calcs values to split")

  order = valid[np.argsort(y[valid], kind="stable")]
  n_val = max(1, min(int(round(m * val_frac)), m - 1))
  val_pos = np.unique(np.round(np.linspace(0, m - 1, n_val)).astype(np.int64))
  is_val = np.zeros(m, dtype=bool)
  is_val[val_pos] = True

  rng = np.random.default_rng(seed)
  train_idx = rng.permutation(order[~is_val])
  val_idx = rng.permutation(order[is_val])
  info = {
    "n_total": int(n_total),
    "n_used": int(m),
    "n_dropped": int(n_total - m),
    "vmin": float(y[valid].min()),
    "vmax": float(y[valid].max()),
  }
  return train_idx.astype(np.int64), val_idx.astype(np.int64), info


def split_reference_to_h5(
  descriptors_h5: str | Path,
  y,
  out_dir: str | Path = ".",
  val_frac: float = 0.1,
  seed: int = 42,
  batch_rows: int = 50_000,
  limit: int | None = None,
  feature_attrs: dict | None = None,
) -> tuple[Path, Path]:
  """Write a value-stratified, shuffled train/val split of the reference library to HDF5.

  Parameters
  ----------
  descriptors_h5 : str or Path
      Reference-library HDF5 with a float32 ``data`` dataset of shape ``(n, dim)``.
  y : array-like
      Teacher value per row, length ``n`` (non-finite entries are dropped).
  out_dir : str or Path, optional
      Directory to write ``train.h5`` / ``val.h5`` into (default: current directory).
  val_frac : float, optional
      Validation fraction (equally-spaced by value rank).
  seed : int, optional
      Shuffle seed.
  batch_rows : int, optional
      Streaming batch size (bounds memory).
  limit : int, optional
      Use only the first N reference rows (development).

  Returns
  -------
  (Path, Path)
      Paths to ``train.h5`` and ``val.h5``.
  """
  import h5py

  set_active_color(_SPLIT_COLOR)
  descriptors_h5 = Path(descriptors_h5)
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  y = np.asarray(y, dtype=np.float32)

  with h5py.File(descriptors_h5, "r") as f:
    data = f["data"]
    n_total = int(data.shape[0])
    if len(y) != n_total:
      raise ValueError(f"reference-calcs length {len(y)} != descriptors rows {n_total}")
    if limit is not None and limit < n_total:
      n_total = int(limit)
      y = y[:n_total]

    rule("olinda · reference split", right=f"{n_total:,} compounds")

    train_shuf, val_shuf, info = split_reference_to_indices(y, val_frac=val_frac, seed=seed)
    m, n_dropped = info["n_used"], info["n_dropped"]
    vmin, vmax = info["vmin"], info["vmax"]
    if n_dropped:
      echo(f"Dropped {n_dropped:,} non-finite value(s)", "warning")
    echo(
      f"Value-stratified split · {val_frac:.0%} validation ({len(train_shuf):,} train · {len(val_shuf):,} val)",
      "run",
    )

    echo("Loading descriptors into memory", "run")
    x_all = np.asarray(data[:n_total], dtype=np.float32)  # one sequential read (fast)

  # File closed; X is in RAM — shuffled gather + sequential writes.
  train_path = out_dir / "train.h5"
  val_path = out_dir / "val.h5"
  _write_split_h5(x_all, y, train_shuf, train_path, batch_rows, "train", attrs=feature_attrs)
  _write_split_h5(x_all, y, val_shuf, val_path, batch_rows, "val", attrs=feature_attrs)

  summary_panel(
    "olinda · reference split",
    [
      ("Total", f"{n_total:,}"),
      ("Used", f"[bold]{m:,}[/]" + (f"  [dim]({n_dropped:,} dropped)[/]" if n_dropped else "")),
      ("Train", f"[bold]{len(train_shuf):,}[/]  [dim]shuffled[/]"),
      ("Val", f"[bold]{len(val_shuf):,}[/]  [dim]{val_frac:.0%} · stratified · shuffled[/]"),
      ("Value range", f"{vmin:.4g} … {vmax:.4g}"),
      ("Output", f"[dim]{train_path}[/]  ·  [dim]{val_path}[/]"),
    ],
    border_style=_SPLIT_COLOR,
    icon="✓",
  )
  return train_path, val_path
