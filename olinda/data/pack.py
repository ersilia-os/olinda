from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from rich.progress import (
  Progress,
  SpinnerColumn,
  BarColumn,
  TextColumn,
  TimeElapsedColumn,
  TimeRemainingColumn,
  MofNCompleteColumn,
)

from olinda.helpers import logger


def _ensure_dir(p: Path) -> None:
  p.mkdir(parents=True, exist_ok=True)


def _fingerprint(items: dict) -> str:
  s = json.dumps(items, sort_keys=True, default=str).encode("utf-8")
  import hashlib

  return hashlib.sha256(s).hexdigest()[:12]


def _infer_x_dim(table: pa.Table, x_col: str) -> int:
  arr = table.column(x_col)
  t = arr.type
  if pa.types.is_fixed_size_list(t):
    return t.list_size
  if pa.types.is_list(t) or pa.types.is_large_list(t):
    if table.num_rows == 0:
      return 0
    first = arr[0].as_py()
    return len(first) if first is not None else 0
  raise ValueError(f"x column type not supported: {t}")


def _cast_x_to_fixed_list(table: pa.Table, x_col: str, x_dim: int) -> pa.Table:
  x = table.column(x_col)
  if (
    pa.types.is_fixed_size_list(x.type)
    and x.type.list_size == x_dim
    and pa.types.is_float32(x.type.value_type)
  ):
    return table

  def to_row(v):
    if v is None:
      return [0.0] * x_dim
    if isinstance(v, (list, tuple)):
      if len(v) != x_dim:
        raise ValueError(f"bad x length: got {len(v)}, expected {x_dim}")
      return [float(z) for z in v]
    if hasattr(v, "to_pylist"):
      vv = v.to_pylist()
      if len(vv) != x_dim:
        raise ValueError(f"bad x length: got {len(vv)}, expected {x_dim}")
      return [float(z) for z in vv]
    raise ValueError("unsupported x element")

  py = [to_row(v.as_py()) for v in x]
  new_x = pa.FixedSizeListArray.from_arrays(
    pa.array([z for row in py for z in row], type=pa.float32()), x_dim
  )

  cols = []
  names = []
  for name in table.column_names:
    if name == x_col:
      cols.append(new_x)
      names.append(x_col)
    else:
      cols.append(table.column(name))
      names.append(name)
  return pa.table(cols, names=names)


def _coerce_cols(table: pa.Table, y_col: str, w_col: str | None, split_col: str) -> pa.Table:
  cols = {}
  for name in table.column_names:
    cols[name] = table.column(name)
  cols[y_col] = cols[y_col].cast(pa.float32())
  if w_col and w_col in cols:
    cols[w_col] = cols[w_col].cast(pa.float32())
  if split_col in cols:
    cols[split_col] = cols[split_col].cast(pa.int8())
  return pa.table([cols[n] for n in cols.keys()], names=list(cols.keys()))


def _make_split(n: int, val_frac: float, seed: int) -> np.ndarray:
  rng = np.random.default_rng(seed)
  split = np.zeros(n, dtype=np.int8)
  idx = rng.permutation(n)
  n_val = int(round(n * val_frac))
  split[idx[:n_val]] = 1
  return split


def _write_shards(
  out_dir: Path,
  table: pa.Table,
  split_col: str,
  shard_rows: int,
  compression: str,
) -> tuple[int, int]:
  tr_dir = out_dir / "train"
  va_dir = out_dir / "val"
  _ensure_dir(tr_dir)
  _ensure_dir(va_dir)

  split = np.asarray(table.column(split_col).to_numpy(zero_copy_only=False)).astype(np.int8)
  train_idx = np.where(split == 0)[0]
  val_idx = np.where(split == 1)[0]

  def write_subset(idx, folder: Path):
    n = len(idx)
    if n == 0:
      return
    k = 0
    while k < n:
      chunk = idx[k : k + shard_rows]
      subt = table.take(pa.array(chunk))
      fn = folder / f"part-{uuid.uuid4().hex[:10]}.parquet"
      pq.write_table(subt, fn, compression=compression, use_dictionary=False)
      k += len(chunk)

  write_subset(train_idx, tr_dir)
  write_subset(val_idx, va_dir)
  return int(len(train_idx)), int(len(val_idx))


def pack_distill_dataset(
  teacher_parquet: str | Path,
  out_dir: str | Path,
  x_col: str = "x",
  y_soft_col: str = "y",
  w_col: str | None = "w",
  split_col: str = "split",
  x_dim: int = 768,
  val_frac: float = 0.1,
  seed: int = 42,
  shard_rows: int = 200_000,
  compression: str = "zstd",
  hard_labels: str | Path | None = None,
  hard_smiles_col: str = "smiles",
  hard_y_col: str = "y",
  hard_weight: float = 1.0,
  featurizer=None,
  hard_batch_rows: int = 8192,
) -> Path:
  out_dir = Path(out_dir)
  _ensure_dir(out_dir)

  meta_path = out_dir / "meta.json"
  if meta_path.exists():
    with open(meta_path, "r") as fp:
      meta = json.load(fp)
    logger.info(f"Found existing packed dataset: {out_dir} (fingerprint={meta.get('fingerprint')})")
    return out_dir

  teacher_parquet = Path(teacher_parquet)
  logger.panel(f"Packing teacher dataset from {teacher_parquet}", title="PACK")

  dset = ds.dataset(str(teacher_parquet), format="parquet")
  schema = dset.schema
  if x_col not in schema.names or y_soft_col not in schema.names:
    raise ValueError(f"teacher parquet must contain columns: {x_col}, {y_soft_col}")

  scanner = dset.scanner(
    columns=[x_col, y_soft_col]
    + ([w_col] if w_col and w_col in schema.names else [])
    + ([split_col] if split_col in schema.names else [])
  )

  total_rows = dset.count_rows()
  logger.info(f"Teacher rows: {total_rows}")

  with Progress(
    SpinnerColumn(),
    TextColumn("[bold]PACK[/bold]"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
  ) as prog:
    task = prog.add_task("writing shards", total=total_rows)

    tables = []
    seen = 0
    for batch in scanner.to_batches():
      t = pa.Table.from_batches([batch])
      if seen == 0:
        inferred = _infer_x_dim(t, x_col)
        if inferred and inferred != x_dim:
          logger.warning(f"inferred x_dim={inferred}, expected x_dim={x_dim}")
      if split_col not in t.column_names:
        split = _make_split(t.num_rows, val_frac=val_frac, seed=seed + seen)
        t = t.append_column(split_col, pa.array(split, type=pa.int8()))
      t = _cast_x_to_fixed_list(t, x_col=x_col, x_dim=x_dim)
      t = _coerce_cols(
        t, y_col=y_soft_col, w_col=w_col if w_col in t.column_names else None, split_col=split_col
      )
      tables.append(t)
      seen += t.num_rows
      prog.update(task, advance=t.num_rows)

    full = pa.concat_tables(tables, promote=True)
    train_rows, val_rows = _write_shards(
      out_dir, full, split_col=split_col, shard_rows=shard_rows, compression=compression
    )

  hard_rows = 0
  if hard_labels is not None:
    if featurizer is None:
      raise ValueError("hard_labels provided but featurizer is None")
    hard_labels = Path(hard_labels)
    logger.panel(f"Packing hard labels from {hard_labels}", title="PACK HARD")

    if hard_labels.suffix.lower() in (".parquet", ".pq"):
      hdf = pd.read_parquet(str(hard_labels))
    else:
      hdf = pd.read_csv(str(hard_labels))

    if hard_smiles_col not in hdf.columns or hard_y_col not in hdf.columns:
      raise ValueError(f"hard labels must contain columns: {hard_smiles_col}, {hard_y_col}")

    if split_col not in hdf.columns:
      hdf[split_col] = _make_split(len(hdf), val_frac=val_frac, seed=seed).astype(np.int8)

    smiles = hdf[hard_smiles_col].astype(str).to_numpy()
    y = hdf[hard_y_col].to_numpy().astype(np.float32)
    split = hdf[split_col].to_numpy().astype(np.int8)
    w = np.full(len(hdf), float(hard_weight), dtype=np.float32)

    tr_dir = out_dir / "train"
    va_dir = out_dir / "val"

    with Progress(
      SpinnerColumn(),
      TextColumn("[bold]PACK HARD[/bold]"),
      BarColumn(),
      MofNCompleteColumn(),
      TimeElapsedColumn(),
      TimeRemainingColumn(),
    ) as prog:
      task = prog.add_task("featurizing+writing", total=len(smiles))
      i = 0
      while i < len(smiles):
        j = min(i + hard_batch_rows, len(smiles))
        X = featurizer.transform(smiles[i:j].tolist()).astype(np.float32)
        if X.shape[1] != x_dim:
          raise ValueError(f"hard featurizer produced dim={X.shape[1]} expected={x_dim}")
        xs = pa.FixedSizeListArray.from_arrays(pa.array(X.reshape(-1), type=pa.float32()), x_dim)

        t = pa.table({
          x_col: xs,
          y_soft_col: pa.array(y[i:j], type=pa.float32()),
          split_col: pa.array(split[i:j], type=pa.int8()),
          "w": pa.array(w[i:j], type=pa.float32()),
          "source": pa.array(["hard"] * (j - i)),
        })

        s = split[i:j]
        tr_idx = np.where(s == 0)[0]
        va_idx = np.where(s == 1)[0]

        if len(tr_idx):
          pq.write_table(
            t.take(pa.array(tr_idx)),
            tr_dir / f"part-hard-{uuid.uuid4().hex[:10]}.parquet",
            compression=compression,
            use_dictionary=False,
          )
        if len(va_idx):
          pq.write_table(
            t.take(pa.array(va_idx)),
            va_dir / f"part-hard-{uuid.uuid4().hex[:10]}.parquet",
            compression=compression,
            use_dictionary=False,
          )

        hard_rows += j - i
        prog.update(task, advance=(j - i))
        i = j

  meta = {
    "format": "olinda.distill.parquet.v1",
    "x_col": x_col,
    "y_col": y_soft_col,
    "w_col": w_col if w_col else None,
    "split_col": split_col,
    "x_dim": int(x_dim),
    "teacher_rows": int(total_rows),
    "hard_rows": int(hard_rows),
    "train_rows": int(train_rows),
    "val_rows": int(val_rows),
    "val_frac": float(val_frac),
    "seed": int(seed),
    "compression": compression,
    "fingerprint": _fingerprint({
      "teacher": str(teacher_parquet),
      "teacher_size": teacher_parquet.stat().st_size if teacher_parquet.exists() else None,
      "x_dim": int(x_dim),
      "x_col": x_col,
      "y_col": y_soft_col,
      "w_col": w_col,
      "split_col": split_col,
      "val_frac": float(val_frac),
      "seed": int(seed),
      "hard_labels": str(hard_labels) if hard_labels else None,
    }),
  }

  with open(meta_path, "w") as fp:
    json.dump(meta, fp, indent=2)

  logger.info(
    "Packed dataset summary: "
    f"teacher_rows={total_rows} "
    f"hard_rows={hard_rows} "
    f"train_rows={train_rows} "
    f"val_rows={val_rows} "
    f"x_dim={x_dim}"
  )

  logger.success(f"Packed dataset written to {out_dir} (train/val shards, meta.json)")
  return out_dir


def pack_feature_table(
  input_path: str | Path,
  out_dir: str | Path,
  y_col: str = "y",
  split_col: str = "split",
  w_col: str | None = None,
  val_frac: float | None = None,
  seed: int = 42,
  shard_rows: int = 200_000,
  compression: str = "zstd",
) -> Path:
  input_path = Path(input_path)
  out_dir = Path(out_dir)
  _ensure_dir(out_dir)

  meta_path = out_dir / "meta.json"
  if meta_path.exists():
    with open(meta_path, "r") as fp:
      meta = json.load(fp)
    logger.info(f"Found existing packed dataset: {out_dir} (fingerprint={meta.get('fingerprint')})")
    return out_dir

  tr_dir = out_dir / "train"
  va_dir = out_dir / "val"
  _ensure_dir(tr_dir)
  _ensure_dir(va_dir)

  rng = np.random.default_rng(seed)
  x_cols = None
  x_dim = None
  train_rows = 0
  val_rows = 0

  def write_parts(table: pa.Table):
    nonlocal train_rows, val_rows
    split = np.asarray(table.column(split_col).to_numpy(zero_copy_only=False)).astype(np.int8)
    tr_idx = np.where(split == 0)[0]
    va_idx = np.where(split == 1)[0]

    def write_subset(idx, folder: Path):
      if len(idx) == 0:
        return
      k = 0
      while k < len(idx):
        chunk = idx[k : k + shard_rows]
        subt = table.take(pa.array(chunk))
        fn = folder / f"part-{uuid.uuid4().hex[:10]}.parquet"
        pq.write_table(subt, fn, compression=compression, use_dictionary=False)
        k += len(chunk)

    write_subset(tr_idx, tr_dir)
    write_subset(va_idx, va_dir)
    train_rows += int(len(tr_idx))
    val_rows += int(len(va_idx))

  def chunk_to_table(df: pd.DataFrame) -> pa.Table:
    nonlocal x_cols, x_dim
    if y_col not in df.columns:
      raise ValueError(f"missing y column: {y_col}")

    if x_cols is None:
      exclude = {y_col, split_col}
      if w_col:
        exclude.add(w_col)
      num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
      if not num_cols:
        raise ValueError("no numeric feature columns found")
      x_cols = num_cols
      x_dim = len(x_cols)

    X = df[x_cols].to_numpy(dtype=np.float32, copy=False)
    y = df[y_col].to_numpy(dtype=np.float32, copy=False)

    if w_col and w_col in df.columns:
      w = df[w_col].to_numpy(dtype=np.float32, copy=False)
    else:
      w = np.ones(len(X), dtype=np.float32)

    if split_col in df.columns:
      split = df[split_col].to_numpy(dtype=np.int8, copy=False)
    elif val_frac and val_frac > 0:
      split = (rng.random(len(X)) < float(val_frac)).astype(np.int8)
    else:
      split = np.zeros(len(X), dtype=np.int8)

    x_flat = pa.array(X.reshape(-1), type=pa.float32())
    x_list = pa.FixedSizeListArray.from_arrays(x_flat, x_dim)

    return pa.table({
      "x": x_list,
      "y": pa.array(y, type=pa.float32()),
      "w": pa.array(w, type=pa.float32()),
      "split": pa.array(split, type=pa.int8()),
    })

  if input_path.suffix.lower() in (".parquet", ".pq"):
    dset = ds.dataset(str(input_path), format="parquet")
    scanner = dset.scanner(batch_size=shard_rows)
    for batch in scanner.to_batches():
      df = pa.Table.from_batches([batch]).to_pandas()
      table = chunk_to_table(df)
      write_parts(table)
  elif input_path.suffix.lower() in (".csv", ".tsv"):
    sep = "\t" if input_path.suffix.lower() == ".tsv" else ","
    for df in pd.read_csv(str(input_path), sep=sep, chunksize=shard_rows):
      table = chunk_to_table(df)
      write_parts(table)
  else:
    raise ValueError("input must be .csv/.tsv/.parquet")

  meta = {
    "format": "olinda.distill.parquet.v1",
    "x_col": "x",
    "y_col": "y",
    "w_col": "w",
    "split_col": "split",
    "x_dim": int(x_dim) if x_dim else 0,
    "train_rows": int(train_rows),
    "val_rows": int(val_rows),
    "val_frac": float(val_frac) if val_frac is not None else 0.0,
    "seed": int(seed),
    "compression": compression,
    "fingerprint": _fingerprint({
      "input": str(input_path),
      "y_col": y_col,
      "split_col": split_col,
      "w_col": w_col,
      "val_frac": val_frac,
      "seed": seed,
    }),
  }

  with open(meta_path, "w") as fp:
    json.dump(meta, fp, indent=2)

  logger.info(f"Packed dataset summary: train_rows={train_rows} val_rows={val_rows} x_dim={x_dim}")
  logger.success(f"Packed dataset written to {out_dir}")
  return out_dir
