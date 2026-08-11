"""The run directory: its manifest, its layout, and the per-column data it holds.

A run distils one or more teacher columns from the same reference library. Every column is trained
independently, so each gets its own directory of artifacts, but they share one copy of the target
vectors and one manifest describing the whole run. A single-column run is simply the one-column case
— there is no separate layout.

    <model-dir>/
      manifest.json      the authoritative description of the run
      targets.h5         one reference-aligned y per column, keyed by column id
      splits.h5          per-column train/val row indices
      columns/<id>/      that column's model, metrics, plots and ground-truth head
      model.onnx         all columns fused into one artifact

The descriptor matrix is deliberately absent: it is the same for every column, so it is read from the
shared reference library rather than copied per run.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MANIFEST_NAME = "manifest.json"
RUN_SCHEMA = "olinda.run.v1"
COLUMNS_DIRNAME = "columns"
TARGETS_NAME = "targets.h5"
SPLITS_NAME = "splits.h5"


def column_id(index: int) -> str:
  """Directory-safe identifier for the *index*-th column.

  Teacher headers are user data — they carry spaces, slashes, and case collisions that are hostile as
  path or HDF5 key components. The real name lives in the manifest and in the ONNX output name.
  """
  return f"c{index}"


def column_dir(model_dir: str | Path, col_id: str) -> Path:
  return Path(model_dir) / COLUMNS_DIRNAME / col_id


def is_run_dir(model_dir: str | Path) -> bool:
  """True iff *model_dir* holds a manifest written by :func:`write_manifest`."""
  return (Path(model_dir) / MANIFEST_NAME).exists()


def write_manifest(model_dir: str | Path, manifest: dict) -> Path:
  """Write the run manifest, atomically and as strict JSON.

  Metrics carry NaN for a degenerate column, and ``json.dump`` writes that as a bare ``NaN`` token
  that only Python accepts — so it is nulled first. The write goes via a temporary file because this
  is rewritten after every column of a run that may last hours; a signal landing mid-write would
  otherwise truncate the file and make the whole run unreadable.
  """
  from olinda.metrics import json_safe

  path = Path(model_dir) / MANIFEST_NAME
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(".json.tmp")
  with open(tmp, "w") as fp:
    json.dump(json_safe(manifest), fp, indent=2)
  os.replace(tmp, path)
  return path


def read_manifest(model_dir: str | Path) -> dict:
  """Load the run manifest, with a clear error for directories from before multi-column support."""
  path = Path(model_dir) / MANIFEST_NAME
  if not path.exists():
    legacy = (Path(model_dir) / "train_meta.json").exists() or (Path(model_dir) / "train.h5").exists()
    hint = (
      " This run directory predates multi-column support — re-run `olinda prepare` to rebuild it."
      if legacy
      else ""
    )
    raise FileNotFoundError(f"no {MANIFEST_NAME} in {model_dir}.{hint}")
  with open(path) as fp:
    return json.load(fp)


def new_manifest(*, soft_labels, hard_labels, reference, features, val_frac, seed, limit) -> dict:
  """Start a manifest for a fresh run; columns are appended by :func:`add_column`."""
  import importlib.metadata

  try:
    version = importlib.metadata.version("olinda")
  except Exception:
    version = "unknown"
  return {
    "schema": RUN_SCHEMA,
    "olinda_version": version,
    "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "reference_library": reference,
    "features": features,
    "split": {"val_frac": val_frac, "seed": seed, "limit": limit},
    "soft_labels": {"path": str(soft_labels)},
    "hard_labels": {"path": str(hard_labels) if hard_labels else None},
    "columns": [],
  }


def add_column(manifest: dict, *, name: str, y, train_idx, val_idx, hard: dict | None = None) -> dict:
  """Append a column's entry to *manifest* and return it."""
  finite = np.asarray(y)[np.isfinite(y)]
  entry = {
    "id": column_id(len(manifest["columns"])),
    "name": name,
    "n_finite": int(finite.size),
    "n_train": int(len(train_idx)),
    "n_val": int(len(val_idx)),
    "value_range": [float(finite.min()), float(finite.max())] if finite.size else [None, None],
    "hard": hard,
    "status": {"soft_trained": False, "hard_trained": False},
  }
  entry["dir"] = f"{COLUMNS_DIRNAME}/{entry['id']}"
  manifest["columns"].append(entry)
  return entry


def find_column(manifest: dict, name_or_id: str) -> dict:
  for col in manifest["columns"]:
    if name_or_id in (col["name"], col["id"]):
      return col
  raise KeyError(f"no column {name_or_id!r} in this run; have {[c['name'] for c in manifest['columns']]}")


# ── per-column data ──────────────────────────────────────────────────────────


def write_targets(model_dir: str | Path, targets: dict) -> Path:
  """Write the reference-aligned target vector for every column, keyed by column id."""
  import h5py

  path = Path(model_dir) / TARGETS_NAME
  path.parent.mkdir(parents=True, exist_ok=True)
  with h5py.File(path, "w") as f:
    for col_id, y in targets.items():
      f.create_dataset(col_id, data=np.asarray(y, dtype=np.float32))
  return path


def read_target(model_dir: str | Path, col_id: str) -> np.ndarray:
  """The full reference-aligned target vector for one column (non-finite entries preserved)."""
  import h5py

  with h5py.File(Path(model_dir) / TARGETS_NAME, "r") as f:
    return np.asarray(f[col_id][:], dtype=np.float32)


def write_splits(model_dir: str | Path, splits: dict) -> Path:
  """Write each column's train/val row indices into one file."""
  import h5py

  path = Path(model_dir) / SPLITS_NAME
  path.parent.mkdir(parents=True, exist_ok=True)
  with h5py.File(path, "w") as f:
    for col_id, (train_idx, val_idx) in splits.items():
      g = f.create_group(col_id)
      g.create_dataset("train_idx", data=np.asarray(train_idx, dtype=np.int64))
      g.create_dataset("val_idx", data=np.asarray(val_idx, dtype=np.int64))
  return path


def read_split(model_dir: str | Path, col_id: str) -> tuple[np.ndarray, np.ndarray]:
  import h5py

  with h5py.File(Path(model_dir) / SPLITS_NAME, "r") as f:
    g = f[col_id]
    return np.asarray(g["train_idx"][:]), np.asarray(g["val_idx"][:])
