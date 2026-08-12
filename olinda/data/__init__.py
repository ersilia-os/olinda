"""Reference-library data: where it lives, how it is read, and how a target is split and weighted."""

from .reference import (
  MAX_COLUMNS as MAX_COLUMNS,
  check_column_budget as check_column_budget,
  load_reference_calcs_frame as load_reference_calcs_frame,
  match_hard_columns as match_hard_columns,
  resolve_smiles_frame as resolve_smiles_frame,
)
from .split import split_reference_to_indices as split_reference_to_indices
from .fetch import (
  OLINDA_HOME as OLINDA_HOME,
  MORGAN_FINGERPRINTS_FILENAME as MORGAN_FINGERPRINTS_FILENAME,
  download_morgan_fingerprints as download_morgan_fingerprints,
)

# The reweighting helpers live behind __getattr__ because ``.dataset`` imports xgboost at module
# scope (for its DataIter), and the commands that only want OLINDA_HOME — `setup`, `prepare` — would
# otherwise pay for the whole boosting stack to read two path constants.
_LAZY = ("apply_bin_weights", "resolve_regression_weights")

__all__ = [
  "MAX_COLUMNS",
  "MORGAN_FINGERPRINTS_FILENAME",
  "OLINDA_HOME",
  "check_column_budget",
  "download_morgan_fingerprints",
  "load_reference_calcs_frame",
  "match_hard_columns",
  "resolve_smiles_frame",
  "split_reference_to_indices",
  *_LAZY,
]


def __getattr__(name):
  if name in _LAZY:
    from olinda.data import dataset

    return getattr(dataset, name)
  raise AttributeError(f"module 'olinda.data' has no attribute {name!r}")
