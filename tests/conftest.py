"""One tiny synthetic run, shared by every test that needs a finished model.

A real olinda run wants a 1.36M-compound reference library and minutes of boosting. Everything the
test suite asserts about the pipeline — that columns stay independent, that the fuse is
numerically faithful, that `clean` does not change predictions — is just as true of 200 rows and
ten distinct molecules, so the fixtures here build that instead.

:func:`build_tiny_model` is also called from CI (the ``report-install`` job in
``.github/workflows/test.yml``) to produce a model with the training stack that is then scored with
only the reporting extra installed. It therefore has to stay importable and runnable outside pytest —
hence plain functions here, and no ``import pytest`` at module scope.
"""

from __future__ import annotations

import os
from pathlib import Path

# Real molecules, repeated to fill the library. Enough *distinct* ones matters more than the row
# count: the teacher targets below are a function of the fingerprint, so a library of ten molecules
# has ten distinct target values however many rows it has, and a booster fitted on six of them
# learns nothing (measured: R² 0.00, one tree, and a degenerate isotonic fit downstream).
SMILES = [
  "CCO",
  "CCN",
  "CCC",
  "c1ccccc1",
  "CC(=O)O",
  "CCOC(=O)C",
  "Clc1ccccc1",
  "COc1ccccc1",
  "CCOCC",
  "OCc1ccccc1",
  "CCCCO",
  "CCCCCC",
  "c1ccc2ccccc2c1",
  "CC(C)Cc1ccccc1",
  "NCCc1ccc(O)cc1",
  "CC(=O)Nc1ccccc1",
  "OC(=O)c1ccccc1O",
  "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
  "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
  "CN1CCC[C@H]1c1cccnc1",
  "Oc1ccc(CCN)cc1O",
  "CSCC[C@H](N)C(=O)O",
  "c1ccc(-c2ccccc2)cc1",
  "ClCCl",
]

LIBRARY_ROWS = 240


def write_library(home: Path):
  """Write a synthetic ``erl0_morgan.h5`` into *home* and return its (smiles, fingerprints).

  The layout is exactly what `olinda setup` downloads: a ``data`` matrix of uint8 Morgan counts and
  a row-aligned ``input`` dataset of SMILES, which is the file's source of truth for order.
  """
  import h5py
  import numpy as np

  from olinda.featurizer import MorganCountFeaturizer

  home.mkdir(parents=True, exist_ok=True)
  smiles = (SMILES * ((LIBRARY_ROWS // len(SMILES)) + 1))[:LIBRARY_ROWS]
  x = MorganCountFeaturizer().transform(smiles).astype(np.uint8)
  with h5py.File(home / "erl0_morgan.h5", "w") as f:
    f.create_dataset("data", data=x)
    f.create_dataset("input", data=np.array([s.encode() for s in smiles]))
  return smiles, x


def write_labels(out_dir: Path, smiles, x, *, columns=("activity", "solubility")):
  """Write a teacher file over the whole library, plus a sparse hard-label file.

  Returns ``(soft_path, hard_path)``. Each teacher column is a different smooth function of the
  fingerprint, so a booster can actually learn it and the columns are not interchangeable.

  The hard file deliberately names its column ``activity`` against a teacher column called
  ``activity_probability``: that suffix relationship is the real-world case (Ersilia writes
  ``*_probability``, a lab writes the bare endpoint), and it is what exercises the suffix matching in
  ``match_hard_columns``.
  """
  import numpy as np
  import pandas as pd

  out_dir.mkdir(parents=True, exist_ok=True)
  frame = {"smiles": smiles}
  for i, name in enumerate(columns):
    # A fixed random projection of the whole fingerprint, not a slice of it: Morgan bits are sparse,
    # so any given 200-bit window is all zeros for small molecules and the "target" comes out
    # constant. Projecting every bit guarantees the value varies across distinct molecules, and a
    # different seed per column makes the columns genuinely different functions.
    weights = np.random.default_rng(i).normal(size=x.shape[1])
    raw = x.astype(np.float64) @ weights
    frame[f"{name}_probability"] = (raw - raw.min()) / (np.ptp(raw) + 1e-9)
  soft = out_dir / "soft.csv"
  pd.DataFrame(frame).to_csv(soft, index=False)

  # A sparse measurement file: a third of the library, labelled by thresholding the first teacher
  # column, so the hard head has real signal and both classes are present.
  take = np.arange(0, len(smiles), 3)
  first = np.asarray(frame[f"{columns[0]}_probability"])[take]
  hard = out_dir / "hard.csv"
  pd.DataFrame({
    "smiles": [smiles[i] for i in take],
    columns[0]: (first > np.median(first)).astype(int),
  }).to_csv(hard, index=False)
  return soft, hard


def run_cli(args):
  """Invoke the real CLI in-process and return the click result.

  Through `CliRunner` rather than a subprocess so a failure surfaces its traceback in the test
  output, and so monkeypatched paths (OLINDA_HOME) are still in effect.
  """
  from click.testing import CliRunner

  from olinda.cli import cli

  return CliRunner().invoke(cli, [str(a) for a in args])


def build_tiny_model(dest, *, backend: str = "xgboost", with_hard: bool = True) -> Path:
  """Fit a complete model into *dest* and return the path to its ``model.onnx``.

  Leaves ``soft.csv`` and ``hard.csv`` beside it, so a caller can score the model against the same
  labels it was trained on. Sets ``OLINDA_HOME`` and ``OLINDA_BACKEND`` in the environment for the
  duration — which is why the pytest fixtures below prefer monkeypatch, and CI, having its own
  process, can just call this.
  """
  dest = Path(dest)
  home = dest / "home"
  smiles, x = write_library(home)
  soft, hard = write_labels(dest, smiles, x)

  previous = {k: os.environ.get(k) for k in ("OLINDA_HOME", "OLINDA_BACKEND")}
  os.environ["OLINDA_HOME"] = str(home)
  os.environ["OLINDA_BACKEND"] = backend
  try:
    import olinda.data as data_pkg
    import olinda.data.fetch as fetch

    fetch.OLINDA_HOME = home
    data_pkg.OLINDA_HOME = home

    model = dest / "model.onnx"
    args = ["fit", "-s", soft, "-m", model]
    if with_hard:
      args += ["-h", hard]
    result = run_cli(args)
    if result.exit_code != 0:
      raise RuntimeError(f"olinda fit failed:\n{result.output}\n{result.exception!r}")
  finally:
    for key, value in previous.items():
      if value is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = value
  return model
