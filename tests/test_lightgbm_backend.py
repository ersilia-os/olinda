"""The LightGBM backend, end to end.

Every other test pins ``OLINDA_BACKEND=xgboost`` for determinism — which left the engine that
*actually runs by default on CPU* (so: every machine without a CUDA GPU) with no coverage at all. A
broken Sequence sampler shipped that way: LightGBM asks for a single row by bare int when it builds
feature bins, and only the slice case was handled.

These mirror the xgboost fixtures deliberately. They are the cheapest guard against the two engines
drifting apart.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import h5py  # noqa: E402
import pandas as pd  # noqa: E402
from click.testing import CliRunner  # noqa: E402

from olinda import OlindaArtifact  # noqa: E402

_SM = [
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
  "CCCC",
  "c1ccncc1",
  "CCOC",
  "CC(C)O",
  "CCCCO",
  "c1ccsc1",
]


def _stage(home, tmp_path, monkeypatch, n_columns=2):
  """A synthetic reference library and teacher file, with the engine forced to LightGBM."""
  import olinda.data as D
  import olinda.data.fetch as F
  from olinda.featurizer import MorganCountFeaturizer

  monkeypatch.setenv("OLINDA_BACKEND", "lightgbm")
  monkeypatch.setattr(F, "OLINDA_HOME", home)
  monkeypatch.setattr(D, "OLINDA_HOME", home)

  smiles = (_SM * 20)[:320]
  x = MorganCountFeaturizer().transform(smiles).astype(np.uint8)
  with h5py.File(home / "erl0_morgan.h5", "w") as f:
    f.create_dataset("data", data=x)
    f.create_dataset("input", data=np.array([s.encode() for s in smiles]))

  frame = {"smiles": smiles}
  for i in range(n_columns):
    v = x[:, 150 * i : 150 * (i + 1) + 200].sum(1).astype(np.float32)
    frame[f"assay{i}_probability"] = (v - v.min()) / (np.ptp(v) + 1e-9)
  soft = tmp_path / "soft.csv"
  pd.DataFrame(frame).to_csv(soft, index=False)
  return soft, smiles, x


def _run(args):
  from olinda.cli import cli

  return CliRunner().invoke(cli, args)


def test_index_sequence_answers_a_single_row(tmp_path):
  """LightGBM's bin sampler indexes with a bare int; a slice-only Sequence raises deep inside it."""
  from olinda.data.matrix import ReferenceMatrix, index_sequence

  matrix = ReferenceMatrix(np.arange(60, dtype=np.uint8).reshape(10, 6))
  idx = np.array([9, 7, 5, 3, 1])
  seq = index_sequence(matrix, idx)

  row = seq[2]  # a bare int, as __sample() passes
  assert row.shape == (6,) and row.dtype == np.float64
  assert np.array_equal(row, matrix.x[5])

  block = seq[1:4]  # a slice, as the streaming path passes
  assert block.shape == (3, 6)
  assert np.array_equal(block, matrix.x[[7, 5, 3]])
  assert len(seq) == 5


def test_fit_and_predict_on_lightgbm(tmp_path, monkeypatch):
  """The default engine on any CPU-only machine has to survive a whole fit."""
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=2)
  onnx = tmp_path / "run.onnx"

  r = _run(["fit", "-s", str(soft), "-m", str(onnx), "--val-frac", "0.2", "--num-boost-round", "40"])
  assert r.exit_code == 0, r.output
  assert onnx.is_file() and not (tmp_path / "run").exists()

  model = OlindaArtifact(onnx)
  assert model.columns == ["assay0_probability", "assay1_probability"]
  assert model.metadata["run"]["backend"] == "lightgbm"
  values = model.run(_SM[:6])[model.columns].to_numpy()
  assert np.isfinite(values).all()
  assert not np.allclose(values[:, 0], values[:, 1])  # independent models, not one broadcast


def test_hard_head_fuses_on_lightgbm(tmp_path, monkeypatch):
  """The blend and its parity gate must hold on this engine too, not only on xgboost."""
  home = tmp_path / "home"
  home.mkdir()
  soft, _, x = _stage(home, tmp_path, monkeypatch, n_columns=1)
  hard_smiles = _SM * 4
  score = x[: len(hard_smiles), :200].sum(1)
  hard = tmp_path / "hard.csv"
  pd.DataFrame({"smiles": hard_smiles, "assay0": (score > np.median(score)).astype(int)}).to_csv(
    hard, index=False
  )

  onnx = tmp_path / "run.onnx"
  r = _run(["fit", "-s", str(soft), "-h", str(hard), "-m", str(onnx), "--num-boost-round", "40"])
  assert r.exit_code == 0, r.output

  model = OlindaArtifact(onnx)
  assert model.has_hard is True
  assert np.isfinite(model.run(_SM[:4])[model.columns[0]].to_numpy()).all()
