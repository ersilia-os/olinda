"""End-to-end `olinda fit` (+ `predict`) on a tiny synthetic reference library, via the real CLI."""

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("xgboost")

import h5py  # noqa: E402
import pandas as pd  # noqa: E402
from click.testing import CliRunner  # noqa: E402

_SM = [
  "CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCOC(=O)C", "Clc1ccccc1", "COc1ccccc1", "CCOCC", "OCc1ccccc1",
]


def _stage_reference(home, tmp_path, monkeypatch):
  """Point OLINDA_HOME at a tmp dir holding a synthetic erl0_morgan.h5 + a row-aligned soft.csv."""
  import olinda.data as D
  import olinda.data.fetch as F
  from olinda.featurizer import MorganCountFeaturizer

  monkeypatch.setenv("OLINDA_BACKEND", "xgboost")  # deterministic engine (no lightgbm dependency in CI)
  monkeypatch.setattr(F, "OLINDA_HOME", home)
  monkeypatch.setattr(D, "OLINDA_HOME", home)

  smiles = (_SM * 20)[:200]
  X = MorganCountFeaturizer().transform(smiles).astype(np.uint8)
  with h5py.File(home / "erl0_morgan.h5", "w") as f:
    f.create_dataset("data", data=X)
    f.create_dataset("input", data=np.array([s.encode() for s in smiles]))
  val = X[:, :120].sum(1).astype(np.float32)
  val = (val - val.min()) / (np.ptp(val) + 1e-9)
  soft = tmp_path / "soft.csv"
  pd.DataFrame({"smiles": smiles, "y": val}).to_csv(soft, index=False)
  return soft


def _run(args):
  from olinda.cli import cli

  return CliRunner().invoke(cli, args)


def test_fit_soft_only_then_predict(tmp_path, monkeypatch):
  home = tmp_path / "home"
  home.mkdir()
  soft = _stage_reference(home, tmp_path, monkeypatch)
  md = tmp_path / "model"

  r = _run(["fit", "-s", str(soft), "-m", str(md), "--val-frac", "0.2", "--num-boost-round", "80"])
  assert r.exit_code == 0, r.output
  assert (md / "model.onnx").exists()

  q = tmp_path / "q.csv"
  pd.DataFrame({"smiles": _SM[:5]}).to_csv(q, index=False)
  out = tmp_path / "pred.csv"
  r2 = _run(["predict", "-m", str(md), "-i", str(q), "-o", str(out)])
  assert r2.exit_code == 0, r2.output
  df = pd.read_csv(out)
  assert list(df.columns) == ["smiles", "prediction", "surrogate"]
  assert len(df) == 5


def test_fit_with_hard_labels_adds_channels(tmp_path, monkeypatch):
  home = tmp_path / "home"
  home.mkdir()
  soft = _stage_reference(home, tmp_path, monkeypatch)
  gt = tmp_path / "gt.csv"
  from olinda.featurizer import MorganCountFeaturizer

  xu = MorganCountFeaturizer().transform(_SM)
  lab = (xu[:, :100].sum(1) > np.median(xu[:, :100].sum(1))).astype(int)
  pd.DataFrame({"smiles": _SM, "label": lab}).to_csv(gt, index=False)
  md = tmp_path / "model"

  r = _run(["fit", "-s", str(soft), "-h", str(gt), "-m", str(md), "--val-frac", "0.2", "--num-boost-round", "80"])
  assert r.exit_code == 0, r.output

  from olinda.onnx_pipeline import OnnxPipeline

  pipe = OnnxPipeline.load(md)
  assert pipe.meta["has_hard"] is True
  ch = pipe.predict_channels(_SM[:4])
  assert set(ch) == {"prediction", "surrogate", "ground_truth", "ground_truth_soft", "applicability"}
