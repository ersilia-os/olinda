"""End-to-end CLI smoke test: pack -> fit -> predict.

Exercises the documented user-facing contract on tiny synthetic data. Uses
``--time-budget 0`` to skip Optuna tuning so the run stays fast and deterministic.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from olinda.cli import cli

X_DIM = 8


def _write_teacher(path, n=400, seed=0):
  rng = np.random.default_rng(seed)
  coef = rng.standard_normal(X_DIM).astype(np.float32)
  X = rng.standard_normal((n, X_DIM)).astype(np.float32)
  y = (X @ coef + rng.standard_normal(n).astype(np.float32) * 0.5).astype(np.float32)
  x_list = pa.array([r.tolist() for r in X], type=pa.list_(pa.float32(), X_DIM))
  pq.write_table(pa.table({"x": x_list, "y": pa.array(y, type=pa.float32())}), str(path))


def test_pack_fit_predict_end_to_end(tmp_path):
  runner = CliRunner()
  teacher = tmp_path / "teacher.parquet"
  _write_teacher(teacher)

  packed = tmp_path / "packed"
  r = runner.invoke(
    cli,
    ["pack", "-i", str(teacher), "-o", str(packed), "--x-dim", str(X_DIM), "--val-frac", "0.25"],
  )
  assert r.exit_code == 0, r.output
  assert (packed / "meta.json").exists()

  model = tmp_path / "model"
  r = runner.invoke(
    cli,
    [
      "fit",
      "--input", str(packed),
      "--out", str(model),
      "--num-boost-round", "40",
      "--early-stopping", "10",
      "--time-budget", "0",
      "--no-onnx",
    ],
  )
  assert r.exit_code == 0, r.output
  assert (model / "xgb.json").exists()
  assert (model / "calibrator.json").exists()

  # Validation report is valid strict JSON and labels its calibration evaluation.
  report_path = model / "validation" / "validation_report.json"
  assert report_path.exists()
  with open(report_path) as fp:
    report = json.load(fp)  # would raise on bare NaN/Infinity tokens
  assert report["calibrated_metrics"]["evaluation"] in {"out_of_fold_5", "in_sample"}

  # Predict on a fresh feature matrix.
  infer = tmp_path / "infer.npy"
  np.save(infer, np.random.default_rng(7).standard_normal((15, X_DIM)).astype(np.float32))
  preds = tmp_path / "preds.csv"
  r = runner.invoke(cli, ["predict", "--model-dir", str(model), "--input", str(infer), "--out", str(preds)])
  assert r.exit_code == 0, r.output
  assert preds.exists()
  with open(preds) as fp:
    lines = [ln for ln in fp.read().splitlines() if ln.strip()]
  assert len(lines) == 16  # header + 15 predictions


def test_smiles_fit_persists_featurizer(tmp_path):
  """Regression test: a Morgan/SMILES student must save its featurizer.

  Without it, ``StudentModel.load(...).predict(smiles=...)`` raises and ``olinda predict``
  silently depends on re-supplied --fp flags. fit recovers the featurizer from the packed
  meta so the saved model is self-describing.
  """
  pytest.importorskip("rdkit")
  from olinda.featurizer import Fingerprint
  from olinda.models import StudentModel

  pool = [
    "CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCCC", "O=C(O)c1ccccc1", "CCOC(=O)C",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "OCC1OC(O)C(O)C(O)C1O",
  ]
  rng = np.random.default_rng(0)
  smiles = [pool[i % len(pool)] for i in range(80)]
  y = rng.uniform(0, 1, size=80).astype(np.float32)
  csv = tmp_path / "teacher.csv"
  csv.write_text("smiles,y\n" + "\n".join(f"{s},{v:.4f}" for s, v in zip(smiles, y)))

  model = tmp_path / "model"
  r = CliRunner().invoke(
    cli,
    ["fit", "--input", str(csv), "--out", str(model), "--smiles-col", "smiles",
     "--fp", "morgan", "--fp-size", "256", "--time-budget", "0", "--no-onnx"],
  )
  assert r.exit_code == 0, r.output

  meta = json.loads((model / "train_meta.json").read_text())
  assert meta.get("featurizer_class") == "Fingerprint"

  student = StudentModel.load(model, featurizer_factory=lambda c, cfg: Fingerprint.from_dict(cfg))
  assert isinstance(student.featurizer, Fingerprint)
  preds = student.predict(smiles=["CCO", "c1ccccc1", "O=C(O)c1ccccc1"], calibrate=False)
  assert preds.shape == (3,)
  assert np.isfinite(preds).all()


def test_fit_without_weight_column_does_not_crash(tmp_path):
  """Regression test: packing without weights must not break fit/validation.

  meta.json records w_col='w' even when no weight column is written; the data
  iterators and validation scan must tolerate the missing column.
  """
  runner = CliRunner()
  teacher = tmp_path / "teacher.parquet"
  _write_teacher(teacher, n=300)
  packed = tmp_path / "packed"
  runner.invoke(
    cli, ["pack", "-i", str(teacher), "-o", str(packed), "--x-dim", str(X_DIM), "--val-frac", "0.3"]
  )

  meta = json.loads((packed / "meta.json").read_text())
  assert meta["w_col"] == "w"  # recorded despite no weight column on disk

  model = tmp_path / "model"
  r = runner.invoke(
    cli,
    ["fit", "--input", str(packed), "--out", str(model),
     "--num-boost-round", "30", "--early-stopping", "10", "--time-budget", "0", "--no-onnx"],
  )
  assert r.exit_code == 0, r.output
  assert (model / "xgb.json").exists()
