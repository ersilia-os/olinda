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
