"""Fused model.onnx: build a soft-only bundle and round-trip it through OnnxPipeline."""

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
xgb = pytest.importorskip("xgboost")

import h5py  # noqa: E402

from olinda.export import build_bundle  # noqa: E402
from olinda.featurizer import MorganCountFeaturizer  # noqa: E402
from olinda.models.bundle import StudentModel  # noqa: E402
from olinda.onnx_pipeline import OnnxPipeline  # noqa: E402

_SMILES = ["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCOC(=O)C", "Clc1ccccc1", "COc1ccccc1"] * 4


def _train_soft_dir(tmp_path):
  feat = MorganCountFeaturizer()
  X = feat.transform(_SMILES).astype(np.float32)
  y = (X[:, :50].sum(1) / 10).astype(np.float32)
  bst = xgb.train({"objective": "reg:squarederror", "max_depth": 3, "eta": 0.3}, xgb.DMatrix(X, label=y), 20)
  StudentModel(
    model=bst,
    featurizer=MorganCountFeaturizer(),
    metadata={"backend": "xgboost", "x_dim": 2048, "task": "regression"},
  ).save(tmp_path)
  with h5py.File(tmp_path / "val.h5", "w") as f:  # enables the soft_calibration stage
    f.create_dataset("x", data=X[:16])
    f.create_dataset("y", data=y[:16])
  return X, y


def test_soft_only_bundle_builds_and_serves(tmp_path):
  _train_soft_dir(tmp_path)
  res = build_bundle(tmp_path)  # parity gate (<1e-4) runs inside; raises on drift
  assert res["has_hard"] is False
  assert (tmp_path / "model.onnx").exists()

  pipe = OnnxPipeline.load(tmp_path)
  ch = pipe.predict_channels(["CCO", "c1ccccc1", "CCN"])
  assert set(ch) == {"prediction", "surrogate"}
  assert np.allclose(ch["prediction"], ch["surrogate"])  # soft-only: prediction == surrogate
  assert np.all(np.isfinite(ch["prediction"]))


def test_bundle_metadata_is_self_describing(tmp_path):
  _train_soft_dir(tmp_path)
  build_bundle(tmp_path)
  meta = OnnxPipeline.load(tmp_path).meta
  assert meta["has_hard"] is False
  assert meta["featurizer"]["rdkit_version"]  # RDKit version embedded
  assert meta["featurizer"].get("fp_size") == 2048
  assert meta["reference_library"]["name"].endswith(".h5")
  assert "hard" not in meta  # no hard head → no hard summary


def test_rdkit_version_mismatch_is_rejected(tmp_path):
  import json

  import onnx

  from olinda.onnx_pipeline import RDKitVersionMismatch

  _train_soft_dir(tmp_path)
  build_bundle(tmp_path)
  # tamper the embedded RDKit version → load must refuse (fingerprints wouldn't be reproducible)
  p = tmp_path / "model.onnx"
  m = onnx.load(str(p))
  for e in m.metadata_props:
    if e.key == "olinda":
      d = json.loads(e.value)
      d["featurizer"]["rdkit_version"] = "0.0.0"
      e.value = json.dumps(d)
  onnx.save(m, str(p))
  with pytest.raises(RDKitVersionMismatch):
    OnnxPipeline.load(tmp_path)
