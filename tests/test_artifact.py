"""The public inference API: a distilled model.onnx must be usable on its own."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("xgboost")

import h5py  # noqa: E402
import onnx  # noqa: E402

from olinda import OnnxArtifact, RDKitVersionMismatch  # noqa: E402

_SM = ["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCOC(=O)C", "Clc1ccccc1", "COc1ccccc1"]


def _build_artifact(tmp_path, monkeypatch):
  """Train a tiny soft-only model and fuse it, returning the model.onnx path."""
  from olinda.export import build_bundle
  from olinda.featurizer import MorganCountFeaturizer
  from olinda.models import StudentModel
  from olinda.train.backend import get_backend, select_backend

  monkeypatch.setenv("OLINDA_BACKEND", "xgboost")
  featurizer = MorganCountFeaturizer()
  smiles = (_SM * 12)[:96]
  x = featurizer.transform(smiles).astype(np.float32)
  y = (x[:, :100].sum(1) / 50.0).astype(np.float32)

  with h5py.File(tmp_path / "val.h5", "w") as f:
    f.create_dataset("x", data=x)
    f.create_dataset("y", data=y)

  name, device, _ = select_backend()
  be = get_backend(name, device)
  dtrain = be.dataset(x, y, None, 64)
  res = be.train(dtrain, dtrain, be.params({"max_bin": 64}), 20, 10, False)
  StudentModel(
    model=res.model,
    backend=name,
    featurizer=featurizer,
    metadata={
      "task": "regression",
      "x_dim": int(x.shape[1]),
      "features": "morgan",
      "backend": name,
      "featurizer": featurizer.to_dict(),
      "featurizer_class": "MorganCountFeaturizer",
    },
  ).save(tmp_path)
  build_bundle(tmp_path)
  return tmp_path / "model.onnx"


def test_artifact_loads_from_the_onnx_alone(tmp_path, monkeypatch):
  """No run directory, no config — just the file."""
  path = _build_artifact(tmp_path, monkeypatch)
  moved = tmp_path / "elsewhere" / "shipped.onnx"
  moved.parent.mkdir()
  moved.write_bytes(path.read_bytes())
  for sibling in tmp_path.iterdir():  # nothing else may be needed
    if sibling.is_file():
      sibling.unlink()

  model = OnnxArtifact(moved)
  df = model.run(_SM[:4])
  assert list(df["smiles"]) == _SM[:4]
  assert "prediction" in df.columns
  assert np.isfinite(df["prediction"].to_numpy()).all()


def test_run_returns_one_column_per_task(tmp_path, monkeypatch):
  """A single-task model is just the one-column case — no channels, no special mode."""
  model = OnnxArtifact(_build_artifact(tmp_path, monkeypatch))
  df = model.run(_SM[:3])
  assert list(df.columns) == ["smiles", *model.columns]
  assert model.n_columns == len(model.columns) == 1
  # the intermediate channels are reachable, just not in the headline frame
  assert set(model.run_channels(_SM[:3])) >= {"prediction", "surrogate"}


def test_artifact_describes_itself(tmp_path, monkeypatch):
  model = OnnxArtifact(_build_artifact(tmp_path, monkeypatch))
  d = model.describe()
  assert d["producer"] == "olinda"
  assert d["trained_at"] and d["trained_at"].endswith("+00:00")
  assert d["rdkit_version"] and d["n_features"] == 2048
  assert model.columns and model.has_ground_truth is False


def test_standard_onnx_provenance_is_set(tmp_path, monkeypatch):
  """Netron and generic ONNX tooling should identify the file without knowing about olinda."""
  m = onnx.load(str(_build_artifact(tmp_path, monkeypatch)), load_external_data=False)
  assert m.producer_name == "olinda"
  assert m.producer_version
  assert "olinda" in m.doc_string


def test_batching_does_not_change_results(tmp_path, monkeypatch):
  model = OnnxArtifact(_build_artifact(tmp_path, monkeypatch))
  many = (_SM * 3)[:20]
  assert np.allclose(
    model.run(many, batch_size=1000)["prediction"].to_numpy(),
    model.run(many, batch_size=3)["prediction"].to_numpy(),
  )


def test_directory_path_is_accepted(tmp_path, monkeypatch):
  _build_artifact(tmp_path, monkeypatch)
  assert len(OnnxArtifact(tmp_path).run(_SM[:2])) == 2


def test_rdkit_mismatch_is_refused(tmp_path, monkeypatch):
  """Fingerprints only reproduce on the exact build, so loading must fail loudly."""
  path = _build_artifact(tmp_path, monkeypatch)
  m = onnx.load(str(path))
  for prop in m.metadata_props:
    if prop.key == "olinda":
      meta = json.loads(prop.value)
      meta["featurizer"]["rdkit_version"] = "0.0.0-not-real"
      prop.value = json.dumps(meta)
  onnx.save(m, str(path))
  with pytest.raises(RDKitVersionMismatch):
    OnnxArtifact(path)


def test_non_olinda_onnx_is_rejected_clearly(tmp_path, monkeypatch):
  path = _build_artifact(tmp_path, monkeypatch)
  m = onnx.load(str(path))
  del m.metadata_props[:]
  onnx.save(m, str(path))
  with pytest.raises(ValueError, match="no olinda metadata"):
    OnnxArtifact(path)


def test_dead_tree_attributes_are_stripped(tmp_path, monkeypatch):
  """nodes_hitrates is a constant the runtime ignores — it must not ship."""
  m = onnx.load(str(_build_artifact(tmp_path, monkeypatch)), load_external_data=False)
  for node in m.graph.node:
    if node.op_type.startswith("TreeEnsemble"):
      names = {a.name for a in node.attribute}
      assert "nodes_hitrates" not in names


def test_inference_path_does_not_import_training_libraries(tmp_path, monkeypatch):
  """The base install carries no gradient-boosting stack, so inference must not reach for one."""
  import subprocess
  import sys

  path = _build_artifact(tmp_path, monkeypatch)
  code = (
    "import sys;"
    "from olinda import OnnxArtifact;"
    f"OnnxArtifact({str(path)!r}).run(['CCO']);"
    "bad=[m for m in ('lightgbm','xgboost','h5py','lazyqsar','onnx','onnxmltools','optuna',"
    "'click','rich_click','tqdm','loguru') if m in sys.modules];"
    "print(','.join(bad))"
  )
  out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
  assert out.stdout.strip() == "", f"inference imported training-only modules: {out.stdout.strip()}"
