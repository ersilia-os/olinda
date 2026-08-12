"""The public inference API: a distilled model.onnx must be usable on its own."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("xgboost")

import onnx  # noqa: E402

from olinda import OlindaArtifact, RDKitVersionMismatch  # noqa: E402

_SM = ["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCOC(=O)C", "Clc1ccccc1", "COc1ccccc1"]


def _build_artifact(tmp_path, monkeypatch, names=("assay_probability",)):
  """Stage a minimal run directory with one trained column per name, fuse it, return model.onnx."""
  from olinda import run as runlib
  from olinda.export import build_bundle
  from olinda.featurizer import MorganCountFeaturizer
  from olinda.models import StudentModel
  from olinda.train.backend import get_backend, select_backend

  monkeypatch.setenv("OLINDA_BACKEND", "xgboost")
  featurizer = MorganCountFeaturizer()
  smiles = (_SM * 12)[:96]
  x = featurizer.transform(smiles).astype(np.float32)

  manifest = runlib.new_manifest(
    soft_labels="soft.csv",
    hard_labels=None,
    reference={"name": "erl0_morgan.h5", "n_rows": len(smiles), "dim": int(x.shape[1])},
    features={"features": "morgan", "radius": 3},
    val_frac=0.2,
    seed=42,
    limit=None,
  )
  name, device, _ = select_backend()
  be = get_backend(name, device)
  targets = {}
  for i, col_name in enumerate(names):
    y = (x[:, 100 * (i + 1) : 100 * (i + 2)].sum(1) / 50.0).astype(np.float32)
    idx = np.arange(len(y))
    entry = runlib.add_column(manifest, name=col_name, y=y, train_idx=idx, val_idx=idx)
    targets[entry["id"]] = y
    dtrain = be.dataset(x, y, None, 64)
    dval = be.dataset(x, y, None, 64, reference=dtrain)
    res = be.train(dtrain, dval, be.params({"max_bin": 64}), 20, 10, False)
    StudentModel(
      model=res.model,
      backend=name,
      featurizer=featurizer,
      metadata={
        "task": "regression",
        "column": col_name,
        "x_dim": int(x.shape[1]),
        "features": "morgan",
        "backend": name,
        "featurizer": featurizer.to_dict(),
        "featurizer_class": "MorganCountFeaturizer",
      },
    ).save(runlib.column_dir(tmp_path, entry["id"]))

  runlib.write_targets(tmp_path, targets)
  runlib.write_splits(tmp_path, {cid: (np.arange(len(smiles)), np.arange(len(smiles))) for cid in targets})
  runlib.write_manifest(tmp_path, manifest)
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

  model = OlindaArtifact(moved)
  df = model.run(_SM[:4])
  assert list(df["smiles"]) == _SM[:4]
  assert df.columns.tolist() == ["smiles", "assay_probability"]
  assert np.isfinite(df["assay_probability"].to_numpy()).all()


def test_run_returns_one_column_per_task(tmp_path, monkeypatch):
  """A single-task model is just the one-column case — no channels, no special mode."""
  model = OlindaArtifact(_build_artifact(tmp_path, monkeypatch))
  df = model.run(_SM[:3])
  assert list(df.columns) == ["smiles", *model.columns]
  assert model.n_columns == len(model.columns) == 1
  assert model.columns == ["assay_probability"]


def test_artifact_describes_itself(tmp_path, monkeypatch):
  model = OlindaArtifact(_build_artifact(tmp_path, monkeypatch))
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
  model = OlindaArtifact(_build_artifact(tmp_path, monkeypatch))
  many = (_SM * 3)[:20]
  assert np.allclose(
    model.run(many, batch_size=1000)["assay_probability"].to_numpy(),
    model.run(many, batch_size=3)["assay_probability"].to_numpy(),
  )


def test_directory_path_is_accepted(tmp_path, monkeypatch):
  _build_artifact(tmp_path, monkeypatch)
  assert len(OlindaArtifact(tmp_path).run(_SM[:2])) == 2


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
    OlindaArtifact(path)


def test_importing_olinda_does_not_check_rdkit(tmp_path, monkeypatch):
  """The version gate belongs to the model, not the package.

  An import-time gate would fire before ``OlindaArtifact`` could be reached at all — making
  ``check_rdkit=False`` unusable and locking every artifact to one global build, however new the
  RDKit its own metadata asks for.
  """
  import subprocess
  import sys

  code = (
    "import rdkit; rdkit.__version__ = '0.0.0-not-real';import olinda; print(olinda.OlindaArtifact.__name__)"
  )
  out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
  assert out.returncode == 0, out.stderr
  assert out.stdout.strip() == "OlindaArtifact"


def test_rdkit_mismatch_can_be_waived_deliberately(tmp_path, monkeypatch):
  """`check_rdkit=False` is the documented escape hatch — it has to actually be reachable."""
  path = _build_artifact(tmp_path, monkeypatch)
  m = onnx.load(str(path))
  for prop in m.metadata_props:
    if prop.key == "olinda":
      meta = json.loads(prop.value)
      meta["featurizer"]["rdkit_version"] = "0.0.0-not-real"
      prop.value = json.dumps(meta)
  onnx.save(m, str(path))
  assert len(OlindaArtifact(path, check_rdkit=False).run(_SM[:2])) == 2


def test_non_olinda_onnx_is_rejected_clearly(tmp_path, monkeypatch):
  path = _build_artifact(tmp_path, monkeypatch)
  m = onnx.load(str(path))
  del m.metadata_props[:]
  onnx.save(m, str(path))
  with pytest.raises(ValueError, match="no olinda metadata"):
    OlindaArtifact(path)


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
    "from olinda import OlindaArtifact;"
    f"OlindaArtifact({str(path)!r}).run(['CCO']);"
    "bad=[m for m in ('lightgbm','xgboost','h5py','lazyqsar','onnx','onnxmltools','optuna',"
    "'click','rich_click','tqdm','loguru') if m in sys.modules];"
    "print(','.join(bad))"
  )
  out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
  assert out.stdout.strip() == "", f"inference imported training-only modules: {out.stdout.strip()}"


# ── unparseable molecules must not look like confident predictions ───────────


def test_unparseable_smiles_yield_nan_and_a_warning(tmp_path, monkeypatch):
  """An all-zero fingerprint scores fine and the gate calls it HIGH-confidence — refuse to report it."""
  import warnings

  model = OlindaArtifact(_build_artifact(tmp_path, monkeypatch))
  mixed = ["CCO", "not_a_smiles", "", "C1CC", "c1ccccc1"]
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    df = model.run(mixed)
  values = df[model.columns[0]].to_numpy()
  assert np.isfinite(values[0]) and np.isfinite(values[4])  # the real molecules still predict
  assert np.isnan(values[1:4]).all()  # garbage, an empty string, and an unclosed ring
  assert caught and "could not be parsed" in str(caught[0].message)
  assert "3 of 5" in str(caught[0].message)


def test_valid_input_warns_about_nothing(tmp_path, monkeypatch):
  import warnings

  model = OlindaArtifact(_build_artifact(tmp_path, monkeypatch))
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    df = model.run(["CCO", "c1ccccc1"])
  assert np.isfinite(df[model.columns[0]].to_numpy()).all()
  assert not [w for w in caught if "could not be parsed" in str(w.message)]


def test_declared_outputs_are_checked_against_the_graph(tmp_path, monkeypatch):
  """A metadata/graph mismatch must fail at load, not as a KeyError mid-prediction."""
  path = _build_artifact(tmp_path, monkeypatch)
  m = onnx.load(str(path))
  for prop in m.metadata_props:
    if prop.key == "olinda":
      meta = json.loads(prop.value)
      meta["columns"][0]["output"] = "no_such_output"
      prop.value = json.dumps(meta)
  onnx.save(m, str(path))
  with pytest.raises(ValueError, match="does not produce"):
    OlindaArtifact(path)
