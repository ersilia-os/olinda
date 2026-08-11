"""Multi-column runs end-to-end, through the real CLI.

A single-column run is the one-column case of these, so these tests also guard the claim that there
is no separate code path for it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("xgboost")

import h5py  # noqa: E402
import onnx  # noqa: E402
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


def _stage(home, tmp_path, monkeypatch, n_columns=3):
  """Point OLINDA_HOME at a synthetic reference library and write an n-column teacher file."""
  import olinda.data as D
  import olinda.data.fetch as F
  from olinda.featurizer import MorganCountFeaturizer

  monkeypatch.setenv("OLINDA_BACKEND", "xgboost")
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


def test_three_columns_fit_and_predict(tmp_path, monkeypatch):
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=3)
  md = tmp_path / "run"

  r = _run(["fit", "-s", str(soft), "-m", str(md), "--val-frac", "0.2", "--num-boost-round", "40"])
  assert r.exit_code == 0, r.output
  assert (md / "model.onnx").exists()

  # the descriptor matrix must never be copied into the run
  assert not (md / "train.h5").exists() and not (md / "val.h5").exists()
  assert not list(md.glob("columns/*/train.h5"))

  model = OlindaArtifact(md)
  expected = [f"assay{i}_probability" for i in range(3)]
  assert model.columns == expected
  df = model.run(_SM[:5])
  assert list(df.columns) == ["smiles", *expected]
  assert np.isfinite(df[expected].to_numpy()).all()


def test_columns_are_independent_models(tmp_path, monkeypatch):
  """Different targets must give different predictions — not one model broadcast across columns."""
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=3)
  md = tmp_path / "run"
  assert (
    _run(["fit", "-s", str(soft), "-m", str(md), "--val-frac", "0.2", "--num-boost-round", "40"]).exit_code
    == 0
  )

  model = OlindaArtifact(md)
  values = model.run(_SM[:8])[model.columns].to_numpy()
  assert not np.allclose(values[:, 0], values[:, 1])


def test_sparse_hard_labels_apply_only_to_matched_columns(tmp_path, monkeypatch):
  """A wide hard file covering some columns gives exactly those a ground-truth head."""
  from olinda import run as runlib

  home = tmp_path / "home"
  home.mkdir()
  soft, _, x = _stage(home, tmp_path, monkeypatch, n_columns=3)

  # hard labels for two of the three columns, named without the teacher's suffix
  hard_smiles = _SM * 4
  hx = x[: len(hard_smiles)]
  frame = {"smiles": hard_smiles}
  for i in (0, 2):
    score = hx[:, 150 * i : 150 * i + 200].sum(1)
    frame[f"assay{i}"] = (score > np.median(score)).astype(int)
  hard = tmp_path / "hard.csv"
  pd.DataFrame(frame).to_csv(hard, index=False)

  md = tmp_path / "run"
  r = _run([
    "fit",
    "-s",
    str(soft),
    "-h",
    str(hard),
    "-m",
    str(md),
    "--val-frac",
    "0.2",
    "--num-boost-round",
    "40",
  ])
  assert r.exit_code == 0, r.output

  manifest = runlib.read_manifest(md)
  by_name = {c["name"]: c for c in manifest["columns"]}
  assert by_name["assay0_probability"]["hard"]["source_column"] == "assay0"
  assert by_name["assay0_probability"]["hard"]["match"] == "suffix"
  assert by_name["assay1_probability"]["hard"] is None  # soft-only, silently

  model = OlindaArtifact(md)
  flags = {c["name"]: c["has_hard"] for c in model.metadata["columns"]}
  assert flags == {"assay0_probability": True, "assay1_probability": False, "assay2_probability": True}
  assert model.has_ground_truth is True
  assert list(model.run(_SM[:4]).columns) == ["smiles", *model.columns]


def test_more_than_ten_columns_is_rejected(tmp_path, monkeypatch):
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=11)
  r = _run(["prepare", "-s", str(soft), "-m", str(tmp_path / "run")])
  assert r.exit_code != 0
  assert "11 value columns" in r.output and "10" in r.output


def test_fused_graph_has_no_duplicate_names(tmp_path, monkeypatch):
  """Per-column prefixes must make every node and initializer unique — the fusion blocker."""
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=3)
  md = tmp_path / "run"
  assert _run(["fit", "-s", str(soft), "-m", str(md), "--num-boost-round", "40"]).exit_code == 0

  m = onnx.load(str(md / "model.onnx"), load_external_data=False)
  init_names = [i.name for i in m.graph.initializer]
  node_names = [n.name for n in m.graph.node if n.name]
  assert len(init_names) == len(set(init_names)), "duplicate initializer names"
  assert len(node_names) == len(set(node_names)), "duplicate node names"
  assert [o.name for o in m.graph.output] == [f"assay{i}_probability" for i in range(3)]
  assert len(m.graph.input) == 1  # every column shares the one fingerprint input


def test_metadata_is_valid_json_and_self_contained(tmp_path, monkeypatch):
  """Strict JSON, and enough on its own to describe the model — metrics may be NaN."""
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=2)
  md = tmp_path / "run"
  assert _run(["fit", "-s", str(soft), "-m", str(md), "--num-boost-round", "40"]).exit_code == 0

  model = OlindaArtifact(md / "model.onnx")

  def reject_constant(c):
    raise ValueError(f"non-finite {c} is not valid JSON")

  parsed = json.loads(model.to_json(), parse_constant=reject_constant)
  assert parsed["producer"] == "olinda"
  assert parsed["trained_at"].endswith("+00:00")
  assert [c["name"] for c in parsed["columns"]] == model.columns
  assert parsed["featurizer"]["rdkit_version"]


def test_export_works_standalone_on_a_trained_run(tmp_path, monkeypatch):
  """`olinda export` must accept a run it trained itself — models live per column, not at the root."""
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=2)
  md = tmp_path / "run"
  assert _run(["fit", "-s", str(soft), "-m", str(md), "--num-boost-round", "40"]).exit_code == 0

  (md / "model.onnx").unlink()
  r = _run(["export", "-m", str(md)])
  assert r.exit_code == 0, r.output
  assert (md / "model.onnx").exists()


def test_export_rejects_an_untrained_run(tmp_path, monkeypatch):
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=2)
  md = tmp_path / "run"
  assert _run(["prepare", "-s", str(soft), "-m", str(md)]).exit_code == 0

  r = _run(["export", "-m", str(md)])  # prepared but never trained
  assert r.exit_code != 0
  assert "not trained yet" in r.output


def test_elapsed_covers_the_whole_command_not_the_last_column(tmp_path, monkeypatch):
  """The per-column timer must not shadow the command-level one."""
  import re

  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=3)
  r = _run(["fit", "-s", str(soft), "-m", str(tmp_path / "run"), "--num-boost-round", "40"])
  assert r.exit_code == 0, r.output
  plain = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", r.output)
  # learn-soft's summary reports one Elapsed; it must not be a per-column figure of 0s
  assert "Elapsed" in plain
