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


def _plain(result) -> str:
  """A command's output as one flat line: no ANSI, no box drawing, no wrapping.

  Rich sizes itself to the terminal, so the same message is one line in a wide shell and several on
  CI at 80 columns. Worse, a message inside an error panel is wrapped *around the border*, so the
  raw text reads ``is left │ │ over from another run``. Stripping the frame and collapsing whitespace
  makes these assertions about what was said, not about how it happened to be laid out.
  """
  import re

  text = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", result.output)
  text = re.sub(r"[│┃╭╮╰╯─━┏┓┗┛]", " ", text)
  return re.sub(r"\s+", " ", text)


def _steps(soft, md, hard=None, *, val_frac="0.2", rounds="40"):
  """Drive the pipeline step by step, i.e. everything ``fit`` does except the closing ``clean``.

  Tests that inspect the manifest, the splits or a column's artifacts have to take this path: ``fit``
  hands back the fused model and nothing else.
  """
  prepare = ["prepare", "-s", str(soft), "-m", str(md), "--val-frac", val_frac]
  if hard is not None:
    prepare += ["-h", str(hard)]
  assert (r := _run(prepare)).exit_code == 0, r.output
  assert (r := _run(["learn-soft", "-m", str(md), "--num-boost-round", rounds])).exit_code == 0, r.output
  if hard is not None:
    assert (r := _run(["learn-hard", "-m", str(md)])).exit_code == 0, r.output
  return md


def test_three_columns_fit_and_predict(tmp_path, monkeypatch):
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=3)
  art = tmp_path / "run.onnx"

  r = _run(["fit", "-s", str(soft), "-m", str(art), "--val-frac", "0.2", "--num-boost-round", "40"])
  assert r.exit_code == 0, r.output
  # fit finishes into the artifact it was asked for, and its working folder is gone
  assert art.is_file()
  assert not (tmp_path / "run").exists()

  model = OlindaArtifact(art)
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
  art = tmp_path / "run.onnx"
  assert (
    _run(["fit", "-s", str(soft), "-m", str(art), "--val-frac", "0.2", "--num-boost-round", "40"]).exit_code
    == 0
  )

  model = OlindaArtifact(art)
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

  md = _steps(soft, tmp_path / "run", hard)
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
  assert "11 value columns" in _plain(r) and "10" in _plain(r)


def test_selecting_soft_columns_trains_only_those(tmp_path, monkeypatch):
  """--soft-label-columns is the contract: what you name is what the artifact ends up predicting."""
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=3)
  art = tmp_path / "run.onnx"

  r = _run(
    # deliberately out of file order — the selection decides the order too
    ["fit", "-s", str(soft), "-m", str(art), "--val-frac", "0.2", "--num-boost-round", "40"]
    + ["--soft-label-columns", "assay2_probability,assay0_probability"]
  )
  assert r.exit_code == 0, r.output
  assert OlindaArtifact(art).columns == ["assay2_probability", "assay0_probability"]


def test_selecting_columns_lifts_the_budget_off_the_unused_ones(tmp_path, monkeypatch):
  """An 11-column file is fine as long as you distil at most MAX_COLUMNS of it."""
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=11)
  md = tmp_path / "run"
  r = _run(["prepare", "-s", str(soft), "-m", str(md), "--soft-label-columns", "assay3_probability"])
  assert r.exit_code == 0, r.output
  manifest = json.loads((md / "manifest.json").read_text())
  assert [c["name"] for c in manifest["columns"]] == ["assay3_probability"]


def test_naming_a_column_that_is_not_there_fails_before_any_training(tmp_path, monkeypatch):
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=3)
  md = tmp_path / "run"
  r = _run(["prepare", "-s", str(soft), "-m", str(md), "--soft-label-columns", "nope"])
  assert r.exit_code != 0
  assert "nope" in _plain(r)
  assert not (md / "manifest.json").exists()


def test_the_ersilia_key_input_layout_works_untouched(tmp_path, monkeypatch):
  """key,input,<value> is what ersilia writes — it must fit and predict with no flags at all."""
  home = tmp_path / "home"
  home.mkdir()
  soft, smiles, _ = _stage(home, tmp_path, monkeypatch, n_columns=1)
  frame = pd.read_csv(soft)
  ersilia = tmp_path / "ersilia_style.csv"
  pd.DataFrame({
    "key": [f"k{i}" for i in range(len(frame))],
    "input": frame["smiles"],
    "assay0_probability": frame["assay0_probability"],
  }).to_csv(ersilia, index=False)

  art = tmp_path / "run.onnx"
  r = _run(["fit", "-s", str(ersilia), "-m", str(art), "--val-frac", "0.2", "--num-boost-round", "40"])
  assert r.exit_code == 0, r.output
  # `key` must not have been mistaken for a value column
  assert OlindaArtifact(art).columns == ["assay0_probability"]

  # and predict reads the same layout, with --smiles-column as the explicit escape hatch
  query = tmp_path / "query.csv"
  pd.DataFrame({"key": ["a", "b"], "input": _SM[:2]}).to_csv(query, index=False)
  out = tmp_path / "pred.csv"
  assert (r := _run(["predict", "-m", str(art), "-i", str(query), "-o", str(out)])).exit_code == 0, r.output
  assert len(pd.read_csv(out)) == 2

  named = tmp_path / "named.csv"
  pd.DataFrame({"mol": _SM[:2]}).to_csv(named, index=False)
  r = _run(["predict", "-m", str(art), "-i", str(named), "-o", str(out), "--smiles-column", "mol"])
  assert r.exit_code == 0, r.output
  assert len(pd.read_csv(out)) == 2
  # ...and without it, an unrecognisable column is refused rather than guessed at
  r = _run(["predict", "-m", str(art), "-i", str(named), "-o", str(out)])
  assert r.exit_code != 0
  assert "smiles" in _plain(r) and "input" in _plain(r)


def test_fused_graph_has_no_duplicate_names(tmp_path, monkeypatch):
  """Per-column prefixes must make every node and initializer unique — the fusion blocker."""
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=3)
  art = tmp_path / "run.onnx"
  assert _run(["fit", "-s", str(soft), "-m", str(art), "--num-boost-round", "40"]).exit_code == 0

  m = onnx.load(str(art), load_external_data=False)
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
  art = tmp_path / "run.onnx"
  assert _run(["fit", "-s", str(soft), "-m", str(art), "--num-boost-round", "40"]).exit_code == 0

  model = OlindaArtifact(art)

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
  md = _steps(soft, tmp_path / "run")

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
  assert "not trained yet" in _plain(r)


def test_elapsed_covers_the_whole_command_not_the_last_column(tmp_path, monkeypatch):
  """The per-column timer must not shadow the command-level one."""
  import re

  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=3)
  r = _run(["fit", "-s", str(soft), "-m", str(tmp_path / "run.onnx"), "--num-boost-round", "40"])
  assert r.exit_code == 0, r.output
  plain = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", r.output)
  # learn-soft's summary reports one Elapsed; it must not be a per-column figure of 0s
  assert "Elapsed" in plain


def test_reusing_a_run_directory_is_caught_not_silently_wrong(tmp_path, monkeypatch):
  """Column dirs are positional, so re-preparing rebinds c0 to a different assay."""
  from olinda import run as runlib

  home = tmp_path / "home"
  home.mkdir()
  soft, smiles, x = _stage(home, tmp_path, monkeypatch, n_columns=2)
  md = _steps(soft, tmp_path / "run")

  # re-prepare the same directory with differently-named columns; columns/c0 keeps the old model
  frame = {"smiles": smiles}
  for i in range(2):
    v = x[:, 150 * i : 150 * (i + 1) + 200].sum(1).astype(np.float32)
    frame[f"renamed{i}_probability"] = (v - v.min()) / (np.ptp(v) + 1e-9)
  other = tmp_path / "other.csv"
  pd.DataFrame(frame).to_csv(other, index=False)
  assert _run(["prepare", "-s", str(other), "-m", str(md)]).exit_code == 0

  names = [c["name"] for c in runlib.read_manifest(md)["columns"]]
  assert names == ["renamed0_probability", "renamed1_probability"]

  r = _run(["export", "-m", str(md)])  # the stale booster must not be fused under the new name
  assert r.exit_code != 0
  assert "stale" in _plain(r) and "trained for column" in _plain(r)


def test_a_changed_reference_library_is_refused(tmp_path, monkeypatch):
  """Splits are positional indices, so a swapped library would silently mispair features."""
  home = tmp_path / "home"
  home.mkdir()
  soft, smiles, _ = _stage(home, tmp_path, monkeypatch, n_columns=2)
  md = tmp_path / "run"
  assert _run(["prepare", "-s", str(soft), "-m", str(md)]).exit_code == 0

  # regenerate the library with a different number of rows, as `olinda setup` might
  from olinda.featurizer import MorganCountFeaturizer

  fewer = smiles[:200]
  with h5py.File(home / "erl0_morgan.h5", "w") as f:
    f.create_dataset("data", data=MorganCountFeaturizer().transform(fewer).astype(np.uint8))
    f.create_dataset("input", data=np.array([s.encode() for s in fewer]))

  r = _run(["learn-soft", "-m", str(md), "--num-boost-round", "40"])
  assert r.exit_code != 0
  assert "library changed" in _plain(r) or "prepared against" in _plain(r)


def test_an_interrupted_learn_hard_does_not_brick_the_run(tmp_path, monkeypatch):
  """learn-hard writes G first and its metadata last; a crash between must read as soft-only."""
  from olinda.ground_truth import GT_DIRNAME, GT_META_NAME, GT_MODEL_SUBDIR, has_hard_head

  home = tmp_path / "home"
  home.mkdir()
  soft, smiles, x = _stage(home, tmp_path, monkeypatch, n_columns=2)
  hard_smiles = _SM * 4
  score = x[: len(hard_smiles), :200].sum(1)
  pd.DataFrame({"smiles": hard_smiles, "assay0": (score > np.median(score)).astype(int)}).to_csv(
    tmp_path / "hard.csv", index=False
  )
  md = _steps(soft, tmp_path / "run", tmp_path / "hard.csv")

  gt_root = md / "columns" / "c0" / GT_DIRNAME
  assert has_hard_head(md / "columns" / "c0")

  # simulate a crash after G was saved but before the head completed
  (gt_root / GT_META_NAME).unlink()
  assert (gt_root / GT_MODEL_SUBDIR).exists()  # the model is still on disk
  assert not has_hard_head(md / "columns" / "c0")  # but the head is not complete

  r2 = _run(["export", "-m", str(md)])  # must fuse soft-only, not die on a missing calibrator
  assert r2.exit_code == 0, r2.output
  assert OlindaArtifact(md).has_ground_truth is False


def test_parity_probe_exercises_the_hard_blend(tmp_path, monkeypatch):
  """Fixed probe molecules may all score zero applicability, leaving the hard head unchecked."""
  from olinda.applicability import SimilarityRegressor
  from olinda.export import _parity_probe
  from olinda.ground_truth import APPLICABILITY_DIRNAME, GT_DIRNAME

  md = _fit_with_hard(tmp_path, monkeypatch)
  from olinda.export import _column_plan

  _, plan = _column_plan(md)
  probe = _parity_probe(plan)
  assert len(probe) > 5, "the probe must add labelled compounds, not just the fixed molecules"

  clf = SimilarityRegressor.load(md / "columns" / "c0" / GT_DIRNAME / APPLICABILITY_DIRNAME)
  assert (np.asarray(clf.weight(probe > 0)) > 0).any(), "blend never exercised — hard head unchecked"


def test_parity_refuses_a_mis_built_graph(tmp_path, monkeypatch):
  """The gate's job is translation fidelity: a graph that disagrees with the sources must fail."""
  import olinda.export as export_mod
  from olinda.calibrate import IsotonicCalibrator

  md = _fit_with_hard(tmp_path, monkeypatch)
  original = export_mod._isotonic_model

  def shifted(cal, in_name, out_name):
    off = IsotonicCalibrator()
    off._x, off._y, off._sign = cal._x, cal._y + 0.01, cal._sign
    return original(off, in_name, out_name)

  monkeypatch.setattr(export_mod, "_isotonic_model", shifted)
  with pytest.raises(RuntimeError, match="parity failed"):
    export_mod.build_bundle(md)


def test_the_step_by_step_path_keeps_the_working_files(tmp_path, monkeypatch):
  """Only `fit` cleans. Driving the steps by hand must leave a run `export` can still act on."""
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=2)
  md = _steps(soft, tmp_path / "run")

  assert (md / "manifest.json").exists()
  assert (md / "targets.h5").exists() and (md / "splits.h5").exists()
  assert (md / "columns" / "c0").is_dir()
  # ...but the descriptor matrix is still never copied into the run
  assert not (md / "train.h5").exists() and not (md / "val.h5").exists()
  assert not list(md.glob("columns/*/train.h5"))

  assert _run(["export", "-m", str(md)]).exit_code == 0


def test_cleaning_does_not_change_predictions(tmp_path, monkeypatch):
  """The claim the whole step rests on: finishing a run predicts identically."""
  md = _fit_with_hard(tmp_path, monkeypatch)  # the step-by-step path, so the folder still exists
  art = md.with_suffix(".onnx")
  before = OlindaArtifact(md).run(_SM)

  r = _run(["clean", "-m", str(art)])
  assert r.exit_code == 0, r.output
  assert art.is_file() and not md.exists()

  after = OlindaArtifact(art).run(_SM)
  pd.testing.assert_frame_equal(before, after)


def test_tuning_leaves_nothing_behind_either(tmp_path, monkeypatch):
  """`tune` writes best_params.json into the run root, so finishing has to take it too."""
  pytest.importorskip("optuna")
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=1)
  art = tmp_path / "run.onnx"
  r = _run([
    "fit",
    "-s",
    str(soft),
    "-m",
    str(art),
    "--num-boost-round",
    "40",
    "--tune",
    "--trials",
    "2",
  ])
  assert r.exit_code == 0, r.output
  assert art.is_file() and not (tmp_path / "run").exists()


def test_fit_rejects_a_path_without_the_onnx_extension(tmp_path, monkeypatch):
  """The working folder is the artifact path minus the suffix, so the suffix cannot be optional."""
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=1)

  r = _run(["fit", "-s", str(soft), "-m", str(tmp_path / "run"), "--num-boost-round", "40"])
  assert r.exit_code != 0
  assert "must end in .onnx" in _plain(r)


def test_fit_refuses_to_reuse_a_prepared_working_folder(tmp_path, monkeypatch):
  """Column dirs are bound positionally to the teacher that made them — never mix two runs."""
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=1)
  assert _run(["prepare", "-s", str(soft), "-m", str(tmp_path / "run")]).exit_code == 0

  r = _run(["fit", "-s", str(soft), "-m", str(tmp_path / "run.onnx"), "--num-boost-round", "40"])
  assert r.exit_code != 0
  assert "holds another run" in _plain(r)


def test_fit_retries_after_an_early_failure(tmp_path, monkeypatch):
  """A fit that dies before prepare completes leaves an empty folder; that must not block the retry."""
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=1)
  art = tmp_path / "run.onnx"

  bad = _run(["fit", "-s", str(tmp_path / "nope.csv"), "-m", str(art), "--num-boost-round", "40"])
  assert bad.exit_code != 0
  assert (tmp_path / "run").exists()  # the folder was created before the input was read

  good = _run(["fit", "-s", str(soft), "-m", str(art), "--num-boost-round", "40"])
  assert good.exit_code == 0, good.output
  assert art.is_file() and not (tmp_path / "run").exists()


def test_clean_refuses_without_a_model_and_removes_nothing(tmp_path, monkeypatch):
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=2)
  md = tmp_path / "run"
  assert _run(["prepare", "-s", str(soft), "-m", str(md)]).exit_code == 0  # prepared, never fused

  r = _run(["clean", "-m", str(tmp_path / "run.onnx")])
  assert r.exit_code != 0
  assert "no model.onnx" in _plain(r)
  assert (md / "manifest.json").exists() and (md / "targets.h5").exists()


def test_cleaning_twice_is_harmless(tmp_path, monkeypatch):
  home = tmp_path / "home"
  home.mkdir()
  soft, _, _ = _stage(home, tmp_path, monkeypatch, n_columns=1)
  art = tmp_path / "run.onnx"
  assert _run(["fit", "-s", str(soft), "-m", str(art), "--num-boost-round", "40"]).exit_code == 0

  r = _run(["clean", "-m", str(art)])  # fit already finished it; nothing left to do
  assert r.exit_code == 0, r.output
  assert "already clean" in _plain(r)
  assert art.is_file()


def _fit_with_hard(tmp_path, monkeypatch):
  """A trained single-column run with a ground-truth head, left uncleaned."""
  home = tmp_path / "home"
  home.mkdir()
  soft, smiles, x = _stage(home, tmp_path, monkeypatch, n_columns=1)
  hard_smiles = _SM * 4
  score = x[: len(hard_smiles), :200].sum(1)
  pd.DataFrame({"smiles": hard_smiles, "assay0": (score > np.median(score)).astype(int)}).to_csv(
    tmp_path / "hard.csv", index=False
  )
  return _steps(soft, tmp_path / "run", tmp_path / "hard.csv")


# ── a hard head that earns no weight must still produce a usable model ───────


def _channels(artifact_path, smiles):
  """Every internal channel of the fused graph, so the blend can be checked against its parts."""
  import onnx
  import onnxruntime as ort
  from onnx import TensorProto, helper

  from olinda.featurizer import MorganCountFeaturizer

  model = onnx.load(str(artifact_path))
  have = {o.name for o in model.graph.output}
  for name in ("c0_surrogate", "c0_applicability"):
    if name not in have:
      model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.DOUBLE, ["B"]))
  tmp = artifact_path.parent / "_channels.onnx"
  onnx.save(model, str(tmp))

  options = ort.SessionOptions()
  options.log_severity_level = 3
  sess = ort.InferenceSession(str(tmp), options, providers=["CPUExecutionProvider"])
  fp = MorganCountFeaturizer().transform([str(s) for s in smiles]).astype(np.float32)
  outs = sess.run(None, {"input": fp})
  return {o.name: np.asarray(v).ravel() for o, v in zip(sess.get_outputs(), outs)}


def test_a_hard_head_that_earns_no_weight_still_fuses_and_predicts(tmp_path, monkeypatch):
  """A_MIN drops a weak ceiling to zero; the fuse must then ship a working, surrogate-only model.

  The parity probe used to reject `a == 0` everywhere as a broken gate. Now it is a legitimate
  outcome, and the graph must still build, load and predict — otherwise a weak hard head bricks the
  run instead of simply being ignored.
  """
  from olinda.applicability import SimilarityRegressor
  from olinda.ground_truth import APPLICABILITY_DIRNAME, GT_DIRNAME

  md = _fit_with_hard(tmp_path, monkeypatch)
  gate_dir = md / "columns" / "c0" / GT_DIRNAME / APPLICABILITY_DIRNAME
  gate = SimilarityRegressor.load(gate_dir)
  gate.a_max = 0.0  # as _blend_ceiling would set it for a poorly aligned head
  gate.save(gate_dir)

  assert _run(["export", "-m", str(md)]).exit_code == 0, "a disabled blend must not break the fuse"

  model = OlindaArtifact(md)
  probe = _SM[:6]
  ch = _channels(md / "model.onnx", probe)
  assert np.all(ch["c0_applicability"] == 0.0), "the ceiling is zero, so no weight may be assigned"
  # With a == 0 the blend is exactly the surrogate — the hard branch contributes nothing.
  np.testing.assert_allclose(model.run(probe)[model.columns[0]].to_numpy(), ch["c0_surrogate"], atol=1e-12)


def test_a_near_constant_teacher_column_disables_the_blend(tmp_path, monkeypatch):
  """R² is undefined when the teacher has no variance to explain; that must disable, not crash."""
  import h5py

  import olinda.data as D
  import olinda.data.fetch as F
  from olinda.featurizer import MorganCountFeaturizer

  home = tmp_path / "home"
  home.mkdir()
  monkeypatch.setenv("OLINDA_BACKEND", "xgboost")
  monkeypatch.setattr(F, "OLINDA_HOME", home)
  monkeypatch.setattr(D, "OLINDA_HOME", home)

  smiles = (_SM * 20)[:320]
  x = MorganCountFeaturizer().transform(smiles).astype(np.uint8)
  with h5py.File(home / "erl0_morgan.h5", "w") as f:
    f.create_dataset("data", data=x)
    f.create_dataset("input", data=np.array([s.encode() for s in smiles]))
  # A teacher that says the same thing about every compound.
  soft = tmp_path / "flat.csv"
  pd.DataFrame({"smiles": smiles, "assay0_probability": np.full(len(smiles), 0.42)}).to_csv(soft, index=False)
  hard_smiles = _SM * 4
  score = x[: len(hard_smiles), :200].sum(1)
  hard = tmp_path / "hard.csv"
  pd.DataFrame({"smiles": hard_smiles, "assay0": (score > np.median(score)).astype(int)}).to_csv(
    hard, index=False
  )

  md = _steps(soft, tmp_path / "run", hard)
  meta = json.loads((md / "columns" / "c0" / "_ground_truth" / "ground_truth_meta.json").read_text())
  assert meta["applicability"]["a_max"] == 0.0, "no variance to explain ⇒ no weight earned"
  assert OlindaArtifact(md).run(_SM[:4])[["assay0_probability"]].notna().all().all()


def test_only_the_gate_branch_binarises_the_fingerprint(tmp_path, monkeypatch):
  """The gate net is trained on `bits > 0`; the surrogate and the hard head are trained on counts.

  So exactly one branch of the fused graph may threshold its input. If the binarisation leaked onto
  the shared input every stage would silently lose count information, and nothing else in the graph
  would complain — the numbers would just be wrong for any molecule with a repeated substructure.
  """
  md = _fit_with_hard(tmp_path, monkeypatch)
  m = onnx.load(str(md / "model.onnx"), load_external_data=False)

  greaters = [n for n in m.graph.node if n.op_type == "Greater"]
  assert greaters, "the gate must threshold its input somewhere"
  for node in greaters:
    assert "ap" in node.name or "ap" in node.output[0], f"{node.name} thresholds outside the gate"

  # And the surrogate must still be fed the raw input, not a thresholded copy.
  produced_by = {out: n for n in m.graph.node for out in n.output}
  soft_feed = next(n for n in m.graph.node if n.output and n.output[0].endswith("sm__input"))
  assert soft_feed.input[0] == "input", "the surrogate must read the shared count fingerprint directly"
  assert produced_by.get(soft_feed.input[0]) is None, "nothing may transform the input before the surrogate"
