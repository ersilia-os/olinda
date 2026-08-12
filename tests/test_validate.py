"""`olinda validate`: scoring a finished artifact against data of your choosing."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("xgboost")
pytest.importorskip("stylia")

import pandas as pd  # noqa: E402
from click.testing import CliRunner  # noqa: E402

# Molecules the model trains on, and a disjoint set to validate against.
_TRAIN = [
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
  "CCCCCC",
  "c1ccco1",
  "CCC(=O)O",
  "CN(C)C",
  "CCCN",
  "c1ccc(cc1)O",
  "CC(C)(C)O",
  "CCCCCCO",
]
_HELD_OUT = [
  "CCCCCCC",
  "CCCCCCCC",
  "c1ccc2ccccc2c1",
  "CC(C)CC",
  "CCOCCO",
  "CCN(CC)CC",
  "c1ccc(cc1)N",
  "CC(=O)N",
  "CCCCCCCCO",
  "c1ccc(cc1)Cl",
  "CCCCC(=O)O",
  "CC(C)C(=O)O",
]


def _run(args):
  from olinda.cli import cli

  return CliRunner().invoke(cli, args)


@pytest.fixture
def fitted(tmp_path, monkeypatch):
  """A trained one-column model with a hard head, plus held-out label files."""
  import h5py

  import olinda.data as D
  import olinda.data.fetch as F
  from olinda.featurizer import MorganCountFeaturizer

  home = tmp_path / "home"
  home.mkdir()
  monkeypatch.setenv("OLINDA_BACKEND", "xgboost")
  monkeypatch.setattr(F, "OLINDA_HOME", home)
  monkeypatch.setattr(D, "OLINDA_HOME", home)

  fz = MorganCountFeaturizer()
  library = (_TRAIN * 17)[:400]
  x = fz.transform(library).astype(np.uint8)
  with h5py.File(home / "erl0_morgan.h5", "w") as f:
    f.create_dataset("data", data=x)
    f.create_dataset("input", data=np.array([s.encode() for s in library]))

  def scaled(smiles):
    v = fz.transform(smiles)[:, :700].sum(1).astype(np.float64)
    return (v - v.min()) / (np.ptp(v) + 1e-9)

  pd.DataFrame({"smiles": library, "assay_probability": scaled(library)}).to_csv(
    tmp_path / "soft.csv", index=False
  )
  hard_smiles = _TRAIN * 3
  score = fz.transform(hard_smiles)[:, :200].sum(1)
  pd.DataFrame({"smiles": hard_smiles, "assay": (score > np.median(score)).astype(int)}).to_csv(
    tmp_path / "hard.csv", index=False
  )

  held = _HELD_OUT * 5
  pd.DataFrame({"smiles": held, "assay_probability": scaled(held)}).to_csv(
    tmp_path / "heldout_soft.csv", index=False
  )
  hs = fz.transform(held)[:, :200].sum(1)
  pd.DataFrame({"smiles": held, "assay": (hs > np.median(hs)).astype(int)}).to_csv(
    tmp_path / "heldout_hard.csv", index=False
  )

  art = tmp_path / "model.onnx"
  r = _run([
    "fit",
    "-s",
    str(tmp_path / "soft.csv"),
    "-h",
    str(tmp_path / "hard.csv"),
    "-m",
    str(art),
    "--num-boost-round",
    "40",
  ])
  assert r.exit_code == 0, r.output
  return tmp_path, art


def test_validate_writes_the_whole_report(fitted):
  tmp_path, art = fitted
  out = tmp_path / "report"
  r = _run([
    "validate",
    "-m",
    str(art),
    "-s",
    str(tmp_path / "heldout_soft.csv"),
    "-h",
    str(tmp_path / "heldout_hard.csv"),
    "-o",
    str(out),
  ])
  assert r.exit_code == 0, r.output

  assert (out / "report.html").exists()
  assert (out / "performance_table.csv").exists()
  metrics = json.loads((out / "metrics.json").read_text())  # strict JSON: no NaN tokens

  # Every figure the report claims must actually be on disk, in both formats.
  figures = [f for group in metrics["figures"].values() for f in group]
  assert figures, "no figures were produced"
  for fig in figures:
    assert (out / fig["png"]).exists(), f"missing {fig['png']}"
    assert (out / fig["pdf"]).exists(), f"missing {fig['pdf']}"

  # And every <img> the page references must resolve, so the report is not silently broken.
  page = (out / "report.html").read_text()
  for src in {s.split('"')[0] for s in page.split('src="')[1:]}:
    assert (out / src).exists(), f"report.html references a missing {src}"


def test_reported_metrics_match_the_metrics_module(fitted):
  """The numbers in the report must be the ones olinda.metrics computes — not a second implementation."""
  from olinda import OlindaArtifact
  from olinda.metrics import binary_metrics, regression_metrics

  tmp_path, art = fitted
  out = tmp_path / "report"
  assert (
    _run([
      "validate",
      "-m",
      str(art),
      "-s",
      str(tmp_path / "heldout_soft.csv"),
      "-h",
      str(tmp_path / "heldout_hard.csv"),
      "-o",
      str(out),
    ]).exit_code
    == 0
  )
  metrics = json.loads((out / "metrics.json").read_text())

  model = OlindaArtifact(art)
  task = model.columns[0]
  soft = pd.read_csv(tmp_path / "heldout_soft.csv")
  pred = model.run(soft["smiles"].tolist())[task].to_numpy()
  expected = regression_metrics(soft["assay_probability"].to_numpy(), pred)
  assert metrics["soft"]["metrics"][task]["r2"] == pytest.approx(expected["r2"])
  assert metrics["soft"]["metrics"][task]["spearman"] == pytest.approx(expected["spearman"])

  hard = pd.read_csv(tmp_path / "heldout_hard.csv")
  hpred = model.run(hard["smiles"].tolist())[task].to_numpy()
  assert metrics["hard"]["metrics"][task]["auroc"] == pytest.approx(
    binary_metrics(hard["assay"].to_numpy(), hpred)["auroc"]
  )


def test_hard_labels_match_by_suffix(fitted):
  """`assay` names `assay_probability`, exactly as prepare matches them."""
  tmp_path, art = fitted
  out = tmp_path / "report"
  assert (
    _run(["validate", "-m", str(art), "-h", str(tmp_path / "heldout_hard.csv"), "-o", str(out)]).exit_code
    == 0
  )
  metrics = json.loads((out / "metrics.json").read_text())
  assert "assay_probability" in metrics["hard"]["metrics"]
  assert metrics.get("soft") is None  # no soft labels were given


def test_extra_columns_are_ignored_not_fatal(fitted):
  """A validation file often carries columns this model knows nothing about."""
  tmp_path, art = fitted
  frame = pd.read_csv(tmp_path / "heldout_soft.csv")
  frame["some_other_assay"] = 0.5
  frame["notes"] = "x"
  frame.to_csv(tmp_path / "wide.csv", index=False)

  r = _run(["validate", "-m", str(art), "-s", str(tmp_path / "wide.csv"), "-o", str(tmp_path / "rep")])
  assert r.exit_code == 0, r.output
  metrics = json.loads((tmp_path / "rep" / "metrics.json").read_text())
  assert list(metrics["soft"]["metrics"]) == ["assay_probability"]
  assert any("ignored" in n for n in metrics["notes"])


def test_validating_on_the_training_library_is_flagged(fitted):
  """Scoring the library the model was distilled from measures fit, not generalisation."""
  tmp_path, art = fitted
  r = _run(["validate", "-m", str(art), "-s", str(tmp_path / "soft.csv"), "-o", str(tmp_path / "rep")])
  assert r.exit_code == 0, r.output
  metrics = json.loads((tmp_path / "rep" / "metrics.json").read_text())
  assert any("reference library" in n for n in metrics["notes"]), metrics["notes"]


def test_internals_are_reported_without_any_labels(fitted):
  """A model can describe itself with no data at all."""
  tmp_path, art = fitted
  out = tmp_path / "rep"
  r = _run(["validate", "-m", str(art), "-o", str(out)])
  assert r.exit_code == 0, r.output
  metrics = json.loads((out / "metrics.json").read_text())
  assert metrics["internals"]["assay_probability"]["n_trees"] > 0
  assert metrics["figures"]["internals"], "the calibration curves should still be drawn"
  assert metrics.get("soft") is None and metrics.get("hard") is None


def test_a_file_with_no_matching_column_is_refused(fitted):
  tmp_path, art = fitted
  pd.DataFrame({"smiles": _HELD_OUT, "unrelated": 1.0}).to_csv(tmp_path / "nope.csv", index=False)
  r = _run(["validate", "-m", str(art), "-s", str(tmp_path / "nope.csv"), "-o", str(tmp_path / "rep")])
  assert r.exit_code != 0
  assert "match this model's tasks" in r.output.replace("\n", " ")


def test_the_report_extra_is_required_with_a_clear_message(monkeypatch):
  """Without the extra, say which package and how to get it — not an ImportError from deep inside."""
  import builtins

  from olinda.report import require_report_extra

  real = builtins.__import__

  def blocked(name, *args, **kwargs):
    if name == "stylia":
      raise ImportError("no stylia")
    return real(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", blocked)
  with pytest.raises(RuntimeError, match=r'pip install "olinda\[report\]"'):
    require_report_extra()


def test_scoring_a_hard_head_on_its_own_training_labels_is_flagged(fitted):
  """Handing back the measurements the head was trained on gives a near-perfect, meaningless AUROC.

  The artifact records how many compounds the head saw, not which, so a matching row count is the
  strongest signal available — and it is worth saying, because the number looks superb either way.
  """
  tmp_path, art = fitted
  out = tmp_path / "insample"
  r = _run(["validate", "-m", str(art), "-h", str(tmp_path / "hard.csv"), "-o", str(out)])
  assert r.exit_code == 0, r.output
  notes = json.loads((out / "metrics.json").read_text())["notes"]
  assert any("in-sample" in n for n in notes), notes


def test_a_differently_sized_hard_set_still_mentions_the_training_size(fitted):
  tmp_path, art = fitted
  out = tmp_path / "heldout"
  assert (
    _run(["validate", "-m", str(art), "-h", str(tmp_path / "heldout_hard.csv"), "-o", str(out)]).exit_code
    == 0
  )
  notes = json.loads((out / "metrics.json").read_text())["notes"]
  assert any("trained on" in n for n in notes), notes
