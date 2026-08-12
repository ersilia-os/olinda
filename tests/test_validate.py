"""`olinda validate`: scoring a finished artifact against data of your choosing.

The model is fitted once for the whole module. Every test here only reads it and writes a report into
its own directory, and a real `fit` is by far the most expensive thing in the suite.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("xgboost")
pytest.importorskip("stylia")

import pandas as pd  # noqa: E402

from tests.conftest import plain, run_cli  # noqa: E402

# The library the model is distilled from, and a set deliberately disjoint from it — held-out data is
# the whole point of `validate`, so the two must not share molecules.
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

TASK = "assay_probability"


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
  """``(dir, model.onnx)`` for a one-column model with a hard head, plus held-out label files.

  Module-scoped, so ``OLINDA_HOME`` is set on the modules that read it rather than monkeypatched, and
  restored on teardown. It stays set for the whole module because `validate` reads the library back to
  decide whether the compounds it was handed are the ones the model was fitted on.
  """
  import h5py

  import olinda.data as data_pkg
  import olinda.data.fetch as fetch
  from olinda.featurizer import MorganCountFeaturizer

  tmp_path = tmp_path_factory.mktemp("validate")
  home = tmp_path / "home"
  home.mkdir()
  saved = (fetch.OLINDA_HOME, data_pkg.OLINDA_HOME, os.environ.get("OLINDA_BACKEND"))
  fetch.OLINDA_HOME = home
  data_pkg.OLINDA_HOME = home
  os.environ["OLINDA_BACKEND"] = "xgboost"

  featurizer = MorganCountFeaturizer()
  library = (_TRAIN * 17)[:400]
  x = featurizer.transform(library).astype(np.uint8)
  with h5py.File(home / "erl0_morgan.h5", "w") as f:
    f.create_dataset("data", data=x)
    f.create_dataset("input", data=np.array([s.encode() for s in library]))

  def scaled(smiles):
    v = featurizer.transform(smiles)[:, :700].sum(1).astype(np.float64)
    return (v - v.min()) / (np.ptp(v) + 1e-9)

  def binary(smiles):
    score = featurizer.transform(smiles)[:, :200].sum(1)
    return (score > np.median(score)).astype(int)

  pd.DataFrame({"smiles": library, TASK: scaled(library)}).to_csv(tmp_path / "soft.csv", index=False)
  hard_smiles = _TRAIN * 3
  pd.DataFrame({"smiles": hard_smiles, "assay": binary(hard_smiles)}).to_csv(
    tmp_path / "hard.csv", index=False
  )
  held = _HELD_OUT * 5
  pd.DataFrame({"smiles": held, TASK: scaled(held)}).to_csv(tmp_path / "heldout_soft.csv", index=False)
  pd.DataFrame({"smiles": held, "assay": binary(held)}).to_csv(tmp_path / "heldout_hard.csv", index=False)

  model = tmp_path / "model.onnx"
  result = run_cli([
    "fit",
    "-s",
    tmp_path / "soft.csv",
    "-h",
    tmp_path / "hard.csv",
    "-m",
    model,
    "--num-boost-round",
    "40",
  ])
  assert result.exit_code == 0, result.output
  yield tmp_path, model

  fetch.OLINDA_HOME, data_pkg.OLINDA_HOME = saved[0], saved[1]
  if saved[2] is None:
    os.environ.pop("OLINDA_BACKEND", None)
  else:
    os.environ["OLINDA_BACKEND"] = saved[2]


def _validate(model, out, *args):
  result = run_cli(["validate", "-m", model, "-o", out, *args])
  assert result.exit_code == 0, result.output
  return json.loads((out / "metrics.json").read_text())  # strict JSON: no NaN tokens


def test_validate_writes_a_report_whose_every_reference_resolves(fitted):
  """A report that names a figure it did not write is worse than no report."""
  tmp_path, model = fitted
  out = tmp_path / "report"
  metrics = _validate(model, out, "-s", tmp_path / "heldout_soft.csv", "-h", tmp_path / "heldout_hard.csv")

  assert (out / "report.html").exists()
  assert (out / "performance_table.csv").exists()

  figures = [f for group in metrics["figures"].values() for f in group]
  assert figures, "no figures were produced"
  for figure in figures:
    assert (out / figure["png"]).exists(), f"missing {figure['png']}"
    assert (out / figure["pdf"]).exists(), f"missing {figure['pdf']}"

  # every <img> the page references must resolve, so the page is not silently broken
  page = (out / "report.html").read_text()
  for src in {s.split('"')[0] for s in page.split('src="')[1:]}:
    assert (out / src).exists(), f"report.html references a missing {src}"


def test_reported_metrics_are_the_ones_olinda_computes(fitted):
  """The numbers in the report must come from olinda.metrics, not a second implementation."""
  from olinda import OlindaArtifact
  from olinda.metrics import binary_metrics, regression_metrics

  tmp_path, model = fitted
  metrics = _validate(
    model,
    tmp_path / "match",
    "-s",
    tmp_path / "heldout_soft.csv",
    "-h",
    tmp_path / "heldout_hard.csv",
  )

  artifact = OlindaArtifact(model)
  soft = pd.read_csv(tmp_path / "heldout_soft.csv")
  expected = regression_metrics(soft[TASK].to_numpy(), artifact.run(soft["smiles"].tolist())[TASK].to_numpy())
  assert metrics["soft"]["metrics"][TASK]["r2"] == pytest.approx(expected["r2"])
  assert metrics["soft"]["metrics"][TASK]["spearman"] == pytest.approx(expected["spearman"])

  hard = pd.read_csv(tmp_path / "heldout_hard.csv")
  predicted = artifact.run(hard["smiles"].tolist())[TASK].to_numpy()
  assert metrics["hard"]["metrics"][TASK]["auroc"] == pytest.approx(
    binary_metrics(hard["assay"].to_numpy(), predicted)["auroc"]
  )


def test_column_matching_is_forgiving_but_never_silent(fitted):
  """A validation file carries whatever it carries; only a file with nothing usable is an error."""
  tmp_path, model = fitted

  # extra columns this model knows nothing about are ignored — and said so
  frame = pd.read_csv(tmp_path / "heldout_soft.csv")
  frame["some_other_assay"] = 0.5
  frame["notes"] = "x"
  frame.to_csv(tmp_path / "wide.csv", index=False)
  metrics = _validate(model, tmp_path / "wide-rep", "-s", tmp_path / "wide.csv")
  assert list(metrics["soft"]["metrics"]) == [TASK]
  assert any("ignored" in note for note in metrics["notes"])

  # a file with no matching column at all is refused, rather than reporting on nothing
  pd.DataFrame({"smiles": _HELD_OUT, "unrelated": 1.0}).to_csv(tmp_path / "nope.csv", index=False)
  result = run_cli(["validate", "-m", model, "-s", tmp_path / "nope.csv", "-o", tmp_path / "nope-rep"])
  assert result.exit_code != 0
  assert "match this model's tasks" in plain(result)


def test_in_sample_scoring_is_flagged_rather_than_reported_as_performance(fitted):
  """The failure mode this guards is a superb number with nothing to say it is meaningless.

  Both halves matter: handing back the reference library measures fit, and handing back the
  measurements the hard head was trained on gives a near-perfect AUROC. The artifact records how many
  compounds the head saw but not which, so a matching row count is the strongest signal available —
  and worth saying, because the number looks excellent either way.
  """
  tmp_path, model = fitted

  library = _validate(model, tmp_path / "libr", "-s", tmp_path / "soft.csv")
  assert any("reference library" in note for note in library["notes"]), library["notes"]

  in_sample = _validate(model, tmp_path / "insample", "-h", tmp_path / "hard.csv")
  assert any("in-sample" in note for note in in_sample["notes"]), in_sample["notes"]

  held_out = _validate(model, tmp_path / "heldout", "-h", tmp_path / "heldout_hard.csv")
  assert any("trained on" in note for note in held_out["notes"]), held_out["notes"]


def test_a_model_can_describe_itself_with_no_labels_at_all(fitted):
  """With neither file you still get the calibration curves, read straight out of the graph."""
  tmp_path, model = fitted
  metrics = _validate(model, tmp_path / "internals")
  assert metrics["internals"][TASK]["n_trees"] > 0
  assert metrics["figures"]["internals"], "the calibration curves should still be drawn"
  assert metrics.get("soft") is None and metrics.get("hard") is None


def test_every_figure_the_report_draws_has_a_title_and_caption():
  """The page reads its captions from this registry, so a new plot without an entry ships bare.

  Guards the pairing rather than the wording: `render` names files ``<task>_<figure>`` and both halves
  contain underscores, so the lookup matches the longest key the name ends with.
  """
  from olinda.report import plots

  for key, (title, text) in plots.FIGURES.items():
    assert title and text, f"{key} has an empty title or caption"
    assert plots.caption(f"some_task_name_{key}") == (title, text), f"{key} is not reachable"

  # the ambiguous pair the longest-match rule exists for
  assert plots.caption("t_soft_calibration")[0] != plots.caption("t_calibration")[0]


def test_the_page_and_the_figures_take_their_colours_from_one_table():
  """`html.py` paints its colour key from `style.hexcol`, which must agree with what the plots draw.

  `style.py` keeps an offline copy of stylia's palette so the page can be written without the report
  extra installed, and names this test as what keeps the two in step — so it has to exist.
  """
  import stylia

  from olinda import style

  for role, name in style.ROLES.items():
    from_plot = stylia.ArticleColors().hex[name]
    assert style.hexcol(role).lower() == from_plot.lower(), f"{role} disagrees with stylia"
    assert style._PAPER_FALLBACK[name].lower() == from_plot.lower(), f"{name} fallback is stale"


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
