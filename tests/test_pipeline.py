"""`olinda fit` end to end, through the real CLI, on a synthetic library.

These are the promises the README makes, exercised the way a user would: distil a teacher, predict
from the artifact, and keep one self-describing file. A multi-column run is the only run — a
single-column model is its one-element case — so everything here uses two or three columns and the
one-column path is covered by the same code.

Both engines run: LightGBM is what `select_backend` picks on any CPU machine, which is most users and
all of CI, and XGBoost is the CUDA path. Pinning one "for determinism" is how the other went
untested for a while.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import pandas as pd  # noqa: E402

from olinda import OlindaArtifact  # noqa: E402
from tests.conftest import peek_internals, plain, run_cli, write_hard, write_soft  # noqa: E402

ROUNDS = ["--num-boost-round", "40"]


def _fit(soft, model, *extra):
  result = run_cli(["fit", "-s", soft, "-m", model, "--val-frac", "0.2", *ROUNDS, *extra])
  assert result.exit_code == 0, result.output
  return result


def _steps(soft, run_dir, hard=None, *extra, stop_after: str | None = None):
  """Drive the pipeline one command at a time — everything `fit` does except the closing `clean`.

  Tests that inspect the manifest or a column's artifacts have to take this path: `fit` deletes the
  working folder and hands back only the fused model. ``stop_after="soft"`` prepares the hard labels
  but leaves `learn-hard` to the caller, which is how a test observes the before-and-after of fusing
  a hard head into an otherwise identical run.
  """
  prepare = ["prepare", "-s", soft, "-m", run_dir, "--val-frac", "0.2", *extra]
  if hard is not None:
    prepare += ["-h", hard]
  assert (r := run_cli(prepare)).exit_code == 0, r.output
  assert (r := run_cli(["learn-soft", "-m", run_dir, *ROUNDS])).exit_code == 0, r.output
  if hard is not None and stop_after != "soft":
    assert (r := run_cli(["learn-hard", "-m", run_dir])).exit_code == 0, r.output
  return run_dir


def test_fit_then_predict(tmp_path, library, backend):
  """The core promise: distil a teacher, and get one file that scores SMILES."""
  home, smiles, x = library
  soft = write_soft(tmp_path, smiles, x)
  model = tmp_path / "model.onnx"
  _fit(soft, model)

  # what you keep is the artifact, and nothing else
  assert model.is_file()
  assert not (tmp_path / "model").exists()

  query = tmp_path / "q.csv"
  pd.DataFrame({"smiles": smiles[:5]}).to_csv(query, index=False)
  out = tmp_path / "pred.csv"
  assert (r := run_cli(["predict", "-m", model, "-i", query, "-o", out])).exit_code == 0, r.output

  df = pd.read_csv(out)
  assert list(df.columns) == ["smiles", "activity_probability", "solubility_probability"]
  assert len(df) == 5
  assert np.isfinite(df.iloc[:, 1:].to_numpy()).all()
  # different targets must give different predictions — not one model broadcast across columns
  assert not np.allclose(df["activity_probability"], df["solubility_probability"])


def test_hard_labels_blend_into_the_prediction(tmp_path, library, backend):
  """A wide measurement file gives exactly the columns it names a hard head, matched by suffix."""
  home, smiles, x = library
  soft = write_soft(tmp_path, smiles, x, columns=("activity", "solubility"))
  hard = write_hard(tmp_path, smiles, x, column="activity")  # 'activity' → 'activity_probability'
  model = tmp_path / "model.onnx"
  _fit(soft, model, "-h", hard)

  artifact = OlindaArtifact(model)
  assert artifact.has_hard is True
  # matched by suffix; the unnamed column stays soft-only, silently
  assert artifact.roles_for("activity_probability") == ["soft", "hard"]
  assert artifact.roles_for("solubility_probability") == ["soft"]

  # A blended column declares exactly one output. S, H_S and the weight a are how that number is
  # computed, not part of it, so the graph keeps them to itself.
  import onnx

  assert [o.name for o in onnx.load(str(model)).graph.output] == artifact.columns

  # ...and that one number must *be* the blend. Read the parts by promoting them on a copy: this is
  # the assertion that would catch a cross-wired branch, and it costs the shipped artifact nothing.
  parts = peek_internals(model, smiles[:6], ("c0_s", "c0_h_s", "c0_a"))
  a = parts["c0_a"]
  np.testing.assert_allclose(
    parts["activity_probability"],
    (1.0 - a) * parts["c0_s"] + a * parts["c0_h_s"],
    atol=1e-9,
  )
  # the hard head has to actually move the answer, or fusing it in was pointless
  assert np.any(a > 0.0), "the hard head earned no weight at all"


def test_the_artifact_describes_itself(tmp_path, library):
  """Strict JSON, and enough on its own to run and interpret the model."""
  home, smiles, x = library
  soft = write_soft(tmp_path, smiles, x)
  hard = write_hard(tmp_path, smiles, x)
  model = tmp_path / "model.onnx"
  _fit(soft, model, "-h", hard)

  artifact = OlindaArtifact(model)

  def reject_constant(c):
    raise ValueError(f"non-finite {c} is not valid JSON")

  # Metrics can be NaN, which Python's json emits as `NaN` and every strict parser rejects.
  meta = json.loads(artifact.to_json(), parse_constant=reject_constant)
  assert meta["schema"] == "olinda.bundle.v1"
  assert meta["producer"] == "olinda"
  assert meta["trained_at"].endswith("+00:00")
  assert meta["featurizer"]["rdkit_version"] == artifact.rdkit_version
  assert [c["name"] for c in meta["columns"]] == artifact.columns

  # Every head states its own task rather than leaving it to be inferred from which metrics appear.
  heads = {h["role"]: h for h in artifact.heads_for("activity_probability")}
  assert heads["soft"]["task"]["type"] == "regression"
  assert heads["hard"]["task"]["type"] == "classification"
  assert heads["hard"]["source"] == {"kind": "measured", "column": "activity"}
  assert heads["soft"]["training"]["n_train"] > 0


def test_clean_leaves_one_file_and_the_same_predictions(tmp_path, library):
  """The claim the whole step rests on: finishing a run predicts identically."""
  home, smiles, x = library
  soft = write_soft(tmp_path, smiles, x)
  hard = write_hard(tmp_path, smiles, x)
  run_dir = _steps(soft, tmp_path / "run", hard)

  before = OlindaArtifact(run_dir).run(smiles)
  assert (r := run_cli(["clean", "-m", tmp_path / "run.onnx"])).exit_code == 0, r.output
  assert (tmp_path / "run.onnx").is_file() and not run_dir.exists()

  after = OlindaArtifact(tmp_path / "run.onnx").run(smiles)
  pd.testing.assert_frame_equal(before, after)


def test_max_samples_limits_every_stage(tmp_path, library):
  """`--max-samples N` means "the first N reference compounds" for learn-hard too, not just the split.

  It used to bound only the student's train/val split, so learn-hard still scored the whole library,
  calibrated on it and trained the gate over it — a "quick check" cost the better part of an hour.
  The README documents the flag as a plumbing check, which is only true if it reaches every stage.
  """
  from olinda import run as runlib
  from olinda.hard import HARD_DIRNAME, HARD_EVAL_NAME, HARD_META_NAME

  home, smiles, x = library
  limit = 120  # of the staged rows
  soft = write_soft(tmp_path, smiles, x, columns=("activity",))
  # labelled compounds from inside the limited view, so the gate has neighbours to learn from
  hard = write_hard(tmp_path, smiles, x, rows=limit)

  run_dir = _steps(soft, tmp_path / "run", hard, "--max-samples", str(limit))
  manifest = runlib.read_manifest(run_dir)
  assert runlib.row_limit(manifest) == limit
  # the library on disk is untouched, and the manifest still records its true size
  assert manifest["reference_library"]["n_rows"] == len(smiles)

  root = run_dir / "columns" / "c0" / HARD_DIRNAME
  # the gate scanned exactly the limited view...
  assert json.loads((root / HARD_META_NAME).read_text())["tanimoto"]["n_ref"] == limit
  # ...and the isotonic map was fitted on it, not on the whole library
  assert json.loads((root / HARD_EVAL_NAME).read_text())["calibration"]["n_reference"] <= limit


def test_a_changed_reference_library_is_refused(tmp_path, library):
  """Splits are positional indices, so a swapped library would silently mispair features and labels."""
  from tests.conftest import write_library

  home, smiles, x = library
  soft = write_soft(tmp_path, smiles, x)
  run_dir = tmp_path / "run"
  assert (r := run_cli(["prepare", "-s", soft, "-m", run_dir, "--val-frac", "0.2"])).exit_code == 0

  write_library(home, n_rows=len(smiles) - 40)  # regenerate it shorter, as `olinda setup` might

  r = run_cli(["learn-soft", "-m", run_dir, *ROUNDS])
  assert r.exit_code != 0
  assert "library changed" in plain(r) or "prepared against" in plain(r)


def test_a_hard_head_that_earns_no_weight_ships_soft_only(tmp_path, library):
  """ "One that loses to the surrogate is dropped, leaving the column soft-only" — the README's claim.

  A teacher with no variance to explain gives an undefined R², so the ceiling comes out zero. That has
  to *disable* the blend, not crash the run and not brick the fuse: with ``a == 0`` the prediction is
  exactly the surrogate, and the graph must still build, load and predict.
  """
  import numpy as np

  from olinda.hard import HARD_DIRNAME, HARD_META_NAME

  home, smiles, x = library
  flat = tmp_path / "flat.csv"
  pd.DataFrame({"smiles": smiles, "activity_probability": np.full(len(smiles), 0.42)}).to_csv(
    flat, index=False
  )
  hard = write_hard(tmp_path, smiles, x)

  # Prepare *with* hard labels but stop after the surrogate, so the soft-only prediction is on record
  # before the hard head is ever fused in.
  run_dir = _steps(flat, tmp_path / "run", hard, stop_after="soft")
  surrogate = OlindaArtifact(run_dir).run(smiles[:6])["activity_probability"].to_numpy()

  assert (r := run_cli(["learn-hard", "-m", run_dir])).exit_code == 0, r.output
  meta = json.loads((run_dir / "columns" / "c0" / HARD_DIRNAME / HARD_META_NAME).read_text())
  assert meta["tanimoto"]["a_max"] == 0.0, "no variance to explain ⇒ no weight earned"

  # The fuse must still produce a working model, and with no weight earned it must predict exactly
  # what the surrogate alone predicted — the hard branch contributes nothing.
  blended = OlindaArtifact(run_dir).run(smiles[:6])["activity_probability"].to_numpy()
  np.testing.assert_allclose(blended, surrogate, atol=1e-12)
  assert np.isfinite(blended).all()


def test_selecting_columns_decides_what_the_model_predicts(tmp_path, library):
  """`--soft-label-columns` is the contract: what you name is what the artifact ends up predicting."""
  home, smiles, x = library
  soft = write_soft(tmp_path, smiles, x, columns=("activity", "solubility", "toxicity"))
  model = tmp_path / "model.onnx"
  # deliberately out of file order — the selection decides the order too
  _fit(soft, model, "--soft-label-columns", "toxicity_probability,activity_probability")
  assert OlindaArtifact(model).columns == ["toxicity_probability", "activity_probability"]

  # naming a column that is not there fails before any training happens
  r = run_cli(["prepare", "-s", soft, "-m", tmp_path / "nope", "--soft-label-columns", "absent"])
  assert r.exit_code != 0
  assert "absent" in plain(r)
  assert not (tmp_path / "nope" / "columns").exists()
