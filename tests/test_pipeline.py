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
import os

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import pandas as pd  # noqa: E402

from olinda import OlindaArtifact  # noqa: E402
from tests.conftest import (  # noqa: E402
    peek_internals,
    plain,
    run_cli,
    write_hard,
    write_soft,
)

ROUNDS = ["--num-boost-round", "40"]


def _fit(soft, model, *extra):
    result = run_cli(
        ["fit", "-s", soft, "-m", model, "--val-frac", "0.2", *ROUNDS, *extra]
    )
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
    assert (r := run_cli(["learn-soft", "-m", run_dir, *ROUNDS])).exit_code == 0, (
        r.output
    )
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
    assert (
        r := run_cli(["predict", "-m", model, "-i", query, "-o", out])
    ).exit_code == 0, r.output

    df = pd.read_csv(out)
    assert list(df.columns) == [
        "smiles",
        "activity_probability",
        "solubility_probability",
    ]
    assert len(df) == 5
    assert np.isfinite(df.iloc[:, 1:].to_numpy()).all()
    # different targets must give different predictions — not one model broadcast across columns
    assert not np.allclose(df["activity_probability"], df["solubility_probability"])


def test_hard_labels_blend_into_the_prediction(tmp_path, library, backend):
    """A wide measurement file gives exactly the columns it names a hard head, matched by suffix."""
    home, smiles, x = library
    soft = write_soft(tmp_path, smiles, x, columns=("activity", "solubility"))
    hard = write_hard(
        tmp_path, smiles, x, column="activity"
    )  # 'activity' → 'activity_probability'
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
    recorded = json.loads(artifact.to_json(), parse_constant=reject_constant)
    assert recorded["schema_version"] == "1.0"
    assert recorded["fingerprint"]["type"] == "qmfp"
    assert recorded["rdkit_version"] == artifact.rdkit_version
    assert artifact.describe()["golden"] == "pass"
    meta = recorded["model"]
    assert meta["producer"] == "olinda"
    assert meta["trained_at"].endswith("+00:00")
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
    assert (r := run_cli(["clean", "-m", tmp_path / "run.onnx"])).exit_code == 0, (
        r.output
    )
    assert (tmp_path / "run.onnx").is_file() and not run_dir.exists()

    after = OlindaArtifact(tmp_path / "run.onnx").run(smiles)
    assert np.array_equal(before, after, equal_nan=True)


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
    assert (
        json.loads((root / HARD_EVAL_NAME).read_text())["calibration"]["n_reference"]
        <= limit
    )


def test_a_changed_reference_library_is_refused(tmp_path, library):
    """Splits are positional indices, so a swapped library would silently mispair features and labels."""
    from tests.conftest import write_library

    home, smiles, x = library
    soft = write_soft(tmp_path, smiles, x)
    run_dir = tmp_path / "run"
    assert (
        r := run_cli(["prepare", "-s", soft, "-m", run_dir, "--val-frac", "0.2"])
    ).exit_code == 0

    write_library(
        home, n_rows=len(smiles) - 40
    )  # regenerate it shorter, as `olinda setup` might

    r = run_cli(["learn-soft", "-m", run_dir, *ROUNDS])
    assert r.exit_code != 0
    assert "library changed" in plain(r) or "prepared against" in plain(r)


def test_a_hard_head_that_earns_no_weight_ships_soft_only(tmp_path, library):
    """ "One that loses to the surrogate is dropped, leaving the column soft-only" — the README's claim.

    A teacher that is noise unrelated to the molecules gives the hard head nothing to explain, so the
    ceiling comes out zero. That has to *disable* the blend, not crash the run and not brick the fuse: with
    ``a == 0`` the prediction is exactly the surrogate, and the graph must still build, load and predict.
    (A truly constant teacher no longer gets this far: `prepare` refuses it.)
    """
    import numpy as np

    from olinda.hard import HARD_DIRNAME, HARD_META_NAME

    home, smiles, x = library
    flat = tmp_path / "flat.csv"
    pd.DataFrame(
        {
            "smiles": smiles,
            "activity_probability": 0.42
            + np.random.default_rng(0).normal(0, 1e-3, len(smiles)),
        }
    ).to_csv(flat, index=False)
    hard = write_hard(tmp_path, smiles, x)

    # Prepare *with* hard labels but stop after the surrogate, so the soft-only prediction is on record
    # before the hard head is ever fused in.
    run_dir = _steps(flat, tmp_path / "run", hard, stop_after="soft")
    surrogate = OlindaArtifact(run_dir).run(smiles[:6])[:, 0]

    assert (r := run_cli(["learn-hard", "-m", run_dir])).exit_code == 0, r.output
    meta = json.loads(
        (run_dir / "columns" / "c0" / HARD_DIRNAME / HARD_META_NAME).read_text()
    )
    assert meta["tanimoto"]["a_max"] == 0.0, "nothing to explain ⇒ no weight earned"

    # The fuse must still produce a working model, and with no weight earned it must predict exactly
    # what the surrogate alone predicted — the hard branch contributes nothing.
    blended = OlindaArtifact(run_dir).run(smiles[:6])[:, 0]
    np.testing.assert_allclose(blended, surrogate, atol=1e-12)
    assert np.isfinite(blended).all()


def test_selecting_columns_decides_what_the_model_predicts(tmp_path, library):
    """`--soft-label-columns` is the contract: what you name is what the artifact ends up predicting."""
    home, smiles, x = library
    soft = write_soft(
        tmp_path, smiles, x, columns=("activity", "solubility", "toxicity")
    )
    model = tmp_path / "model.onnx"
    # deliberately out of file order — the selection decides the order too
    _fit(
        soft, model, "--soft-label-columns", "toxicity_probability,activity_probability"
    )
    assert OlindaArtifact(model).columns == [
        "toxicity_probability",
        "activity_probability",
    ]

    # naming a column that is not there fails before any training happens
    r = run_cli(
        [
            "prepare",
            "-s",
            soft,
            "-m",
            tmp_path / "nope",
            "--soft-label-columns",
            "absent",
        ]
    )
    assert r.exit_code != 0
    assert "absent" in plain(r)
    assert not (tmp_path / "nope" / "columns").exists()


def test_prepare_refuses_a_directory_that_already_holds_a_run(tmp_path, library):
    """Re-preparing over an old run would fuse its boosters and hard heads under the new run's names."""
    home, smiles, x = library
    soft = write_soft(tmp_path, smiles, x, columns=("activity",))
    run = tmp_path / "run"
    assert run_cli(["prepare", "-s", soft, "-m", run]).exit_code == 0
    again = run_cli(["prepare", "-s", soft, "-m", run])
    assert again.exit_code != 0
    assert "already holds a prepared run" in plain(again)


def test_a_missing_input_file_is_a_usage_error_not_a_traceback(tmp_path, library):
    result = run_cli(["prepare", "-s", tmp_path / "nope.csv", "-m", tmp_path / "run"])
    assert result.exit_code == 2
    assert "does not exist" in plain(result)
    assert not (tmp_path / "run").exists()


def test_a_fit_that_fails_in_prepare_leaves_no_folder_behind(tmp_path, library):
    home, smiles, x = library
    short = tmp_path / "short.csv"
    pd.DataFrame({"smiles": smiles[:5], "y": [0.1] * 5}).to_csv(short, index=False)
    result = run_cli(["fit", "-s", short, "-m", tmp_path / "out" / "model.onnx"])
    assert result.exit_code != 0
    assert "must have" in plain(result)
    assert not (tmp_path / "out" / "model").exists()


def test_prepare_refuses_a_constant_column(tmp_path, library):
    home, smiles, x = library
    flat = tmp_path / "flat.csv"
    pd.DataFrame({"smiles": smiles, "y": [0.5] * len(smiles)}).to_csv(flat, index=False)
    result = run_cli(["prepare", "-s", flat, "-m", tmp_path / "run"])
    assert result.exit_code != 0
    assert "is constant" in plain(result)


def test_prepare_refuses_a_column_with_almost_no_finite_values(tmp_path, library):
    home, smiles, x = library
    sparse = tmp_path / "sparse.csv"
    values = [float("nan")] * len(smiles)
    values[:3] = [0.1, 0.2, 0.3]
    pd.DataFrame({"smiles": smiles, "y": values}).to_csv(sparse, index=False)
    result = run_cli(["prepare", "-s", sparse, "-m", tmp_path / "run"])
    assert result.exit_code != 0
    assert "finite value" in plain(result)


def test_learn_soft_refuses_a_model_too_big_for_one_onnx_file_before_training(
    tmp_path, library
):
    """The 2 GB limit is only met after every column trains; the worst case is known beforehand."""
    home, smiles, x = library
    soft = write_soft(tmp_path, smiles, x, columns=("activity",))
    run = tmp_path / "run"
    assert run_cli(["prepare", "-s", soft, "-m", run]).exit_code == 0
    result = run_cli(["learn-soft", "-m", run, "--num-boost-round", "200000"])
    assert result.exit_code != 0
    assert "--num-boost-round" in plain(result) and "limit" in plain(result)
    assert not (run / "columns" / "c0" / "train_meta.json").exists()


def test_boosters_converted_in_parallel_give_the_same_predictions(
    tmp_path, library, backend, monkeypatch
):
    """Large boosters are converted in worker processes; the model they make must be the one a serial run makes.

    The booster is trained once and fused twice. Training two boosters would compare two trainings, and
    multithreaded boosting is not bit-for-bit repeatable across machines.
    """
    import shutil

    import onnxruntime as ort

    from olinda.export import heads
    from olinda.featurizer.codec import pack
    from olinda.train.backend import CANONICAL_DEFAULTS

    home, smiles, x = library
    monkeypatch.setitem(CANONICAL_DEFAULTS, "min_child_weight", 1.0)
    soft = write_soft(tmp_path, smiles, x)
    monkeypatch.setattr(
        heads, "_PARALLEL_MIN_BYTES", 0
    )  # every booster counts as large
    parallel = tmp_path / "parallel"
    assert (
        run_cli(["prepare", "-s", soft, "-m", parallel, "--val-frac", "0.2"]).exit_code
        == 0
    )
    assert (
        run_cli(["learn-soft", "-m", parallel, "--num-boost-round", "40"]).exit_code
        == 0
    )

    serial = tmp_path / "serial"
    shutil.copytree(parallel, serial)
    for kept in serial.glob("columns/*/" + heads.SOFT_ONNX_NAME):
        kept.unlink()  # so the fuse has to convert again
    (serial / "model.onnx").unlink()
    monkeypatch.setattr(heads, "_PARALLEL_MIN_BYTES", 10**12)  # none does
    assert run_cli(["export", "-m", serial]).exit_code == 0

    probe = pack(x[:50].astype("uint8"))
    outs = [
        ort.InferenceSession(str(run / "model.onnx")).run(None, {"input": probe})
        for run in (parallel, serial)
    ]
    for a, b in zip(*outs):
        assert np.array_equal(a, b)


def test_conversion_workers_are_bounded_by_free_memory():
    """Four 260 MB boosters would need ~7 GB each; a machine without that must convert fewer at once."""
    from olinda.export import heads

    big = 260_000_000
    assert heads._convert_workers(8, big, available=64 * 2**30) == min(
        8, os.cpu_count() or 1, heads._MAX_CONVERT_WORKERS
    )
    assert heads._convert_workers(8, big, available=16 * 2**30) <= 1
    assert heads._convert_workers(8, big, available=0) == 1
    assert heads._convert_workers(8, 10**6, available=None) <= 2  # unknown is scarce


def test_a_killed_conversion_leaves_no_file_to_be_trusted(tmp_path):
    """The converter writes in place; the file the fuse trusts must only ever appear whole."""
    from olinda.export import heads

    class Dies:
        def to_onnx(self, model, path, input_dim):
            path.write_bytes(b"half a model")
            raise MemoryError

    final = tmp_path / heads.SOFT_ONNX_NAME
    with pytest.raises(MemoryError):
        heads._convert_to(Dies(), None, 4, final)
    assert list(tmp_path.iterdir()) == []

    class Works:
        def to_onnx(self, model, path, input_dim):
            path.write_bytes(b"whole")

    heads._convert_to(Works(), None, 4, final)
    assert final.read_bytes() == b"whole" and [p.name for p in tmp_path.iterdir()] == [
        heads.SOFT_ONNX_NAME
    ]


def test_an_unreadable_kept_conversion_is_made_again(tmp_path, library, monkeypatch):
    """A corrupt ``s.onnx`` newer than its booster passes the freshness test, so it is also read."""
    import onnx

    from olinda.export import heads
    from olinda.models.bundle import StudentModel
    from olinda.train.backend import CANONICAL_DEFAULTS

    home, smiles, x = library
    monkeypatch.setitem(CANONICAL_DEFAULTS, "min_child_weight", 1.0)
    soft = write_soft(tmp_path, smiles, x, columns=("activity",))
    run = tmp_path / "run"
    assert run_cli(["prepare", "-s", soft, "-m", run]).exit_code == 0
    assert run_cli(["learn-soft", "-m", run, "--num-boost-round", "20"]).exit_code == 0
    column = run / "columns" / "c0"
    sm = StudentModel.load(column)
    (column / heads.SOFT_ONNX_NAME).write_bytes(b"truncated")
    proto = heads._soft_model_proto(sm, int(sm.metadata["x_dim"]), column)
    onnx.checker.check_model(proto)
    assert (column / heads.SOFT_ONNX_NAME).stat().st_size > 100


def _write_continuous_hard(
    tmp_path, smiles, x, column="activity", scale=40.0, skewed=False
):
    """Measurements on their own scale (here ~5 to ~45), not a probability: a concentration, say.

    ``skewed`` makes them positive with a long right tail, which lazy-qsar trains with a Tweedie objective.
    """
    from tests.conftest import _target

    take = np.arange(0, len(smiles), 3)
    rng = np.random.default_rng(3)
    values = 5.0 + scale * _target(x, 0)[take] + rng.normal(0, 0.5, len(take))
    if skewed:
        values = 10.0 * np.exp(
            4.0 * _target(x, 0)[take] + rng.normal(0, 0.1, len(take))
        )
    path = tmp_path / "hard_continuous.csv"
    pd.DataFrame({"smiles": [smiles[i] for i in take], column: values}).to_csv(
        path, index=False
    )
    return path


@pytest.mark.parametrize("skewed", [False, True], ids=["gaussian", "skewed-positive"])
def test_continuous_hard_labels_blend_into_the_prediction(tmp_path, library, skewed):
    """A regression head is fused, calibrated onto the teacher's scale, and capped lower than a classifier."""
    from olinda.hard import HARD_DIRNAME, HARD_META_NAME
    from olinda.tanimoto import A_CEILING_REGRESSION

    home, smiles, x = library
    soft = write_soft(tmp_path, smiles, x, columns=("activity", "solubility"))
    hard = _write_continuous_hard(tmp_path, smiles, x, skewed=skewed)
    run_dir = _steps(soft, tmp_path / "run", hard)
    if skewed:  # the case whose export drops the link function: make sure it is the one being run
        from olinda.hard import HARD_MODEL_SUBDIR

        fit = json.loads(
            (
                run_dir
                / "columns"
                / "c0"
                / HARD_DIRNAME
                / HARD_MODEL_SUBDIR
                / "xgboost.json"
            ).read_text()
        )
        assert fit["params"]["objective"] == "reg:tweedie"

    meta = json.loads(
        (run_dir / "columns" / "c0" / HARD_DIRNAME / HARD_META_NAME).read_text()
    )
    assert meta["task"] == "regression"
    assert 0.0 < meta["tanimoto"]["a_max"] <= A_CEILING_REGRESSION
    assert meta["tanimoto"]["a_max_ceiling"] == A_CEILING_REGRESSION

    artifact = OlindaArtifact(run_dir)
    heads = {h["role"]: h for h in artifact.heads_for("activity_probability")}
    assert heads["hard"]["task"]["type"] == "regression"
    assert artifact.roles_for("solubility_probability") == ["soft"]

    # the published number is still the blend of the surrogate and the *calibrated* head, and that
    # calibrated head lives on the teacher's scale rather than the labels' (5 to 45)
    parts = peek_internals(
        run_dir / "model.onnx", smiles[:12], ("c0_s", "c0_h_s", "c0_a")
    )
    a = parts["c0_a"]
    np.testing.assert_allclose(
        parts["activity_probability"],
        (1.0 - a) * parts["c0_s"] + a * parts["c0_h_s"],
        atol=1e-9,
    )
    assert parts["c0_h_s"].max() <= 1.0 + 1e-9
    assert np.any(a > 0.0), "the hard head earned no weight at all"


def test_a_binary_head_keeps_its_own_ceiling(tmp_path, library):
    """Adding a regression head must not move the classification path."""
    from olinda.hard import HARD_DIRNAME, HARD_META_NAME
    from olinda.tanimoto import A_CEILING

    home, smiles, x = library
    soft = write_soft(tmp_path, smiles, x, columns=("activity",))
    run_dir = _steps(soft, tmp_path / "run", write_hard(tmp_path, smiles, x))
    meta = json.loads(
        (run_dir / "columns" / "c0" / HARD_DIRNAME / HARD_META_NAME).read_text()
    )
    assert meta["task"] == "binary"
    assert meta["tanimoto"]["a_max_ceiling"] == A_CEILING


@pytest.mark.parametrize(
    "values,fragment",
    [
        (lambda n: np.full(n, 3.5), "single value"),
        (lambda n: np.linspace(0, 1, n)[:10], "needs at least"),
    ],
)
def test_unusable_continuous_hard_labels_are_refused_at_prepare(
    tmp_path, library, values, fragment
):
    home, smiles, x = library
    soft = write_soft(tmp_path, smiles, x, columns=("activity",))
    take = np.arange(0, 90)
    vals = values(len(take))
    hard = tmp_path / "bad.csv"
    pd.DataFrame(
        {"smiles": [smiles[i] for i in take[: len(vals)]], "activity": vals + 0.1}
    ).to_csv(hard, index=False)
    result = run_cli(["prepare", "-s", soft, "-m", tmp_path / "run", "-h", hard])
    assert result.exit_code != 0 and fragment in plain(result)
    assert "Traceback" not in result.output


def test_the_blend_ceiling_can_be_lowered_but_not_raised_past_the_default():
    from olinda.hard.gate import _blend_ceiling
    from olinda.tanimoto import A_CEILING

    assert _blend_ceiling(0.9) == A_CEILING  # the classifier's, unchanged
    assert _blend_ceiling(0.9, 0.25) == 0.25
    assert _blend_ceiling(0.15, 0.25) == pytest.approx(0.15)  # R² still binds
    assert _blend_ceiling(0.05, 0.25) == 0.0  # and still switches the blend off
    assert _blend_ceiling(float("nan"), 0.25) == 0.0


def test_a_regressor_export_may_be_offset_but_not_reordered(tmp_path):
    """A converter that adds a constant is harmless (the map is fitted on the ONNX); one that scrambles is not."""
    from olinda.hard import train as hard_train

    margin = np.linspace(-2, 3, 50)

    class Booster:
        def inplace_predict(self, X, iteration_range, predict_type):
            return margin[: len(X)]

    class Model:
        params_ = {"objective": "reg:tweedie"}
        booster_ = Booster()
        best_iteration_ = 0

    X = np.zeros((50, 4))

    class Offset:
        def run(self, X):
            return margin[: len(X)] + 113.78

    class Scrambled:
        def run(self, X):
            return np.random.default_rng(0).permutation(margin[: len(X)])

    hard_train._check_regressor_export(Model(), Offset(), X)
    with pytest.raises(RuntimeError, match="does not follow its booster"):
        hard_train._check_regressor_export(Model(), Scrambled(), X)
