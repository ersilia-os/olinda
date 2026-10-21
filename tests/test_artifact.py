"""The public inference API: a distilled model.onnx must be usable, and honest, on its own.

These are tests of the *file*, not of training — how a model was fitted is irrelevant to a caller who
was handed one. So they fuse a minimal artifact once per session and then only read it, except where
a guard can only be reached by making the file lie about itself.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("xgboost")

from olinda import GoldenMismatch, GoldenMismatchWarning, OlindaArtifact  # noqa: E402
from tests.conftest import corrupt_metadata  # noqa: E402

_SM = [
    "CCO",
    "CCN",
    "CCC",
    "c1ccccc1",
    "CC(=O)O",
    "CCOC(=O)C",
    "Clc1ccccc1",
    "COc1ccccc1",
]


def test_the_onnx_is_the_only_input_required(tmp_path, artifact):
    """No run directory, no config, no sidecar files — move the file anywhere and it still runs."""
    moved = tmp_path / "elsewhere" / "shipped.onnx"
    moved.parent.mkdir()
    moved.write_bytes(artifact.read_bytes())

    model = OlindaArtifact(moved)
    values = model.run(_SM[:4])
    assert values.shape == (4, 1)
    assert model.columns == ["assay_probability"]
    assert np.isfinite(values).all()


def test_run_returns_one_column_per_task(artifact):
    """A single-task model is just the one-column case — no channels, no special mode."""
    model = OlindaArtifact(artifact)
    values = model.run(_SM[:3])
    assert values.shape == (3, model.n_columns)
    assert values.dtype == np.float64
    assert model.n_columns == len(model.columns) == 1
    assert model.columns == ["assay_probability"]


def test_parallel_featurization_changes_nothing(artifact):
    """Workers featurize ahead of the graph; the rows, their order and every number must be the same."""
    model = OlindaArtifact(artifact)
    smiles = [*_SM, "not a smiles", *_SM]
    with pytest.warns(RuntimeWarning, match="could not be parsed"):
        serial = model.run(smiles, batch_size=3)
    with pytest.warns(RuntimeWarning, match="could not be parsed"):
        parallel = model.run(smiles, batch_size=3, workers=2)
    assert np.array_equal(parallel, serial, equal_nan=True)


def test_the_artifact_describes_itself(artifact):
    model = OlindaArtifact(artifact)
    described = model.describe()
    assert described["producer"] == "olinda"
    assert described["trained_at"].endswith("+00:00")
    assert described["rdkit_version"] == model.rdkit_version
    assert described["featurizer_rdkit_version"] == model.rdkit_version
    assert described["golden"] == "pass"
    assert described["n_features"] == 2048
    assert model.has_hard is False
    assert model.roles_for("assay_probability") == ["soft"]
    assert model.task_for("assay_probability")["type"] == "regression"


def test_a_directory_is_accepted_as_well_as_the_file(artifact):
    assert len(OlindaArtifact(artifact.parent).run(_SM[:2])) == 2


def test_batching_does_not_change_results(artifact):
    """batch_size bounds memory on large inputs; it must not be observable in the answers."""
    model = OlindaArtifact(artifact)
    many = (_SM * 3)[:20]
    np.testing.assert_allclose(
        model.run(many, batch_size=1000), model.run(many, batch_size=3)
    )


def test_a_tampered_golden_prediction_warns_or_raises_in_strict_mode(
    tmp_path, artifact
):
    """The golden check is the arbiter of whether this install reproduces the model.

    A prediction that no longer matches means the graph is not running as built: warn by default, so a
    caller is told, and refuse in strict mode, so a pipeline can insist.
    """

    def shift(golden):
        name = next(iter(golden["outputs"]))
        golden["outputs"][name][0] += 1.0

    path = corrupt_metadata(
        artifact, tmp_path / "shifted.onnx", shift, key="olinda.golden"
    )
    with pytest.warns(GoldenMismatchWarning, match="score differently"):
        model = OlindaArtifact(path)
    assert model.describe()["golden"].startswith("fail")
    with pytest.raises(GoldenMismatch):
        OlindaArtifact(path, strict=True)


def test_a_fingerprint_digest_mismatch_is_blamed_on_the_featurizer(tmp_path, artifact):
    """Different bytes are a featurizer problem, not a graph one, and the message must say which."""

    def scramble(golden):
        golden["fp_sha256"][0] = "0" * 16

    path = corrupt_metadata(
        artifact, tmp_path / "digest.onnx", scramble, key="olinda.golden"
    )
    with pytest.warns(GoldenMismatchWarning, match="featurize differently"):
        OlindaArtifact(path)


def test_an_rdkit_version_difference_only_warns(tmp_path, artifact):
    """A version string proves nothing on its own: the golden vectors decide, and here they agree."""
    path = corrupt_metadata(
        artifact,
        tmp_path / "version.onnx",
        lambda _: "0.0.0",
        key="olinda.rdkit_version",
    )
    with pytest.warns(RuntimeWarning, match="RDKit 0.0.0"):
        model = OlindaArtifact(path)
    assert model.describe()["golden"] == "pass"
    assert len(model.run(_SM[:2])) == 2


def test_an_olinda_1_model_is_refused_with_a_way_forward(tmp_path, artifact):
    """A file with only the old ``olinda`` key fed Morgan counts from rdkit; it cannot run here."""
    import onnx

    model = onnx.load(str(artifact))
    del model.metadata_props[:]
    legacy = model.metadata_props.add()
    legacy.key, legacy.value = "olinda", "{}"
    path = tmp_path / "legacy.onnx"
    onnx.save(model, str(path))
    with pytest.raises(ValueError, match="olinda 1.x"):
        OlindaArtifact(path)


@pytest.mark.parametrize(
    ("key", "mutate", "message"),
    [
        ("olinda.schema_version", None, "no olinda metadata"),
        ("olinda.schema_version", lambda _: "2.0", "schema"),
        (
            "olinda.model",
            lambda meta: meta["columns"][0].update(output="no_such_output"),
            "does not produce",
        ),
        ("olinda.fingerprint", lambda spec: spec.update(type="morgan"), "qmfp only"),
    ],
    ids=[
        "no metadata",
        "unknown schema",
        "output the graph lacks",
        "unknown fingerprint",
    ],
)
def test_a_file_that_lies_about_itself_is_refused_at_load(
    tmp_path, artifact, key, mutate, message
):
    """Every mismatch between the metadata and the graph must fail at load, not mid-prediction.

    A KeyError from inside `run()` on the thousandth molecule is the outcome being prevented here, and
    an unknown schema is refused rather than guessed at: the graph would be fine, but this code cannot
    say what its metadata means.
    """
    path = corrupt_metadata(artifact, tmp_path / "broken.onnx", mutate, key=key)
    with pytest.raises(ValueError, match=message):
        OlindaArtifact(path)


def test_unparseable_smiles_yield_nan_and_a_warning(artifact):
    """An all-zero fingerprint scores perfectly happily — refuse to report it as a prediction."""
    import warnings

    model = OlindaArtifact(artifact)
    mixed = ["CCO", "not_a_smiles", "", "C1CC", "c1ccccc1"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = model.run(mixed)[:, 0]
    assert np.isfinite(values[0]) and np.isfinite(
        values[4]
    )  # the real molecules still predict
    assert np.isnan(values[1:4]).all()  # garbage, an empty string, and an unclosed ring
    assert "3 of 5" in str(caught[0].message)

    # and the inverse: clean input must not cry wolf
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.run(["CCO", "c1ccccc1"])
    assert not [w for w in caught if "could not be parsed" in str(w.message)]


def test_inputs_that_would_mislead_are_refused(artifact):
    """``run("CCO")`` must not silently score C, C and O as three molecules.

    A string is a valid sequence, so without this guard the most natural mistake a caller can make —
    one molecule instead of a list of them — returns an array of confident nonsense. A set is refused
    for the same reason in reverse: it would scramble the row order against the caller's input.
    """
    model = OlindaArtifact(artifact)
    with pytest.raises(TypeError, match="single string"):
        model.run("CCO")
    with pytest.raises(TypeError, match="order matters"):
        model.run({"CCO", "CCC"})
    with pytest.raises(TypeError, match="sequence of SMILES"):
        model.run(42)
    for bad in (
        0,
        -1,
    ):  # zero used to reach range() and fail with 'arg 3 must not be zero'
        with pytest.raises(ValueError, match="at least 1"):
            model.run(["CCO"], batch_size=bad)


def test_every_ordinary_sequence_type_is_accepted(artifact):
    """Lists, tuples, numpy arrays, pandas Series and generators are all reasonable inputs."""
    import pandas as pd

    model = OlindaArtifact(artifact)
    expected = model.run(["CCO", "c1ccccc1"])
    for given in (
        ("CCO", "c1ccccc1"),
        np.array(["CCO", "c1ccccc1"]),
        pd.Series(["CCO", "c1ccccc1"]),
        iter(["CCO", "c1ccccc1"]),
    ):
        np.testing.assert_allclose(model.run(given), expected)
