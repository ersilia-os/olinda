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

from olinda import OlindaArtifact, RDKitVersionMismatch  # noqa: E402
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
    df = model.run(_SM[:4])
    assert list(df["smiles"]) == _SM[:4]
    assert df.columns.tolist() == ["smiles", "assay_probability"]
    assert np.isfinite(df["assay_probability"].to_numpy()).all()


def test_run_returns_one_column_per_task(artifact):
    """A single-task model is just the one-column case — no channels, no special mode."""
    model = OlindaArtifact(artifact)
    df = model.run(_SM[:3])
    assert list(df.columns) == ["smiles", *model.columns]
    assert model.n_columns == len(model.columns) == 1
    assert model.columns == ["assay_probability"]


def test_the_artifact_describes_itself(artifact):
    model = OlindaArtifact(artifact)
    described = model.describe()
    assert described["producer"] == "olinda"
    assert described["trained_at"].endswith("+00:00")
    assert described["rdkit_version"] == model.rdkit_version
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
        model.run(many, batch_size=1000)["assay_probability"].to_numpy(),
        model.run(many, batch_size=3)["assay_probability"].to_numpy(),
    )


def test_an_rdkit_mismatch_is_refused_but_can_be_waived(tmp_path, artifact):
    """Fingerprints only reproduce on the exact build, so loading must fail loudly — and deliberately not.

    `check_rdkit=False` is the documented escape hatch, so it has to actually be reachable: a mismatch
    that raised from somewhere the flag could not suppress would leave a legitimate caller stuck.
    """

    def wrong_rdkit(meta):
        meta["featurizer"]["rdkit_version"] = "0.0.0-not-real"

    path = corrupt_metadata(artifact, tmp_path / "mismatch.onnx", wrong_rdkit)
    with pytest.raises(RDKitVersionMismatch):
        OlindaArtifact(path)
    assert len(OlindaArtifact(path, check_rdkit=False).run(_SM[:2])) == 2


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (None, "no olinda metadata"),
        (lambda meta: meta.update(schema="olinda.bundle.v99"), "schema"),
        (
            lambda meta: meta["columns"][0].update(output="no_such_output"),
            "does not produce",
        ),
    ],
    ids=["no metadata", "unknown schema", "output the graph lacks"],
)
def test_a_file_that_lies_about_itself_is_refused_at_load(
    tmp_path, artifact, mutate, message
):
    """Every mismatch between the metadata and the graph must fail at load, not mid-prediction.

    A KeyError from inside `run()` on the thousandth molecule is the outcome being prevented here, and
    an unknown schema is refused rather than guessed at: the graph would be fine, but this code cannot
    say what its metadata means.
    """
    path = corrupt_metadata(artifact, tmp_path / "broken.onnx", mutate)
    with pytest.raises(ValueError, match=message):
        OlindaArtifact(path)


def test_unparseable_smiles_yield_nan_and_a_warning(artifact):
    """An all-zero fingerprint scores perfectly happily — refuse to report it as a prediction."""
    import warnings

    model = OlindaArtifact(artifact)
    mixed = ["CCO", "not_a_smiles", "", "C1CC", "c1ccccc1"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = model.run(mixed)[model.columns[0]].to_numpy()
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
    one molecule instead of a list of them — returns a DataFrame of confident nonsense. A set is refused
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
    expected = model.run(["CCO", "c1ccccc1"])[model.columns[0]].to_numpy()
    for given in (
        ("CCO", "c1ccccc1"),
        np.array(["CCO", "c1ccccc1"]),
        pd.Series(["CCO", "c1ccccc1"]),
        iter(["CCO", "c1ccccc1"]),
    ):
        np.testing.assert_allclose(
            model.run(given)[model.columns[0]].to_numpy(), expected
        )
