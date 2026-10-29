"""`olinda tune` runs one study per soft column, and `learn-soft` picks each column's result up.

Columns train independently, so tuning must too: one shared study would fit a learning rate to whichever
column came first and impose it on the rest.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("optuna")
pytest.importorskip("onnx")

from olinda import run as runlib  # noqa: E402
from tests.conftest import plain, run_cli, write_soft  # noqa: E402


def _prepared(tmp_path, library):
    home, smiles, x = library
    soft = write_soft(tmp_path, smiles, x, columns=("activity", "solubility"))
    run_dir = tmp_path / "run"
    assert (
        r := run_cli(["prepare", "-s", soft, "-m", run_dir, "--val-frac", "0.2"])
    ).exit_code == 0, r.output
    return run_dir


def test_tune_writes_one_params_file_per_column(tmp_path, library, backend):
    run_dir = _prepared(tmp_path, library)
    r = run_cli(["tune", "-m", run_dir, "--trials", "3"])
    assert r.exit_code == 0, r.output

    for col in runlib.read_manifest(run_dir)["columns"]:
        params = json.loads(
            (runlib.column_dir(run_dir, col["id"]) / runlib.PARAMS_NAME).read_text()
        )
        assert {"learning_rate", "min_split_gain", "backend"} <= set(params)
    # nothing is left at run level, where it would apply to every column alike
    assert not (run_dir / runlib.PARAMS_NAME).exists()

    # learn-soft finds them (per-column file first), rather than reporting built-in defaults
    r = run_cli(["learn-soft", "-m", run_dir, "--num-boost-round", "20"])
    assert r.exit_code == 0, r.output
    assert "using built-in defaults" not in plain(r)


def test_tune_can_target_a_single_column(tmp_path, library, backend):
    run_dir = _prepared(tmp_path, library)
    r = run_cli(
        ["tune", "-m", run_dir, "--trials", "3", "--column", "solubility_probability"]
    )
    assert r.exit_code == 0, r.output
    manifest = runlib.read_manifest(run_dir)
    tuned = [
        (runlib.column_dir(run_dir, c["id"]) / runlib.PARAMS_NAME).exists()
        for c in manifest["columns"]
    ]
    assert tuned == [False, True]


def test_tune_names_an_unknown_column(tmp_path, library):
    run_dir = _prepared(tmp_path, library)
    r = run_cli(["tune", "-m", run_dir, "--column", "absent"])
    assert r.exit_code != 0
    assert "absent" in plain(r)
