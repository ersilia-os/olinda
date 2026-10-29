"""The export parity gate: it must tolerate float rounding at an isotonic step, and refuse a real fault."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from olinda.calibrate import IsotonicCalibrator  # noqa: E402
from olinda.export.parity import _RAW_SLACK, _random_rows, _step_envelope  # noqa: E402
from tests.conftest import run_cli, write_soft  # noqa: E402


def test_the_envelope_spans_the_step_a_rounding_error_could_land_on():
    raw = np.linspace(0, 1, 200)
    cal = IsotonicCalibrator().fit(
        raw=raw, target=np.round(raw * 4) / 4, increasing=True
    )
    at_step = np.array([0.125])  # the boundary between the 0 and 0.25 plateaus
    lo, hi = _step_envelope(cal, at_step)
    assert lo[0] < hi[0], "a score on a step boundary must admit both sides of it"
    flat = np.array([0.5])
    lo, hi = _step_envelope(cal, flat)
    assert hi[0] - lo[0] < 1e-9, (
        "away from a step the envelope is as tight as the output itself"
    )
    assert _RAW_SLACK > 5e-7  # looser than the float32 error observed on real ensembles


def test_random_probe_rows_are_deterministic_sparse_counts():
    a, b = _random_rows(50, 2048), _random_rows(50, 2048)
    assert (a == b).all()
    assert a.max() <= 15 and (a > 0).sum(axis=1).min() >= 5


def test_a_corrupted_leaf_stops_the_export(tmp_path, library, backend, monkeypatch):
    """One wrong leaf in one column's trees must fail the export rather than ship a quietly wrong model."""
    import onnx

    home, smiles, x = library
    soft = write_soft(tmp_path, smiles, x, columns=("activity", "solubility"))
    run = tmp_path / "run"
    assert (
        run_cli(["prepare", "-s", soft, "-m", run, "--val-frac", "0.2"]).exit_code == 0
    )
    # at the default min_child_weight a 240-row library never splits: one constant leaf, which the
    # calibrator clamps, so a corrupted leaf would be invisible to any check
    from olinda.train.backend import CANONICAL_DEFAULTS

    monkeypatch.setitem(CANONICAL_DEFAULTS, "min_child_weight", 1.0)
    assert run_cli(["learn-soft", "-m", run, "--num-boost-round", "40"]).exit_code == 0
    assert run_cli(["export", "-m", run]).exit_code == 0
    (run / "model.onnx").unlink()

    path = run / "columns" / "c1" / "s.onnx"
    m = onnx.load(str(path))
    tampered = False
    for node in m.graph.node:
        if node.op_type == "TreeEnsembleRegressor":
            for attr in node.attribute:
                if attr.name == "target_weights":
                    w = np.array(attr.floats)
                    w[len(w) // 2] += 0.5
                    del attr.floats[:]
                    attr.floats.extend(w.tolist())
                    tampered = True
    assert tampered, "no tree ensemble found to corrupt"
    onnx.save(m, str(path))

    result = run_cli(["export", "-m", run])
    assert result.exit_code != 0, result.output
    assert not (run / "model.onnx").exists()


def test_the_parity_scale_grows_with_the_values_and_never_shrinks_the_tolerance():
    from olinda.export.parity import value_scale

    assert value_scale(np.array([0.2, 0.9])) == 1.0, (
        "probabilities keep the absolute tolerance"
    )
    assert value_scale(np.array([-4000.0, 10.0])) == 4000.0
    assert value_scale(np.array([])) == 1.0


def test_a_column_in_the_thousands_still_exports(
    tmp_path, library, backend, monkeypatch
):
    """float32 rounding on values near 5,000 is ~5e-4, which an absolute 1e-4 check would reject."""
    import pandas as pd

    from olinda.train.backend import CANONICAL_DEFAULTS

    home, smiles, x = library
    soft = write_soft(tmp_path, smiles, x, columns=("activity",))
    frame = pd.read_csv(soft)
    value = [c for c in frame.columns if c != "smiles"][0]
    frame[value] = 3000 + 6000 * (frame[value] - frame[value].min()) / (
        frame[value].max() - frame[value].min()
    )
    frame.to_csv(soft, index=False)
    monkeypatch.setitem(CANONICAL_DEFAULTS, "min_child_weight", 1.0)
    run = tmp_path / "run"
    assert (
        run_cli(["prepare", "-s", soft, "-m", run, "--val-frac", "0.2"]).exit_code == 0
    )
    assert run_cli(["learn-soft", "-m", run, "--num-boost-round", "40"]).exit_code == 0
    assert (run / "model.onnx").is_file()
