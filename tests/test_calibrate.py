"""The isotonic calibrator: orientation is detected from the data, and a reversed one is not silent."""

from __future__ import annotations

import numpy as np

from olinda import calibrate
from olinda.calibrate import IsotonicCalibrator


class _Spy:
    """Stands in for the logger and keeps what was warned, since the console binds its stream at import."""

    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(str(message))

    def debug(self, message):
        pass


def test_a_decreasing_relationship_is_fitted_and_warned_about(monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(calibrate, "logger", spy)
    raw = np.linspace(0, 1, 50)
    cal = IsotonicCalibrator().fit(raw=raw, target=1 - raw, increasing="auto")
    assert np.all(np.diff(cal.transform(raw)) <= 1e-12)
    assert any("opposite" in w for w in spy.warnings)


def test_an_increasing_relationship_is_quiet(monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(calibrate, "logger", spy)
    raw = np.linspace(0, 1, 50)
    IsotonicCalibrator().fit(raw=raw, target=raw, increasing="auto")
    assert spy.warnings == []
