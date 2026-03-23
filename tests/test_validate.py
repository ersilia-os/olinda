"""Tests for validation metrics and plots."""
from __future__ import annotations

import numpy as np
import pytest

from olinda.validate import _auc_roc, _plot_roc_curve


class TestAucRoc:
  def test_perfect_classifier(self):
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert _auc_roc(y, p, threshold=0.5) == pytest.approx(1.0)

  def test_random_classifier(self):
    rng = np.random.default_rng(42)
    y = rng.choice([0.0, 1.0], size=10_000)
    p = rng.uniform(size=10_000)
    auc = _auc_roc(y, p, threshold=0.5)
    assert 0.45 < auc < 0.55

  def test_single_class_returns_nan(self):
    y = np.array([1.0, 1.0, 1.0])
    p = np.array([0.5, 0.6, 0.7])
    assert np.isnan(_auc_roc(y, p, threshold=0.5))


class TestPlotRocCurve:
  def test_saves_png(self, tmp_path):
    rng = np.random.default_rng(0)
    y = rng.choice([0.0, 1.0], size=200)
    p = y * 0.6 + rng.uniform(size=200) * 0.4  # noisy but correlated
    out = tmp_path / "roc.png"
    _plot_roc_curve(y, p, out)
    assert out.exists()
    assert out.stat().st_size > 0

  def test_skips_single_class(self, tmp_path):
    y = np.ones(50)
    p = np.random.default_rng(1).uniform(size=50)
    out = tmp_path / "roc.png"
    _plot_roc_curve(y, p, out, threshold=0.5)
    assert not out.exists()  # should not produce a file

  def test_threshold_binarizes_correctly(self, tmp_path):
    """With a custom threshold, the curve should still be valid."""
    rng = np.random.default_rng(2)
    y = rng.uniform(0, 10, size=500)
    p = y + rng.normal(0, 1, size=500)
    out = tmp_path / "roc_thresh.png"
    _plot_roc_curve(y, p, out, threshold=5.0)
    assert out.exists()
