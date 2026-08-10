"""ONNX stage builders: isotonic calibrator and the applicability NB gate round-trip through ONNX."""

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import onnxruntime as ort  # noqa: E402

from olinda.applicability import A_HIGH, A_LOW, ApplicabilityClassifier, BernoulliNB  # noqa: E402
from olinda.calibrate import IsotonicCalibrator  # noqa: E402
from olinda.export import applicability_to_onnx, isotonic_to_onnx  # noqa: E402


def _counts(bits, y):
  bits, y = np.asarray(bits, float), np.asarray(y)
  cn = np.array([(y == 0).sum(), (y == 1).sum()], float)
  on = np.stack([bits[y == 0].sum(0), bits[y == 1].sum(0)])
  return cn, on


@pytest.mark.parametrize("relation", ["increasing", "decreasing"])
def test_isotonic_onnx_matches_transform(tmp_path, relation):
  rng = np.random.default_rng(0)
  g = rng.random(4000)
  s = (0.9 * g if relation == "increasing" else 0.9 * (1 - g)) + rng.standard_normal(4000) * 0.04
  cal = IsotonicCalibrator().fit(g, s, increasing="auto")
  info = isotonic_to_onnx(cal, tmp_path / "iso.onnx")
  assert info["max_abs_diff"] <= 1e-4  # includes the out-of-range margins probed inside isotonic_to_onnx

  # independent check on fresh raw inputs (incl. out of range)
  sess = ort.InferenceSession((tmp_path / "iso.onnx").read_bytes(), providers=["CPUExecutionProvider"])
  raw = np.linspace(-0.2, 1.2, 1500).astype(np.float32)
  got = sess.run(None, {"input": raw})[0]
  assert np.max(np.abs(got - cal.transform(raw))) <= 1e-4


def test_applicability_onnx_matches_weight_exactly(tmp_path):
  rng = np.random.default_rng(1)
  d = 128
  gt = (rng.random((30, d)) < 0.05).astype(np.float32)  # sparse "similar" positives
  neg = (rng.random((300, d)) < 0.15).astype(np.float32)  # denser negatives
  bits = np.vstack([neg, gt])
  y = np.array([0] * len(neg) + [1] * len(gt))
  clf = ApplicabilityClassifier(
    BernoulliNB.from_counts(*_counts(bits, y)),
    BernoulliNB.from_counts(*_counts(bits, y)),
    a_low=A_LOW,
    a_high=A_HIGH,
  )
  info = applicability_to_onnx(clf, d, tmp_path / "ad.onnx")
  assert info["max_abs_diff"] == 0.0

  sess = ort.InferenceSession((tmp_path / "ad.onnx").read_bytes(), providers=["CPUExecutionProvider"])
  q = np.vstack([gt[:5], (rng.random((5, d)) < 0.15).astype(np.float32)])
  got = sess.run(None, {"input": q})[0]
  assert np.array_equal(got, clf.weight(q > 0))
  assert set(np.unique(got).tolist()) <= {0.0, A_LOW, A_HIGH}
