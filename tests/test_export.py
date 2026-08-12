"""ONNX stage builders: the isotonic calibrator and ``T`` + its ramp round-trip through ONNX."""

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import onnxruntime as ort  # noqa: E402

from olinda.tanimoto import A_CEILING, T_HI, T_LO, TanimotoRegressor  # noqa: E402
from olinda.calibrate import IsotonicCalibrator  # noqa: E402
from olinda.export import tanimoto_to_onnx, isotonic_to_onnx  # noqa: E402


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


def _fitted_gate(d=64, n=3000, seed=1):
  """A similarity gate trained on a synthetic target, through the real fit path."""
  rng = np.random.default_rng(seed)
  bits = (rng.random((n, d)) < 0.15).astype(np.float32)
  # A target with real structure across the whole [0, 1] range, so the ramp is exercised end to end
  # rather than saturating at one end.
  sim = np.clip(bits[:, :12].sum(1) / 12.0 + rng.normal(0, 0.03, n), 0, 1)

  def batches(shuffler):
    idx = np.arange(n - 500)
    shuffler.shuffle(idx)
    for start in range(0, len(idx), 256):
      take = idx[start : start + 256]
      yield bits[take], sim[take]

  gate = TanimotoRegressor.fit(batches, d, (bits[-500:], sim[-500:]), seed=seed)
  return gate, bits, d


def test_the_gate_graph_binarises_its_input(tmp_path):
  """The net sees ``bits > 0``; the fused graph carries counts. Without that they quietly diverge."""
  clf, bits, d = _fitted_gate()
  tanimoto_to_onnx(clf, d, tmp_path / "ad.onnx")
  sess = ort.InferenceSession((tmp_path / "ad.onnx").read_bytes(), providers=["CPUExecutionProvider"])

  counts = bits[:64].copy()
  counts[counts > 0] = 3.0  # the same substructures, each seen three times
  as_counts = np.asarray(sess.run(None, {"input": counts})[0]).ravel()
  as_bits = np.asarray(sess.run(None, {"input": bits[:64]})[0]).ravel()
  np.testing.assert_allclose(as_counts, as_bits, atol=1e-9)


def test_tanimoto_onnx_matches_weight(tmp_path):
  """The fused gate must agree with the Python weight() — this is what the export parity gate checks."""
  clf, bits, d = _fitted_gate()
  info = tanimoto_to_onnx(clf, d, tmp_path / "ad.onnx")
  assert info["max_abs_diff"] < 1e-9

  sess = ort.InferenceSession((tmp_path / "ad.onnx").read_bytes(), providers=["CPUExecutionProvider"])
  got = np.asarray(sess.run(None, {"input": bits[:64]})[0]).ravel()
  np.testing.assert_allclose(got, clf.weight(bits[:64] > 0), atol=1e-9)


def test_tanimoto_onnx_respects_the_ramp_bounds(tmp_path):
  """Whatever the trees extrapolate to, the weight stays inside [0, a_max]."""
  clf, bits, d = _fitted_gate()
  tanimoto_to_onnx(clf, d, tmp_path / "ad.onnx")
  sess = ort.InferenceSession((tmp_path / "ad.onnx").read_bytes(), providers=["CPUExecutionProvider"])
  probe = np.vstack([np.zeros((4, d), np.float32), np.ones((4, d), np.float32), bits[:32]])
  got = np.asarray(sess.run(None, {"input": probe})[0]).ravel()
  assert got.min() >= 0.0 and got.max() <= A_CEILING + 1e-12


def test_a_max_of_zero_switches_the_gate_off_in_onnx(tmp_path):
  """The 'hard head has not earned any weight' case has to survive into the graph."""
  clf, bits, d = _fitted_gate()
  clf.a_max = 0.0
  tanimoto_to_onnx(clf, d, tmp_path / "off.onnx")
  sess = ort.InferenceSession((tmp_path / "off.onnx").read_bytes(), providers=["CPUExecutionProvider"])
  got = np.asarray(sess.run(None, {"input": bits[:32]})[0]).ravel()
  assert np.all(got == 0.0)


def test_similarity_regressor_roundtrips(tmp_path):
  clf, bits, _ = _fitted_gate()
  clf.save(tmp_path / "gate")
  back = TanimotoRegressor.load(tmp_path / "gate")
  assert (back.a_max, back.sim_lo, back.sim_hi) == (clf.a_max, T_LO, T_HI)
  np.testing.assert_allclose(back.weight(bits[:50] > 0), clf.weight(bits[:50] > 0), atol=1e-12)
