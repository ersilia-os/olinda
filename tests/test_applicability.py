"""Learned applicability gate: exact Tanimoto-NN labelling, and the similarity → blend-weight ramp."""

import numpy as np
import pytest

from olinda.applicability import (
  A_CEILING,
  A_MIN,
  SIM_HI,
  SIM_LO,
  prepare_gt_bits,
  ramp,
  tanimoto_nn,
)


# ── tanimoto_nn (training-time reference labeling) ────────────────────────────


def test_tanimoto_nn_identical_disjoint_and_empty_query():
  gt = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
  q = np.array(
    [
      [1, 1, 0, 0],  # identical to GT row 0 → 1.0
      [0, 0, 0, 0],  # empty fingerprint → 0.0 (no divide-by-zero)
      [1, 0, 0, 0],  # |∩|=1, |∪|=2 vs row 0 → 0.5
      [1, 1, 1, 1],  # |∩|=2, |∪|=4 → 0.5
    ],
    dtype=np.uint8,
  )
  assert np.allclose(tanimoto_nn(q, gt), [1.0, 0.0, 0.5, 0.5])


def test_tanimoto_nn_empty_gt_and_counts_binarized():
  assert tanimoto_nn(np.ones((3, 4), dtype=np.uint8), np.zeros((0, 4), dtype=np.uint8)).tolist() == [0, 0, 0]
  # count fingerprints are binarized before comparison
  assert np.isclose(tanimoto_nn(np.array([[2, 0, 0, 0]]), np.array([[5, 0, 0, 0]]))[0], 1.0)


# ── BernoulliNB ───────────────────────────────────────────────────────────────


def _counts(bits, y):
  """Sufficient statistics (class_counts, feat_on_counts) for a two-class Bernoulli NB."""
  bits, y = np.asarray(bits, float), np.asarray(y)
  cn = np.array([(y == 0).sum(), (y == 1).sum()], dtype=float)
  on = np.stack([bits[y == 0].sum(0), bits[y == 1].sum(0)])
  return cn, on


def _unblocked_reference(query_bits, prepared):
  """The pre-optimisation formulation: one full (queries x labelled) pass, explicit 0/0 guard."""
  q = (np.asarray(query_bits) > 0).astype(np.float32)
  g, g_card = prepared
  m = q.shape[0]
  if g.shape[0] == 0 or m == 0:
    return np.zeros(m, dtype=np.float64)
  inter = q @ g.T
  union = q.sum(axis=1)[:, None] + g_card - inter
  with np.errstate(divide="ignore", invalid="ignore"):
    tan = np.where(union > 0, inter / union, 0.0)
  return tan.max(axis=1).astype(np.float64)


def test_blocked_search_matches_the_unblocked_formulation():
  """Blocking is a memory optimisation, so it must be numerically indistinguishable."""
  rng = np.random.default_rng(0)
  queries = (rng.random((300, 256)) < 0.05).astype(np.uint8)
  labelled = (rng.random((2500, 256)) < 0.05).astype(np.uint8)  # spans several blocks
  prepared = prepare_gt_bits(labelled)
  assert np.array_equal(tanimoto_nn(queries, prepared=prepared), _unblocked_reference(queries, prepared))


def test_degenerate_fingerprints_score_zero():
  """Flooring the denominator must behave exactly like the explicit 0/0 guard it replaced."""
  rng = np.random.default_rng(1)
  labelled = (rng.random((50, 256)) < 0.05).astype(np.uint8)
  empty = np.zeros((3, 256), dtype=np.uint8)
  assert np.array_equal(tanimoto_nn(empty, prepared=prepare_gt_bits(labelled)), np.zeros(3))
  assert np.array_equal(tanimoto_nn(empty, prepared=prepare_gt_bits(empty)), np.zeros(3))
  assert np.array_equal(tanimoto_nn(labelled[:4], prepared=prepare_gt_bits(np.zeros((0, 256)))), np.zeros(4))


# ── the ramp (similarity → blend weight) ─────────────────────────────────────


def test_ramp_is_zero_below_the_lower_knee():
  """Far from the labelled chemistry the hard head must not speak at all."""
  assert ramp([0.0, 0.1, SIM_LO - 1e-9]).tolist() == [0.0, 0.0, 0.0]


def test_ramp_saturates_at_the_ceiling():
  assert ramp([SIM_HI, 0.9, 1.0]).tolist() == [A_CEILING] * 3


def test_ramp_is_continuous_across_the_knees():
  """The bucketed gate this replaced jumped 0.33 → 0.66 across sim_hi; nothing may jump now."""
  s = np.linspace(0.0, 1.0, 2001)
  a = ramp(s)
  assert np.all(np.diff(a) >= -1e-12), "the ramp must be non-decreasing"
  assert np.max(np.diff(a)) < 0.01, "no step should be visible at this resolution"


def test_ramp_is_linear_between_the_knees():
  mid = (SIM_LO + SIM_HI) / 2
  assert ramp([mid])[0] == pytest.approx(A_CEILING / 2)


def test_a_max_of_zero_disables_the_blend_everywhere():
  """A hard head that has not earned any weight must be switched off, however similar the query."""
  assert ramp([0.0, 0.5, 1.0], a_max=0.0).tolist() == [0.0, 0.0, 0.0]


# ── the blend ceiling (how far the hard head is trusted at all) ──────────────


def test_a_barely_aligned_head_is_not_merged_at_all():
  """Under A_MIN there is no point mixing it in — a token weight risks more than it can add."""
  from olinda.ground_truth import _blend_ceiling

  assert _blend_ceiling(0.0) == 0.0
  assert _blend_ceiling(1e-4) == 0.0
  assert _blend_ceiling(A_MIN - 1e-9) == 0.0
  assert _blend_ceiling(A_MIN) == pytest.approx(A_MIN)  # the floor itself still merges


def test_a_negative_r2_disables_rather_than_inverts():
  """Worse than predicting the teacher's mean is a reason to switch off, not to run backwards."""
  from olinda.ground_truth import _blend_ceiling

  assert _blend_ceiling(-0.4) == 0.0
  assert _blend_ceiling(-99.0) == 0.0  # R² is unbounded below
  assert _blend_ceiling(float("nan")) == 0.0  # constant teacher column, or a hard set too small


def test_the_ceiling_can_only_lower_trust():
  """R² caps A_CEILING; it never licenses more weight than the hard ceiling allows."""
  from olinda.ground_truth import _blend_ceiling

  assert _blend_ceiling(0.3) == pytest.approx(0.3)
  assert _blend_ceiling(0.99) == pytest.approx(A_CEILING)
  assert _blend_ceiling(1.0) == pytest.approx(A_CEILING)


def test_the_ceiling_is_monotone_in_alignment():
  """Better agreement must never buy less trust."""
  from olinda.ground_truth import _blend_ceiling

  values = [_blend_ceiling(r2) for r2 in np.linspace(-0.5, 1.0, 200)]
  assert all(b <= a for a, b in zip(values[1:], values[:-1]))
