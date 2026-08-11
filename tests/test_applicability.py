"""Learned applicability gate: Tanimoto-NN labeling, Bernoulli-NB, and the bucket→weight mapping."""

import numpy as np

from olinda.applicability import (
  A_HIGH,
  A_LOW,
  ApplicabilityClassifier,
  BernoulliNB,
  prepare_gt_bits,
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


def test_bernoulli_nb_learns_separable_feature():
  # feature 0 on ⇒ class 1, off ⇒ class 0.
  bits = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
  y = np.array([1, 1, 0, 0])
  nb = BernoulliNB.from_counts(*_counts(bits, y))
  assert nb.predict(np.array([[1, 0]]))[0] == 1
  assert nb.predict(np.array([[0, 1]]))[0] == 0


def test_bernoulli_nb_uniform_prior_beats_imbalance():
  # 1000 negatives, 5 positives — an empirical prior would never predict positive, but a query matching the
  # positive feature profile must still be classified positive under the uniform prior.
  d = 6
  neg = np.zeros((1000, d), dtype=np.float32)
  neg[:, 0] = 1  # negatives carry feature 0
  pos = np.zeros((5, d), dtype=np.float32)
  pos[:, 1] = 1  # positives carry feature 1
  bits = np.vstack([neg, pos])
  y = np.array([0] * 1000 + [1] * 5)
  nb = BernoulliNB.from_counts(*_counts(bits, y))
  assert nb.predict(np.array([[0, 1, 0, 0, 0, 0]]))[0] == 1


def test_bernoulli_nb_roundtrips():
  bits = np.array([[1, 0], [0, 1]], dtype=np.float32)
  nb = BernoulliNB.from_counts(*_counts(bits, np.array([1, 0])))
  nb2 = BernoulliNB.from_dict(nb.to_dict())
  q = np.array([[1, 0], [0, 1]])
  assert np.array_equal(nb.predict(q), nb2.predict(q))


# ── ApplicabilityClassifier (bucket → weight) ─────────────────────────────────


class _ConstNB:
  """Stub NB that always predicts a fixed class — to test the ordinal bucket→weight logic."""

  def __init__(self, cls):
    self._cls = cls

  def predict(self, bits):
    return np.full(len(bits), self._cls)


def test_weight_bucket_mapping_and_high_precedence():
  bits = np.zeros((1, 3))
  # HIGH fires (regardless of low) → A_HIGH
  assert ApplicabilityClassifier(_ConstNB(1), _ConstNB(1)).weight(bits)[0] == A_HIGH
  # only LOW fires → A_LOW
  assert ApplicabilityClassifier(_ConstNB(1), _ConstNB(0)).weight(bits)[0] == A_LOW
  # neither fires → 0
  assert ApplicabilityClassifier(_ConstNB(0), _ConstNB(0)).weight(bits)[0] == 0.0
  # HIGH takes precedence even if the (nominally nested) low classifier says 0
  assert ApplicabilityClassifier(_ConstNB(0), _ConstNB(1)).weight(bits)[0] == A_HIGH


def test_applicability_classifier_roundtrips(tmp_path):
  bits = np.array([[1, 0], [0, 1]], dtype=np.float32)
  low = BernoulliNB.from_counts(*_counts(bits, np.array([0, 1])))
  high = BernoulliNB.from_counts(*_counts(bits, np.array([0, 1])))
  clf = ApplicabilityClassifier(low, high, a_low=A_LOW, a_high=A_HIGH)
  p = tmp_path / "ad.json"
  clf.save(p)
  reloaded = ApplicabilityClassifier.load(p)
  q = np.array([[1, 0], [0, 1]])
  assert np.allclose(reloaded.weight(q), clf.weight(q))
  assert (reloaded.a_low, reloaded.a_high) == (A_LOW, A_HIGH)


# ── the blocked nearest-neighbour search must not change any number ───────────


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
