"""Learned applicability gate for the hard-label signal.

``learn-hard`` calibrates a hard-label model ``G`` onto the teacher's soft-label scale
(:mod:`olinda.ground_truth`). The hard signal is only trustworthy for queries close to the labeled
training set, so at predict time the fused ``model.onnx`` blends the surrogate ``S``
with the calibrated hard signal ``G_soft`` using a weight ``a`` that is 0 by default and rises only near the
labeled chemistry:

    prediction = (1 - a) * surrogate + a * ground_truth_soft

Rather than search the labeled set for each query at predict time, the gate is **learned once** at
``learn-hard`` time: every reference-library compound is bucketed by its 1-NN Tanimoto similarity to the
labeled set into NOT SIMILAR / LOW / HIGH, and two simple Bernoulli **Naive-Bayes** classifiers are fit on
binarized Morgan features — ``clf_low`` ("at least LOW") and ``clf_high`` ("HIGH"). At predict time the two
classifiers place a query in a bucket (two dot products, no fingerprint comparison), which sets ``a``.

:func:`tanimoto_nn` is the training-time labeling helper. :class:`BernoulliNB` is the tiny NB engine and
:class:`ApplicabilityClassifier` bundles the two classifiers plus the bucket→weight mapping.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Similarity buckets (1-NN Tanimoto to the labeled set): HIGH >= SIM_HI, LOW in [SIM_LO, SIM_HI),
# NOT SIMILAR < SIM_LO. Weights favour the surrogate everywhere except close to the labeled chemistry.
SIM_LO: float = 0.4
SIM_HI: float = 0.7
A_LOW: float = 0.33
A_HIGH: float = 0.66
NB_ALPHA: float = 1.0  # Laplace smoothing for the Bernoulli likelihoods

# Labelled-set columns per similarity block. Only the row-wise maximum survives, so scoring the whole
# labelled set at once would build several (queries x labelled) arrays — measured at 6.9 GB for one
# 50k-row chunk — to produce a single vector. Blocking keeps the working set in cache: ~1.45x faster
# and ~7x less memory, with bit-identical output. 1024 measured best across 256-4096.
_NN_BLOCK = 1024


def prepare_gt_bits(gt_bits: np.ndarray):
  """Binarize the labelled set once, returning ``(g, g_card)`` for reuse across many query chunks.

  The reference library is scanned in chunks, so without this the labelled-set matrix and its
  cardinalities are rebuilt for every chunk — tens of megabytes of identical work per pass.
  """
  g = (np.asarray(gt_bits) > 0).astype(np.float32)
  return g, g.sum(axis=1)[None, :]


def tanimoto_nn(query_bits: np.ndarray, gt_bits: np.ndarray = None, *, prepared=None) -> np.ndarray:
  """Max Tanimoto similarity of each query to the labeled (ground-truth) set.

  Brute-force and exact — used at ``learn-hard`` time to label the reference library. Operates on binary
  fingerprints; ``|A ∩ B| = A·Bᵀ`` and ``|A ∪ B| = |A| + |B| - |A ∩ B|``.

  Parameters
  ----------
  query_bits : np.ndarray
      ``(m, d)`` fingerprints of the queries (binarized: nonzero ⇒ on).
  gt_bits : np.ndarray
      ``(n, d)`` fingerprints of the labeled set (binarized).

  Returns
  -------
  np.ndarray
      ``(m,)`` float64 nearest-neighbour Tanimoto similarity. ``0`` for an all-zero query fingerprint
      (e.g. an unparseable SMILES) and ``0`` when the labeled set is empty.
  """
  q = (np.asarray(query_bits) > 0).astype(np.float32)
  g, g_card = prepared if prepared is not None else prepare_gt_bits(gt_bits)
  m, n = q.shape[0], g.shape[0]
  if n == 0 or m == 0:
    return np.zeros(m, dtype=np.float64)

  q_card = q.sum(axis=1)[:, None]  # (m, 1)
  best = np.zeros(m, dtype=np.float32)
  for start in range(0, n, _NN_BLOCK):
    gb = g[start : start + _NN_BLOCK]
    inter = q @ gb.T  # (m, block) intersection counts
    union = q_card + g_card[:, start : start + _NN_BLOCK] - inter
    # union is 0 only when both fingerprints are empty, and an empty query already has inter == 0,
    # so flooring the denominator yields the same 0 as an explicit 0/0 guard, without a mask array.
    np.maximum(union, 1e-9, out=union)
    np.divide(inter, union, out=inter)  # in place: the intersection block is dead after this
    np.maximum(best, inter.max(axis=1), out=best)
  return best.astype(np.float64)


class BernoulliNB:
  """A minimal two-class Bernoulli Naive-Bayes over binary features.

  Fit from sufficient statistics (per-class row counts and per-class feature-on counts) so it can be
  trained by streaming — no full feature matrix is held in memory. Uniform class priors by default, so a
  heavily imbalanced negative class does not swamp the rare positives; the decision is then driven by the
  class-conditional feature likelihoods.
  """

  def __init__(self, log_prior: np.ndarray, log_theta: np.ndarray, log_neg_theta: np.ndarray) -> None:
    self.log_prior_ = np.asarray(log_prior, dtype=np.float64)  # (2,)
    self.log_theta_ = np.asarray(log_theta, dtype=np.float64)  # (2, d) log P(x_j=1 | c)
    self.log_neg_theta_ = np.asarray(log_neg_theta, dtype=np.float64)  # (2, d) log P(x_j=0 | c)

  @classmethod
  def from_counts(
    cls,
    class_counts: np.ndarray,
    feat_on_counts: np.ndarray,
    *,
    alpha: float = NB_ALPHA,
    uniform_prior: bool = True,
  ) -> "BernoulliNB":
    """Build the classifier from streamed sufficient statistics.

    Parameters
    ----------
    class_counts : np.ndarray
        ``(2,)`` number of training rows in class 0 and class 1.
    feat_on_counts : np.ndarray
        ``(2, d)`` number of rows in each class for which feature ``j`` is on.
    alpha : float
        Laplace smoothing added to on/off counts.
    uniform_prior : bool
        ``True`` uses a uniform class prior (imbalance-robust); ``False`` uses the empirical prior.
    """
    n = np.asarray(class_counts, dtype=np.float64)  # (2,)
    on = np.asarray(feat_on_counts, dtype=np.float64)  # (2, d)
    theta = (on + alpha) / (n[:, None] + 2.0 * alpha)  # P(x_j=1 | c), smoothed
    theta = np.clip(theta, 1e-12, 1.0 - 1e-12)
    if uniform_prior:
      log_prior = np.log(np.full(2, 0.5))
    else:
      total = float(n.sum()) or 1.0
      log_prior = np.log(np.clip(n / total, 1e-12, 1.0))
    return cls(log_prior, np.log(theta), np.log1p(-theta))

  def _log_posterior(self, bits: np.ndarray) -> np.ndarray:
    x = (np.asarray(bits) > 0).astype(np.float64)  # (m, d)
    # log P(c) + Σ_j [x_j log θ_cj + (1-x_j) log(1-θ_cj)]
    return self.log_prior_[None, :] + x @ self.log_theta_.T + (1.0 - x) @ self.log_neg_theta_.T

  def predict(self, bits: np.ndarray) -> np.ndarray:
    """Predicted class (0/1) per row, ``argmax`` of the log-posterior."""
    return self._log_posterior(bits).argmax(axis=1)

  def to_dict(self) -> dict:
    return {
      "log_prior": self.log_prior_.tolist(),
      "log_theta": self.log_theta_.tolist(),
      "log_neg_theta": self.log_neg_theta_.tolist(),
    }

  @classmethod
  def from_dict(cls, d: dict) -> "BernoulliNB":
    return cls(d["log_prior"], d["log_theta"], d["log_neg_theta"])


class ApplicabilityClassifier:
  """Two Bernoulli-NB classifiers + the bucket→weight mapping that gates the hard signal.

  ``clf_high`` predicts HIGH similarity, ``clf_low`` predicts at-least-LOW. A query is HIGH if ``clf_high``
  fires (HIGH takes precedence), else LOW if ``clf_low`` fires, else NOT SIMILAR — mapping to blend weights
  ``a_high`` / ``a_low`` / ``0``.
  """

  def __init__(
    self,
    clf_low: BernoulliNB,
    clf_high: BernoulliNB,
    *,
    a_low: float = A_LOW,
    a_high: float = A_HIGH,
    sim_lo: float = SIM_LO,
    sim_hi: float = SIM_HI,
  ) -> None:
    self.clf_low = clf_low
    self.clf_high = clf_high
    self.a_low = float(a_low)
    self.a_high = float(a_high)
    self.sim_lo = float(sim_lo)
    self.sim_hi = float(sim_hi)

  def weight(self, bits: np.ndarray) -> np.ndarray:
    """Blend weight ``a`` per row: ``a_high`` if HIGH, else ``a_low`` if LOW, else ``0`` (float64)."""
    high = self.clf_high.predict(bits) == 1
    low = self.clf_low.predict(bits) == 1
    return np.where(high, self.a_high, np.where(low, self.a_low, 0.0)).astype(np.float64)

  def to_dict(self) -> dict:
    return {
      "type": "bernoulli_nb_2clf",
      "sim_lo": self.sim_lo,
      "sim_hi": self.sim_hi,
      "a_low": self.a_low,
      "a_high": self.a_high,
      "clf_low": self.clf_low.to_dict(),
      "clf_high": self.clf_high.to_dict(),
    }

  @classmethod
  def from_dict(cls, d: dict) -> "ApplicabilityClassifier":
    return cls(
      BernoulliNB.from_dict(d["clf_low"]),
      BernoulliNB.from_dict(d["clf_high"]),
      a_low=d.get("a_low", A_LOW),
      a_high=d.get("a_high", A_HIGH),
      sim_lo=d.get("sim_lo", SIM_LO),
      sim_hi=d.get("sim_hi", SIM_HI),
    )

  def save(self, path: str | Path) -> None:
    with open(Path(path), "w") as fp:
      json.dump(self.to_dict(), fp)

  @classmethod
  def load(cls, path: str | Path) -> "ApplicabilityClassifier":
    with open(Path(path)) as fp:
      return cls.from_dict(json.load(fp))
