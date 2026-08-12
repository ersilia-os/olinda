"""Learned applicability gate for the hard-label signal.

``learn-hard`` calibrates a hard-label model ``G`` onto the teacher's soft-label scale
(:mod:`olinda.ground_truth`). The hard signal is only trustworthy for queries close to the labeled
training set, so at predict time the fused ``model.onnx`` blends the surrogate ``S``
with the calibrated hard signal ``G_soft`` using a weight ``a`` that is 0 by default and rises only near the
labeled chemistry:

    prediction = (1 - a) * surrogate + a * ground_truth_soft

Rather than search the labeled set for each query at predict time, the gate is **learned once** at
``learn-hard`` time: :func:`tanimoto_nn` gives every reference-library compound its exact 1-NN Tanimoto
to the labeled set, and a small gradient-boosted :class:`SimilarityRegressor` learns to predict that
number from the fingerprint alone. At predict time one tree ensemble estimates the similarity and a
linear ramp turns it into ``a`` — no fingerprint comparison, and the labeled set never leaves the run.

``a`` is a product of two measured quantities:

    a = a_max * ramp(predicted similarity)

The ramp says *where* the hard signal may speak; ``a_max`` says *how loudly*, and is earned by the hard
head beating the surrogate out-of-fold on the labelled compounds (:mod:`olinda.ground_truth`). A head
that loses gets ``a_max = 0`` and the model ships soft-only.

This replaced a pair of Bernoulli Naive-Bayes classifiers over similarity *buckets*. Measured on a real
run, that gate opened the hard channel on 32.6% of the library against a true rate of ~7%, with 0.013
precision on its top bucket: "within 0.4 Tanimoto of any of 7,684 compounds" is a union of balls, and a
Naive-Bayes decision is one linear boundary. Regressing the similarity keeps the magnitude, drops the
thresholds, and makes ``a`` continuous.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Knees of the ramp, in **raw** 1-NN Tanimoto to the labelled set: a = 0 below SIM_LO, rising linearly
# to a_max at SIM_HI. Same constants the bucket edges used, so the intent is unchanged — what goes is
# the cliff between them.
#
# Raw, deliberately, though the case for transforming is real and worth restating before anyone
# reopens it. Baldi & Nasr (JCIM 2010, doi:10.1021/ci100010v) show the Tanimoto distribution is not
# invariant — it depends on how many bits the query sets — and that the *maximum* over a set, which is
# exactly what tanimoto_nn returns, follows an extreme-value distribution that shifts with the size of
# that set. So a fixed 0.4 does not mean the same thing for a model with 500 labelled compounds as for
# one with 50,000.
#
# Mapping onto the library's own similarity percentiles would fix that for free (the gate sweep already
# computes the full distribution) but trades one flaw for a worse one: a percentile always promotes the
# top of whatever is present, so a labelled set with no relationship to the query chemistry would still
# hand full weight to its least-unrelated tail. Absolute Tanimoto fails the safe way — it simply stays
# quiet — and keeps the knees legible to a chemist reading the metadata.
SIM_LO: float = 0.4
SIM_HI: float = 0.7

# Ceiling on the blend weight. Even an exact match keeps a third of the surrogate, because the hard head
# is trained on far less data and the surrogate carries the teacher's whole view of the library.
A_CEILING: float = 0.66

# Below this the hard head is not mixed in at all. A few percent of a weakly aligned signal cannot move
# a prediction enough to be worth the risk of moving it the wrong way, so the dead zone (0, A_MIN)
# collapses to zero and the model ships soft-only rather than carrying a token hard branch.
A_MIN: float = 0.1

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


# Capacity of the similarity regressor — an upper bound, not a fixed cost. These are the point where
# accuracy stops paying for size on a full-library run: measured there, 300 rounds at this depth gave
# R² 0.46 for 1.4 MB and 2.6 µs/molecule, where doubling the leaves bought R² 0.51 for twice the size.
#
# The *proportion* of the artifact this represents is not general — it was a few percent beside a
# 42.5 MB surrogate, but a small run (`--max-samples`, or a small library) produces a small surrogate
# and the gate would loom much larger against it. What keeps that in hand is early stopping rather
# than these constants: with little data the fit converges in a handful of rounds, and a toy fixture
# produces a single-tree gate. So read these as "no larger than", and check the fitted tree count in
# the run's metadata if size matters.
#
# Depth rather than a leaf count because that is the canonical knob both backends translate — LightGBM
# derives num_leaves = min(2**max_depth - 1, 255), so 6 gives 63.
GATE_ROUNDS: int = 300
GATE_MAX_DEPTH: int = 6
GATE_MAX_BIN: int = 64

_META_NAME = "gate_meta.json"


def ramp(similarity, *, a_max: float = A_CEILING, sim_lo: float = SIM_LO, sim_hi: float = SIM_HI):
  """Map predicted similarity onto a blend weight: 0 below ``sim_lo``, ``a_max`` at ``sim_hi``.

  Linear in between, so two compounds either side of a knee no longer receive very different weights —
  the bucketed gate this replaced jumped from 0.33 to 0.66 across ``sim_hi``.
  """
  s = np.asarray(similarity, dtype=np.float64).ravel()
  span = max(float(sim_hi) - float(sim_lo), 1e-9)
  return float(a_max) * np.clip((s - float(sim_lo)) / span, 0.0, 1.0)


class SimilarityRegressor:
  """Predicts a compound's 1-NN Tanimoto to the labelled set, and turns it into a blend weight.

  The model is a small gradient-boosted regressor over the same binarised Morgan features everything
  else uses, trained on the reference library with :func:`tanimoto_nn` as the target. It is a *stand-in*
  for a nearest-neighbour search: approximate by construction, but it keeps the labelled fingerprints
  out of the shipped artifact and costs one tree ensemble per query instead of a scan.

  Attributes
  ----------
  model : object
      The trained booster, in the backend's native form.
  backend : str
      Which engine trained it, so it can be reloaded and converted to ONNX.
  a_max : float
      Ceiling on the blend weight — earned from the hard head's out-of-fold margin over the surrogate,
      so a head that does not beat the surrogate yields 0 and disables the blend entirely.
  """

  def __init__(
    self,
    model,
    backend: str,
    *,
    a_max: float = A_CEILING,
    sim_lo: float = SIM_LO,
    sim_hi: float = SIM_HI,
    metrics: dict | None = None,
  ) -> None:
    self.model = model
    self.backend = str(backend)
    self.a_max = float(a_max)
    self.sim_lo = float(sim_lo)
    self.sim_hi = float(sim_hi)
    self.metrics = dict(metrics or {})

  def predict_similarity(self, bits: np.ndarray) -> np.ndarray:
    """Estimated 1-NN Tanimoto per row, clipped to [0, 1] — the target's own range."""
    from olinda.train.backend import get_backend

    x = (np.asarray(bits) > 0).astype(np.float32)
    raw = get_backend(self.backend, "cpu").predict(self.model, x)
    return np.clip(np.asarray(raw, dtype=np.float64).ravel(), 0.0, 1.0)

  def weight(self, bits: np.ndarray) -> np.ndarray:
    """Blend weight ``a`` per row (float64)."""
    return ramp(self.predict_similarity(bits), a_max=self.a_max, sim_lo=self.sim_lo, sim_hi=self.sim_hi)

  def save(self, directory: str | Path) -> None:
    """Write the booster plus the ramp parameters into *directory*."""
    from olinda.train.backend import get_backend

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    get_backend(self.backend, "cpu").save(self.model, directory)
    with open(directory / _META_NAME, "w") as fp:
      json.dump(
        {
          "type": "similarity_regressor",
          "backend": self.backend,
          "a_max": self.a_max,
          "sim_lo": self.sim_lo,
          "sim_hi": self.sim_hi,
          "metrics": self.metrics,
        },
        fp,
        indent=2,
      )

  @classmethod
  def load(cls, directory: str | Path) -> "SimilarityRegressor":
    from olinda.train.backend import get_backend

    directory = Path(directory)
    with open(directory / _META_NAME) as fp:
      meta = json.load(fp)
    backend = meta.get("backend", "lightgbm")
    return cls(
      get_backend(backend, "cpu").load(directory),
      backend,
      a_max=meta.get("a_max", A_CEILING),
      sim_lo=meta.get("sim_lo", SIM_LO),
      sim_hi=meta.get("sim_hi", SIM_HI),
      metrics=meta.get("metrics"),
    )
