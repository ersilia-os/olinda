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


# Capacity of the similarity gate. A small MLP rather than a gradient-boosted tree, because Tanimoto is
# (x·g)/(|x|+|g|−x·g) maximised over the labelled set — a ratio of inner products. A dense layer computes
# exactly those weighted overlaps; axis-aligned tree splits have to reconstruct a 2048-term sum out of
# thresholds on single bits, and measurably cannot. On a real run, against the 9.2% of the library that
# genuinely deserves weight:
#
#   tree, tuned and loss-weighted   R² 0.08   recall 0.38   precision 0.64   0.6 MB
#   MLP (64,)                       R² 0.48   recall 0.44   precision 0.62   0.5 MB
#   MLP (256, 64)                   R² 0.61   recall 0.52   precision 0.71   2.2 MB
#
# The tree could buy recall only by giving up precision and R²; the MLP improves all three at once.
#
# Size scales with the layer widths alone, not with the library — 2048x256 + 256x64 + biases as float32.
# Predicting is two MatMuls, so it stays cheap next to the surrogate's tree traversal.
GATE_HIDDEN: tuple[int, ...] = (256, 64)
# Trained by streaming mini-batches off the resident uint8 library, so the whole reference set is used
# without ever materialising it as float32 (which would be ~11 GB). One batch is ~34 MB.
GATE_MAX_EPOCHS: int = 15
GATE_PATIENCE: int = 3
GATE_BATCH: int = 4096
GATE_LEARNING_RATE: float = 1e-3

_MODEL_NAME = "gate.onnx"

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

  A small MLP over the same binarised Morgan features everything else uses, trained on the reference
  library with :func:`tanimoto_nn` as the target. It stands in for a nearest-neighbour search: an
  approximation by construction, but it keeps the labelled fingerprints out of the shipped artifact —
  which is a hard requirement, since the artifact is what gets distributed — and costs two matrix
  multiplies per query instead of a scan over the labelled set.

  The trained net is held **as ONNX**, not as a pickled estimator: it is destined for the fused graph
  anyway, onnxruntime is already a base dependency, and it avoids persisting a pickle whose validity
  depends on the scikit-learn version that happens to be installed later.

  Attributes
  ----------
  onnx_bytes : bytes
      The serialised network, input ``"input"`` (float32, ``[B, d]``), output ``"variable"``.
  a_max : float
      Ceiling on the blend weight, earned from how well the calibrated hard head reproduces the
      teacher's scale (:func:`olinda.ground_truth._blend_ceiling`). Zero disables the blend entirely.
  """

  def __init__(
    self,
    onnx_bytes: bytes,
    *,
    a_max: float = A_CEILING,
    sim_lo: float = SIM_LO,
    sim_hi: float = SIM_HI,
    metrics: dict | None = None,
  ) -> None:
    self.onnx_bytes = bytes(onnx_bytes)
    self.a_max = float(a_max)
    self.sim_lo = float(sim_lo)
    self.sim_hi = float(sim_hi)
    self.metrics = dict(metrics or {})
    self._session = None

  @classmethod
  def fit(cls, batches, n_features: int, validation, *, seed: int = 0, echo=None, **kwargs):
    """Train the gate by streaming mini-batches, and keep the result as ONNX.

    Parameters
    ----------
    batches : callable
        ``batches(rng)`` yields ``(bits, similarity)`` pairs for one shuffled epoch. Taking a factory
        rather than an array is what lets the caller stream from the resident uint8 library — dense
        float32 for the whole reference set would be ~11 GB, and there is no reason to hold it.
    n_features : int
        Fingerprint width, needed to declare the ONNX input.
    validation : tuple
        ``(bits, similarity)`` held out for early stopping. Materialised, so keep it modest.

    The target is the similarity itself, not the ramp applied to it. Regressing the ramp output looks
    tempting — it is what the gate actually needs — but it is zero for ~91% of the library, so the net
    emits small positive values everywhere and *any* positive value opens the gate: measured, 84% of
    the library at 0.11 precision. Regressing similarity and ramping afterwards keeps the signal.
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.neural_network import MLPRegressor

    val_x = (np.asarray(validation[0]) > 0).astype(np.float32)
    val_y = np.asarray(validation[1], dtype=np.float64).ravel()

    net = MLPRegressor(
      hidden_layer_sizes=GATE_HIDDEN,
      activation="relu",
      solver="adam",
      learning_rate_init=GATE_LEARNING_RATE,
      random_state=seed,
    )
    rng = np.random.default_rng(seed)
    best, best_state, waited = np.inf, None, 0
    for epoch in range(GATE_MAX_EPOCHS):
      for bits, target in batches(rng):
        net.partial_fit((np.asarray(bits) > 0).astype(np.float32), np.asarray(target, dtype=np.float64))
      loss = float(np.mean((net.predict(val_x) - val_y) ** 2))
      if echo:
        echo(f"    epoch {epoch + 1}/{GATE_MAX_EPOCHS} · val MSE {loss:.6f}", "info")
      # Early stopping is hand-rolled because sklearn's own only applies to `fit`, which would need the
      # whole library in memory. Keep the best weights rather than the last: Adam can step past a good
      # solution, and the epoch that stops the run is by definition not the best one.
      if loss < best - 1e-7:
        best, waited = loss, 0
        best_state = ([c.copy() for c in net.coefs_], [b.copy() for b in net.intercepts_])
      else:
        waited += 1
        if waited >= GATE_PATIENCE:
          break
    if best_state is not None:
      net.coefs_, net.intercepts_ = best_state

    onx = convert_sklearn(net, initial_types=[("input", FloatTensorType([None, int(n_features)]))])
    return cls(onx.SerializeToString(), **kwargs)

  def _run(self):
    import onnxruntime as ort

    if self._session is None:
      options = ort.SessionOptions()
      options.log_severity_level = 3
      self._session = ort.InferenceSession(self.onnx_bytes, options, providers=["CPUExecutionProvider"])
    return self._session

  def predict_similarity(self, bits: np.ndarray) -> np.ndarray:
    """Estimated 1-NN Tanimoto per row, clipped to [0, 1] — the target's own range."""
    sess = self._run()
    x = (np.asarray(bits) > 0).astype(np.float32)
    raw = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    return np.clip(np.asarray(raw, dtype=np.float64).ravel(), 0.0, 1.0)

  def weight(self, bits: np.ndarray) -> np.ndarray:
    """Blend weight ``a`` per row (float64)."""
    return ramp(self.predict_similarity(bits), a_max=self.a_max, sim_lo=self.sim_lo, sim_hi=self.sim_hi)

  def save(self, directory: str | Path) -> None:
    """Write the network and the ramp parameters into *directory*."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / _MODEL_NAME).write_bytes(self.onnx_bytes)
    with open(directory / _META_NAME, "w") as fp:
      json.dump(
        {
          "type": "similarity_mlp",
          "hidden": list(GATE_HIDDEN),
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
    directory = Path(directory)
    with open(directory / _META_NAME) as fp:
      meta = json.load(fp)
    return cls(
      (directory / _MODEL_NAME).read_bytes(),
      a_max=meta.get("a_max", A_CEILING),
      sim_lo=meta.get("sim_lo", SIM_LO),
      sim_hi=meta.get("sim_hi", SIM_HI),
      metrics=meta.get("metrics"),
    )
