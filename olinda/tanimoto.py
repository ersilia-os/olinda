"""``T`` — the learned stand-in for a Tanimoto search against the labelled set.

``learn-hard`` calibrates a hard-label model ``H`` onto the teacher's soft-label scale
(:mod:`olinda.hard`). The hard signal is only trustworthy for queries close to the labeled
training set, so at predict time the fused ``model.onnx`` blends the surrogate ``S``
with the calibrated hard signal ``G_soft`` using a weight ``a`` that is 0 by default and rises only near the
labeled chemistry:

    prediction = (1 - a) * surrogate + a * h_s

Rather than search the labeled set for each query at predict time, the gate is **learned once** at
``learn-hard`` time: :func:`tanimoto_nn` gives every reference-library compound its exact 1-NN Tanimoto
to the labeled set, and a small :class:`TanimotoRegressor` — an MLP over the same fingerprint —
learns to predict that number from the fingerprint alone. At predict time two matrix multiplies estimate
the similarity and a linear ramp turns it into ``a`` — no fingerprint comparison, and the labeled set
never leaves the run.

``a`` is a product of two measured quantities:

    a = a_max * ramp(predicted similarity)

The ramp says *where* the hard signal may speak; ``a_max`` says *how loudly*, and is earned by the hard
head beating the surrogate out-of-fold on the labelled compounds (:mod:`olinda.hard`). A head
that loses gets ``a_max = 0`` and the model ships soft-only.

This replaced a pair of Bernoulli Naive-Bayes classifiers over similarity *buckets*. Measured on a real
run, that gate opened the hard channel on 32.6% of the library against a true rate of ~7%, with 0.013
precision on its top bucket: "within 0.4 Tanimoto of any of 7,684 compounds" is a union of balls, and a
Naive-Bayes decision is one linear boundary. Regressing the similarity keeps the magnitude, drops the
thresholds, and makes ``a`` continuous.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# Knees of the ramp, in **raw** 1-NN Tanimoto to the labelled set: a = 0 below T_LO, rising linearly
# to a_max at T_HI. Same constants the bucket edges used, so the intent is unchanged — what goes is
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
T_LO: float = 0.4
T_HI: float = 0.7

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


def prepare_hard_bits(hard_bits: np.ndarray):
  """Binarize the labelled set once, returning ``(g, g_card)`` for reuse across many query chunks.

  The reference library is scanned in chunks, so without this the labelled-set matrix and its
  cardinalities are rebuilt for every chunk — tens of megabytes of identical work per pass.
  """
  g = (np.asarray(hard_bits) > 0).astype(np.float32)
  return g, g.sum(axis=1)[None, :]


def tanimoto_nn(query_bits: np.ndarray, hard_bits: np.ndarray = None, *, prepared=None) -> np.ndarray:
  """Max Tanimoto similarity of each query to the labeled (hard-label) set.

  Brute-force and exact — used at ``learn-hard`` time to label the reference library. Operates on binary
  fingerprints; ``|A ∩ B| = A·Bᵀ`` and ``|A ∪ B| = |A| + |B| - |A ∩ B|``.

  Parameters
  ----------
  query_bits : np.ndarray
      ``(m, d)`` fingerprints of the queries (binarized: nonzero ⇒ on).
  hard_bits : np.ndarray
      ``(n, d)`` fingerprints of the labeled set (binarized).

  Returns
  -------
  np.ndarray
      ``(m,)`` float64 nearest-neighbour Tanimoto similarity. ``0`` for an all-zero query fingerprint
      (e.g. an unparseable SMILES) and ``0`` when the labeled set is empty.
  """
  q = (np.asarray(query_bits) > 0).astype(np.float32)
  g, g_card = prepared if prepared is not None else prepare_hard_bits(hard_bits)
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
T_HIDDEN: tuple[int, ...] = (256, 64)
# Trained by streaming mini-batches off the resident uint8 library, so the whole reference set is used
# without ever materialising it as float32 (which would be ~11 GB). One batch is ~34 MB.
T_MAX_EPOCHS: int = 15
T_PATIENCE: int = 3
T_BATCH: int = 4096
# How many rows one Adam step sees. Deliberately *not* T_BATCH: that one sizes the read off the
# library and wants to be large, while this one sets how often the optimiser steps and wants to be
# small. Fitting quality here is governed by the number of steps, not by how many rows are seen —
# at one step per 4096-row read the gate plateaus 14 points of recall short no matter how many
# epochs it runs (0.84 against 0.98), because it takes 8x fewer steps for the same data:
#
#   step rows     lr      R²     recall   relative time
#          256   1e-3   0.988     0.98        1.9x
#          512   2e-3   0.986     0.97        3.0x     <- shipped
#         1024   3e-3   0.981     0.91        4.3x
#         4096   1e-3   0.969     0.84        5.5x
#
# 512 with the learning rate scaled to match is the knee: recall within a point of the smallest step
# tested, at a third of the time. The lr moves with the batch because a larger batch averages more
# samples per step, so it can afford a proportionally bigger one.
T_SGD_BATCH: int = 512
T_LEARNING_RATE: float = 2e-3
# L2 penalty on the weights, matching what scikit-learn's MLPRegressor applies by default. Not
# decoration: dropping it was measured to cost real accuracy on a near-constant target, where an
# undamped net keeps enough prediction noise to swamp a target whose whole spread is 0.01.
T_L2: float = 1e-4

_MODEL_NAME = "t.onnx"

_META_NAME = "t_meta.json"

# Adam, standard values. Exposed as constants only so the update rule below reads like the paper.
_BETA1, _BETA2, _EPS = 0.9, 0.999, 1e-8

# The gate emits its own ONNX rather than going through a converter, so it pins its own opset. Every op
# it uses (MatMul, Add, Relu) is ancient; the fuse re-stamps this to the bundle's opset anyway. The IR
# version is pinned for the same reason it is elsewhere: onnxruntime refuses a graph newer than itself.
_T_OPSET = 16
_T_IR_VERSION = 10


class _MLP:
  """A plain float32 MLP with ReLU hidden layers, a linear output, and Adam — trained by mini-batch.

  Hand-rolled rather than :class:`sklearn.neural_network.MLPRegressor`, purely for speed. The gate sees
  every one of the library's ~1.35M compounds each epoch, which is ~4.4 TFLOP of dense matrix product;
  that is BLAS-bound work, and sklearn spends most of it on per-call input validation and on promoting
  the target to float64. Measured at matched quality — same architecture, same number of Adam steps —
  this runs about **3x faster**, which turns a 13-minute gate into a ~4-minute one. It also drops
  scikit-learn and skl2onnx from the training extra, since :meth:`to_onnx` writes the graph directly.

  Matched quality is the load-bearing part of that claim. Handing sklearn a 4096-row batch does not
  perform one update: it splits the array into its own 200-row minibatches and steps ~21 times. Taking
  a single step per batch instead looks 12x faster and quietly fits a worse net, so this class steps
  every :data:`T_SGD_BATCH` rows to keep the step count comparable.

  Nothing here is novel and it is deliberately minimal: Adam, ReLU, and an L2 penalty, matching what
  the scikit-learn defaults gave. If the gate ever needs a schedule or dropout, take the dependency
  back rather than growing this class.
  """

  def __init__(
    self,
    n_features: int,
    hidden=T_HIDDEN,
    *,
    lr: float = T_LEARNING_RATE,
    l2: float = T_L2,
    seed: int = 0,
  ):
    rng = np.random.default_rng(seed)
    sizes = [int(n_features), *[int(h) for h in hidden], 1]
    # Glorot uniform: with ReLU and a fan-in of 2048 the naive N(0,1) init saturates the first layer.
    self.weights, self.biases = [], []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
      limit = np.sqrt(6.0 / (fan_in + fan_out))
      self.weights.append(rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32))
      self.biases.append(np.zeros(fan_out, dtype=np.float32))
    self._mw = [np.zeros_like(w) for w in self.weights]
    self._vw = [np.zeros_like(w) for w in self.weights]
    self._mb = [np.zeros_like(b) for b in self.biases]
    self._vb = [np.zeros_like(b) for b in self.biases]
    self.lr = float(lr)
    self.l2 = float(l2)
    self._steps = 0

  def partial_fit(self, x: np.ndarray, y: np.ndarray) -> None:
    """One Adam step on a single mini-batch of float32 features and targets."""
    depth = len(self.weights)
    acts = [x]
    for i in range(depth):
      z = acts[-1] @ self.weights[i] + self.biases[i]
      acts.append(np.maximum(z, 0, out=z) if i < depth - 1 else z)

    # d(MSE)/d(output). The 2/batch factor keeps the step size independent of how the caller batches.
    delta = (acts[-1] - y.reshape(-1, 1).astype(np.float32)) * np.float32(2.0 / len(x))
    self._steps += 1
    bias1 = 1.0 - _BETA1**self._steps
    bias2 = 1.0 - _BETA2**self._steps
    scale = np.float32(self.l2 * 2.0 / len(x))  # delta already carries 2/batch; keep the penalty in step
    for i in range(depth - 1, -1, -1):
      grad_w = acts[i].T @ delta
      if self.l2:
        grad_w += scale * self.weights[i]  # biases stay unpenalised, as they should
      grad_b = delta.sum(axis=0)
      if i:  # propagate before the update, while the weights are still the ones used in the forward pass
        delta = (delta @ self.weights[i].T) * (acts[i] > 0)
      for grad, param, m, v in (
        (grad_w, self.weights[i], self._mw[i], self._vw[i]),
        (grad_b, self.biases[i], self._mb[i], self._vb[i]),
      ):
        m *= _BETA1
        m += (1.0 - _BETA1) * grad
        v *= _BETA2
        v += (1.0 - _BETA2) * grad * grad
        param -= np.float32(self.lr) * (m / bias1) / (np.sqrt(v / bias2) + _EPS)

  def predict(self, x: np.ndarray) -> np.ndarray:
    """Forward pass only, returning one value per row."""
    a = np.asarray(x, dtype=np.float32)
    for i in range(len(self.weights)):
      a = a @ self.weights[i] + self.biases[i]
      if i < len(self.weights) - 1:
        np.maximum(a, 0, out=a)
    return a.ravel()

  def state(self):
    """A deep copy of the parameters, for restoring the best epoch after early stopping."""
    return [w.copy() for w in self.weights], [b.copy() for b in self.biases]

  def restore(self, state) -> None:
    self.weights, self.biases = state

  def to_onnx(self, n_features: int, *, input_name: str = "input", output_name: str = "variable"):
    """Serialise as an ONNX model: ``MatMul``/``Add`` per layer with ``Relu`` between."""
    from onnx import TensorProto, helper, numpy_helper

    nodes, initializers = [], []
    current = input_name
    for i, (w, b) in enumerate(zip(self.weights, self.biases)):
      initializers.append(numpy_helper.from_array(w, f"t_w{i}"))
      initializers.append(numpy_helper.from_array(b, f"t_b{i}"))
      last = i == len(self.weights) - 1
      nodes.append(helper.make_node("MatMul", [current, f"t_w{i}"], [f"t_z{i}"]))
      # Add broadcasts [B, out] + [out] natively; the final Add writes straight to the graph output.
      nodes.append(helper.make_node("Add", [f"t_z{i}", f"t_b{i}"], [output_name if last else f"t_a{i}"]))
      if not last:
        nodes.append(helper.make_node("Relu", [f"t_a{i}"], [f"t_h{i}"]))
        current = f"t_h{i}"

    graph = helper.make_graph(
      nodes,
      "tanimoto_regressor",
      [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, ["B", int(n_features)])],
      [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, ["B", 1])],
      initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _T_OPSET)])
    model.ir_version = _T_IR_VERSION
    return model


def ramp(similarity, *, a_max: float = A_CEILING, sim_lo: float = T_LO, sim_hi: float = T_HI):
  """Map predicted similarity onto a blend weight: 0 below ``sim_lo``, ``a_max`` at ``sim_hi``.

  Linear in between, so two compounds either side of a knee no longer receive very different weights —
  the bucketed gate this replaced jumped from 0.33 to 0.66 across ``sim_hi``.
  """
  s = np.asarray(similarity, dtype=np.float64).ravel()
  span = max(float(sim_hi) - float(sim_lo), 1e-9)
  return float(a_max) * np.clip((s - float(sim_lo)) / span, 0.0, 1.0)


class TanimotoRegressor:
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
      teacher's scale (:func:`olinda.hard._blend_ceiling`). Zero disables the blend entirely.
  """

  def __init__(
    self,
    onnx_bytes: bytes,
    *,
    a_max: float = A_CEILING,
    sim_lo: float = T_LO,
    sim_hi: float = T_HI,
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
    val_x = (np.asarray(validation[0]) > 0).astype(np.float32)
    val_y = np.asarray(validation[1], dtype=np.float32).ravel()

    net = _MLP(int(n_features), T_HIDDEN, lr=T_LEARNING_RATE, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, waited = np.inf, None, 0
    started = time.time()
    for epoch in range(T_MAX_EPOCHS):
      for bits, target in batches(rng):
        # One read off the library, several optimiser steps: see T_SGD_BATCH for why the two sizes
        # are decoupled. The rows arrive already shuffled, so slicing them in order is a fair sample.
        x = (np.asarray(bits) > 0).astype(np.float32)
        y = np.asarray(target, dtype=np.float32)
        for start in range(0, len(x), T_SGD_BATCH):
          net.partial_fit(x[start : start + T_SGD_BATCH], y[start : start + T_SGD_BATCH])
      loss = float(np.mean((net.predict(val_x) - val_y) ** 2))
      # Early stopping is hand-rolled because sklearn's own only applies to `fit`, which would need the
      # whole library in memory. Keep the best weights rather than the last: Adam can step past a good
      # solution, and the epoch that stops the run is by definition not the best one.
      improved = loss < best - 1e-7
      if echo:
        rmse = np.sqrt(loss)  # in similarity units, which is the number a reader can actually judge
        echo(
          f"  epoch {epoch + 1:>2}/{T_MAX_EPOCHS} · val MSE {loss:.6f} · ±{rmse:.3f} similarity"
          f"{' · best' if improved else f' · no gain ({waited + 1}/{T_PATIENCE})'}"
          f" [dim]· {time.time() - started:.0f}s[/]",
          "info",
        )
      if improved:
        best, waited = loss, 0
        best_state = net.state()
      else:
        waited += 1
        if waited >= T_PATIENCE:
          if echo:
            echo(f"  stopped early · no improvement for {T_PATIENCE} epochs", "info")
          break
    if best_state is not None:
      net.restore(best_state)

    return cls(net.to_onnx(int(n_features)).SerializeToString(), **kwargs)

  def _run(self):
    import onnxruntime as ort

    if self._session is None:
      options = ort.SessionOptions()
      options.log_severity_level = 3
      self._session = ort.InferenceSession(self.onnx_bytes, options, providers=["CPUExecutionProvider"])
    return self._session

  def predict_tanimoto(self, bits: np.ndarray) -> np.ndarray:
    """Estimated 1-NN Tanimoto per row, clipped to [0, 1] — the target's own range."""
    sess = self._run()
    x = (np.asarray(bits) > 0).astype(np.float32)
    raw = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    return np.clip(np.asarray(raw, dtype=np.float64).ravel(), 0.0, 1.0)

  def weight(self, bits: np.ndarray) -> np.ndarray:
    """Blend weight ``a`` per row (float64)."""
    return ramp(self.predict_tanimoto(bits), a_max=self.a_max, sim_lo=self.sim_lo, sim_hi=self.sim_hi)

  def save(self, directory: str | Path) -> None:
    """Write the network and the ramp parameters into *directory*."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / _MODEL_NAME).write_bytes(self.onnx_bytes)
    with open(directory / _META_NAME, "w") as fp:
      json.dump(
        {
          "type": "tanimoto_mlp",
          "hidden": list(T_HIDDEN),
          "a_max": self.a_max,
          "sim_lo": self.sim_lo,
          "sim_hi": self.sim_hi,
          "metrics": self.metrics,
        },
        fp,
        indent=2,
      )

  @classmethod
  def load(cls, directory: str | Path) -> "TanimotoRegressor":
    directory = Path(directory)
    with open(directory / _META_NAME) as fp:
      meta = json.load(fp)
    return cls(
      (directory / _MODEL_NAME).read_bytes(),
      a_max=meta.get("a_max", A_CEILING),
      sim_lo=meta.get("sim_lo", T_LO),
      sim_hi=meta.get("sim_hi", T_HI),
      metrics=meta.get("metrics"),
    )
