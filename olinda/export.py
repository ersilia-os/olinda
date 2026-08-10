"""Fuse a trained model dir into a single, self-describing ``model.onnx``.

Every olinda-owned transform downstream of the RDKit featurizer is ONNX-able, so a bundle collapses to ONE
graph that runs on onnxruntime alone. Two shapes:

- **soft-only**: ``fp → soft_model → [soft_correction] → prediction`` (= surrogate).
- **hard present**: fuse ``soft_model → [soft_correction]`` (surrogate), ``hard_model → [G score] →
  hard_correction`` (ground_truth_soft), ``applicability`` (weight ``a``) and ``blender``
  (``(1-a)·soft + a·hard``). Outputs: ``prediction``, ``surrogate``, ``ground_truth``, ``ground_truth_soft``,
  ``applicability``.

The **featurizer config + provenance travel inside ``model.onnx`` metadata** (``metadata_props["olinda"]``),
so the file is self-describing — a consumer reads the Morgan config (and RDKit version) to build the 2048-count
fingerprint in Python (no ONNX op for featurization) and runs the single graph.

The hard head is task-aware: a **classifier** exposes ``probabilities`` (we take column 1); a **regressor**
would expose ``variable`` directly (seam — only classifier is enabled today, see :func:`olinda.ground_truth`).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from olinda.console import echo, rule, success, summary_panel
from olinda.ground_truth import (
  APPLICABILITY_NAME,
  CALIBRATOR_NAME,
  GT_DIRNAME,
  GT_META_NAME,
  GT_MODEL_SUBDIR,
  has_hard_head,
)

MODEL_NAME = "model.onnx"
_IR_VERSION = 10  # onnxruntime in this env caps the model IR version at 10
_OPSET = 16
_ISOTONIC_TOL = 1e-5  # knot-thinning target — kept below _PARITY_TOL so ONNX float error has headroom
_ISOTONIC_MAX_KNOTS = 4096
_PARITY_TOL = 1e-4  # build_bundle raises if the fused graph drifts from the Python reference beyond this
_SAMPLE_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "OCc1ccccc1"]


def _save(model, path: Path) -> None:
  import onnx

  model.ir_version = _IR_VERSION
  onnx.checker.check_model(model)
  onnx.save(model, str(path))


# ── isotonic calibrator → ONNX ────────────────────────────────────────────────


def _rdp_vertical(x: np.ndarray, y: np.ndarray, tol: float) -> np.ndarray:
  """Ramer–Douglas–Peucker on the ``(x, y)`` polyline with a VERTICAL tolerance; return the kept-point mask.

  Bounding the deviation at the anchors bounds it everywhere (both maps are piecewise-linear over the
  anchors) — unlike grid sampling, which only controls the sampled points.
  """
  n = len(x)
  keep = np.zeros(n, dtype=bool)
  keep[0] = keep[-1] = True
  stack = [(0, n - 1)]
  while stack:
    i, j = stack.pop()
    if j <= i + 1:
      continue
    xs = x[i : j + 1]
    dx = x[j] - x[i]
    chord = y[i] + (y[j] - y[i]) * ((xs - x[i]) / dx) if dx != 0 else np.full(len(xs), y[i])
    d = np.abs(y[i : j + 1] - chord)
    k = int(d.argmax())
    if d[k] > tol:
      keep[i + k] = True
      stack.append((i, i + k))
      stack.append((i + k, j))
  return keep


def _thin_isotonic(x: np.ndarray, y: np.ndarray, tol: float, max_knots: int):
  """Simplify the monotone map to within ``tol`` via vertical-distance RDP (knots are true anchors)."""
  if len(x) <= 2:
    return x, y
  t = tol
  for _ in range(40):
    keep = _rdp_vertical(x, y, t)
    if int(keep.sum()) <= max_knots:
      break
    t *= 2
  return x[keep], y[keep]


def _isotonic_model(cal, in_name: str, out_name: str):
  """ONNX ``ModelProto``: raw(float32) → ``np.interp(sign*raw, x, y)`` in float64 (searchsorted + one step)."""
  from onnx import TensorProto, helper

  xk, yk = _thin_isotonic(cal._x, cal._y, _ISOTONIC_TOL, _ISOTONIC_MAX_KNOTS)
  xk = xk.astype(np.float64)
  yk = yk.astype(np.float64)
  slope = np.zeros_like(xk)
  slope[:-1] = (yk[1:] - yk[:-1]) / (xk[1:] - xk[:-1])  # slope[-1]=0 → end-clamp
  n = len(xk)
  ct = helper.make_tensor
  init = [
    ct("sign", TensorProto.DOUBLE, [1], [float(cal._sign)]),
    ct("xk", TensorProto.DOUBLE, [n], xk),
    ct("yk", TensorProto.DOUBLE, [n], yk),
    ct("slope", TensorProto.DOUBLE, [n], slope),
    ct("x0", TensorProto.DOUBLE, [1], [float(xk[0])]),
    ct("xlast", TensorProto.DOUBLE, [1], [float(xk[-1])]),
    ct("one_i", TensorProto.INT64, [1], [1]),
    ct("nm1", TensorProto.INT64, [1], [n - 1]),
    ct("zero_i", TensorProto.INT64, [1], [0]),
    ct("axis1", TensorProto.INT64, [1], [1]),
  ]
  nodes = [
    helper.make_node("Cast", [in_name], ["rawd"], to=TensorProto.DOUBLE),
    helper.make_node("Mul", ["rawd", "sign"], ["t_raw"]),
    helper.make_node("Max", ["t_raw", "x0"], ["t_lo"]),  # clamp into [x0, xlast] → out-of-range holds ends
    helper.make_node("Min", ["t_lo", "xlast"], ["t"]),
    helper.make_node("Unsqueeze", ["t", "axis1"], ["t2"]),
    helper.make_node("LessOrEqual", ["xk", "t2"], ["le"]),
    helper.make_node("Cast", ["le"], ["lei"], to=TensorProto.INT64),
    helper.make_node("ReduceSum", ["lei", "axis1"], ["cnt"], keepdims=0),
    helper.make_node("Sub", ["cnt", "one_i"], ["idx0"]),
    helper.make_node("Max", ["idx0", "zero_i"], ["idxc"]),
    helper.make_node("Min", ["idxc", "nm1"], ["idx"]),
    helper.make_node("Gather", ["xk", "idx"], ["x_i"]),
    helper.make_node("Gather", ["yk", "idx"], ["y_i"]),
    helper.make_node("Gather", ["slope", "idx"], ["m_i"]),
    helper.make_node("Sub", ["t", "x_i"], ["dx"]),
    helper.make_node("Mul", ["dx", "m_i"], ["step"]),
    helper.make_node("Add", ["y_i", "step"], [out_name]),
  ]
  g = helper.make_graph(
    nodes,
    "isotonic",
    [helper.make_tensor_value_info(in_name, TensorProto.FLOAT, ["B"])],
    [helper.make_tensor_value_info(out_name, TensorProto.DOUBLE, ["B"])],
    initializer=init,
  )
  return helper.make_model(g, opset_imports=[helper.make_opsetid("", _OPSET)])


def isotonic_to_onnx(cal, out_path: Path, *, in_name: str = "input", out_name: str = "output") -> dict:
  """Save a standalone isotonic ONNX and self-check vs ``cal.transform``; return ``{knots, max_abs_diff}``."""
  import onnxruntime as ort

  model = _isotonic_model(cal, in_name, out_name)
  _save(model, out_path)
  span = float(cal._x.max() - cal._x.min()) or 1.0
  grid = np.linspace(float(cal._x.min()) - 0.1 * span, float(cal._x.max()) + 0.1 * span, 2000)
  raw = (cal._sign * grid).astype(np.float32)
  sess = ort.InferenceSession(out_path.read_bytes(), providers=["CPUExecutionProvider"])
  got = sess.run(None, {in_name: raw})[0]
  n_knots = int(len(_thin_isotonic(cal._x, cal._y, _ISOTONIC_TOL, _ISOTONIC_MAX_KNOTS)[0]))
  return {"knots": n_knots, "max_abs_diff": float(np.max(np.abs(got - cal.transform(raw))))}


# ── applicability NB gate → ONNX ──────────────────────────────────────────────


def _nb_linear(nb):
  """(W (d,2), B (2,)) such that argmax(bits @ W + B) == BernoulliNB.predict(bits)."""
  W = (nb.log_theta_ - nb.log_neg_theta_).T
  B = nb.log_prior_ + nb.log_neg_theta_.sum(axis=1)
  return W.astype(np.float64), B.astype(np.float64)


def _applicability_model(clf, n_features: int, in_name: str, out_name: str):
  """ONNX ``ModelProto``: fp(float32) → blend weight ``a`` (double) via two linear NB scorers + bucket."""
  from onnx import TensorProto, helper

  Wl, Bl = _nb_linear(clf.clf_low)
  Wh, Bh = _nb_linear(clf.clf_high)
  ct = helper.make_tensor
  init = [
    ct("Wl", TensorProto.DOUBLE, list(Wl.shape), Wl.ravel()),
    ct("Bl", TensorProto.DOUBLE, [2], Bl),
    ct("Wh", TensorProto.DOUBLE, list(Wh.shape), Wh.ravel()),
    ct("Bh", TensorProto.DOUBLE, [2], Bh),
    ct("zero", TensorProto.FLOAT, [1], [0.0]),
    ct("a_low", TensorProto.DOUBLE, [1], [clf.a_low]),
    ct("a_high", TensorProto.DOUBLE, [1], [clf.a_high]),
    ct("a_zero", TensorProto.DOUBLE, [1], [0.0]),
  ]
  nodes = [
    helper.make_node("Greater", [in_name, "zero"], ["onb"]),
    helper.make_node("Cast", ["onb"], ["x"], to=TensorProto.DOUBLE),
    helper.make_node("MatMul", ["x", "Wl"], ["sl0"]),
    helper.make_node("Add", ["sl0", "Bl"], ["sl"]),
    helper.make_node("ArgMax", ["sl"], ["low_c"], axis=1, keepdims=0),
    helper.make_node("MatMul", ["x", "Wh"], ["sh0"]),
    helper.make_node("Add", ["sh0", "Bh"], ["sh"]),
    helper.make_node("ArgMax", ["sh"], ["high_c"], axis=1, keepdims=0),
    helper.make_node("Cast", ["high_c"], ["highb"], to=TensorProto.BOOL),
    helper.make_node("Cast", ["low_c"], ["lowb"], to=TensorProto.BOOL),
    helper.make_node("Where", ["lowb", "a_low", "a_zero"], ["a_low_or_0"]),
    helper.make_node("Where", ["highb", "a_high", "a_low_or_0"], [out_name]),
  ]
  g = helper.make_graph(
    nodes,
    "applicability",
    [helper.make_tensor_value_info(in_name, TensorProto.FLOAT, ["B", int(n_features)])],
    [helper.make_tensor_value_info(out_name, TensorProto.DOUBLE, ["B"])],
    initializer=init,
  )
  return helper.make_model(g, opset_imports=[helper.make_opsetid("", _OPSET)])


def applicability_to_onnx(clf, n_features: int, out_path: Path, *, in_name: str = "input") -> dict:
  """Save a standalone applicability ONNX and self-check vs ``clf.weight``; return ``{max_abs_diff}``."""
  import onnxruntime as ort

  _save(_applicability_model(clf, n_features, in_name, "applicability"), out_path)
  rng = np.random.default_rng(0)
  fp = (rng.random((64, n_features)) < 0.1).astype(np.float32)
  sess = ort.InferenceSession(out_path.read_bytes(), providers=["CPUExecutionProvider"])
  got = sess.run(None, {in_name: fp})[0]
  return {"max_abs_diff": float(np.max(np.abs(got - clf.weight(fp > 0))))}


# ── small structural graphs (blender, prob→G) ─────────────────────────────────


def _blender_model():
  """ONNX ``ModelProto``: (soft, hard, a) → ``(1-a)*soft + a*hard`` (all double)."""
  from onnx import TensorProto, helper

  vi = lambda n: helper.make_tensor_value_info(n, TensorProto.DOUBLE, ["B"])  # noqa: E731
  nodes = [
    helper.make_node("Sub", ["one", "a"], ["oma"]),
    helper.make_node("Mul", ["oma", "soft"], ["p0"]),
    helper.make_node("Mul", ["a", "hard"], ["p1"]),
    helper.make_node("Add", ["p0", "p1"], ["prediction"]),
  ]
  g = helper.make_graph(
    nodes,
    "blender",
    [vi("soft"), vi("hard"), vi("a")],
    [vi("prediction")],
    initializer=[helper.make_tensor("one", TensorProto.DOUBLE, [1], [1.0])],
  )
  return helper.make_model(g, opset_imports=[helper.make_opsetid("", _OPSET)])


def _prob1_model(in_name: str, out_name: str):
  """ONNX ``ModelProto``: classifier ``probabilities`` (B,2) → column 1 (B,) (a classifier's positive score)."""
  from onnx import TensorProto, helper

  nodes = [helper.make_node("Gather", [in_name, "one_idx"], [out_name], axis=1)]
  g = helper.make_graph(
    nodes,
    "prob1",
    [helper.make_tensor_value_info(in_name, TensorProto.FLOAT, ["B", 2])],
    [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, ["B"])],
    initializer=[helper.make_tensor("one_idx", TensorProto.INT64, [], [1])],  # scalar → drops axis 1
  )
  return helper.make_model(g, opset_imports=[helper.make_opsetid("", _OPSET)])


def _soft_model_proto(sm, x_dim: int):
  """The surrogate regressor as an ONNX ``ModelProto`` (via the backend exporter, through a temp file)."""
  import onnx

  from olinda.train.backend import get_backend

  with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "s.onnx"
    get_backend(sm.backend, "cpu").to_onnx(sm.model, p, x_dim)
    return onnx.load(str(p))


# ── fusion ────────────────────────────────────────────────────────────────────


def _fit_soft_calibration(sm, model_dir: Path):
  """Fit the surrogate isotonic correction on ``val.h5`` (raw S → soft labels); ``None`` if unavailable."""
  import h5py

  from olinda.calibrate import IsotonicCalibrator

  val = model_dir / "val.h5"
  if not val.exists():
    return None
  with h5py.File(val, "r") as f:
    vx = np.asarray(f["x"][:], dtype=np.float32)
    vy = np.asarray(f["y"][:], dtype=np.float64)
  if len(vy) < 4 or not np.isfinite(vy).any():
    return None
  raw = np.asarray(sm.predict(X=vx, calibrate=False)).ravel()
  return IsotonicCalibrator().fit(raw=raw, target=vy)  # increasing (S already predicts the soft target)


def _bundle_metadata(
  model_dir: Path, featurizer: dict, featurizer_class: str, has_hard: bool, outputs
) -> dict:
  """Provenance embedded in ``model.onnx`` metadata: featurizer (+ RDKit), reference lib, hard-head summary."""
  import importlib.metadata

  import rdkit

  from olinda.data.fetch import MORGAN_FINGERPRINTS_FILENAME, MORGAN_FINGERPRINTS_URL

  try:
    olinda_version = importlib.metadata.version("olinda")
  except Exception:
    olinda_version = "unknown"
  md = {
    "olinda_version": olinda_version,
    "featurizer": {**featurizer, "rdkit_version": rdkit.__version__},
    "featurizer_class": featurizer_class,
    "reference_library": {"name": MORGAN_FINGERPRINTS_FILENAME, "url": MORGAN_FINGERPRINTS_URL},
    "has_hard": bool(has_hard),
    "outputs": list(outputs),
  }
  if has_hard:
    with open(model_dir / GT_DIRNAME / GT_META_NAME) as fp:
      gtm = json.load(fp)
    task = gtm.get("task", "binary")
    md["hard"] = {
      "n_train": gtm.get("n"),
      "task": "regression" if task == "regression" else "classification",
      "lazyqsar_version": gtm.get("lazyqsar_version"),
    }
  return md


def _toposort(nodes: list, available: set) -> list:
  """Order nodes so every input is produced earlier (ONNX requires topological order after our bridging)."""
  ordered: list = []
  avail = set(available)
  pending = list(nodes)
  while pending:
    rest = []
    progressed = False
    for nd in pending:
      if all(i == "" or i in avail for i in nd.input):
        ordered.append(nd)
        avail.update(nd.output)
        progressed = True
      else:
        rest.append(nd)
    if not progressed:
      raise RuntimeError(f"cannot topologically order nodes (unresolved: {[n.name for n in rest][:5]})")
    pending = rest
  return ordered


def _fuse(model_dir: Path):
  """Assemble the single fused ``model.onnx`` ModelProto for *model_dir* (soft-only or hard)."""
  import onnx
  from onnx import TensorProto, helper
  from onnx.compose import add_prefix

  from olinda.applicability import ApplicabilityClassifier
  from olinda.calibrate import IsotonicCalibrator
  from olinda.featurizer import featurizer_from_meta
  from olinda.models.bundle import StudentModel

  sm = StudentModel.load(model_dir, featurizer_factory=featurizer_from_meta)
  featurizer = sm.metadata.get("featurizer") or {}
  featurizer_class = sm.metadata.get("featurizer_class", "MorganCountFeaturizer")
  n_features = int(featurizer.get("fp_size", sm.metadata.get("x_dim", 2048)))
  x_dim = int(sm.metadata.get("x_dim") or n_features)
  has_hard = has_hard_head(model_dir)

  nodes: list = []
  inits: list = []
  opset: dict = {"": _OPSET}

  def collect(model, prefix: str) -> None:
    m = add_prefix(model, prefix + "__")
    nodes.extend(m.graph.node)
    inits.extend(m.graph.initializer)
    for op in m.opset_import:
      opset[op.domain] = max(opset.get(op.domain, 0), op.version)

  ident = lambda src, dst: helper.make_node("Identity", [src], [dst], name=f"br_{dst}")  # noqa: E731
  cast_d = lambda src, dst: helper.make_node("Cast", [src], [dst], to=TensorProto.DOUBLE, name=f"br_{dst}")  # noqa: E731
  # XGBoost ONNX outputs are (N,1); flatten to (N,) so they line up with the (N,) calibration/gate stages.
  inits.append(helper.make_tensor("flat_shape", TensorProto.INT64, [1], [-1]))
  flat = lambda src, dst: helper.make_node("Reshape", [src, "flat_shape"], [dst], name=f"fl_{dst}")  # noqa: E731

  # soft branch: input → soft_model → [soft_correction] → "surrogate" (double)
  collect(_soft_model_proto(sm, x_dim), "sm")
  nodes.append(ident("input", "sm__input"))
  nodes.append(flat("sm__variable", "soft_raw"))
  soft_raw = "soft_raw"  # float32 (N,)
  cal_soft = _fit_soft_calibration(sm, model_dir)
  if cal_soft is not None:
    collect(_isotonic_model(cal_soft, "in", "out"), "sc")
    nodes.append(ident(soft_raw, "sc__in"))
    surrogate_src = "sc__out"  # double
  else:
    surrogate_src = soft_raw  # float32
  nodes.append(cast_d(surrogate_src, "surrogate"))  # canonical double surrogate

  outputs = ["prediction", "surrogate"]
  out_vi = {}

  if has_hard:
    gt_root = model_dir / GT_DIRNAME
    with open(gt_root / GT_META_NAME) as fp:
      hard_task = json.load(fp).get("task", "binary")

    collect(onnx.load(str(gt_root / GT_MODEL_SUBDIR / "xgboost.onnx")), "hm")
    nodes.append(ident("input", "hm__float_input"))
    if hard_task == "regression":  # seam: G regressor exposes a single "variable" output, no slice
      nodes.append(flat("hm__variable", "g_reg"))
      g_src = "g_reg"
    else:  # classifier: take probabilities[:, 1]
      collect(_prob1_model("p", "g"), "pr")
      nodes.append(ident("hm__probabilities", "pr__p"))
      g_src = "pr__g"
    nodes.append(cast_d(g_src, "ground_truth"))

    gcal = IsotonicCalibrator.load(gt_root / CALIBRATOR_NAME)
    collect(_isotonic_model(gcal, "in", "out"), "hc")
    nodes.append(ident(g_src, "hc__in"))
    nodes.append(ident("hc__out", "ground_truth_soft"))

    clf = ApplicabilityClassifier.load(gt_root / APPLICABILITY_NAME)
    collect(_applicability_model(clf, n_features, "input", "applicability"), "ap")
    nodes.append(ident("input", "ap__input"))
    nodes.append(ident("ap__applicability", "applicability"))

    collect(_blender_model(), "bl")
    nodes.append(ident("surrogate", "bl__soft"))
    nodes.append(ident("ground_truth_soft", "bl__hard"))
    nodes.append(ident("applicability", "bl__a"))
    nodes.append(ident("bl__prediction", "prediction"))

    outputs = ["prediction", "surrogate", "ground_truth", "ground_truth_soft", "applicability"]
    for name in outputs:
      out_vi[name] = helper.make_tensor_value_info(name, TensorProto.DOUBLE, ["B"])
  else:
    nodes.append(ident("surrogate", "prediction"))
    for name in outputs:
      out_vi[name] = helper.make_tensor_value_info(name, TensorProto.DOUBLE, ["B"])

  graph = helper.make_graph(
    _toposort(nodes, {"input"} | {i.name for i in inits}),
    "olinda_model",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["B", int(n_features)])],
    [out_vi[n] for n in outputs],
    initializer=inits,
  )
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid(d, v) for d, v in opset.items()])
  md = _bundle_metadata(model_dir, featurizer, featurizer_class, has_hard, outputs)
  entry = model.metadata_props.add()
  entry.key, entry.value = "olinda", json.dumps(md)
  return model, has_hard, outputs


def build_bundle(model_dir: str | Path) -> dict:
  """Fuse *model_dir* into ``model.onnx`` and gate on numeric parity vs the Python reference. Returns paths."""
  import onnxruntime as ort

  from olinda.featurizer import featurizer_from_meta
  from olinda.models.bundle import StudentModel

  model_dir = Path(model_dir)
  rule("olinda · export", style="green", right=str(model_dir))
  echo("fusing pipeline → model.onnx", "run")
  model, has_hard, outputs = _fuse(model_dir)
  _save(model, model_dir / MODEL_NAME)

  # parity: run the fused graph vs the Python composition of the exact sources
  echo("checking parity: model.onnx vs Python reference", "run")
  sm = StudentModel.load(model_dir, featurizer_factory=featurizer_from_meta)
  fp = sm.featurizer.transform(_SAMPLE_SMILES).astype(np.float32)
  raw = np.asarray(sm.predict(X=fp, calibrate=False)).ravel()
  cal_soft = _fit_soft_calibration(sm, model_dir)
  surrogate_ref = (
    np.asarray(cal_soft.transform(raw)).ravel() if cal_soft is not None else raw.astype(np.float64)
  )
  ref = {"surrogate": surrogate_ref, "prediction": surrogate_ref}

  if has_hard:
    from lazyqsar.base.xgboost import BaseXGBArtifact

    from olinda.applicability import ApplicabilityClassifier
    from olinda.calibrate import IsotonicCalibrator

    gt_root = model_dir / GT_DIRNAME
    g = np.asarray(BaseXGBArtifact.load(str(gt_root / GT_MODEL_SUBDIR)).run(fp))[:, 1].astype(np.float64)
    gsoft = np.asarray(IsotonicCalibrator.load(gt_root / CALIBRATOR_NAME).transform(g)).ravel()
    a = np.asarray(ApplicabilityClassifier.load(gt_root / APPLICABILITY_NAME).weight(fp > 0)).ravel()
    ref = {
      "surrogate": surrogate_ref,
      "ground_truth": g,
      "ground_truth_soft": gsoft,
      "applicability": a,
      "prediction": (1.0 - a) * surrogate_ref + a * gsoft,
    }

  sess = ort.InferenceSession((model_dir / MODEL_NAME).read_bytes(), providers=["CPUExecutionProvider"])
  got = {o.name: np.asarray(v).ravel() for o, v in zip(sess.get_outputs(), sess.run(None, {"input": fp}))}
  parity = {k: float(np.max(np.abs(got[k] - ref[k]))) for k in outputs}
  worst = max(parity.values())
  if worst > _PARITY_TOL:
    raise RuntimeError(f"model.onnx parity failed: max abs diff {worst:.2e} > {_PARITY_TOL:.0e} ({parity})")

  rows = [
    ("Head", "soft + hard (blend)" if has_hard else "soft only"),
    ("Outputs", " · ".join(outputs)),
    ("Parity (max)", f"[bold]{worst:.2e}[/] ≤ {_PARITY_TOL:.0e}"),
    ("Saved", f"[dim]{model_dir / MODEL_NAME}[/]"),
  ]
  summary_panel("olinda · export", rows, border_style="green", icon="✓")
  success(f"fused model.onnx built and parity-checked → [dim]{model_dir / MODEL_NAME}[/]")
  return {"model": str(model_dir / MODEL_NAME), "has_hard": has_hard, "parity": parity}
