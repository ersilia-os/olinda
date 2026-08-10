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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from olinda.console import echo, path as cpath, rule, success, summary_panel
from olinda.metrics import json_safe
from olinda.ground_truth import (
  APPLICABILITY_NAME,
  CALIBRATOR_NAME,
  GT_DIRNAME,
  GT_META_NAME,
  GT_MODEL_SUBDIR,
  has_hard_head,
)

MODEL_NAME = "model.onnx"
BUNDLE_SCHEMA = "olinda.bundle.v1"
PRODUCER_NAME = "olinda"
_IR_VERSION = 10  # onnxruntime in this env caps the model IR version at 10
_OPSET = 16
_ISOTONIC_TOL = 1e-5  # knot-thinning target — kept below _PARITY_TOL so ONNX float error has headroom
_ISOTONIC_MAX_KNOTS = 4096
_PARITY_TOL = 1e-4  # build_bundle raises if the fused graph drifts from the Python reference beyond this
_SAMPLE_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "OCc1ccccc1"]


_PROTOBUF_LIMIT = 2 * 1024**3  # a single serialized ModelProto cannot exceed this


def _strip_dead_attributes(model) -> int:
  """Drop tree attributes that carry no information; returns the bytes saved.

  ``nodes_hitrates`` is emitted as one float per node and, for boosters converted from LightGBM or
  XGBoost, is uniformly ``1.0`` — on a real model that is 1.5M copies of the same constant, about an
  eighth of the file, which onnxruntime ignores. It is optional in the ai.onnx.ml schema, so removing
  it leaves predictions bit-identical.

  ``nodes_missing_value_tracks_true`` looks similar but genuinely carries both values, so it stays.
  """
  import numpy as np

  saved = 0
  for node in model.graph.node:
    if not node.op_type.startswith("TreeEnsemble"):
      continue
    keep = []
    for attr in node.attribute:
      if attr.name == "nodes_hitrates" and len(attr.floats) and np.all(np.asarray(attr.floats) == 1.0):
        saved += attr.ByteSize()
        continue
      keep.append(attr)
    if len(keep) != len(node.attribute):
      del node.attribute[:]
      node.attribute.extend(keep)
  return saved


def _save(model, path: Path) -> None:
  import onnx

  model.ir_version = _IR_VERSION
  _strip_dead_attributes(model)
  size = model.ByteSize()
  if size >= _PROTOBUF_LIMIT:
    raise ValueError(
      f"fused model is {size / 1e9:.2f} GB, over ONNX's {_PROTOBUF_LIMIT / 1e9:.2f} GB protobuf limit. "
      "Reduce the number of columns or --num-boost-round."
    )
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


def _bundle_metadata(manifest: dict, plan: list, featurizer: dict, featurizer_class: str, outputs) -> dict:
  """Everything a consumer needs, embedded in ``model.onnx`` so the file is the only input required.

  Carries the featurizer (and the RDKit build it must run under), when the model was trained, the
  reference library it was distilled from, and one entry per column describing that task.
  """
  import importlib.metadata

  import rdkit

  from olinda.data.fetch import MORGAN_FINGERPRINTS_FILENAME, MORGAN_FINGERPRINTS_URL

  try:
    olinda_version = importlib.metadata.version("olinda")
  except Exception:
    olinda_version = "unknown"

  columns = []
  for entry in plan:
    col = {
      "name": entry["name"],
      "id": entry["id"],
      "output": entry["output"],
      "has_hard": bool(entry["has_hard"]),
      "metrics": entry.get("metrics"),
    }
    if entry["has_hard"]:
      gtm_path = entry["dir"] / GT_DIRNAME / GT_META_NAME
      gtm = json.loads(gtm_path.read_text()) if gtm_path.exists() else {}
      task = gtm.get("task", "binary")
      col["hard"] = {
        "source_column": (entry.get("hard_meta") or {}).get("source_column"),
        "n_train": gtm.get("n"),
        "task": "regression" if task == "regression" else "classification",
        "lazyqsar_version": gtm.get("lazyqsar_version"),
      }
    columns.append(col)

  return {
    "schema": BUNDLE_SCHEMA,
    "producer": "olinda",
    "olinda_version": olinda_version,
    "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "featurizer": {**featurizer, "rdkit_version": rdkit.__version__},
    "featurizer_class": featurizer_class,
    "reference_library": {"name": MORGAN_FINGERPRINTS_FILENAME, "url": MORGAN_FINGERPRINTS_URL},
    "n_columns": len(columns),
    "columns": columns,
    "outputs": list(outputs),
    "has_hard": any(c["has_hard"] for c in columns),
    "backend": manifest.get("backend"),
    "split": manifest.get("split"),
  }


def _toposort(nodes: list, available: set) -> list:
  """Order *nodes* so every input is produced before it is consumed.

  ``add_prefix`` leaves each sub-model's nodes contiguous, and the Identity bridges that wire columns
  together are appended afterwards, so the raw list is not in dependency order. Kahn's algorithm over
  a producer map keeps this linear — a repeated-scan sort would be quadratic in the node count, which
  grows with the number of columns.
  """
  producer: dict = {}
  for i, nd in enumerate(nodes):
    for out in nd.output:
      producer[out] = i

  indegree = [0] * len(nodes)
  dependents: dict = {i: [] for i in range(len(nodes))}
  for i, nd in enumerate(nodes):
    for inp in nd.input:
      if inp in available or inp not in producer:
        continue
      j = producer[inp]
      if j != i:
        dependents[j].append(i)
        indegree[i] += 1

  queue = [i for i, d in enumerate(indegree) if d == 0]
  order: list = []
  while queue:
    i = queue.pop()
    order.append(nodes[i])
    for k in dependents[i]:
      indegree[k] -= 1
      if indegree[k] == 0:
        queue.append(k)

  if len(order) != len(nodes):
    unresolved = [nodes[i].name for i, d in enumerate(indegree) if d > 0][:5]
    raise ValueError(f"fused graph has a dependency cycle or a missing producer near: {unresolved}")
  return order


def _column_plan(model_dir: Path) -> tuple[dict, list]:
  """The run manifest plus, per column, where its artifacts live and what its graph output is called.

  Built once and shared by the fuse and the parity check, so both describe exactly the same sources.
  """
  from olinda import run as runlib

  manifest = runlib.read_manifest(model_dir)
  plan = []
  seen: set[str] = set()
  for col in manifest["columns"]:
    col_dir = runlib.column_dir(model_dir, col["id"])
    output = col["name"]
    if output in seen:
      raise ValueError(f"two columns would produce the output name {output!r}")
    seen.add(output)
    plan.append({
      "id": col["id"],
      "name": col["name"],
      "output": output,
      "dir": col_dir,
      "has_hard": has_hard_head(col_dir),
      "hard_meta": col.get("hard"),
      "metrics": col.get("metrics"),
    })
  if not plan:
    raise ValueError(f"{model_dir} has no columns — run `olinda prepare` first")
  return manifest, plan


def _fuse(model_dir: Path):
  """Assemble the fused ``model.onnx`` ModelProto: every column, sharing one input tensor.

  Each column contributes an independent sub-pipeline under its own ``c{i}_`` prefix, so a run with
  one column is simply the one-column case of a run with many. The graph exposes exactly one output
  per column — its blended prediction, named after the column.
  """
  import onnx
  from onnx import TensorProto, helper
  from onnx.compose import add_prefix

  from olinda.applicability import ApplicabilityClassifier
  from olinda.calibrate import IsotonicCalibrator
  from olinda.featurizer import featurizer_from_meta
  from olinda.models.bundle import StudentModel

  manifest, plan = _column_plan(model_dir)

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
  # Tree ONNX outputs are (N,1); flatten to (N,) so they line up with the (N,) calibration/gate stages.
  # Added once for the whole graph — a per-column copy would collide on name and fail the checker.
  inits.append(helper.make_tensor("flat_shape", TensorProto.INT64, [1], [-1]))
  flat = lambda src, dst: helper.make_node("Reshape", [src, "flat_shape"], [dst], name=f"fl_{dst}")  # noqa: E731

  featurizer: dict = {}
  featurizer_class = "MorganCountFeaturizer"
  n_features = 2048
  outputs: list[str] = []

  for entry in plan:
    p = entry["id"] + "_"  # every node and tensor of this column is namespaced by it
    sm = StudentModel.load(entry["dir"], featurizer_factory=featurizer_from_meta)
    featurizer = sm.metadata.get("featurizer") or featurizer
    featurizer_class = sm.metadata.get("featurizer_class", featurizer_class)
    n_features = int(featurizer.get("fp_size", sm.metadata.get("x_dim", n_features)))
    x_dim = int(sm.metadata.get("x_dim") or n_features)

    # soft branch: input → soft_model → [soft correction] → surrogate (double)
    collect(_soft_model_proto(sm, x_dim), f"{p}sm")
    nodes.append(ident("input", f"{p}sm__input"))
    nodes.append(flat(f"{p}sm__variable", f"{p}soft_raw"))
    soft_raw = f"{p}soft_raw"
    if sm.calibrator is not None:  # fitted during learn-soft and loaded from calibrator.json
      collect(_isotonic_model(sm.calibrator, "in", "out"), f"{p}sc")
      nodes.append(ident(soft_raw, f"{p}sc__in"))
      surrogate_src = f"{p}sc__out"
    else:
      surrogate_src = soft_raw
    nodes.append(cast_d(surrogate_src, f"{p}surrogate"))

    if entry["has_hard"]:
      gt_root = entry["dir"] / GT_DIRNAME
      with open(gt_root / GT_META_NAME) as fp:
        hard_task = json.load(fp).get("task", "binary")

      collect(onnx.load(str(gt_root / GT_MODEL_SUBDIR / "xgboost.onnx")), f"{p}hm")
      nodes.append(ident("input", f"{p}hm__float_input"))
      if hard_task == "regression":  # seam: a G regressor exposes a single "variable" output
        nodes.append(flat(f"{p}hm__variable", f"{p}g_reg"))
        g_src = f"{p}g_reg"
      else:  # classifier: take probabilities[:, 1]
        collect(_prob1_model("p", "g"), f"{p}pr")
        nodes.append(ident(f"{p}hm__probabilities", f"{p}pr__p"))
        g_src = f"{p}pr__g"
      nodes.append(cast_d(g_src, f"{p}ground_truth"))

      gcal = IsotonicCalibrator.load(gt_root / CALIBRATOR_NAME)
      collect(_isotonic_model(gcal, "in", "out"), f"{p}hc")
      nodes.append(ident(g_src, f"{p}hc__in"))
      nodes.append(ident(f"{p}hc__out", f"{p}ground_truth_soft"))

      clf = ApplicabilityClassifier.load(gt_root / APPLICABILITY_NAME)
      collect(_applicability_model(clf, n_features, "input", "applicability"), f"{p}ap")
      nodes.append(ident("input", f"{p}ap__input"))
      nodes.append(ident(f"{p}ap__applicability", f"{p}applicability"))

      collect(_blender_model(), f"{p}bl")
      nodes.append(ident(f"{p}surrogate", f"{p}bl__soft"))
      nodes.append(ident(f"{p}ground_truth_soft", f"{p}bl__hard"))
      nodes.append(ident(f"{p}applicability", f"{p}bl__a"))
      nodes.append(ident(f"{p}bl__prediction", f"{p}prediction"))
    else:
      nodes.append(ident(f"{p}surrogate", f"{p}prediction"))

    nodes.append(ident(f"{p}prediction", entry["output"]))
    outputs.append(entry["output"])

  out_vi = [helper.make_tensor_value_info(n, TensorProto.DOUBLE, ["B"]) for n in outputs]
  graph = helper.make_graph(
    _toposort(nodes, {"input"} | {i.name for i in inits}),
    "olinda_model",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["B", int(n_features)])],
    out_vi,
    initializer=inits,
  )
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid(d, v) for d, v in opset.items()])
  md = _bundle_metadata(manifest, plan, featurizer, featurizer_class, outputs)

  # Standard ONNX provenance, so the file identifies itself to any tool (Netron, hub tooling, the
  # onnx CLI) without them having to know about the custom metadata key below.
  model.producer_name = PRODUCER_NAME
  model.producer_version = str(md.get("olinda_version", ""))
  model.domain = "io.ersilia.olinda"
  model.doc_string = (
    f"Distilled model produced by olinda {md.get('olinda_version')} on {md.get('trained_at')}. "
    f"Input: {int(n_features)}-d Morgan count fingerprint "
    f"(RDKit {md['featurizer'].get('rdkit_version')}). Outputs: {', '.join(outputs)}."
  )
  entry_prop = model.metadata_props.add()
  # json_safe first: metrics can carry NaN, which strict JSON parsers in other languages reject.
  entry_prop.key, entry_prop.value = "olinda", json.dumps(json_safe(md))
  return model, plan, outputs


def build_bundle(model_dir: str | Path) -> dict:
  """Fuse a run directory into ``model.onnx``, gated on numeric parity against the Python pipeline.

  Every column is checked independently: the graph's output for that column must match a pure-Python
  recomposition of exactly the same artifacts, so a mis-wired subgraph cannot slip through.
  """
  import onnxruntime as ort

  from olinda.featurizer import featurizer_from_meta
  from olinda.models.bundle import StudentModel

  model_dir = Path(model_dir)
  rule("olinda · export", style="green", right=str(model_dir))
  model, plan, outputs = _fuse(model_dir)
  echo(f"fusing {len(plan)} column(s) → model.onnx", "run")
  _save(model, model_dir / MODEL_NAME)

  echo("checking parity: model.onnx vs Python reference", "run")
  ref: dict = {}
  fp = None
  for entry in plan:
    sm = StudentModel.load(entry["dir"], featurizer_factory=featurizer_from_meta)
    if fp is None:
      fp = sm.featurizer.transform(_SAMPLE_SMILES).astype(np.float32)
    raw = np.asarray(sm.predict(X=fp, calibrate=False)).ravel()
    surrogate = (
      np.asarray(sm.calibrator.transform(raw)).ravel()
      if sm.calibrator is not None
      else raw.astype(np.float64)
    )
    if not entry["has_hard"]:
      ref[entry["output"]] = surrogate
      continue

    from lazyqsar.base.xgboost import BaseXGBArtifact

    from olinda.applicability import ApplicabilityClassifier
    from olinda.calibrate import IsotonicCalibrator

    gt_root = entry["dir"] / GT_DIRNAME
    g = np.asarray(BaseXGBArtifact.load(str(gt_root / GT_MODEL_SUBDIR)).run(fp))[:, 1].astype(np.float64)
    gsoft = np.asarray(IsotonicCalibrator.load(gt_root / CALIBRATOR_NAME).transform(g)).ravel()
    a = np.asarray(ApplicabilityClassifier.load(gt_root / APPLICABILITY_NAME).weight(fp > 0)).ravel()
    ref[entry["output"]] = (1.0 - a) * surrogate + a * gsoft

  sess = ort.InferenceSession((model_dir / MODEL_NAME).read_bytes(), providers=["CPUExecutionProvider"])
  got = {o.name: np.asarray(v).ravel() for o, v in zip(sess.get_outputs(), sess.run(None, {"input": fp}))}
  parity = {k: float(np.max(np.abs(got[k] - ref[k]))) for k in outputs}
  worst = max(parity.values())
  if worst > _PARITY_TOL:
    offender = max(parity, key=parity.get)
    raise RuntimeError(
      f"model.onnx parity failed on column {offender!r}: max abs diff {worst:.2e} > {_PARITY_TOL:.0e}"
    )

  n_hard = sum(1 for e in plan if e["has_hard"])
  size_mb = (model_dir / MODEL_NAME).stat().st_size / 1e6
  summary_panel(
    "olinda · export",
    [
      ("Columns", f"[bold]{len(plan)}[/] · {n_hard} with a hard head"),
      ("Outputs", " · ".join(outputs)),
      ("Parity (max)", f"[bold]{worst:.2e}[/] ≤ {_PARITY_TOL:.0e}"),
      ("Size", f"[bold]{size_mb:.1f} MB[/]"),
      ("Saved", f"[dim]{cpath(model_dir / MODEL_NAME)}[/]"),
    ],
    border_style="green",
    icon="✓",
  )
  success(f"fused model.onnx built and parity-checked → [dim]{model_dir / MODEL_NAME}[/]")
  return {"model": str(model_dir / MODEL_NAME), "columns": outputs, "parity": parity}
