"""What a ``model.onnx`` can say about itself, read back out of the graph.

The fuse writes each isotonic map into the graph as plain initializers — the anchor points it
interpolates between — so the calibration curves a run learned are recoverable from the artifact
alone, with no run directory and no training data. Same for the tree counts.

Names follow the per-column prefixes ``_fuse`` assigns (:mod:`olinda.export`): ``c0_sc__xk`` is the
surrogate correction for column ``c0``, ``c0_hc__xk`` the ground-truth head's map onto the teacher's
scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SOFT_STAGE = "sc"  # surrogate correction: raw student output → teacher scale
HARD_STAGE = "hc"  # ground truth: G's probability → teacher scale


def _initializers(graph) -> dict:
  from onnx import numpy_helper

  return {i.name: numpy_helper.to_array(i) for i in graph.initializer}


def _curve(init: dict, prefix: str, stage: str) -> dict | None:
  """The anchor points of one isotonic stage, or ``None`` if the model has no such stage."""
  xk, yk = init.get(f"{prefix}{stage}__xk"), init.get(f"{prefix}{stage}__yk")
  if xk is None or yk is None:
    return None
  sign = float(np.asarray(init.get(f"{prefix}{stage}__sign", 1.0)).ravel()[0])
  # The graph stores the map over sign*raw, so undo the flip to get back to the model's own scale.
  x = np.asarray(xk, dtype=np.float64) * sign
  y = np.asarray(yk, dtype=np.float64)
  order = np.argsort(x)
  return {"x": x[order], "y": y[order], "sign": sign, "n_anchors": int(len(x))}


def _tree_counts(graph) -> dict:
  """``{column_prefix: {"n_trees", "n_nodes"}}`` summed over that column's TreeEnsemble nodes."""
  out: dict[str, dict] = {}
  for node in graph.node:
    if not node.op_type.startswith("TreeEnsemble"):
      continue
    prefix = node.name.split("_", 1)[0] + "_" if "_" in node.name else ""
    treeids = next((a for a in node.attribute if a.name == "nodes_treeids"), None)
    if treeids is None:
      continue
    ids = np.asarray(treeids.ints)
    entry = out.setdefault(prefix, {"n_trees": 0, "n_nodes": 0})
    entry["n_trees"] += int(len(np.unique(ids)))
    entry["n_nodes"] += int(len(ids))
  return out


def describe_graph(model_onnx: str | Path) -> dict:
  """Per-column internals of a fused artifact: its calibration curves and tree sizes.

  Parameters
  ----------
  model_onnx : str or Path
      A fused ``model.onnx``.

  Returns
  -------
  dict
      ``{column_name: {"id", "soft_calibration", "hard_calibration", "n_trees", "n_nodes"}}``. Each
      calibration is ``{"x", "y", "sign", "n_anchors"}`` or ``None`` when that stage is absent — a
      column with too small a validation split has no surrogate correction, and a soft-only column
      has no ground-truth map.
  """
  import json

  import onnx

  model = onnx.load(str(model_onnx), load_external_data=False)
  raw = next((p.value for p in model.metadata_props if p.key == "olinda"), None)
  if not raw:
    raise ValueError(f"{model_onnx} carries no olinda metadata — is it an olinda artifact?")
  meta = json.loads(raw)

  init = _initializers(model.graph)
  trees = _tree_counts(model.graph)

  out = {}
  for col in meta.get("columns", []):
    prefix = f"{col['id']}_"
    out[col["name"]] = {
      "id": col["id"],
      "soft_calibration": _curve(init, prefix, SOFT_STAGE),
      "hard_calibration": _curve(init, prefix, HARD_STAGE),
      **trees.get(prefix, {"n_trees": 0, "n_nodes": 0}),
    }
  return out
