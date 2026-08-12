"""Stitch every column's stages into ONE graph, with one output per column.

``S``, ``H_S`` and the weight ``a`` stay internal tensors here: they are how the answer is computed,
not part of the answer. Each column's nodes and tensors are namespaced by its id so ten columns can
share a graph without colliding.
"""

from __future__ import annotations

import json
from pathlib import Path

from olinda.export.gate import _tanimoto_model
from olinda.export.graph import _OPSET, _toposort
from olinda.export.heads import _blender_model, _prob1_model, _soft_model_proto
from olinda.export.isotonic import _isotonic_model
from olinda.export.metadata import PRODUCER_NAME, _bundle_metadata
from olinda.export.parity import _assert_model_belongs_to
from olinda.hard.layout import (
    H_TO_S_NAME,
    HARD_DIRNAME,
    HARD_META_NAME,
    HARD_MODEL_SUBDIR,
    TANIMOTO_DIRNAME,
    has_hard_head,
)
from olinda.metrics import json_safe


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
        plan.append(
            {
                "id": col["id"],
                "name": col["name"],
                "output": output,
                "dir": col_dir,
                "has_hard": has_hard_head(col_dir),
                "hard_meta": col.get("hard"),
                "metrics": col.get("metrics"),
                "training": {
                    "n_finite": col.get("n_finite"),
                    "n_train": col.get("n_train"),
                    "n_val": col.get("n_val"),
                    "value_range": col.get("value_range"),
                },
            }
        )
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

    from olinda.calibrate import IsotonicCalibrator
    from olinda.featurizer import featurizer_from_meta
    from olinda.models.bundle import StudentModel
    from olinda.tanimoto import TanimotoRegressor

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

    def ident(src: str, dst: str):
        return helper.make_node("Identity", [src], [dst], name=f"br_{dst}")

    def cast_d(src: str, dst: str):
        return helper.make_node(
            "Cast", [src], [dst], to=TensorProto.DOUBLE, name=f"br_{dst}"
        )

    # Tree ONNX outputs are (N,1); flatten to (N,) so they line up with the (N,) calibration/gate stages.
    # Added once for the whole graph — a per-column copy would collide on name and fail the checker.
    inits.append(helper.make_tensor("flat_shape", TensorProto.INT64, [1], [-1]))

    def flat(src: str, dst: str):
        return helper.make_node("Reshape", [src, "flat_shape"], [dst], name=f"fl_{dst}")

    featurizer: dict = {}
    featurizer_class = "MorganCountFeaturizer"
    n_features = 2048
    outputs: list[str] = []

    for entry in plan:
        p = (
            entry["id"] + "_"
        )  # every node and tensor of this column is namespaced by it
        sm = StudentModel.load(entry["dir"], featurizer_factory=featurizer_from_meta)
        _assert_model_belongs_to(sm, entry)
        featurizer = sm.metadata.get("featurizer") or featurizer
        featurizer_class = sm.metadata.get("featurizer_class", featurizer_class)
        n_features = int(
            featurizer.get("fp_size", sm.metadata.get("x_dim", n_features))
        )
        x_dim = int(sm.metadata.get("x_dim") or n_features)

        # soft branch: input → soft_model → [soft correction] → surrogate (double)
        collect(_soft_model_proto(sm, x_dim), f"{p}sm")
        nodes.append(ident("input", f"{p}sm__input"))
        nodes.append(flat(f"{p}sm__variable", f"{p}soft_raw"))
        soft_raw = f"{p}soft_raw"
        if (
            sm.calibrator is not None
        ):  # fitted during learn-soft and loaded from calibrator.json
            collect(_isotonic_model(sm.calibrator, "in", "out"), f"{p}sc")
            nodes.append(ident(soft_raw, f"{p}sc__in"))
            surrogate_src = f"{p}sc__out"
        else:
            surrogate_src = soft_raw
        nodes.append(cast_d(surrogate_src, f"{p}s"))

        if entry["has_hard"]:
            hard_root = entry["dir"] / HARD_DIRNAME
            with open(hard_root / HARD_META_NAME) as fp:
                hard_task = json.load(fp).get("task", "binary")

            collect(
                onnx.load(str(hard_root / HARD_MODEL_SUBDIR / "xgboost.onnx")), f"{p}hm"
            )
            nodes.append(ident("input", f"{p}hm__float_input"))
            if (
                hard_task == "regression"
            ):  # seam: an H regressor exposes a single "variable" output
                nodes.append(flat(f"{p}hm__variable", f"{p}g_reg"))
                g_src = f"{p}g_reg"
            else:  # classifier: take probabilities[:, 1]
                collect(_prob1_model("p", "g"), f"{p}pr")
                nodes.append(ident(f"{p}hm__probabilities", f"{p}pr__p"))
                g_src = f"{p}pr__g"

            gcal = IsotonicCalibrator.load(hard_root / H_TO_S_NAME)
            collect(_isotonic_model(gcal, "in", "out"), f"{p}hc")
            nodes.append(ident(g_src, f"{p}hc__in"))
            nodes.append(ident(f"{p}hc__out", f"{p}h_s"))

            clf = TanimotoRegressor.load(hard_root / TANIMOTO_DIRNAME)
            collect(_tanimoto_model(clf, n_features, "input", "a"), f"{p}t")
            nodes.append(ident("input", f"{p}t__input"))
            nodes.append(ident(f"{p}t__a", f"{p}a"))

            collect(_blender_model(), f"{p}bl")
            nodes.append(ident(f"{p}s", f"{p}bl__soft"))
            nodes.append(ident(f"{p}h_s", f"{p}bl__hard"))
            nodes.append(ident(f"{p}a", f"{p}bl__a"))
            nodes.append(ident(f"{p}bl__prediction", f"{p}prediction"))
        else:
            nodes.append(ident(f"{p}s", f"{p}prediction"))

        nodes.append(ident(f"{p}prediction", entry["output"]))
        outputs.append(entry["output"])

    # One output per column and nothing else. ``S``, ``H_S`` and ``a`` remain as internal tensors — the
    # blender consumes them — but they are working, not product: declaring them would invite callers to
    # depend on a wiring we need to stay free to change.
    out_vi = [
        helper.make_tensor_value_info(n, TensorProto.DOUBLE, ["B"]) for n in outputs
    ]
    graph = helper.make_graph(
        _toposort(nodes, {"input"} | {i.name for i in inits}),
        "olinda_model",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, ["B", int(n_features)]
            )
        ],
        out_vi,
        initializer=inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid(d, v) for d, v in opset.items()]
    )
    md = _bundle_metadata(manifest, plan, featurizer, featurizer_class)

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
