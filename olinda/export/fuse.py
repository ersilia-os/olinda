"""Stitch every column's stages into ONE graph, with one output per column.

The graph's input is qmfp bytes, Morgan counts packed 4 bits per slot by olinda's featurizer; a decoder at the
front turns them into the float fingerprint every head reads. ``S``, ``H_S`` and the weight ``a`` stay
internal tensors: they are how the answer is computed, not part of the answer. Each column's nodes and
tensors are namespaced by its id so ten columns can share a graph without colliding.
"""

from __future__ import annotations

import json
from pathlib import Path

from olinda.export.decoder import DECODED, qmfp_decoder_nodes
from olinda.export.gate import _tanimoto_model
from olinda.export.graph import _OPSET, _toposort
from olinda.export.heads import (
    _blender_model,
    _prob1_model,
    _soft_model_proto,
    prime_soft_conversions,
)
from olinda.export.isotonic import _isotonic_model
from olinda.export.metadata import PRODUCER_NAME, SCHEMA_VERSION, _bundle_metadata
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


def _collect(nodes: list, inits: list, opset: dict, model, prefix: str) -> None:
    """Absorb one stage's nodes and initializers, namespaced by *prefix* so columns cannot collide."""
    from onnx.compose import add_prefix

    m = add_prefix(model, prefix + "__")
    nodes.extend(m.graph.node)
    inits.extend(m.graph.initializer)
    for op in m.opset_import:
        opset[op.domain] = max(opset.get(op.domain, 0), op.version)


def _ident(src: str, dst: str):
    """A pass-through edge, which is how one stage's output becomes the next one's input."""
    from onnx import helper

    return helper.make_node("Identity", [src], [dst], name=f"br_{dst}")


def _cast_d(src: str, dst: str):
    """Widen to float64: the calibration and blend stages all compute in double."""
    from onnx import TensorProto, helper

    return helper.make_node(
        "Cast", [src], [dst], to=TensorProto.DOUBLE, name=f"br_{dst}"
    )


def _flat(src: str, dst: str):
    """Reshape a tree's (N,1) output to (N,), so it lines up with the downstream stages."""
    from onnx import helper

    return helper.make_node("Reshape", [src, "flat_shape"], [dst], name=f"fl_{dst}")


def _wire_soft_branch(
    nodes: list, inits: list, opset: dict, sm, entry: dict, x_dim: int, p: str
) -> None:
    """Wire fingerprint → soft_model → [soft correction] → surrogate (double) for one column.

    Ends at ``{p}s``, the surrogate ``S`` that the hard branch, if any, blends against.
    """
    _collect(nodes, inits, opset, _soft_model_proto(sm, x_dim, entry["dir"]), f"{p}sm")
    nodes.append(_ident(DECODED, f"{p}sm__input"))
    nodes.append(_flat(f"{p}sm__variable", f"{p}soft_raw"))
    soft_raw = f"{p}soft_raw"
    if (
        sm.calibrator is not None
    ):  # fitted during learn-soft and loaded from calibrator.json
        _collect(
            nodes, inits, opset, _isotonic_model(sm.calibrator, "in", "out"), f"{p}sc"
        )
        nodes.append(_ident(soft_raw, f"{p}sc__in"))
        surrogate_src = f"{p}sc__out"
    else:
        surrogate_src = soft_raw
    nodes.append(_cast_d(surrogate_src, f"{p}s"))


def _wire_hard_branch(
    nodes: list, inits: list, opset: dict, entry: dict, n_features: int, p: str
) -> None:
    """Wire ``H`` → ``H_S``, the Tanimoto gate ``a`` and the blend into ``{p}prediction``.

    Consumes the surrogate ``{p}s`` laid down by :func:`_wire_soft_branch`.
    """
    import onnx

    from olinda.calibrate import IsotonicCalibrator
    from olinda.tanimoto import TanimotoRegressor

    hard_root = entry["dir"] / HARD_DIRNAME
    with open(hard_root / HARD_META_NAME) as fp:
        hard_task = json.load(fp).get("task", "binary")

    _collect(
        nodes,
        inits,
        opset,
        onnx.load(str(hard_root / HARD_MODEL_SUBDIR / "xgboost.onnx")),
        f"{p}hm",
    )
    nodes.append(_ident(DECODED, f"{p}hm__float_input"))
    if (
        hard_task == "regression"
    ):  # seam: an H regressor exposes a single "variable" output
        nodes.append(_flat(f"{p}hm__variable", f"{p}g_reg"))
        g_src = f"{p}g_reg"
    else:  # classifier: take probabilities[:, 1]
        _collect(nodes, inits, opset, _prob1_model("p", "g"), f"{p}pr")
        nodes.append(_ident(f"{p}hm__probabilities", f"{p}pr__p"))
        g_src = f"{p}pr__g"

    gcal = IsotonicCalibrator.load(hard_root / H_TO_S_NAME)
    _collect(nodes, inits, opset, _isotonic_model(gcal, "in", "out"), f"{p}hc")
    nodes.append(_ident(g_src, f"{p}hc__in"))
    nodes.append(_ident(f"{p}hc__out", f"{p}h_s"))

    clf = TanimotoRegressor.load(hard_root / TANIMOTO_DIRNAME)
    _collect(
        nodes, inits, opset, _tanimoto_model(clf, n_features, "input", "a"), f"{p}t"
    )
    nodes.append(_ident(DECODED, f"{p}t__input"))
    nodes.append(_ident(f"{p}t__a", f"{p}a"))

    _collect(nodes, inits, opset, _blender_model(), f"{p}bl")
    nodes.append(_ident(f"{p}s", f"{p}bl__soft"))
    nodes.append(_ident(f"{p}h_s", f"{p}bl__hard"))
    nodes.append(_ident(f"{p}a", f"{p}bl__a"))
    nodes.append(_ident(f"{p}bl__prediction", f"{p}prediction"))


def _load_students(model_dir: Path, plan: list):
    """Load every column's student and return ``(students, validated fingerprint spec)``.

    One graph, one input, so every column must have been trained on the same fingerprint.
    """
    from olinda.featurizer import validate_spec
    from olinda.models.bundle import StudentModel

    students = []
    for entry in plan:
        sm = StudentModel.load(entry["dir"])
        _assert_model_belongs_to(sm, entry)
        students.append(sm)
    specs = {
        json.dumps(sm.metadata.get("fingerprint"), sort_keys=True) for sm in students
    }
    if len(specs) != 1:
        raise ValueError(
            f"{model_dir}: columns were trained on different fingerprints; re-run `olinda prepare`"
        )
    return students, validate_spec(students[0].metadata["fingerprint"])


def _assemble_model(
    nodes: list, inits: list, opset: dict, outputs: list, spec: dict, manifest, plan
):
    """Close the graph over its single input and one output per column, and stamp the metadata."""
    from onnx import TensorProto, helper

    from olinda.featurizer import n_input_bytes

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
                "input", TensorProto.UINT8, ["B", n_input_bytes(spec)]
            )
        ],
        out_vi,
        initializer=inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid(d, v) for d, v in opset.items()]
    )
    md = _bundle_metadata(manifest, plan)

    # Standard ONNX provenance, so the file identifies itself to any tool (Netron, hub tooling, the
    # onnx CLI) without them having to know about the custom metadata key below.
    model.producer_name = PRODUCER_NAME
    model.producer_version = str(md.get("olinda_version", ""))
    model.domain = "io.ersilia.olinda"
    model.doc_string = (
        f"Distilled model produced by olinda {md.get('olinda_version')} on {md.get('trained_at')}. "
        f"Input: uint8 ({n_input_bytes(spec)},) qmfp bytes: Morgan radius "
        f"{spec['morgan']['radius']} counts clipped at 15, two 4-bit slots per byte. "
        f"Outputs: {', '.join(outputs)}."
    )
    # json_safe first: metrics can carry NaN, which strict JSON parsers in other languages reject. The
    # golden vectors are added by build_bundle, which needs the finished graph to compute them.
    set_props(
        model,
        {
            "olinda.schema_version": SCHEMA_VERSION,
            "olinda.fingerprint": json.dumps(spec),
            "olinda.model": json.dumps(json_safe(md)),
        },
    )
    return model


def _fuse(model_dir: Path):
    """Assemble the fused ``model.onnx`` ModelProto: every column, sharing one input tensor.

    Each column contributes an independent sub-pipeline under its own ``c{i}_`` prefix, so a run with
    one column is simply the one-column case of a run with many. The graph exposes exactly one output
    per column — its blended prediction, named after the column.
    """
    from onnx import TensorProto, helper

    manifest, plan = _column_plan(model_dir)

    nodes: list = []
    inits: list = []
    opset: dict = {"": _OPSET}

    # Tree ONNX outputs are (N,1); flatten to (N,) so they line up with the (N,) calibration/gate stages.
    # Added once for the whole graph — a per-column copy would collide on name and fail the checker.
    inits.append(helper.make_tensor("flat_shape", TensorProto.INT64, [1], [-1]))

    students, spec = _load_students(model_dir, plan)
    n_features = int(spec["slots"])

    # The decoder sits in front of everything: uint8 bytes in, the float fingerprint (DECODED) out.
    dec_nodes, dec_inits = qmfp_decoder_nodes(spec, in_name="input")
    nodes.extend(dec_nodes)
    inits.extend(dec_inits)
    outputs: list[str] = []

    prime_soft_conversions(
        [
            (sm, entry["dir"], int(sm.metadata.get("x_dim") or n_features))
            for entry, sm in zip(plan, students)
        ]
    )
    for entry, sm in zip(plan, students):
        p = (
            entry["id"] + "_"
        )  # every node and tensor of this column is namespaced by it
        x_dim = int(sm.metadata.get("x_dim") or n_features)

        _wire_soft_branch(nodes, inits, opset, sm, entry, x_dim, p)
        if entry["has_hard"]:
            _wire_hard_branch(nodes, inits, opset, entry, n_features, p)
        else:
            nodes.append(_ident(f"{p}s", f"{p}prediction"))

        nodes.append(_ident(f"{p}prediction", entry["output"]))
        outputs.append(entry["output"])

    model = _assemble_model(nodes, inits, opset, outputs, spec, manifest, plan)
    return model, plan, outputs


def set_props(model, props: dict) -> None:
    """Set ``metadata_props`` entries, replacing any with the same key."""
    keep = [p for p in model.metadata_props if p.key not in props]
    del model.metadata_props[:]
    model.metadata_props.extend(keep)
    for key, value in props.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
