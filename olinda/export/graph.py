"""Generic ONNX plumbing: opset and IR pins, saving, renaming, topological order.

Nothing here knows what olinda's graphs mean. It is the layer every stage builder and the fuse both
sit on.
"""

from __future__ import annotations

from pathlib import Path

_IR_VERSION = 10  # onnxruntime in this env caps the model IR version at 10
_OPSET = 16
_PROTOBUF_LIMIT = 2 * 1024**3  # a single serialized ModelProto cannot exceed this


_BYTES_PER_TREE = 18_000
"""What one tree adds to the fused file: measured at 150 MB for a 9,970-tree column and 304 MB for three."""


def check_fused_size(n_columns: int, num_boost_round: int) -> None:
    """Refuse a run whose largest possible model would not fit in one ONNX file.

    The limit is on the *serialized* graph and is only met once every column has trained, which is hours in
    for a full-size run. A column grows to at most ``num_boost_round`` trees, so the worst case is known
    before anything trains.
    """
    worst = n_columns * num_boost_round * _BYTES_PER_TREE
    if worst >= _PROTOBUF_LIMIT:
        fits = int(_PROTOBUF_LIMIT // (n_columns * _BYTES_PER_TREE))
        raise ValueError(
            f"{n_columns} column(s) of up to {num_boost_round:,} trees would make a model of about "
            f"{worst / 1e9:.1f} GB, over ONNX's {_PROTOBUF_LIMIT / 1e9:.1f} GB limit. Use "
            f"--num-boost-round {fits - fits % 100} or less, or distil fewer columns at a time."
        )


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
            if (
                attr.name == "nodes_hitrates"
                and len(attr.floats)
                and np.all(np.asarray(attr.floats) == 1.0)
            ):
                saved += attr.ByteSize()
                continue
            keep.append(attr)
        if len(keep) != len(node.attribute):
            del node.attribute[:]
            node.attribute.extend(keep)
    return saved


def _finalise(model) -> None:
    """Stamp, trim and check a fused model in memory, so it can be run before it is written.

    build_bundle needs the finished graph to compute the golden vectors, which then go into the same
    file: checking here and writing once afterwards saves a second serialisation of a large model.
    """
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


def _write(model, path: Path) -> None:
    """Write a model :func:`_finalise` has already checked."""
    import onnx

    onnx.save(model, str(path))


def _rename_input(model, new_name: str) -> None:
    """Point a converted booster's graph at ``new_name`` so the fuse can wire it up like any other stage."""
    old = model.graph.input[0].name
    model.graph.input[0].name = new_name
    for node in model.graph.node:
        node.input[:] = [new_name if i == old else i for i in node.input]


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
        raise ValueError(
            f"fused graph has a dependency cycle or a missing producer near: {unresolved}"
        )
    return order
