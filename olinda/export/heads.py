"""The small structural graphs: the blender, the probability picker, and the soft model's proto."""

from __future__ import annotations

import tempfile
from pathlib import Path

from olinda.export.graph import _OPSET


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
        initializer=[
            helper.make_tensor("one_idx", TensorProto.INT64, [], [1])
        ],  # scalar → drops axis 1
    )
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", _OPSET)])


SOFT_ONNX_NAME = "s.onnx"


def _soft_model_proto(sm, x_dim: int, column_dir=None):
    """The surrogate regressor as an ONNX ``ModelProto``.

    Converting a booster walks every node in Python, roughly 15 s or more per column, and a run fuses
    more than once (`learn-soft`, then `learn-hard`, then any `export`). So the conversion is kept as
    ``s.onnx`` in *column_dir* and reused for as long as it is newer than the model file it came from.
    """
    import onnx

    from olinda.train.backend import get_backend

    backend = get_backend(sm.backend, "cpu")
    if column_dir is not None:
        cached = Path(column_dir) / SOFT_ONNX_NAME
        source = Path(column_dir) / backend.model_file
        if (
            cached.exists()
            and source.exists()
            and cached.stat().st_mtime >= source.stat().st_mtime
        ):
            return onnx.load(str(cached))
        backend.to_onnx(sm.model, cached, x_dim)
        return onnx.load(str(cached))

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / SOFT_ONNX_NAME
        backend.to_onnx(sm.model, p, x_dim)
        return onnx.load(str(p))
