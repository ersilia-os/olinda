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

_PARALLEL_MIN_BYTES = 5_000_000
"""A booster file at least this large is worth converting in its own process.

Conversion costs ~17 ms per tree, so a 10,000-tree column is nearly three minutes of one core, whereas a
test-sized model is milliseconds and would only pay for starting a process.
"""

_MAX_CONVERT_WORKERS = 4
"""Most boosters converted at once. Each worker holds one whole booster and its ONNX graph, so this is a
memory bound as much as a CPU one."""


def _cached_soft_path(backend, column_dir) -> Path | None:
    """The kept conversion of a column's booster, if it is newer than the booster it came from."""
    cached = Path(column_dir) / SOFT_ONNX_NAME
    source = Path(column_dir) / backend.model_file
    if (
        cached.exists()
        and source.exists()
        and cached.stat().st_mtime >= source.stat().st_mtime
    ):
        return cached
    return None


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
        cached = _cached_soft_path(backend, column_dir)
        if cached is None:
            cached = Path(column_dir) / SOFT_ONNX_NAME
            backend.to_onnx(sm.model, cached, x_dim)
        return onnx.load(str(cached))

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / SOFT_ONNX_NAME
        backend.to_onnx(sm.model, p, x_dim)
        return onnx.load(str(p))


def _convert_in_worker(column_dir: str, x_dim: int) -> str:
    """Convert one column's booster in this process, leaving ``s.onnx`` beside it (a pool worker's job)."""
    from olinda.models.bundle import StudentModel

    _soft_model_proto(StudentModel.load(Path(column_dir)), x_dim, column_dir)
    return column_dir


def prime_soft_conversions(jobs) -> None:
    """Convert the large boosters among *jobs* side by side, so the fuse finds each ``s.onnx`` already kept.

    *jobs* is ``[(student, column_dir, x_dim), ...]``. Conversion is single-core Python, and a run fuses
    after every column has trained, so with several large columns the other cores otherwise sit idle for
    minutes. Each worker loads its booster from disk instead of being sent it, and a failure here is not
    an error: the fuse converts whatever is still missing in-process, and reports the real fault.
    """
    from olinda.train.backend import get_backend

    todo = []
    for sm, column_dir, x_dim in jobs:
        backend = get_backend(sm.backend, "cpu")
        source = Path(column_dir) / backend.model_file
        if _cached_soft_path(backend, column_dir) is not None or not source.exists():
            continue
        if source.stat().st_size >= _PARALLEL_MIN_BYTES:
            todo.append((str(column_dir), x_dim))
    if len(todo) < 2:
        return

    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor

    from olinda.console import echo

    workers = min(len(todo), os.cpu_count() or 1, _MAX_CONVERT_WORKERS)
    echo(
        f"converting {len(todo)} boosters to ONNX · [dim]{workers} at a time[/]", "run"
    )
    try:
        # spawn, not fork: the boosters' OpenMP threads do not survive a fork
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            list(pool.map(_convert_in_worker, *zip(*todo)))
    except Exception as exc:  # noqa: BLE001
        echo(f"parallel conversion failed ({exc}); converting one by one", "warning")
