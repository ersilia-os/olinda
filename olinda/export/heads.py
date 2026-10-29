"""The small structural graphs: the blender, the probability picker, and the soft model's proto."""

from __future__ import annotations

import os
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
"""Most boosters converted at once, however much memory there is."""

_CONVERT_BYTES_PER_FILE_BYTE = 25
_CONVERT_BASE_BYTES = 400_000_000
"""What one conversion peaks at, measured on LightGBM boosters of 300 to 1,200 trees: about 0.53 MB per
tree on top of ~200 MB, which is ~22x the booster file (~25 KB a tree). The factor is rounded up, so a
10,000-tree column (~260 MB on disk) is budgeted ~7 GB. The same peak is paid by a serial conversion; running
several side by side multiplies it, which is what :func:`_convert_workers` keeps inside the free memory."""

_MEMORY_SHARE = 0.7
"""Share of the memory that is free that conversions may plan to use, leaving the rest to the parent."""

_PARTIAL_SUFFIX = ".partial"


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


def _available_memory() -> int | None:
    """Bytes of memory free right now, or ``None`` when this platform cannot say."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None


def _convert_workers(n_jobs: int, largest_bytes: int, available: int | None) -> int:
    """How many boosters may convert at once: bounded by cores, the cap, and the memory they would need.

    One means "serial". Unknown memory is treated as scarce rather than plentiful, because the failure it
    guards against, the OS killing a worker, costs the whole conversion.
    """
    by_cpu = min(n_jobs, os.cpu_count() or 1, _MAX_CONVERT_WORKERS)
    if available is None:
        return min(by_cpu, 2)
    per_worker = largest_bytes * _CONVERT_BYTES_PER_FILE_BYTE + _CONVERT_BASE_BYTES
    return max(1, min(by_cpu, int(available * _MEMORY_SHARE // per_worker)))


def _convert_to(backend, model, x_dim: int, final: Path) -> None:
    """Convert *model* so that *final* only ever appears whole.

    The converter writes in place, and a process killed mid-write (an out-of-memory kill, Ctrl-C) would
    leave a truncated file that the newer-than-the-booster test would then trust. So it writes beside
    *final* and renames, which is atomic on one filesystem.
    """
    partial = final.with_name(f"{final.name}.{os.getpid()}{_PARTIAL_SUFFIX}")
    try:
        backend.to_onnx(model, partial, x_dim)
        os.replace(partial, final)
    finally:
        partial.unlink(missing_ok=True)


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
        if cached is not None:
            try:
                return onnx.load(str(cached))
            except Exception:  # noqa: BLE001
                cached.unlink(missing_ok=True)  # unreadable, so convert it again below
        cached = Path(column_dir) / SOFT_ONNX_NAME
        _convert_to(backend, sm.model, x_dim, cached)
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
    largest = 0
    for sm, column_dir, x_dim in jobs:
        backend = get_backend(sm.backend, "cpu")
        source = Path(column_dir) / backend.model_file
        for stray in Path(column_dir).glob(f"*{_PARTIAL_SUFFIX}"):
            stray.unlink(missing_ok=True)  # left by a conversion that was killed
        if _cached_soft_path(backend, column_dir) is not None or not source.exists():
            continue
        if source.stat().st_size >= _PARALLEL_MIN_BYTES:
            todo.append((str(column_dir), x_dim))
            largest = max(largest, source.stat().st_size)
    if len(todo) < 2:
        return

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from olinda.console import echo

    workers = _convert_workers(len(todo), largest, _available_memory())
    if workers < 2:
        return  # not enough free memory to hold two conversions; the fuse converts one at a time
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
        for column_dir, _ in todo:  # a killed worker may have left half a file
            for stray in Path(column_dir).glob(f"*{_PARTIAL_SUFFIX}"):
                stray.unlink(missing_ok=True)
