"""The isotonic calibrator as an ONNX graph, thinned to a manageable number of knots.

Both the surrogate's own correction and the H -> S map are isotonic, so both come through here.
"""

from __future__ import annotations

import numpy as np

from olinda.export.graph import _OPSET

_ISOTONIC_TOL = 1e-5  # knot-thinning target — kept below _PARITY_TOL so ONNX float error has headroom
_ISOTONIC_MAX_KNOTS = 4096


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
        chord = (
            y[i] + (y[j] - y[i]) * ((xs - x[i]) / dx)
            if dx != 0
            else np.full(len(xs), y[i])
        )
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
    # The knot index is count(xk <= t) - 1. Counting against all n knots is a B x n comparison per batch
    # (4,096 knots: a 134 MB int64 temporary and ~9 ms per 4,096 rows). Instead count the knots in two
    # steps: against every k-th knot to find t's group, then within that one group. Every group before it
    # lies wholly at or below t and every group after it wholly above, so the count is exact, with
    # B x (n/k + k) comparisons. Padding the last group with +inf keeps the table rectangular.
    k = max(1, int(np.ceil(np.sqrt(n))))
    groups = int(np.ceil(n / k))
    table = np.full(groups * k, np.inf)
    table[:n] = xk
    ct = helper.make_tensor
    init = [
        ct("sign", TensorProto.DOUBLE, [1], [float(cal._sign)]),
        ct("xk", TensorProto.DOUBLE, [n], xk),
        ct("yk", TensorProto.DOUBLE, [n], yk),
        ct("slope", TensorProto.DOUBLE, [n], slope),
        ct("x0", TensorProto.DOUBLE, [1], [float(xk[0])]),
        ct("xlast", TensorProto.DOUBLE, [1], [float(xk[-1])]),
        ct("coarse", TensorProto.DOUBLE, [groups], xk[::k]),
        ct("fine", TensorProto.DOUBLE, [groups, k], table),
        ct("k_i", TensorProto.INT64, [1], [k]),
        ct("gm1", TensorProto.INT64, [1], [groups - 1]),
        ct("one_i", TensorProto.INT64, [1], [1]),
        ct("nm1", TensorProto.INT64, [1], [n - 1]),
        ct("zero_i", TensorProto.INT64, [1], [0]),
        ct("axis1", TensorProto.INT64, [1], [1]),
    ]
    nodes = [
        helper.make_node("Cast", [in_name], ["rawd"], to=TensorProto.DOUBLE),
        helper.make_node("Mul", ["rawd", "sign"], ["t_raw"]),
        helper.make_node(
            "Max", ["t_raw", "x0"], ["t_lo"]
        ),  # clamp into [x0, xlast] → out-of-range holds ends
        helper.make_node("Min", ["t_lo", "xlast"], ["t"]),
        helper.make_node("Unsqueeze", ["t", "axis1"], ["t2"]),
        # step 1: which group of k knots t falls in
        helper.make_node("LessOrEqual", ["coarse", "t2"], ["le_c"]),
        helper.make_node("Cast", ["le_c"], ["lei_c"], to=TensorProto.INT64),
        helper.make_node("ReduceSum", ["lei_c", "axis1"], ["cnt_c"], keepdims=0),
        helper.make_node("Sub", ["cnt_c", "one_i"], ["grp0"]),
        helper.make_node("Max", ["grp0", "zero_i"], ["grp1"]),
        helper.make_node("Min", ["grp1", "gm1"], ["grp"]),
        # step 2: how many knots of that group lie at or below t
        helper.make_node("Gather", ["fine", "grp"], ["window"], axis=0),
        helper.make_node("LessOrEqual", ["window", "t2"], ["le_f"]),
        helper.make_node("Cast", ["le_f"], ["lei_f"], to=TensorProto.INT64),
        helper.make_node("ReduceSum", ["lei_f", "axis1"], ["cnt_f"], keepdims=0),
        helper.make_node("Mul", ["grp", "k_i"], ["base"]),
        helper.make_node("Add", ["base", "cnt_f"], ["cnt"]),
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
