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
        helper.make_node(
            "Max", ["t_raw", "x0"], ["t_lo"]
        ),  # clamp into [x0, xlast] → out-of-range holds ends
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
