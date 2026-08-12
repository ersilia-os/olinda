"""T and its ramp as an ONNX graph: two matrix multiplies standing in for a similarity search."""

from __future__ import annotations

from olinda.export.graph import _IR_VERSION, _OPSET, _rename_input


def _tanimoto_model(clf, n_features: int, in_name: str, out_name: str):
    """ONNX ``ModelProto``: fp(float32) → blend weight ``a`` (double).

    Three parts spliced into one graph: binarise the incoming count fingerprint, run the gate network,
    ramp its output into a weight.

    The binarisation is not optional. The net is trained and evaluated on ``bits > 0``
    (:meth:`TanimotoRegressor.predict_tanimoto`), while the fused graph carries the shared *count*
    fingerprint, so without it the graph would feed counts to a net that has only ever seen indicators
    and quietly disagree with its own Python reference on any molecule with a repeated substructure.

    The ramp clips to [0, 1] before scaling: a net can extrapolate past the target's range, and a
    similarity above 1 would otherwise push ``a`` past its ceiling.
    """
    import onnx
    from onnx import TensorProto, helper

    net = onnx.load_from_string(clf.onnx_bytes)

    # Rewire the net's own input so the binarisation can sit in front of it.
    inner = "t_bits"
    _rename_input(net, inner)
    del net.graph.input[:]
    net.graph.input.append(
        helper.make_tensor_value_info(
            in_name, TensorProto.FLOAT, ["B", int(n_features)]
        )
    )

    net_out = net.graph.output[0].name
    span = max(float(clf.sim_hi) - float(clf.sim_lo), 1e-9)
    ct = helper.make_tensor
    net.graph.initializer.extend(
        [
            ct("t_thr", TensorProto.FLOAT, [1], [0.0]),
            ct("t_zero", TensorProto.DOUBLE, [1], [0.0]),
            ct("t_one", TensorProto.DOUBLE, [1], [1.0]),
            ct("t_lo", TensorProto.DOUBLE, [1], [float(clf.sim_lo)]),
            ct("t_span", TensorProto.DOUBLE, [1], [span]),
            ct("t_max", TensorProto.DOUBLE, [1], [float(clf.a_max)]),
            ct("t_flat", TensorProto.INT64, [1], [-1]),
        ]
    )
    head = [
        helper.make_node("Greater", [in_name, "t_thr"], ["t_on"]),
        helper.make_node("Cast", ["t_on"], [inner], to=TensorProto.FLOAT),
    ]
    tail = [
        helper.make_node("Cast", [net_out], ["t_simd"], to=TensorProto.DOUBLE),
        helper.make_node("Reshape", ["t_simd", "t_flat"], ["t_sim1"]),
        helper.make_node("Clip", ["t_sim1", "t_zero", "t_one"], ["t_sim"]),
        helper.make_node("Sub", ["t_sim", "t_lo"], ["t_shift"]),
        helper.make_node("Div", ["t_shift", "t_span"], ["t_frac0"]),
        helper.make_node("Clip", ["t_frac0", "t_zero", "t_one"], ["t_frac"]),
        helper.make_node("Mul", ["t_frac", "t_max"], [out_name]),
    ]
    existing = list(net.graph.node)
    del net.graph.node[:]
    net.graph.node.extend(head + existing + tail)

    del net.graph.output[:]
    net.graph.output.append(
        helper.make_tensor_value_info(out_name, TensorProto.DOUBLE, ["B"])
    )
    # The gate pins its own opset when it serialises itself; re-stamp it to the bundle's. Every op used
    # here (MatMul, Relu, Add, Cast, Reshape, Clip, Greater) exists at _OPSET, and unlike the boosted
    # stages there is no ai.onnx.ml domain to preserve, so replacing the whole list is safe.
    del net.opset_import[:]
    net.opset_import.append(helper.make_opsetid("", _OPSET))
    net.ir_version = _IR_VERSION
    return net
