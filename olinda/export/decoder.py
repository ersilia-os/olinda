"""The qmfp decoder at the front of every fused graph: packed bytes in, float slot values out.

The model's input is the packed 4-bit Morgan counts olinda's WebAssembly RDKit returns, so a runtime
passes those bytes straight to onnxruntime and holds no fingerprint logic of its own. One ``Gather``
on a 256-row table decodes both slots of every byte at once; unpacking with BitShift/Mod was about
4x slower under onnxruntime-web.
"""

from __future__ import annotations

import numpy as np

from olinda.featurizer.codec import byte_table

DECODED = "qmfp_x"
"""The decoded ``float32 (N, slots)`` tensor every head reads instead of the raw input."""


def qmfp_decoder_nodes(spec: dict, in_name: str = "input"):
    """``(nodes, initializers)`` turning ``uint8 (N, slots/2)`` into ``float32 (N, slots)`` named :data:`DECODED`.

    Names carry a ``qmfp_`` prefix, never a column's ``c<n>_``, so tools that parse per-column
    prefixes out of the graph keep working. The reshape copies the batch dimension (``0``) rather
    than inferring it (``-1``), which stays unambiguous for an empty batch.
    """
    from onnx import TensorProto, helper, numpy_helper

    slots = int(spec["slots"])
    inits = [
        numpy_helper.from_array(byte_table(), "qmfp_table"),
        numpy_helper.from_array(np.array([0, slots], dtype=np.int64), "qmfp_shape"),
    ]
    nodes = [
        helper.make_node(
            "Cast", [in_name], ["qmfp_idx"], to=TensorProto.INT32, name="qmfp_cast"
        ),
        helper.make_node(
            "Gather",
            ["qmfp_table", "qmfp_idx"],
            ["qmfp_pairs"],
            axis=0,
            name="qmfp_gather",
        ),
        helper.make_node(
            "Reshape", ["qmfp_pairs", "qmfp_shape"], [DECODED], name="qmfp_reshape"
        ),
    ]
    return nodes, inits
