"""Build the web page's example model: a real olinda model file whose "prediction" is a count.

The graph is olinda's own front end (the qmfp decoder) followed by a sum over the 2048 slot counts,
so its one output, ``count_sum``, is the total of a molecule's Morgan counts (each clipped at 15). It
learns nothing; it exists so the page can be tried without a model of your own.

Everything else is real: the ``olinda.*`` metadata, the fingerprint spec, and golden vectors computed
from this graph, so the page's golden check runs on it exactly as on a distilled model. Built in CI
(``.github/workflows/pages.yml``), never committed.

    python scripts/make_example_model.py OUT.onnx
"""

import json
import sys
from datetime import datetime, timezone

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from olinda.export.decoder import DECODED, qmfp_decoder_nodes
from olinda.export.fuse import set_props
from olinda.export.graph import _IR_VERSION, _OPSET
from olinda.export.metadata import PRODUCER_NAME, SCHEMA_VERSION
from olinda.featurizer import QMFP_SPEC, QmfpFeaturizer, n_input_bytes
from olinda.golden import compute_golden
from olinda.metrics import json_safe

OUTPUT = "count_sum"


def build() -> onnx.ModelProto:
    """The graph: uint8 bytes → decoded counts → their sum, as float64 like every olinda output."""
    nodes, inits = qmfp_decoder_nodes(QMFP_SPEC, in_name="input")
    inits.append(helper.make_tensor("sum_axes", TensorProto.INT64, [1], [1]))
    nodes += [
        helper.make_node("ReduceSum", [DECODED, "sum_axes"], ["sum_f"], keepdims=0),
        helper.make_node("Cast", ["sum_f"], [OUTPUT], to=TensorProto.DOUBLE),
    ]
    graph = helper.make_graph(
        nodes,
        "olinda_example",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.UINT8, ["B", n_input_bytes(QMFP_SPEC)]
            )
        ],
        [helper.make_tensor_value_info(OUTPUT, TensorProto.DOUBLE, ["B"])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    model.ir_version = _IR_VERSION
    model.producer_name = PRODUCER_NAME
    model.doc_string = (
        "olinda example model: the sum of a molecule's Morgan counts (radius 3, 2048 slots, each "
        "count clipped at 15). It learns nothing; it shows the web page and the model format."
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out = sys.argv[1]
    model = build()
    featurizer = QmfpFeaturizer(QMFP_SPEC)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    def run(packed):
        return {OUTPUT: session.run(None, {"input": packed})[0]}

    columns = [
        {
            "id": "c0",
            "name": OUTPUT,
            "output": OUTPUT,
            "task": {"type": "regression", "range": None, "units": "count"},
            "heads": [],
        }
    ]
    set_props(
        model,
        {
            "olinda.schema_version": SCHEMA_VERSION,
            "olinda.fingerprint": json.dumps(QMFP_SPEC),
            "olinda.rdkit_version": featurizer.rdkit_version,
            "olinda.golden": json.dumps(json_safe(compute_golden(run, featurizer))),
            "olinda.model": json.dumps(
                {
                    "producer": PRODUCER_NAME,
                    "example": True,
                    "trained_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "columns": columns,
                }
            ),
        },
    )
    onnx.save(model, out)

    # The example must load and score like any olinda model.
    from olinda import OlindaArtifact

    check = OlindaArtifact(out, strict=True)
    y = check.run(["CCO", "c1ccccc1"])
    expected = np.minimum(featurizer.transform(["CCO", "c1ccccc1"]), 15).sum(axis=1)
    assert np.array_equal(y[:, 0], expected), (y, expected)
    print(
        f"{out}: count_sum(CCO, c1ccccc1) = {y[:, 0].tolist()} · golden {check.describe()['golden']}"
    )


if __name__ == "__main__":
    main()
