"""The qmfp fingerprint: its byte layout, the decoder inside the graph, and agreement with RDKit.

The layout is the contract between olinda and every runtime that feeds a model, so it is tested
directly rather than only through a fitted model.
"""

from __future__ import annotations

import numpy as np
import pytest

from olinda.featurizer.codec import LEVEL_VALUES, byte_table, decode, pack
from olinda.featurizer.spec import QMFP_SPEC, default_spec, n_input_bytes, validate_spec

# Every byte MinimalLib can emit: two thermometer nibbles, each one of 0x0, 0x1, 0x3, 0x7, 0xF.
_NIBBLES = [0x0, 0x1, 0x3, 0x7, 0xF]
_VALID_BYTES = np.array(
    [[lo | (hi << 4) for hi in _NIBBLES for lo in _NIBBLES]], dtype=np.uint8
)


def test_every_byte_minimallib_can_emit_round_trips():
    """Decoding then packing gives back each of the 25 thermometer bytes unchanged."""
    assert np.array_equal(pack(decode(_VALID_BYTES)), _VALID_BYTES)


def test_slot_values_round_trip_and_the_even_slot_is_the_low_nibble():
    rng = np.random.default_rng(0)
    values = rng.choice(LEVEL_VALUES, size=(64, 2048)).astype(np.uint8)
    assert np.array_equal(decode(pack(values)), values)
    # 0x71: low nibble 0x1 → even slot level 1 → 1; high nibble 0x7 → odd slot level 3 → 4
    assert byte_table()[0x71].tolist() == [1.0, 4.0]


def test_pack_refuses_a_value_that_is_not_a_level():
    with pytest.raises(ValueError, match="0, 1, 2, 4 or 8"):
        pack(np.array([[3, 0]], dtype=np.uint8))


def test_the_spec_is_self_consistent_and_guards_its_layout():
    assert validate_spec(default_spec()) is not None
    assert n_input_bytes(QMFP_SPEC) == 1024
    for change in (
        {"useCountSimulation": False},
        {"nBits": 2048},
        {"levels": [0, 1, 2, 3, 4]},
    ):
        with pytest.raises(ValueError):
            validate_spec({**QMFP_SPEC, **change})
    with pytest.raises(ValueError, match="qmfp only"):
        validate_spec({**QMFP_SPEC, "type": "morgan"})


def test_the_graph_decoder_is_the_numpy_decoder():
    """Every byte value, a random batch, and an empty batch: the ONNX decoder must equal decode()."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    from olinda.export.decoder import DECODED, qmfp_decoder_nodes

    nb = n_input_bytes(QMFP_SPEC)
    nodes, inits = qmfp_decoder_nodes(QMFP_SPEC)
    graph = helper.make_graph(
        nodes,
        "decoder",
        [helper.make_tensor_value_info("input", TensorProto.UINT8, ["N", nb])],
        [helper.make_tensor_value_info(DECODED, TensorProto.FLOAT, ["N", 2 * nb])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    rng = np.random.default_rng(1)
    every_byte = np.tile(np.arange(256, dtype=np.uint8), nb // 256)[None, :]
    for x in (every_byte, rng.integers(0, 256, (50, nb), dtype=np.uint8)):
        assert np.array_equal(
            session.run(None, {"input": x})[0], decode(x).astype(np.float32)
        )
    empty = session.run(None, {"input": np.zeros((0, nb), np.uint8)})[0]
    assert empty.shape == (0, 2 * nb)


def test_quantized_counts_are_rdkit_count_simulation():
    """The claim the format rests on: count simulation is the true counts, clipped and quantized."""
    pytest.importorskip("rdkit")
    from tests.rdkit_oracle import RDKitOracle

    smiles = [
        "CCO",
        "c1ccccc1O",
        "C" * 40,
        "OCC(O)C(O)C(O)C(O)CO",
        "CC(=O)Oc1ccccc1C(=O)O",
    ]
    oracle = RDKitOracle(QMFP_SPEC)
    simulated, v1 = oracle.simulation_bytes(smiles)
    counted, v2 = oracle.count_bytes(smiles)
    assert v1.all() and np.array_equal(v1, v2)
    assert np.array_equal(simulated, counted)
    # the long chain reaches level 4 (a count of 8 or more) somewhere
    assert decode(simulated[2:3]).max() == 8


def test_the_webassembly_featurizer_is_rdkit_byte_for_byte():
    """olinda's featurizer is RDKit compiled to WebAssembly; it must equal full RDKit exactly.

    Over the golden set (which covers every way a featurizer can go wrong) and the test library. A
    difference means the WebAssembly build drifted from the RDKit release it claims, which would
    change every prediction while every other test still passed.
    """
    pytest.importorskip("rdkit")
    from olinda.featurizer import QmfpFeaturizer
    from olinda.golden import GOLDEN_SMILES
    from tests.conftest import SMILES
    from tests.rdkit_oracle import RDKitOracle

    smiles = GOLDEN_SMILES + SMILES
    featurizer = QmfpFeaturizer(QMFP_SPEC)
    got, got_valid = featurizer.featurize_bytes(smiles)
    want, want_valid = RDKitOracle(QMFP_SPEC).simulation_bytes(smiles)
    assert np.array_equal(got_valid, want_valid)
    assert np.array_equal(got, want)

    import rdkit

    assert featurizer.rdkit_version == rdkit.__version__, (
        "the WebAssembly module and the dev rdkit pin must be the same RDKit release"
    )


def test_the_same_molecule_written_differently_featurizes_the_same():
    from olinda.featurizer import QmfpFeaturizer

    pairs = [
        ("c1ccccc1N(=O)=O", "O=[N+]([O-])c1ccccc1"),
        ("C1=CC=CC=C1O", "c1ccccc1O"),
        ("C1=CNC=C1", "c1cc[nH]c1"),
        ("[H]C([H])([H])O", "CO"),
    ]
    featurizer = QmfpFeaturizer(QMFP_SPEC)
    for a, b in pairs:
        packed, valid = featurizer.featurize_bytes([a, b])
        assert valid.all() and np.array_equal(packed[0], packed[1]), (a, b)


def test_invalid_and_blank_smiles_keep_their_rows():
    from olinda.featurizer import QmfpFeaturizer

    smiles = ["CCO", "", "   ", "not_a_smiles", "C(C)(C)(C)(C)C", "c1ccccc1"]
    packed, valid = QmfpFeaturizer(QMFP_SPEC).featurize_bytes(smiles)
    assert packed.shape == (len(smiles), 1024)
    assert valid.tolist() == [True, False, False, False, False, True]
    assert not packed[~valid].any()
    empty, empty_valid = QmfpFeaturizer(QMFP_SPEC).featurize_bytes([])
    assert empty.shape == (0, 1024) and empty_valid.shape == (0,)
