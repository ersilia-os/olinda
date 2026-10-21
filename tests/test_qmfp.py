"""The qmfp fingerprint: its byte layout, the decoder inside the graph, and agreement with RDKit.

The layout is the contract between olinda and every runtime that feeds a model, so it is tested
directly rather than only through a fitted model.
"""

from __future__ import annotations

import numpy as np
import pytest

from olinda.featurizer.codec import COUNT_CLIP, byte_table, decode, pack
from olinda.featurizer.spec import QMFP_SPEC, default_spec, n_input_bytes, validate_spec


def test_every_byte_round_trips():
    """With 4-bit counts every one of the 256 byte values is a valid pair of slots."""
    every_byte = np.arange(256, dtype=np.uint8)[None, :]
    assert np.array_equal(pack(decode(every_byte)), every_byte)


def test_counts_round_trip_clip_at_15_and_the_even_slot_is_the_low_nibble():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 16, size=(64, 2048)).astype(np.uint8)
    assert np.array_equal(decode(pack(counts)), counts)
    # 0x71: low nibble 1 → even slot count 1; high nibble 7 → odd slot count 7
    assert byte_table()[0x71].tolist() == [1.0, 7.0]
    # true counts above 15 are stored as 15
    assert decode(pack(np.array([[16, 200]]))).tolist() == [[COUNT_CLIP, COUNT_CLIP]]


def test_the_spec_is_self_consistent_and_guards_its_layout():
    assert validate_spec(default_spec()) is not None
    assert n_input_bytes(QMFP_SPEC) == 1024
    for change in (
        {"encoding": "thermometer"},
        {"count_clip": 8},
        {"slots": 1024},
    ):
        with pytest.raises(ValueError):
            validate_spec({**QMFP_SPEC, **change})
    with pytest.raises(ValueError, match="qmfp only"):
        validate_spec({**QMFP_SPEC, "type": "morgan"})
    # a spec from the earlier count-simulation layout has no encoding: refused, not misread
    old = {k: v for k, v in QMFP_SPEC.items() if k != "encoding"}
    with pytest.raises(ValueError, match="missing"):
        validate_spec(old)


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


def test_the_oracle_bytes_are_rdkit_counts_clipped_at_15():
    """The format is RDKit's folded Morgan counts, clipped at 15: nothing else."""
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
    raw, valid = oracle.counts(smiles)
    packed, valid2 = oracle.count_bytes(smiles)
    assert valid.all() and np.array_equal(valid, valid2)
    assert np.array_equal(decode(packed), np.minimum(raw, COUNT_CLIP))
    # the long chain's CH2 environments count far above 15, so clipping is exercised
    assert raw[2].max() > COUNT_CLIP and decode(packed[2:3]).max() == COUNT_CLIP


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
    want, want_valid = RDKitOracle(QMFP_SPEC).count_bytes(smiles)
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
