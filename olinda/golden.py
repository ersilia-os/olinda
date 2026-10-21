"""Golden vectors: molecules every model scores at build time and every runtime re-scores on load.

A model records which RDKit release featurized it, but a version string proves nothing about the
fingerprints a runtime will actually compute. The golden set does: each model carries these SMILES with
a digest of their input bytes and their predictions, and :class:`~olinda.artifact.OlindaArtifact`
featurizes and scores them again when it loads. Matching digests mean the runtime's RDKit reproduces
the training fingerprints; matching predictions mean the graph runs as it did when it was built.

The list is fixed and public. It never includes a model's own labelled compounds, which the file
must not reveal. It covers what a featurizer can get wrong: explicit hydrogens, isotopes, charges and
salts, nitro forms, Kekulé against aromatic input, organometallics, stereochemistry, macrocycles, a
chain long enough to reach the top count level, and SMILES that must be rejected.

numpy and the standard library only: it runs on the inference path.
"""

from __future__ import annotations

import hashlib

import numpy as np

GOLDEN_SMILES = [
    # explicit hydrogen atoms, which MinimalLib parses with different options
    "[H]C([H])([H])O",
    "C[C@@H](O)[H]",
    "[2H]C",
    "[H+]",
    # isotopes, charges, salts
    "[13CH4]",
    "C[N+](C)(C)C",
    "[Na+].[Cl-]",
    "[Na+].[O-]C(=O)C",
    "CC(=O)[O-].[K+]",
    "[N-]=[N+]=NC",
    # the same molecule written two ways must give the same bytes
    "c1ccccc1N(=O)=O",
    "O=[N+]([O-])c1ccccc1",
    "C1=CC=CC=C1O",
    "c1ccccc1O",
    "c1cc[nH]c1",
    "C1=CNC=C1",
    # unusual atoms
    "[Hg](C)C",
    "[He]",
    "N.N.Cl[Pt]Cl",
    # stereochemistry, which this fingerprint ignores
    "C[C@H](N)C(=O)O",
    "C[C@@H](N)C(=O)O",
    "F/C=C/F",
    # small aromatics and heterocycles
    "c1ccccc1",
    "c1ccncc1",
    "c1ccsc1",
    "c1ccoc1",
    "c1nnn[nH]1",
    "c1ccc2ccccc2c1",
    "c1cc2ccc3cccc4ccc(c1)c2c34",
    # rings and cages
    "C1CCCCCCCCCCC1",
    "C1CCCCCCCCCCCCCCCCCCCCCCCCCCCCC1",
    "C1C2CC3CC1CC(C2)C3",
    # drugs
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CC(=O)Nc1ccc(O)cc1",
    "NNC(=O)c1ccncc1",
    "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "Cc1ncc([N+](=O)[O-])n1CCO",
    "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
    "CC1CCC2C(C)C(=O)OC3OC4(C)CCC1C32OO4",
    "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
    "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1",
    "CC12CCC3c4ccc(O)cc4CCC3C1CCC2O",
    "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C",
    "OCC1OC(O)C(O)C(O)C1O",
    "NC(=O)c1ccc[n+](c1)C1OC(COP(=O)([O-])OP(=O)(O)OCC2OC(n3cnc4c(N)ncnc34)C(O)C2O)C(O)C1O",
    # long chains: counts of 8 and more, the top level
    "C" * 60,
    "OCCOCCOCCOCCOCCOCCOCCO",
    # must be rejected
    "C(C)(C)(C)(C)C",
    "c1ccccc",
    "not_a_smiles",
    "",
]

TOLERANCE = {"atol": 1e-5, "rtol": 1e-5}


def _digest(row: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(row, dtype=np.uint8).tobytes()
    ).hexdigest()[:16]


def _score(run, packed: np.ndarray, valid: np.ndarray) -> dict:
    """Run the graph on the bytes, with NaN where a SMILES was invalid, as a runtime reports it."""
    scored = {
        name: np.asarray(v, dtype=np.float64).ravel() for name, v in run(packed).items()
    }
    for values in scored.values():
        values[~valid] = np.nan
    return scored


def compute_golden(run, featurizer) -> dict:
    """The ``olinda.golden`` payload for a finished graph.

    Parameters
    ----------
    run : callable
        ``uint8 bytes (N, B) -> {output_name: (N,) array}``: the fused graph.
    featurizer : QmfpFeaturizer
        The featurizer the model was trained with.
    """
    packed, valid = featurizer.featurize_bytes(GOLDEN_SMILES)
    scored = _score(run, packed, valid)
    return {
        "smiles": list(GOLDEN_SMILES),
        "valid": valid.tolist(),
        "fp_sha256": [_digest(row) for row in packed],
        "outputs": {
            name: [None if np.isnan(x) else float(x) for x in values]
            for name, values in scored.items()
        },
        **TOLERANCE,
    }


def check_golden(golden: dict, run, featurizer) -> dict:
    """Re-featurize and re-score the golden set, and say what, if anything, differs.

    Returns ``{"n", "bytes_differ": [smiles...], "outputs_differ": [smiles...]}``; both lists are
    empty when everything agrees. A molecule whose bytes differ is a featurizer problem (another RDKit
    release, other parse options); one whose bytes agree but whose prediction differs is a graph or
    onnxruntime problem. Telling the two apart is the point of storing the digests.
    """
    smiles = list(golden["smiles"])
    packed, valid = featurizer.featurize_bytes(smiles)
    want_valid = np.asarray(golden["valid"], dtype=bool)
    bytes_bad = (valid != want_valid) | np.array(
        [_digest(row) != d for row, d in zip(packed, golden["fp_sha256"])]
    )
    scored = _score(run, packed, valid)
    out_bad = np.zeros(len(smiles), dtype=bool)
    for name, want in golden["outputs"].items():
        want = np.array([np.nan if x is None else x for x in want], dtype=np.float64)
        got = scored.get(name)
        if got is None:
            out_bad[:] = True
            continue
        out_bad |= ~np.isclose(
            got,
            want,
            atol=golden.get("atol", 1e-5),
            rtol=golden.get("rtol", 1e-5),
            equal_nan=True,
        )
    return {
        "n": len(smiles),
        "bytes_differ": [s for s, b in zip(smiles, bytes_bad) if b],
        "outputs_differ": [
            s for s, b, o in zip(smiles, bytes_bad, out_bad) if o and not b
        ],
    }
