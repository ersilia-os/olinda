"""The fingerprint specification every olinda model records and every runtime obeys.

A model's ``olinda.fingerprint`` metadata is this dictionary. Runtimes read it rather than
hard-coding featurization, so the file alone says how to turn SMILES into its input: the
Morgan options for RDKit MinimalLib's ``get_morgan_fp_as_bytes`` (and RDKit.js's
``get_morgan_fp_as_uint8array``), and the ``get_mol`` options to parse with.

Parsing has one exception, and the spec carries it. MinimalLib keeps explicit hydrogen atoms
(``[H]``, ``[2H]``, ``[H+]``) in the graph when ``removeHs`` is false, which changes the
fingerprint, so SMILES matching ``explicit_h_regex`` are parsed with ``get_mol_explicit_h``
instead, which leaves ``removeHs`` at its default.
"""

from __future__ import annotations

import copy

QMFP_SPEC = {
    "type": "qmfp",
    "radius": 3,
    "nBits": 8192,
    "useCountSimulation": True,
    "slots": 2048,
    "levels": [0, 1, 2, 4, 8],
    "get_mol": {"assignStereo": False, "removeHs": False},
    "get_mol_explicit_h": {"assignStereo": False},
    "explicit_h_regex": r"\[\d*H[^a-z]",
    "input": "fp_bytes",
}


def default_spec() -> dict:
    """A fresh copy of :data:`QMFP_SPEC`, safe for the caller to keep or modify."""
    return copy.deepcopy(QMFP_SPEC)


def validate_spec(spec: dict) -> dict:
    """Return *spec* if this olinda can honour it, else raise ``ValueError`` saying why.

    Only the layout is fixed: four count-simulation bits per slot with RDKit's default bounds,
    which MinimalLib cannot change. The radius and the slot count are free, and travel with the
    model.
    """
    if not isinstance(spec, dict) or spec.get("type") != "qmfp":
        raise ValueError(
            f"unsupported fingerprint {spec!r}: this olinda reads qmfp only"
        )
    missing = [k for k in QMFP_SPEC if k not in spec]
    if missing:
        raise ValueError(f"fingerprint spec is missing {missing}")
    if not spec["useCountSimulation"]:
        raise ValueError("qmfp needs useCountSimulation")
    if list(spec["levels"]) != QMFP_SPEC["levels"]:
        raise ValueError(
            f"qmfp levels must be {QMFP_SPEC['levels']}, got {spec['levels']}"
        )
    if int(spec["nBits"]) != 4 * int(spec["slots"]) or int(spec["slots"]) % 2:
        raise ValueError(
            f"qmfp needs nBits = 4 x slots and an even slot count, got nBits={spec['nBits']}, "
            f"slots={spec['slots']}"
        )
    return spec


def n_input_bytes(spec: dict) -> int:
    """Width of the model input: one bit per count-simulation bit, eight to a byte."""
    return int(spec["nBits"]) // 8
