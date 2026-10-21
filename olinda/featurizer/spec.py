"""The fingerprint specification every olinda model records and every runtime obeys.

A model's ``olinda.fingerprint`` metadata is this dictionary. Runtimes read it rather than
hard-coding featurization, so the file alone says how to turn SMILES into its input:

- parse with RDKit MinimalLib's ``get_mol`` and the ``get_mol`` options;
- compute RDKit's folded Morgan **count** fingerprint with the ``morgan`` options (the generator
  arguments, under the names RDKit MinimalLib uses);
- clip every count at ``count_clip`` and pack two slots per byte (``encoding: count4``, see
  :mod:`olinda.featurizer.codec`).

olinda does all three with its own WebAssembly build of RDKit (:mod:`olinda.featurizer.wasm`), which a
browser or Node runtime can run too. Stock RDKit.js cannot produce counts, only bit vectors.

Parsing has one exception, and the spec carries it. MinimalLib keeps explicit hydrogen atoms
(``[H]``, ``[2H]``, ``[H+]``) in the graph when ``removeHs`` is false, which changes the
fingerprint, so SMILES matching ``explicit_h_regex`` are parsed with ``get_mol_explicit_h``
instead, which leaves ``removeHs`` at its default.
"""

from __future__ import annotations

import copy

QMFP_SPEC = {
    "type": "qmfp",
    "encoding": "count4",
    "slots": 2048,
    "count_clip": 15,
    "morgan": {
        "radius": 3,
        "nBits": 2048,
        "useChirality": False,
        "useBondTypes": True,
        "includeRedundantEnvironments": False,
        "onlyNonzeroInvariants": False,
    },
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

    Only the packing is fixed: 4-bit counts clipped at 15, two slots per byte. The Morgan options and
    the slot count travel with the model. A spec without ``encoding: count4`` predates this layout
    (RDKit count simulation) and is refused rather than misread.
    """
    if not isinstance(spec, dict) or spec.get("type") != "qmfp":
        raise ValueError(
            f"unsupported fingerprint {spec!r}: this olinda reads qmfp only"
        )
    missing = [k for k in QMFP_SPEC if k not in spec]
    if missing:
        raise ValueError(f"fingerprint spec is missing {missing}")
    if spec["encoding"] != "count4" or int(spec["count_clip"]) != 15:
        raise ValueError(
            f"qmfp encoding must be count4 with counts clipped at 15, got {spec['encoding']!r} "
            f"clipped at {spec['count_clip']}"
        )
    slots = int(spec["slots"])
    if slots % 2 or int(spec["morgan"]["nBits"]) != slots:
        raise ValueError(
            f"qmfp needs an even slot count equal to morgan.nBits, got slots={slots}, "
            f"nBits={spec['morgan']['nBits']}"
        )
    return spec


def n_input_bytes(spec: dict) -> int:
    """Width of the model input: two 4-bit slots per byte."""
    return int(spec["slots"]) // 2
