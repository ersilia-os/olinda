"""SMILES to qmfp fingerprints: the one featurizer olinda uses, for training and for inference.

A qmfp fingerprint is RDKit's folded Morgan count fingerprint, each count clipped at 15 and packed as
4 bits (see :mod:`olinda.featurizer.codec` for the layout). olinda computes it through
:mod:`olinda.featurizer.wasm`, RDKit compiled to WebAssembly, so neither training nor inference needs
the rdkit package, and both produce exactly the same bytes.
"""

from __future__ import annotations

import json
import re

import numpy as np

from olinda.featurizer.codec import decode
from olinda.featurizer.spec import (
    QMFP_SPEC,
    default_spec,
    n_input_bytes,
    validate_spec,
)

__all__ = [
    "QMFP_SPEC",
    "QmfpFeaturizer",
    "default_spec",
    "n_input_bytes",
    "validate_spec",
]


class QmfpFeaturizer:
    """Featurize SMILES as a model's fingerprint spec says to.

    Parameters
    ----------
    spec : dict, optional
        The ``olinda.fingerprint`` spec; defaults to :data:`QMFP_SPEC`.

    Notes
    -----
    Every row of the input comes back, in order. A SMILES that is blank or that RDKit cannot parse
    or sanitize gives an all-zero row and ``valid = False``; the caller decides what that means.

    The object holds only its spec, so it pickles cheaply into worker processes, each of which builds
    its own WebAssembly instance on first use.
    """

    def __init__(self, spec: dict | None = None):
        self.spec = validate_spec(spec if spec is not None else default_spec())
        self._explicit_h = re.compile(self.spec["explicit_h_regex"])
        self._fp_details = json.dumps(
            {**self.spec["morgan"], "countEncoding": self.spec["encoding"]}
        )
        self._mol_details = json.dumps(self.spec["get_mol"])
        self._mol_details_h = json.dumps(self.spec["get_mol_explicit_h"])
        self.n_bytes = n_input_bytes(self.spec)

    def __getstate__(self) -> dict:
        return {"spec": self.spec}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["spec"])

    @property
    def rdkit_version(self) -> str:
        """The RDKit release that computes these fingerprints."""
        from olinda.featurizer import wasm

        return wasm.rdkit_version()

    def featurize_bytes(self, smiles) -> tuple[np.ndarray, np.ndarray]:
        """``(uint8 (N, slots/2), bool (N,))``: the model's input bytes and which rows are valid."""
        from olinda.featurizer import wasm

        smiles = [str(s) for s in smiles]
        out = np.zeros((len(smiles), self.n_bytes), dtype=np.uint8)
        valid = np.zeros(len(smiles), dtype=bool)
        # MinimalLib keeps explicit hydrogen atoms when removeHs is off, which changes the
        # fingerprint, so those SMILES are parsed with the spec's other options. Blank SMILES and
        # embedded NULs are invalid before they reach RDKit.
        groups: dict[bool, list[int]] = {False: [], True: []}
        for i, s in enumerate(smiles):
            if s.strip() and "\0" not in s:
                groups[bool(self._explicit_h.search(s))].append(i)
        for has_h, idx in groups.items():
            if not idx:
                continue
            o, v = wasm.featurize(
                [smiles[i] for i in idx],
                self._mol_details_h if has_h else self._mol_details,
                self._fp_details,
                self.n_bytes,
            )
            out[idx], valid[idx] = o, v
        return out, valid

    def transform(self, smiles) -> np.ndarray:
        """``uint8 (N, slots)`` slot counts in 0-15: what training reads and models see."""
        packed, _ = self.featurize_bytes(smiles)
        return decode(packed)
