"""Full RDKit as the reference that olinda's WebAssembly featurizer must match, byte for byte.

Test-only: the package itself featurizes through RDKit MinimalLib compiled to WebAssembly and
never imports rdkit. This module mirrors what that module does with a SMILES, so a mismatch
means the WebAssembly build drifted, not that two parsers legitimately disagree:

- ``MinimalLib::mol_from_input`` parses with ``sanitize=False`` and the ``removeHs`` option,
  then runs full sanitization; a failure at either step is an invalid molecule.
- the fingerprint is RDKit's folded Morgan count fingerprint, each count clipped at 15 and packed
  two slots per byte (:mod:`olinda.featurizer.codec`).
"""

from __future__ import annotations

import re

import numpy as np

from olinda.featurizer.codec import pack


class RDKitOracle:
    """Featurize SMILES with full RDKit, exactly as olinda's WebAssembly module would under *spec*."""

    def __init__(self, spec: dict):
        from rdkit import RDLogger
        from rdkit.Chem import rdFingerprintGenerator as rfg

        RDLogger.DisableLog("rdApp.*")
        self.spec = spec
        self.slots = int(spec["slots"])
        self._explicit_h = re.compile(spec["explicit_h_regex"])
        m = spec["morgan"]
        self._counts = rfg.GetMorganGenerator(
            radius=int(m["radius"]),
            fpSize=int(m["nBits"]),
            includeChirality=bool(m["useChirality"]),
            useBondTypes=bool(m["useBondTypes"]),
            countSimulation=False,
            includeRedundantEnvironments=bool(m["includeRedundantEnvironments"]),
        )

    def parse(self, smiles: str):
        """The molecule MinimalLib would build, or ``None`` where it would refuse."""
        from rdkit import Chem

        if not smiles or not smiles.strip():
            return None
        ps = Chem.SmilesParserParams()
        ps.sanitize = False
        ps.removeHs = bool(self._explicit_h.search(smiles))
        try:
            mol = Chem.MolFromSmiles(smiles, ps)
            if mol is None or mol.GetNumAtoms() == 0:
                return None
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None

    def counts(self, smiles: list[str]):
        """True folded counts, ``(N, slots)``, before clipping, and which rows parsed."""
        out = np.zeros((len(smiles), self.slots), dtype=np.int64)
        valid = np.zeros(len(smiles), dtype=bool)
        for i, smi in enumerate(smiles):
            mol = self.parse(smi)
            if mol is None:
                continue
            for k, c in (
                self._counts.GetCountFingerprint(mol).GetNonzeroElements().items()
            ):
                out[i, k] = c
            valid[i] = True
        return out, valid

    def count_bytes(self, smiles: list[str]):
        """The model's input bytes: counts clipped at 15, two slots per byte."""
        counts, valid = self.counts(smiles)
        return pack(counts), valid
