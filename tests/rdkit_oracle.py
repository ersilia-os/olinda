"""Full RDKit as the reference that olinda's WebAssembly featurizer must match, byte for byte.

Test-only: the package itself featurizes through RDKit MinimalLib compiled to WebAssembly and
never imports rdkit. This module mirrors what MinimalLib does with a SMILES, so a mismatch
means the WebAssembly build drifted, not that two parsers legitimately disagree:

- ``MinimalLib::mol_from_input`` parses with ``sanitize=False`` and the ``removeHs`` option,
  then runs full sanitization; a failure at either step is an invalid molecule.
- ``MinimalLib::morgan_fp_as_bitvect`` uses RDKit's Morgan generator with count simulation.

It also derives the same bytes the other way round, from true folded counts, which is the
claim the qmfp format rests on: count simulation is the counts, quantized.
"""

from __future__ import annotations

import re

import numpy as np

from olinda.featurizer.codec import COUNT_TO_LEVEL, pack_levels


class RDKitOracle:
    """Featurize SMILES with full RDKit, exactly as MinimalLib would under *spec*."""

    def __init__(self, spec: dict):
        from rdkit import RDLogger
        from rdkit.Chem import rdFingerprintGenerator as rfg

        RDLogger.DisableLog("rdApp.*")
        self.spec = spec
        self.slots = int(spec["slots"])
        self._explicit_h = re.compile(spec["explicit_h_regex"])
        self._sim = rfg.GetMorganGenerator(
            radius=int(spec["radius"]), fpSize=int(spec["nBits"]), countSimulation=True
        )
        self._counts = rfg.GetMorganGenerator(
            radius=int(spec["radius"]), fpSize=self.slots
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

    def simulation_bytes(self, smiles: list[str]):
        """Count-simulation bits packed little-endian, as ``get_morgan_fp_as_bytes`` returns them."""
        out = np.zeros((len(smiles), self.slots // 2), dtype=np.uint8)
        valid = np.zeros(len(smiles), dtype=bool)
        for i, smi in enumerate(smiles):
            mol = self.parse(smi)
            if mol is None:
                continue
            bits = self._sim.GetFingerprintAsNumPy(mol).astype(np.uint8)
            out[i] = np.packbits(bits, bitorder="little")
            valid[i] = True
        return out, valid

    def count_bytes(self, smiles: list[str]):
        """The same bytes from true folded counts: clip at 8, quantize to a level, pack."""
        levels = np.zeros((len(smiles), self.slots), dtype=np.uint8)
        valid = np.zeros(len(smiles), dtype=bool)
        for i, smi in enumerate(smiles):
            mol = self.parse(smi)
            if mol is None:
                continue
            for k, c in (
                self._counts.GetCountFingerprint(mol).GetNonzeroElements().items()
            ):
                levels[i, k] = COUNT_TO_LEVEL[min(c, 8)]
            valid[i] = True
        return pack_levels(levels), valid
