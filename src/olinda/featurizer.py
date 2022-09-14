"""Featurizer for SMILES."""

from abc import ABC
from typing import Any, List

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


from olinda.utils import get_package_root_path


class Featurizer(ABC):
    def featurize(self: "Featurizer", batch: Any) -> Any:
        """Featurize input batch.

        Args:
            batch (Any): batch of smiles

        Returns:
            Any: featurized outputs
        """
        pass


class Flat2Grid(Featurizer):
    def __init__(self: "Flat2Grid") -> None:
        self.transformer = joblib.load(get_package_root_path() / "flat2grid.joblib")
        self.name = "flat2grid"

    def featurize(self: "Flat2Grid", batch: Any) -> Any:
        """Featurize input batch.

        Args:
            batch (Any): batch of smiles

        Returns:
            Any: featurized outputs
        """

        mols = [Chem.MolFromSmiles(smi) for smi in batch]
        ecfps = self.ecfp_counts(mols)
        return self.transformer.transform(ecfps)

    def ecfp_counts(self: "Flat2Grid", mols: List) -> List:
        """Create ECFPs from batch of smiles.

        Args:
            mols (List): batch of molecules

        Returns:
            List: batch of ECFPs
        """
        fps = [
            AllChem.GetMorganFingerprint(
                mol, radius=3, useCounts=True, useFeatures=True
            )
            for mol in mols
        ]
        nfp = np.zeros((len(fps), 1024), np.uint8)
        for i, fp in enumerate(fps):
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % 1024
                nfp[i, nidx] += int(v)
        return nfp
