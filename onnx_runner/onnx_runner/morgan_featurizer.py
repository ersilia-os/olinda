from abc import ABC
from typing import Any, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

class MorganFeaturizer(object):
    def __init__(self: "MorganFeaturizer") -> None:
        self.name = "morganfeaturizer"
        
    def featurize(self: "MorganFeaturizer", batch: Any) -> Any:
        """Featurize input batch.

        Args:
            batch (Any): batch of smiles

        Returns:
            Any: featurized outputs
        """
        mols = [Chem.MolFromSmiles(smi) for smi in batch if smi is not None]
        ecfps = self.ecfp_counts(mols)
        return ecfps
    
    def ecfp_counts(self: "MorganFeaturizer", mols: List) -> List:
        """Create ECFPs from batch of smiles.

        Args:
            mols (List): batch of molecules

        Returns:
            List: batch of ECFPs
        """
        fps = [
            AllChem.GetMorganFingerprint(
                mol, radius=3, useCounts=True, useFeatures=True
            ) if mol is not None else None
            for mol in mols
        ]
        
        nfp = []
        for fp in fps:
            if fp is not None:
                tmp = np.zeros((1024), np.float32)
                for idx, v in fp.GetNonzeroElements().items():
                    tmp[idx % 1024] += int(v)
                nfp.append(tmp)
            else:
                nfp.append(None)
        return np.array(nfp)
