import os
import pandas as pd
from zairachem.descriptors.eosce import EosceEmbedder
from ersilia import ErsiliaModel
import shutil

class DescriptorCalculator():
    def __init__(self, smiles_csv, output_path):
        self.smiles_path = smiles_csv
        self.output_path = output_path
        
    def calculate(self):
        os.makedirs(os.path.join(self.output_path, "descriptors"))
        os.makedirs(os.path.join(self.output_path, "data"))
        
        shutil.copy(self.smiles_path, os.path.join(self.output_path, "reference_library.csv"))
        self.data_path = os.path.join(self.output_path, "data", "data.csv")
        df = pd.read_csv(os.path.join(self.output_path, "reference_library.csv"))
        self.smiles_list = df["SMILES"].to_list()
        df.rename(columns = {"SMILES":"smiles"}, inplace=True)
        
        indx_list = [i for i, smi in enumerate(self.smiles_list)]
        cmpd_list = ["CID" + str(i).zfill(4) for i in indx_list]
        mapping = pd.DataFrame(list(zip(indx_list, indx_list, cmpd_list)), columns=["orig_idx", "uniq_idx", "compound_id"])    
        mapping.to_csv(os.path.join(self.output_path, "data", "mapping.csv"))
        
        df["compound_id"] = cmpd_list
        df.to_csv(self.data_path, index=False)
        
        #raw descriptors ersilia api
        base_desc = ["cc-signaturizer", "grover-embedding", "molfeat-chemgpt", "mordred", "rdkit-fingerprint"]
        for desc in base_desc:
            print(desc)
            path = os.path.join(self.output_path, "descriptors", desc)
            os.makedirs(path)
            with ErsiliaModel(desc) as em:
                em.api(input=self.data_path, output=os.path.join(path, "raw.h5"))
        
        print("Ersilia Compound Embeddings")
        eosce = EosceEmbedder()
        eosce.calculate(self.smiles_list, os.path.join(self.output_path, "descriptors", "eosce.h5"))
        
        #cp grover reference to reference.h5
        shutil.copy(os.path.join(self.output_path, "descriptors", "grover-embedding", "raw.h5"), os.path.join(self.output_path, "descriptors", "reference.h5"))
