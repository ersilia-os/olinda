import os
import csv
import pandas as pd
import numpy as np
from zairachem.descriptors.eosce import EosceEmbedder
from ersilia import ErsiliaModel
import shutil

class DescriptorCalculator():
    def __init__(self, smiles_csv, output_path):
        self.smiles_path = smiles_csv
        self.output_path = output_path
        os.makedirs(os.path.join(self.output_path, "descriptors"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "data"), exist_ok=True)
        self.data_path = os.path.join(self.output_path, "data", "data.csv
        
    def calculate(self):
        self._screen_smiles()
        
        self.df = pd.read_csv(os.path.join(self.output_path, "reference_library.csv"))
        self.smiles_list = self.df["SMILES"].to_list()
        
        self._data_files()
        self._eosce()
        self._molmap()
        
        #raw descriptors ersilia api
        base_desc = ["cc-signaturizer", "grover-embedding", "molfeat-chemgpt", "mordred", "rdkit-fingerprint"]
        for desc in base_desc:
            print(desc)
            path = os.path.join(self.output_path, "descriptors", desc)
            os.makedirs(path)
            with ErsiliaModel(desc) as em:
                em.api(input=self.data_path, output=os.path.join(path, "raw.h5"))
        
        #cp grover reference to reference.h5
        shutil.copy(os.path.join(self.output_path, "descriptors", "grover-embedding", "raw.h5"), os.path.join(self.output_path, "descriptors", "reference.h5"))
    
    def _data_files(self):
        indx_list = [i for i, smi in enumerate(self.smiles_list)]
        cmpd_list = ["CID" + str(i).zfill(4) for i in indx_list]
        
        self.df.rename(columns = {"SMILES":"smiles"}, inplace=True)
        self.df["compound_id"] = cmpd_list
        self.df.to_csv(self.data_path, index=False)
        
        mapping = pd.DataFrame(list(zip(indx_list, indx_list, cmpd_list)), columns=["orig_idx", "uniq_idx", "compound_id"])    
        mapping.to_csv(os.path.join(self.output_path, "data", "mapping.csv"))
        
    def _eosce(self):
        print("Ersilia Compound Embeddings")
        eosce = EosceEmbedder()
        eosce.calculate(self.smiles_list, os.path.join(self.output_path, "descriptors", "eosce.h5"))
        
    def _molmap(self):
        with ErsiliaModel("bidd-molmap-desc") as mdl:
            X1 = mdl.run(input=self.smiles_list, output="numpy")
            X1 = X1.reshape(X1.shape[0], 37, 37, 1)

        with ErsiliaModel("bidd-molmap-fps") as mdl:
            X2 = mdl.run(input=self.smiles_list, output="numpy")
            X2 = X2.reshape(X2.shape[0], 37, 36, 1)
        
        with open(os.path.join(self.output_path, "descriptors", "bidd_molmap_desc.np"), "wb") as f1:
            np.save(f1, X1)

        with open(os.path.join(self.output_path, "descriptors", "bidd_molmap_fps.np"), "wb") as f2:
            np.save(f2, X2)
    
    def _screen_smiles(self):
        with ErsiliaModel("eos7w6n") as em:
                em.api(input=self.smiles_path, output=os.path.join(self.output_path, "descriptors", "eos7w6n_raw.csv"))
        raw_smiles_path = os.path.join(self.output_path, "descriptors", "eos7w6n_raw.csv")
        smiles_removed = []
        indxs_removed = []
        
        with open(raw_smiles_path, "r") as csv_file:
            datareader = csv.reader(csv_file)
            next(datareader)
            for i, row in enumerate(datareader):
                if row[2] == "":
                    print("Bad molecule found: ", row[1])
                    smiles_removed.append(row[1])
                    indxs_removed.append(i)
        
            df = pd.read_csv(raw_smiles_path)
            df.drop(df.index[indxs_removed], axis=0, inplace=True)
            df = df[["input"]]
            df.rename(columns = {"input":"SMILES"}, inplace=True)
            df.to_csv(os.path.join(self.output_path, "reference_library.csv"), index=False)
        
        os.remove(raw_smiles_path)
        
        with open(os.path.join(self.output_path, "removed_smiles.csv"), "w") as removed_smiles_file:
            for smi in smiles_removed:
                removed_smiles_file.write(smi + "\n")
        
