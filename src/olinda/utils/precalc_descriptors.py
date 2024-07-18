import os
import csv
import pandas as pd
import numpy as np
import shutil
import h5py

from zairachem.descriptors.eosce import EosceEmbedder
from zairachem.tools.melloddy.pipeline import MelloddyTunerPredictPipeline
from zairachem.setup.standardize import Standardize
from zairachem.setup.merge import DataMergerForPrediction
from zairachem.setup.clean import SetupCleaner

from zairachem.setup.check import SetupChecker

from ersilia import ErsiliaModel

DATA_SUBFOLDER = "data"

class DescriptorCalculator():
    def __init__(self, smiles_csv, output_path):
        self.smiles_path = smiles_csv
        self.output_path = output_path
        os.makedirs(os.path.join(self.output_path, "descriptors"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "data"), exist_ok=True)
        self.data_path = os.path.join(self.output_path, "data", "data.csv")
        
    def calculate(self):
        # prepare input files and filter out problem compounds from grover calculations
        self._data_files()
        
        self.df = pd.read_csv(os.path.join(self.output_path, "reference_library.csv"))
        self.smiles_list = self.df["SMILES"].to_list()
        
        self._eosce()
        self._molmap()
        
        # rest of raw descriptors ersilia api
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
        raw_df = pd.read_csv(self.smiles_path)
        indx_list = [i for i, smi in enumerate(raw_df["SMILES"].to_list())]
        cmpd_list = ["CID" + str(i).zfill(4) for i in indx_list]
        
        raw_df.rename(columns = {"SMILES":"smiles"}, inplace=True)
        raw_df["compound_id"] = cmpd_list
        raw_df.to_csv(os.path.join(self.output_path, "data", "compounds.csv"), index=False)
        
        mapping = pd.DataFrame(list(zip(indx_list, indx_list, cmpd_list)), columns=["orig_idx", "uniq_idx", "compound_id"])    
        mapping.to_csv(os.path.join(self.output_path, "data", "mapping.csv"))
        
        print("Mellody Tuner")
        mp = MellodyPrecalculator(self.output_path)
        mp.run()
        
        self._screen_smiles()
        
        self.df = pd.read_csv(os.path.join(self.output_path, "data", "data.csv"))
        self.df.rename(columns = {"smiles":"SMILES"}, inplace=True)
        self.df[["SMILES"]].to_csv(os.path.join(self.output_path, "reference_library.csv"), index=False)
        
    def _eosce(self):
        print("Ersilia Compound Embeddings")
        eosce = EosceEmbedder()
        eosce.calculate(self.smiles_list, os.path.join(self.output_path, "descriptors", "eosce.h5"))
        
    def _molmap(self):
        print("bidd-molmap-desc")
        with ErsiliaModel("bidd-molmap-desc") as mdl:
            X1 = mdl.run(input=self.smiles_list, output="numpy")
            X1 = X1.reshape(X1.shape[0], 37, 37, 1)
        
        print("bidd-molmap-fps")
        with ErsiliaModel("bidd-molmap-fps") as mdl:
            X2 = mdl.run(input=self.smiles_list, output="numpy")
            X2 = X2.reshape(X2.shape[0], 37, 36, 1)
        
        with open(os.path.join(self.output_path, "descriptors", "bidd_molmap_desc.np"), "wb") as f1:
            np.save(f1, X1)

        with open(os.path.join(self.output_path, "descriptors", "bidd_molmap_fps.np"), "wb") as f2:
            np.save(f2, X2)
    
    def _screen_smiles(self):
        print("Check SMILES with Grover")
        raw_smiles_path = os.path.join(self.output_path, "descriptors", "eos7w6n_initial.h5")
        with ErsiliaModel("eos7w6n") as em:
                em.api(input=self.data_path, output=raw_smiles_path)
        
        with h5py.File(raw_smiles_path, "r") as data_file:
            keys = data_file["Keys"][:]
            inputs = data_file["Inputs"][:]
            features = data_file["Features"][:]
            values = data_file["Values"][:]
        
        drop_indxs = [i for i, row in enumerate(np.isnan(values)) if True in row]    
        
        # filter out problematic smiles from data files
        smiles_strings = [smi.decode("utf-8") for smi in np.delete(inputs, drop_indxs)]    
        indx_list = [i for i, smi in enumerate(smiles_strings)]
        cmpd_list = ["CID" + str(i).zfill(4) for i in indx_list]
        
        df = pd.DataFrame(list(zip(cmpd_list, smiles_strings)), columns=["compound_id", "smiles"])
        df.to_csv(self.data_path, index=False)
        
        mapping = pd.DataFrame(list(zip(indx_list, indx_list, cmpd_list)), columns=["orig_idx", "uniq_idx", "compound_id"])    
        mapping.to_csv(os.path.join(self.output_path, "data", "mapping.csv"))
        
        os.remove(raw_smiles_path)

class MellodyPrecalculator():
    def __init__(self, output_dir):
        self.output_dir = output_dir
        
    def _melloddy_tuner_run(self):
        MelloddyTunerPredictPipeline(
            os.path.join(self.output_dir, DATA_SUBFOLDER)
            ).run(has_tasks=False)

    def _standardize(self):
        Standardize(os.path.join(self.output_dir, DATA_SUBFOLDER)).run()
        
    def _merge(self):
        DataMergerForPrediction(os.path.join(self.output_dir, DATA_SUBFOLDER)).run(False)

    def _clean(self):
        SetupCleaner(os.path.join(self.output_dir, DATA_SUBFOLDER)).run()
    
    def run(self):
        self._melloddy_tuner_run()
        self._standardize()
        self._merge()
        self._clean()
        
