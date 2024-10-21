"""ZairaChem predictor"""
import zairachem
import ersilia
from ersilia import ErsiliaModel

from zairachem.setup.prediction import PredictSetup
from zairachem.descriptors.describe import Describer
from zairachem.estimators.pipe import EstimatorPipeline
from zairachem.pool.pool import Pooler
from zairachem.finish.finisher import Finisher
from zairachem.reports.report import Reporter

import pandas as pd
import os
from os import devnull
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import shutil
import json
import sys
import warnings
import logging
import glob
from pathlib import Path

class ZairaChemPredictor(object):
    def __init__(self, input_file, model_dir, output_dir, clean, flush):
        self.input_file = input_file
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.clean = clean
        self.flush = flush
        self.precalc_path = os.path.dirname(self.input_file)
    
    def predict(self):
        print("ZairaChem: Setup")
        with HiddenPrints():
            self.s = PredictSetup(
	        input_file=self.input_file,
	        output_dir=self.output_dir,
	        model_dir=self.model_dir,
	        time_budget=60, 
            )
            self.data_files(self.s)            
        
        print("ZairaChem: Describe")
        with HiddenPrints():
            d = Describer(path=self.output_dir)
            self.run_descriptors(d)
        
        print("ZairaChem: Estimate")
        with HiddenPrints():
            e = EstimatorPipeline(path=self.output_dir)
            e.run()
        
        print("ZairaChem: Pool")
        with HiddenPrints():
            p = Pooler(path=self.output_dir)
            p.run()
        
        print("ZairaChem: Report")
        with HiddenPrints():
            r = Reporter(path=self.output_dir)
            r._output_table()
        
        print("ZairaChem: Finish")
        with HiddenPrints():
            f = Finisher(path=self.output_dir, clean=self.clean, flush=self.flush)
            f.run()
        
        return self.clean_output(self.output_dir)
 
    def data_files(self, s):
        s._initialize()
        s._normalize_input()
        
        #update mapping file
        shutil.copy(os.path.join(self.precalc_path, "data", "mapping.csv"), os.path.join(self.output_dir, "data"))
        shutil.copy(os.path.join(self.precalc_path, "data", "data.csv"), os.path.join(self.output_dir, "data"))
        shutil.copy(os.path.join(self.precalc_path, "data", "data_schema.json"), os.path.join(self.output_dir, "data"))
        
        s._check()
 
    def run_descriptors(self, d: Describer) -> None:   
        d.reset_time()
        self.precalc_descriptors()
        d._treated_descriptions()
        d._manifolds()
        d.update_elapsed_time()
        
    def precalc_descriptors(self) -> None:
        shutil.copytree(os.path.join(self.precalc_path, "descriptors", "grover-embedding"), os.path.join(self.output_dir, "descriptors", "grover-embedding"))
        done = ["grover-embedding"]
    
        precalc_descs = [os.path.basename(desc_path) for desc_path in list(glob.glob(os.path.join(self.precalc_path, "descriptors", "*")))]        
        #raw descriptors
        with open(os.path.join(self.model_dir, "descriptors", "done_eos.json"), "r") as calculated_desc_file:
            parameters = json.load(calculated_desc_file)
            for desc in parameters:
                if desc in precalc_descs and desc != "grover-embedding":
                    shutil.copytree(os.path.join(self.precalc_path, "descriptors", desc), os.path.join(self.output_dir, "descriptors", desc))
                    done.append(desc)
                elif desc != "grover-embedding":
                    #make folder and copy output h5
                    with ErsiliaModel(desc) as em_api:
                        os.makedirs(os.path.join(self.output_dir, "descriptors", desc))
                        em_api.api(input=self.input_file, output=os.path.join(self.output_dir, "descriptors", desc, "raw.h5"))
                        done.append(desc)
        
        #copy remaining manifolds, ersilia compound embeddings
        for f in ["eosce.h5", "bidd_molmap_desc.np", "bidd_molmap_fps.np"]:
            shutil.copy(os.path.join(self.precalc_path, "descriptors", f), os.path.join(self.output_dir, "descriptors"))
        
        #update json descriptor file
        with open(os.path.join(self.output_dir, "descriptors", "done_eos.json"), "w") as done_file:
            json.dump(done, done_file)

    
    def clean_output(self, path):
        results = pd.read_csv(os.path.join(path, "output.csv"))
        col_names = results.columns.values.tolist()
        
        clf_col = ""
        for c in col_names:
            if "clf" in c and "bin" not in c:
                clf_col = c
        
        results.rename({clf_col: 'pred'}, axis=1, inplace=True)
        return results[["smiles", 'pred']].dropna()
        
@contextmanager
def HiddenPrints():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            with warnings.catch_warnings():
                logging.config.dictConfig({
                'version': 1,
                'disable_existing_loggers': True
                })
                warnings.simplefilter('ignore')
                try:
                    yield (err, out)
                finally:
                    logging.config.dictConfig({
                    'version': 1,
                    'disable_existing_loggers': False
                    })
                    warnings.simplefilter('default')        
