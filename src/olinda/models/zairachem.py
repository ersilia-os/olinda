"""ZairaChem predictor"""
import zairachem
import ersilia

from zairachem.setup.prediction import PredictSetup
from zairachem.descriptors.describe import Describer
from zairachem.estimators.pipe import EstimatorPipeline
from zairachem.pool.pool import Pooler
from zairachem.finish.finisher import Finisher
from zairachem.reports.report import Reporter
import pandas as pd
import os
import shutil
import json

class ZairaChemPredictor(object):
    def __init__(self, input_file, model_dir, output_dir, clean, flush):
        self.input_file = input_file
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.clean = clean
        self.flush = flush

    def predict(self):
        self.s = PredictSetup(
            input_file=self.input_file,
            output_dir=self.output_dir,
            model_dir=self.model_dir,
            time_budget=60, 
        )
        self.s.setup()
        d = Describer(path=self.output_dir)
        d.run()
        #self.run_descriptors(d)        
        e = EstimatorPipeline(path=self.output_dir)
        e.run()
        p = Pooler(path=self.output_dir)
        p.run()
        r = Reporter(path=self.output_dir)
        r.run()
        f = Finisher(path=self.output_dir, clean=self.clean, flush=self.flush)
        f.run()
        return self.clean_output()
#WIP - precalc descriptors
"""        
    def run_descriptors(self, d: Describer) -> None:
        d.reset_time()
        self.precalc_descriptors()
        d._treated_descriptions()
        d.update_elapsed_time()
        
    def precalc_descriptors(self) -> None:
        path = get_workspace_path()
        precalc_descs = list(path.glob("*"))
        done = []
        
        #raw descriptors
        with open(os.path.join(self.model_dir, "data", "parameters.json"), "r") as param_file:
            parameters = json.load(param_file)
            for desc in parameters["ersilia_hub"]:
                if desc in precalc_descs:
                    shutil.copytree(os.path.join(path, desc), os.path.join(self.model_dir, "descriptors"))
                    done.append(desc)
                else:
                    ### Make folder and copy output h5
                    with ErsiliaModel(model_id) as em_api:
                        em_api.run(x, output="h5")
                        done.append(desc)
        
        #remaining manifolds, ersilia compound embeddings and reference embedding
        for f in precalc_descriptors:
            if "." in f and f != "done_eos.json":
                shutil.copy(os.path.join(path, f), os.path.join(self.model_dir, "descriptors"))
        
        #update json
        with open(os.path.join(self.model_dir, "descriptors", "done_eos.json"), "r") as done_file:
            done_eos = json.load(done_file)
            #UPDATE FILE AND WRITE
"""

    def clean_output(self):
        results = pd.read_csv(os.path.join(self.output_dir, "output.csv"))
        col_names = results.columns.values.tolist()
        
        clf_col = ""
        for c in col_names:
            if "clf" in c and "bin" not in c:
                clf_col = c
        
        results.rename({clf_col: 'pred'}, axis=1, inplace=True)
        return results[["smiles", 'pred']]
