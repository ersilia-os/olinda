import onnx
import onnxruntime as rt
from .morgan_featurizer import MorganFeaturizer

class onnx_runner(object):
    def __init__(self, model_path, featurizer=MorganFeaturizer()):
        self.onnx_model = self.load(model_path)
        self.featurizer = featurizer

    def load(self, model_path):
        return onnx.load(model_path)

    def predict(self, smiles_list):
        if type(smiles_list) == str:
            smiles_list = [smiles_list]
        X = self._featurize(smiles_list)
        onnx_rt = rt.InferenceSession(self.onnx_model.SerializeToString())
        output_names = [n.name for n in self.onnx_model.graph.output]

        preds = []
        for i, smi in enumerate(smiles_list):
            if X[i] is not None:
                pred = onnx_rt.run(output_names, {"input": [X[i]]})
                preds.append(pred[0][0][0]) #remove tensorflow nesting
            else:
                preds.append("")
        return preds

    def _featurize(self, smiles_list):
        return self.featurizer.featurize(smiles_list)
