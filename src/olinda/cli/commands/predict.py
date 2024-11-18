import click
import pandas as pd
import onnx_runner

from . import olinda_cli
from ...featurizer import MorganFeaturizer

def predict_cmd():
    @olinda_cli.command(help="Run ONNX model predictions")
    @click.option("--input_file", "-i", type=click.STRING, help="Path to .csv file with 'smiles' column")
    @click.option("--model", "-m", type=click.STRING, help="Path to ONNX model")
    @click.option("--output_path", "-o", type=click.STRING, help="Path and file name for onnx model predictions")

    def predict(input_file, model, output_path):
        df = pd.read_csv(input_file)
        smiles_list = df['smiles'].tolist()
        
        onnx_model = onnx_runner.ONNX_Runner(model)
        output = onnx_model.predict(smiles_list)

        df["pred"] = output
        df.to_csv(output_path, index=False)
