"""Distillation module."""

from warnings import filterwarnings
filterwarnings(action="ignore")

import os
from pathlib import Path
import shutil
import math
from typing import Any, Optional

from cbor2 import dump
import joblib
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from loguru import logger

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import zipfile

import tensorflow as tf
import tf2onnx
import onnx
import pandas as pd

from olinda.data import ReferenceSmilesDM, FeaturizedSmilesDM, GenericOutputDM
from olinda.featurizer import Featurizer, MorganFeaturizer, Flat2Grid
from olinda.generic_model import GenericModel
from olinda.tuner import ModelTuner, KerasTuner
from olinda.utils.utils import calculate_cbor_size, get_workspace_path
from olinda.utils.s3 import ProgressPercentage

### TODO: Improve object-oriented setup of distillation code segments
class Distiller(object):
    def __init__(self,
        featurizer: Optional[Featurizer] = MorganFeaturizer(),
        tuner: ModelTuner = KerasTuner(),
        reference_smiles_dm: Optional[ReferenceSmilesDM] = None,
        featurized_smiles_dm: Optional[FeaturizedSmilesDM] = None,
        generic_output_dm: Optional[GenericOutputDM] = None,
        num_data: int = 100000,
        clean: bool = True,
        test: bool = False,
    ):
        """
        Args:
        featurizer (Optional[Featurizer]): Featurizer to use.
        working_dir (Path): Path to model workspace directory.
        clean (bool): Clean workspace before starting.
        tuner (ModelTuner): Tuner to use for selecting and optimizing student model.
        reference_smiles_dm (Optional[ReferenceSmilesDM]): Reference SMILES datamodules.
        featurized_smiles_dm (Optional[FeaturizedSmilesDM]): Reference Featurized SMILES datamodules.
        generic_output_dm (Optional[GenericOutputDM]): Precalculated training dataset for student model.
        test (bool): Run a test distillation on a smaller fraction of the dataset.
        """     
        self.working_dir = get_workspace_path()
        self.featurizer = featurizer
        self.tuner = tuner
        self.reference_smiles_dm = reference_smiles_dm
        self.featurized_smiles_dm = featurized_smiles_dm
        self.generic_output_dm = generic_output_dm
        self.num_data = num_data
        if test:
            self.num_data = self.num_data // 10
        self.clean = clean
        self.test = test      

    def distill(self, model: Any) -> pl.LightningModule:
        """Distill models.
        
        Args:
            model (Any): Teacher Model.
        Returns:
            pl.LightningModule: Student Model.
        """
        
        if self.clean is True:
            clean_workspace(Path(self.working_dir), reference=True)
        
        # Convert model to a generic model
        model = GenericModel(model)
        if model.type == "zairachem":
            fetch_ref_library()
            ref_library = os.path.join(os.path.expanduser("~"), "olinda", "precalculated_descriptors", "olinda_reference_library.csv")
            precalc_smiles_df = pd.read_csv(ref_library, header=None)
            ref_data = len(precalc_smiles_df)
            self.reference_smiles_dm = ReferenceSmilesDM(num_data=self.num_data)
            self.reference_smiles_dm.prepare_data()
            self.reference_smiles_dm.setup("train")
            
            if self.num_data > ref_data:
                self.num_data = ref_data  
            zairachem_folds = math.ceil(self.num_data / 50000)
            fetch_descriptors(zairachem_folds)
            
            self.featurized_smiles_dm = gen_featurized_smiles(self.reference_smiles_dm, self.featurizer, self.working_dir, num_data=self.num_data, clean=self.clean)
            self.featurized_smiles_dm.setup("train")       
            student_training_dm = gen_model_output(model, self.featurized_smiles_dm, self.working_dir, self.num_data, self.clean)
        else:
            student_training_dm = self.generic_output_dm
            
        if student_training_dm is None:
            # Prepare reference smiles datamodule
            if reference_smiles_dm is None:
                self.reference_smiles_dm = ReferenceSmilesDM(num_data=self.num_data)
            self.reference_smiles_dm.prepare_data()
            self.reference_smiles_dm.setup("train")
        
            # Generate student model training dataset
            student_training_dm = gen_training_dataset(
                model,
                self.featurizer,
                self.reference_smiles_dm,
                self.featurized_smiles_dm,
                self.working_dir,
                self.num_data,
                self.clean,
            )
        
        # Select and Train student model
        student_model = self.tuner.fit(student_training_dm)
        model_onnx = convert_to_onnx(student_model, self.featurizer)
    
        return model_onnx

def fetch_ref_library():
    s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
    bucket = s3.Bucket('olinda')
    
    path = os.path.join(os.path.expanduser("~"), "olinda", "precalculated_descriptors")
    os.makedirs(path, exist_ok=True)
    
    lib_name = "olinda_reference_library.csv"
    ref_lib = os.path.join(path, lib_name)
    # if no reference library or the size differs to the S3 bucket version
    if os.path.exists(ref_lib) == False or bucket.Object(key=lib_name).content_length != os.path.getsize(ref_lib):
        if os.path.exists(ref_lib):
            os.remove(ref_lib)
        bucket.download_file(
                "olinda_reference_library.csv", ref_lib,
                Callback=ProgressPercentage(bucket, "olinda_reference_library.csv")
                )

def fetch_descriptors(
    num_folds: int
    ):
    """Check if required precalculated descriptor folds are on disk and fetch missing folds 
    
    Args:
        num_folds (int): Number of folds of 50k precalculated descriptors
    """
    
    s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
    bucket = s3.Bucket('olinda')
    
    path = os.path.join(os.path.expanduser("~"), "olinda", "precalculated_descriptors")
    
    for i in range(num_folds):
        fold = "olinda_reference_descriptors_" + str(i*50) + "_" + str((i+1)*50) + "k"
        dest = os.path.join(path, fold)
        fold_zip = fold + ".zip"
        dest_zip = dest + ".zip"
        
        if os.path.exists(dest) == False:
            print("Downloading precalculated descriptors: fold " + str(i+1))
            bucket.download_file(
                fold_zip, dest_zip,
                Callback=ProgressPercentage(bucket, fold_zip)
                )
            with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
                zip_ref.extractall(path)
            assert os.path.exists(dest)
            os.remove(dest_zip)

def gen_training_dataset(
    model: pl.LightningModule,
    featurizer: Featurizer,
    reference_smiles_dm: pl.LightningDataModule,
    featurized_smiles_dm: Optional[pl.LightningDataModule],
    working_dir: Path,
    num_data,
    clean: bool = False,
    ) -> pl.LightningDataModule:
    """Generate dataset for training and evaluating student model.

    Args:
        model (pl.LightningModule): Teacher model.
        featurizer (Featurizer): Featurizer to use.
        reference_smiles_dm (pl.LightningDataModule): Reference SMILES to use as inputs.
        featurized_smiles_dm (Optional[FeaturizedSmilesDM]): Reference Featurized SMILES datamodules.
        working_dir (Path): Path to model workspace directory.
        clean (bool): Clean workspace before starting.

    Returns:
        pl.LightningDataModule: generated Dataset
    """

    # Generate featurized smiles
    if featurized_smiles_dm is None:
        if featurizer is None:
            raise Exception("Featurizer cannont be None.")
        featurized_smiles_dm = gen_featurized_smiles(
            reference_smiles_dm, featurizer, working_dir, num_data, clean,
        )
    
    featurized_smiles_dm.setup("train")
    # Generate model outputs and save to a file
    model_output_dm = gen_model_output(featurized_smiles_dm, model, working_dir, clean)

    return model_output_dm


def gen_featurized_smiles(
    reference_smiles_dm: pl.LightningDataModule,
    featurizer: Featurizer,
    working_dir: Path,
    num_data,
    clean: bool = False,
) -> pl.LightningDataModule:
    """Generate featurized smiles representation dataset.

    Args:
        reference_smiles_dm (pl.LightningDataModule): Reference SMILES datamodule.
        featurizer (Featurizer): Featurizer to use.
        working_dir (Path): Path to model workspace directory.
        clean (bool): Clean workspace before starting.

    Returns:
        pl.LightningDataModule: Dateset with featurized smiles.
    """

    if clean is True:
        clean_workspace(Path(working_dir), featurizer=featurizer)
        reference_smiles_dl = reference_smiles_dm.train_dataloader()
    else:
        try:
            reference_smiles_dl = joblib.load(
                Path(working_dir / "reference" / "reference_smiles_dl.joblib")
            )
        except Exception:
            reference_smiles_dl = reference_smiles_dm.train_dataloader()

    # Save dataloader for resuming
    joblib.dump(
        reference_smiles_dl,
        Path(working_dir) / "reference" / "reference_smiles_dl.joblib",
    )

    # calculate stop_step
    try:
        with open(
            Path(working_dir)
            / "reference"
            / f"featurized_smiles_{(type(featurizer).__name__.lower())}.cbor",
            "rb",
        ) as feature_stream:
            stop_step = calculate_cbor_size(feature_stream)
    except Exception:
        stop_step = 0

    with open(
        Path(working_dir)
        / "reference"
        / f"featurized_smiles_{(type(featurizer).__name__.lower())}.cbor",
        "wb",
    ) as feature_stream:
        for i, batch in tqdm(
            enumerate(iter(reference_smiles_dl)),
            total=reference_smiles_dl.length,
            desc="Featurizing",
        ):  
            if i < stop_step // len(batch[0]):
                continue   
            if i >= reference_smiles_dl.length:
                break
            if i >= num_data:
                break
                     
            output = featurizer.featurize(batch[1])
            for j, elem in enumerate(batch[0]):
                dump((elem.tolist(), batch[1][j], output[j].tolist()), feature_stream)

    featurized_smiles_dm = FeaturizedSmilesDM(Path(working_dir), featurizer, num_data=num_data)
    
    return featurized_smiles_dm

def gen_model_output(
    model: GenericModel,
    featurized_smiles_dm: pl.LightningDataModule,
    working_dir: Path,
    ref_size: int,
    clean: bool = False,
) -> pl.LightningDataModule:

    """Generate featurized smiles representation dataset.

    Args:
        featurized_smiles_dm ( pl.LightningDataModule): Featurized SMILES to use as inputs.
        model (GenericModel): Wrapped Teacher model.
        working_dir (Path): Path to model workspace directory.
        clean (bool): Clean workspace before starting.

    Returns:
        pl.LightningDataModule: Dateset with featurized smiles.
    """

    os.makedirs(os.path.join(working_dir, model.name), exist_ok=True)
    if clean is True:
        clean_workspace(Path(working_dir), model=model)
        featurized_smiles_dl = featurized_smiles_dm.train_dataloader()
    else:
        try:
            featurized_smiles_dl = joblib.load(
                Path(working_dir / (model.name) / "featurized_smiles_dl.joblib")
            )
        except Exception:
            featurized_smiles_dl = featurized_smiles_dm.train_dataloader()

    # Save dataloader for resuming
    joblib.dump(
        featurized_smiles_dl,
        Path(working_dir / (model.name) / "featurized_smiles_dl.joblib"),
    )

    # calculate stop step
    try:
        with open(
            Path(working_dir) / (model.name) / "model_output.cbor",
            "rb",
        ) as output_stream:
            stop_step = calculate_cbor_size(output_stream)
    except Exception:
        stop_step = 0
    
    with open(
        Path(working_dir) / (model.name) / "model_output.cbor", "wb"
    ) as output_stream:
    
        if model.type == "zairachem":
            output = pd.DataFrame(columns = ["smiles", 'pred'])
            for i in range(math.ceil(ref_size/50000)):
                logger.info("Getting ZairaChem predictions for fold " + str(i+1) + " of " + str(math.ceil(ref_size/50000)))
                folder = os.path.join(os.path.expanduser("~"), "olinda", "precalculated_descriptors", "olinda_reference_descriptors_" + str(i*50) + "_" + str((i+1)*50) + "k")
                smiles_input_path = os.path.join(os.path.expanduser("~"), "olinda", "precalculated_descriptors", folder, "reference_library.csv")
                preds = model(smiles_input_path)
                output = pd.concat([output, preds])    
            output = output[["smiles", "pred"]]   
             
            # save to zairachem model folder
            zaira_distill_path = os.path.join(model.name, "distill")
            if os.path.exists(zaira_distill_path) == False:
                os.mkdir(zaira_distill_path)
            output.to_csv(os.path.join(zaira_distill_path, "reference_predictions.csv"), index=False)
            
            # correct wrong zairachem predictions before training Olinda
            training_output = model.get_training_preds()
            for i, row in enumerate(training_output.iterrows()):
                if row[1]["pred"] >= 0.0 and row[1]["pred"] <= 0.5 and row[1]["true"] == 1.0:
                    training_output.at[i, "pred"] = 1.0 - row[1]["pred"]
                elif row[1]["pred"] >= 0.5 and row[1]["pred"] <= 1.0 and row[1]["true"] == 0:
                    training_output.at[i, "pred"] = 1.0 - row[1]["pred"]
                    
            """
            # weight by data source: training/reference
            # inverse of proportion of training compounds to all compounds
            train_weight = 1 #round((training_output.shape[0] + ref_size) / len(training_output), 2)
            """
            
            # inverse of ratio of predicted active to inactive 
            y_bin_train = [1 if val > 0.5 else 0 for val in training_output["pred"]]
            y_bin_ref = [1 if val > 0.5 else 0 for val in output["pred"]]
            active_weight = (y_bin_train.count(0) + y_bin_ref.count(0)) / (y_bin_train.count(1) + y_bin_ref.count(1))
            
            print("Creating model prediction files")
            train_counter = 0
            morganFeat = MorganFeaturizer()
            for i, row in training_output.dropna().iterrows():
                fp = morganFeat.featurize([row["smiles"]])
                if fp is None:
                    continue
                weight = active_weight
                dump((i, row["smiles"], fp[0].tolist(), [row["pred"]], [weight]), output_stream)
                train_counter += 1        
            
        ref_counter = 0        
        for i, batch in tqdm(
            enumerate(iter(featurized_smiles_dl)),
            total=featurized_smiles_dl.length,
            desc="Creating model output",
        ):
            if i < stop_step // len(batch[0]):
                continue

            if model.type == "zairachem":
                # final dataset a multiple of batch
                combined_count = train_counter + featurized_smiles_dl.length*len(batch[0])
                target_count = combined_count // len(batch[0]) * len(batch[0]) 
                
                for j, elem in enumerate(batch[1]):
                    if ref_counter + train_counter == target_count:
                        break
                    if not output[output["smiles"] == elem].empty:
                        pred_val = output[output["smiles"] == elem]["pred"].iloc[0]
                        if pred_val > 0.5:
                            weight = active_weight
                        else:
                            weight = 1
                        dump((j, elem, batch[2][j], [pred_val], [weight]), output_stream)
                        ref_counter += 1
                
            elif model.type == "ersilia":
                output = model(batch[1])
                for j, elem in enumerate(batch[1]):
                    dump((j, elem, batch[2][j], [output[j].tolist()]), output_stream)

            else:
            	output = model(torch.tensor(batch[2]))
            	for j, elem in enumerate(batch[1]):
            	    dump((j, elem, batch[2][j], output[j].tolist()), output_stream)

        # Remove zairachem folder
        if os.path.exists(os.path.join(get_workspace_path(), "zairachem_output_dir")):
            shutil.rmtree(os.path.join(get_workspace_path(), "zairachem_output_dir"))

    if model.type == "zairachem":
        model_output_dm = GenericOutputDM(Path(working_dir / (model.name)), zaira_training_size = training_output.shape[0])
    else:
        model_output_dm = GenericOutputDM(Path(working_dir / (model.name)))   
        
    return model_output_dm

def convert_to_onnx(
    model: pl.LightningModule,
    featurizer: Featurizer,
) -> onnx.onnx_ml_pb2.ModelProto:
    """Convert student model to ONNX format

    Args:
        model (GenericModel): Wrapped Student model.
        featurizer (Featurizer): Featurizer to test data shape.

    Returns:
        onnx.onnx_ml_pb2.ModelProto: ONNX formatted model
    """
    
    example = featurizer.featurize(["CCCOC"])
    
    spec = (tf.TensorSpec(example.shape, featurizer.tf_dtype, name="input"),)
    model_onnx, _ = tf2onnx.convert.from_keras(model.nn, input_signature=spec)
    model_onnx = GenericModel(model_onnx)
    return model_onnx


def clean_workspace(
    working_dir: Path, model: GenericModel = None, featurizer: Featurizer = None, reference: bool = False
) -> None:
    """Clean workspace.

    Args:
        working_dir (Path): Path of the working directory.
        model (GenericModel): Wrapped Teacher model.
        featurizer (Featurizer): Featurizer to use.
    """
    
    curr_ref_smiles_path = Path(working_dir) / "reference" / "reference_smiles.csv"
    orig_ref_smiles_path = os.path.join(os.path.expanduser("~"), "olinda", "precalculated_descriptors", "olinda_reference_library.csv")
    
    if model:
        shutil.rmtree(Path(working_dir) / (model.name), ignore_errors=True)
        os.makedirs(Path(working_dir) / (model.name), exist_ok=True)

    if featurizer and os.path.exists(Path(working_dir) / "reference" / "reference_smiles_dl.joblib"):
        os.remove(Path(working_dir) / "reference" / "reference_smiles_dl.joblib")
        os.remove(Path(working_dir) / "reference" / f"featurized_smiles_{type(featurizer).__name__.lower()}.cbor"
        )
    
    if reference and os.path.exists(curr_ref_smiles_path):
        if os.path.exists(orig_ref_smiles_path):
            curr_df = pd.read_csv(curr_ref_smiles_path, header=None, names=["SMILES"])
            orig_df = pd.read_csv(orig_ref_smiles_path)
            if not curr_df.equals(orig_df):
                shutil.rmtree(Path(working_dir) / "reference")
        else:
            shutil.rmtree(Path(working_dir) / "reference")

