"""Distillation module."""

import os
from pathlib import Path
import shutil
import glob
from typing import Any, Optional

from cbor2 import dump
import joblib
import pytorch_lightning as pl
import torch
from tqdm import tqdm

import tensorflow as tf
import tf2onnx
import onnx
import pandas as pd

from olinda.data import ReferenceSmilesDM, FeaturizedSmilesDM, GenericOutputDM
from olinda.featurizer import Featurizer, MorganFeaturizer, Flat2Grid
from olinda.generic_model import GenericModel
from olinda.tuner import ModelTuner, KerasTuner
from olinda.utils import calculate_cbor_size, get_workspace_path


def distill(
    model: Any,
    working_dir: Path = get_workspace_path(),
    featurizer: Optional[Featurizer] = MorganFeaturizer(),
    clean: bool = True,
    tuner: ModelTuner = KerasTuner(),
    reference_smiles_dm: Optional[ReferenceSmilesDM] = None,
    featurized_smiles_dm: Optional[FeaturizedSmilesDM] = None,
    generic_output_dm: Optional[GenericOutputDM] = None,
    test: bool = False,
    num_data: int = 1999380,
) -> pl.LightningModule:
    """Distill models.

    Args:
        model (Any): Teacher Model.
        featurizer (Optional[Featurizer]): Featurizer to use.
        working_dir (Path): Path to model workspace directory.
        clean (bool): Clean workspace before starting.
        tuner (ModelTuner): Tuner to use for selecting and optimizing student model.
        reference_smiles_dm (Optional[ReferenceSmilesDM]): Reference SMILES datamodules.
        featurized_smiles_dm (Optional[FeaturizedSmilesDM]): Reference Featurized SMILES datamodules.
        generic_output_dm (Optional[GenericOutputDM]): Precalculated training dataset for student model.
        test (bool): Run a test distillation on a smaller fraction of the dataset.
        num_data: (int) : Set the number of ChEMBL training points to use (up to 1999380)
    Returns:
        pl.LightningModule: Student Model.
    """
    
    if clean is True:
        clean_workspace(Path(working_dir), reference=True)
    
    # Convert model to a generic model
    model = GenericModel(model)
    if model.type == "zairachem":
        ref_library = Path(os.path.join(__file__, "..", ".." , ".." , "data" , "olinda_reference_library.csv")).resolve()
        precalc_smiles_df = pd.read_csv(ref_library, header=None)
        ref_size = len(precalc_smiles_df)
        reference_smiles_dm = ReferenceSmilesDM(num_data=ref_size)
        reference_smiles_dm.prepare_data()
        reference_smiles_dm.setup("train")
        featurized_smiles_dm = gen_featurized_smiles(reference_smiles_dm, featurizer, working_dir, num_data=ref_size, clean=clean)
        featurized_smiles_dm.setup("train")       
        student_training_dm = gen_model_output(featurized_smiles_dm, model, working_dir, ref_size, clean)
    else:
        student_training_dm = generic_output_dm
        
    if student_training_dm is None:
        # Prepare reference smiles datamodule
        if reference_smiles_dm is None:
            reference_smiles_dm = ReferenceSmilesDM(num_data=num_data)
        reference_smiles_dm.prepare_data()
        if not test:
            reference_smiles_dm.setup("train")
        else:
            reference_smiles_dm.setup("val")

        # Generate student model training dataset
        student_training_dm = gen_training_dataset(
            model,
            featurizer,
            reference_smiles_dm,
            featurized_smiles_dm,
            working_dir,
            num_data,
            clean,
        )

    # Select and Train student model
    student_model = tuner.fit(student_training_dm)
    model_onnx = convert_to_onnx(student_model, featurizer)

    return model_onnx


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
                     
            output = featurizer.featurize(batch[1])
            for j, elem in enumerate(batch[0]):
                dump((elem.tolist(), batch[1][j], output[j].tolist()), feature_stream)

    featurized_smiles_dm = FeaturizedSmilesDM(Path(working_dir), featurizer)
    
    return featurized_smiles_dm

def gen_model_output(
    featurized_smiles_dm: pl.LightningDataModule,
    model: GenericModel,
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
    
    os.makedirs(Path(working_dir) / model.name, exist_ok=True)
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
            train_counter = 0
            
            training_output = model.get_training_preds()
            # inverse of proportion of training compounds
            train_weight = round((training_output.shape[0] + ref_size) / len(training_output), 2)
            
            morganFeat = MorganFeaturizer()
            for i, row in training_output.iterrows():
                fp = morganFeat.featurize([row["smiles"]])
                if fp is None:
                    continue
                dump((i, row["smiles"], fp[0].tolist(), [[row["pred"]]], [train_weight]), output_stream)
                train_counter += 1
            
            output = pd.DataFrame(columns = ["smiles", 'pred'])
            for i, dir_item in enumerate(glob.glob(os.path.join(os.path.expanduser("~"), "olinda", "precalculated_descriptors", "*"))):
                if os.path.isdir(dir_item):
                    folder = os.path.join(os.path.expanduser("~"), "olinda", "precalculated_descriptors", "olinda_reference_descriptors_" + str(i*50) + "_" + str((i+1)*50) + "k")
                    preds = model(os.path.join(os.path.expanduser("~"), "olinda", "precalculated_descriptors", folder, "reference_library.csv"))
                    output = pd.concat([output, preds])
            
        ref_counter = 0        
        for i, batch in tqdm(
            enumerate(iter(featurized_smiles_dl)),
            total=featurized_smiles_dl.length,
            desc="Creating model output",
        ):
            if i < stop_step // len(batch[0]):
                continue

            if model.type == "zairachem":
                # final dataset a multiple of 32
                combined_count = train_counter + featurized_smiles_dl.length*len(batch[0])
                target_count = combined_count // len(batch[0]) * len(batch[0]) 
                
                for j, elem in enumerate(batch[1]):
                    if ref_counter + train_counter == target_count:
                        break
                    if not output[output["smiles"] == elem].empty:
                        dump((j, elem, batch[2][j], [[output[output["smiles"] == elem]["pred"].iloc[0]]], [1.0]), output_stream)
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
        shutil.rmtree(os.path.join(get_workspace_path(), "zairachem_output_dir"))

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
    orig_ref_smiles_path = Path(os.path.join(__file__, "..", ".." , ".." , "data" , "olinda_reference_library.csv")).resolve()
    
    if model:
        shutil.rmtree(Path(working_dir) / (model.name), ignore_errors=True)
        os.makedirs(Path(working_dir) / (model.name), exist_ok=True)

    if featurizer and os.path.exists(Path(working_dir) / "reference" / "reference_smiles_dl.joblib"):
        os.remove(Path(working_dir) / "reference" / "reference_smiles_dl.joblib")
        os.remove(Path(working_dir) / "reference" / f"featurized_smiles_{type(featurizer).__name__.lower()}.cbor"
        )
    
    if reference and os.path.exists(curr_ref_smiles_path):
        curr_df = pd.read_csv(curr_ref_smiles_path, header=None, names=["SMILES"])
        orig_df = pd.read_csv(orig_ref_smiles_path)
        if not curr_df.equals(orig_df):
            shutil.rmtree(Path(working_dir) / "reference")
