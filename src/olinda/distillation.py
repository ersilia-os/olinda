"""Distillation module."""

import os
from pathlib import Path
import shutil
from typing import Any, Optional

from cbor2 import dump
import joblib
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm

from olinda.data import ReferenceSmilesDM, FeaturizedSmilesDM, GenericOutputDM
from olinda.featurizer import Featurizer
from olinda.generic_model import GenericModel
from olinda.utils import calculate_stop_step


def distill(
    model: Any,
    featurizer: Optional[Featurizer],
    working_dir: Path,
    clean: bool = False,
    reference_smiles_dm: Optional[ReferenceSmilesDM] = None,
    featurized_smiles_dm: Optional[FeaturizedSmilesDM] = None,
) -> pl.LightningModule:
    """Distill models.

    Args:
        model (Any): Teacher Model.
        featurizer (Optional[Featurizer]): Featurizer to use.
        working_dir (Path): Path to model workspace directory.
        clean (bool): Clean workspace before starting.
        reference_smiles_dm (Optional[ReferenceSmilesDM]): Reference SMILES datamodules.
        featurized_smiles_dm (Optional[FeaturizedSmilesDM]): Reference Featurized SMILES datamodules.

    Returns:
        pl.LightningModule: Student Model.
    """

    # Convert model to a generic model
    model = GenericModel(model)

    # Prepare reference smiles datamodule
    if reference_smiles_dm is None:
        reference_smiles_dm = ReferenceSmilesDM()
    reference_smiles_dm.prepare_data()
    reference_smiles_dm.setup("train")

    # Generate student model training dataset
    student_training_dm = gen_training_dataset(
        model, featurizer, reference_smiles_dm, featurized_smiles_dm, working_dir, clean
    )

    # Select and Train student model
    student_model = pl.LightningModule()
    trainer = pl.Trainer()
    trainer.fit(student_model, student_training_dm)

    return student_model


def gen_training_dataset(
    model: pl.LightningModule,
    featurizer: Featurizer,
    reference_smiles_dm: pl.LightningDataModule,
    featurized_smiles_dm: Optional[pl.LightningDataModule],
    working_dir: Path,
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
            reference_smiles_dm, featurizer, working_dir, clean
        )

    # Generate model outputs and save to a file
    model_output_dm = gen_model_output(featurized_smiles_dm, model, working_dir, clean)

    return model_output_dm


def gen_featurized_smiles(
    reference_smiles_dm: pl.LightningDataModule,
    featurizer: Featurizer,
    working_dir: Path,
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
    # Extract transformer from the featurizer
    transformer = featurizer.transformer

    if clean is True:
        clean_workspace(working_dir, featurizer=featurizer)
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
        Path(working_dir / "reference" / "reference_smiles_dl.joblib"),
    )

    # calculate stop_step
    try:
        with open(
            working_dir / "reference" / f"featurized_smiles_{featurizer.name}.cbor",
            "rb",
        ) as feature_stream:
            stop_step = calculate_stop_step(feature_stream)
    except Exception:
        stop_step = 0

    with open(
        working_dir / "reference" / f"featurized_smiles_{featurizer.name}.cbor", "wb"
    ) as feature_stream:
        for i, batch in tqdm(enumerate((reference_smiles_dl))):
            if i < stop_step:
                continue
            output = transformer.transform(batch[1])
            for j, elem in enumerate(batch):
                dump((elem, batch[1][j], output[j]), feature_stream)

    featurized_smiles_dm = FeaturizedSmilesDM(Path(working_dir / "reference"))
    return featurized_smiles_dm


def gen_model_output(
    featurized_smiles_dm: DataLoader,
    model: GenericModel,
    working_dir: Path,
    clean: bool = False,
) -> pl.LightningDataModule:
    """Generate featurized smiles representation dataset.

    Args:
        featurized_smiles_dm (DataLoader): Featurized SMILES to use as inputs.
        model (GenericModel): Wrapped Teacher model.
        working_dir (Path): Path to model workspace directory.
        clean (bool): Clean workspace before starting.

    Returns:
        pl.LightningDataModule: Dateset with featurized smiles.
    """
    if clean is True:
        clean_workspace(working_dir, model=model)
        featurized_smiles_dl = featurized_smiles_dm.train_dataloader()
    else:
        featurized_smiles_dl = joblib.load(
            Path(working_dir / str(model) / "featurized_smiles_dl.joblib")
        )

    # Save dataloader for resuming
    joblib.dump(
        featurized_smiles_dl,
        Path(working_dir / str(model) / "featurized_smiles_dl.joblib"),
    )

    # calculate stop step
    with open(working_dir / "reference" / "model_output.cbor", "rb") as output_stream:
        stop_step = calculate_stop_step(output_stream)

    with open(working_dir / "reference" / "model_output.cbor", "wb") as output_stream:
        for i, batch in tqdm(enumerate(iter(featurized_smiles_dm))):
            if i < stop_step:
                continue
            output = model(batch[0])
            for j, elem in enumerate(batch[0]):
                dump((elem, batch[1][j], output[j]), output_stream)

    model_output_dm = GenericOutputDM(Path(working_dir / str(model)))
    return model_output_dm


def clean_workspace(
    working_dir: Path, model: GenericModel, featurizer: Featurizer
) -> None:
    """Clean workspace.

    Args:
        working_dir (Path): Path of the working directory.
        model (GenericModel): Wrapped Teacher model.
        featurizer (Featurizer): Featurizer to use.
    """

    if model:
        shutil.rmtree(working_dir / str(model), ignore_errors=True)
        os.makedirs(working_dir / str(model), exist_ok=True)

    if featurizer:
        os.remove(Path(working_dir / "reference" / "reference_smiles_dl.joblib"))
        os.remove(
            Path(working_dir / "reference" / f"featurized_smiles_{featurizer}.cbor")
        )
