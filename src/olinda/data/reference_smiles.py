"""Reference SMILES datamodule."""

from pathlib import Path
from typing import Any, Optional, Union
import shutil
import os

from cbor2 import dump
import pandas as pd
import pytorch_lightning as pl
import requests
from torch.utils.data import DataLoader
from tqdm import tqdm
import webdataset as wds

from olinda.utils import get_workspace_path


class ReferenceSmilesDM(pl.LightningDataModule):
    """Reference SMILES datamodule."""

    def __init__(
        self: "ReferenceSmilesDM",
        num_data: int,
        workspace: Union[str, Path] = None,
        batch_size: int = 32,
        num_workers: int = 1,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ) -> None:
        """Init.

        Args:
            workspace (Union[str, Path]): URLs or Path to the data files.
                Defaults to local workspace.
            batch_size (int): batch size. Defaults to 32.
            num_workers (int): workers to use. Defaults to 2.
            transform (Optional[Any]): Tranforms for the inputs. Defaults to None.
            target_transform (Optional[Any]): Transforms for the target. Defaults to None.
        """
        super().__init__()
        self.workspace = workspace or get_workspace_path()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.target_transform = target_transform
        self.num_data = num_data

    def prepare_data(self: "ReferenceSmilesDM") -> None:
        """Prepare data."""
        # Check if data files already present
        if (
            Path(self.workspace / "reference" / "reference_smiles.csv").is_file()
            is False
        ):
            ref_path = os.path.join(os.path.expanduser("~"), "olinda", "precalculated_descriptors", "olinda_reference_library.csv")
            # check if reference files not already present
            if os.path.exists(Path(self.workspace / "reference" / "reference_smiles.csv")) == False or os.path.getsize(ref_path) != os.path.getsize(ref_path):
                df = pd.read_csv(ref_path)
                df.to_csv(Path(self.workspace / "reference" / "reference_smiles.csv"), header=False, index=False)
            
        # Check if processed data files already present
        if (
            Path(self.workspace / "reference" / "reference_smiles.cbor").is_file()
            is False
            or Path(
                self.workspace / "reference" / "reference_smiles_truncated.cbor"
            ).is_file()
            is False
        ):
            # preprocess csv into a cbor file
            df = pd.read_csv(self.workspace / "reference" / "reference_smiles.csv", header=None)
            truncated_df = df.iloc[:self.num_data]
            with open(
                self.workspace / "reference" / "reference_smiles.cbor", "wb"
            ) as stream:
                for i, row in tqdm(
                    df.iterrows(),
                    total=df.shape[0],
                    desc="Creating reference smiles dataset",
                ):
                    dump((i, str(row.to_list()[0])), stream)
            with open(
                self.workspace / "reference" / "reference_smiles_truncated.cbor", "wb"
            ) as stream:
                for i, row in tqdm(
                    truncated_df.iterrows(),
                    total=truncated_df.shape[0],
                    desc="Creating truncated reference smiles dataset",
                ):
                    dump((i, str(row.to_list()[0])), stream)

    def setup(self: "ReferenceSmilesDM", stage: Optional[str] = "train") -> None:
        """Setup dataloaders.

        Args:
            stage (Optional[str]): Optional pipeline state
        """
        if stage == "train":
            self.dataset_size = self.num_data
            #shuffle = 5000
            self.dataset = wds.DataPipeline(
                wds.SimpleShardList(
                    str(
                        (
                            self.workspace / "reference" / "reference_smiles.cbor"
                        ).absolute()
                    )
                ),
                wds.cbors2_to_samples(),
                #wds.shuffle(shuffle),
                wds.batched(self.batch_size, partial=False),
            )
        elif stage == "val":
            self.dataset_size = self.num_data//10
            #shuffle = 5000
            self.dataset = wds.DataPipeline(
                wds.SimpleShardList(
                    str(
                        (
                            self.workspace
                            / "reference"
                            / "reference_smiles_truncated.cbor"
                        ).absolute()
                    )
                ),
                wds.cbors2_to_samples(),
                #wds.shuffle(shuffle),
                wds.batched(self.batch_size, partial=False),
            )

    def train_dataloader(self: "ReferenceSmilesDM") -> DataLoader:
        """Train dataloader.

        Returns:
            DataLoader: train dataloader
        """
        loader = wds.WebLoader(
            self.dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
        )

        loader.length = (self.dataset_size * self.num_workers) // self.batch_size

        return loader

    def val_dataloader(self: "ReferenceSmilesDM") -> DataLoader:
        """Val dataloader.

        Returns:
            DataLoader: val dataloader
        """
        loader = wds.WebLoader(
            self.dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
        )

        loader.length = (self.dataset_size * self.num_workers) // self.batch_size

        return loader

    def teardown(self: "ReferenceSmilesDM", stage: Optional[str] = None) -> None:
        """Teardown of data module."""
        pass
