"""Generic datamodule for teacher model output and featurized inputs."""

from pathlib import Path
from typing import Any, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import webdataset as wds

from olinda.utils import calculate_cbor_size

import numpy as np
class Segmenter:
    def __init__(self, only_X: bool, only_Y: bool, weights: bool) -> None:
        self.only_X = only_X
        self.only_Y = only_Y
        self.weights = weights

    def segment_dataset(self, iterator: Any) -> Any:
        """Segment dataset."""
        for sample in iterator:
            _, _, featurized_smile, output, weight = sample
            if self.only_X and not self.only_Y:
                yield featurized_smile
            elif self.only_Y and not self.only_X:
                yield output
            elif self.only_X and self.only_Y and not self.weights:
                yield featurized_smile, output
            elif self.only_X and self.only_Y and self.weights:
                yield featurized_smile, output, weight
            else:
                yield sample


class GenericOutputDM(pl.LightningDataModule):
    """Generic teacher model output datamodule."""

    def __init__(
        self: "GenericOutputDM",
        model_dir: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 1,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        zaira_training_size = 0,
    ) -> None:
        """Init.

        Args:
            model_dir (Union[str, Path]): Path to the data files.
            batch_size (int): batch size. Defaults to 32.
            num_workers (int): workers to use. Defaults to 2.
            transform (Optional[Any]): Tranforms for the inputs. Defaults to None.
            target_transform (Optional[Any]): Transforms for the target. Defaults to None.
        """
        super().__init__()
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.target_transform = target_transform
        self.zaira_training_size = zaira_training_size

    def setup(
        self: "GenericOutputDM",
        stage: Optional[str],
        only_X: bool = False,
        only_Y: bool = False,
        weights: bool = True,
        batched: bool = True,
        smaller_set: bool = False,
    ) -> None:
        """Setup dataloaders.

        Args:
            stage (Optional[str]): Optional pipeline state
            only_X (bool): Returns only X part of the dataset
            only_Y (bool): Returns only Y part of the dataset
            batched (bool): Create batches
        """
        
        # Check if data files are available
        file_path = Path(self.model_dir) / "model_output.cbor"
        if file_path.is_file() is not True:
            raise Exception(f"Data file not available at {file_path.absolute()}")

        with open(file_path, "rb") as fp:
            dataset_size = calculate_cbor_size(fp)
        
        if stage == "train":
            if smaller_set:
                self.train_dataset_size = dataset_size // 10
            else:
                self.train_dataset_size = dataset_size
            shuffle = 5000
            self.dataset = wds.DataPipeline(
            wds.SimpleShardList(
                str((Path(self.model_dir) / "model_output.cbor").absolute())
                ),
                wds.cbors2_to_samples(),
                Segmenter(only_X, only_Y, weights).segment_dataset,
                wds.shuffle(shuffle),
                )
            self.dataset.with_epoch(self.train_dataset_size)
        elif stage == "val":
            if self.zaira_training_size >= 0:
                self.val_dataset_size = self.zaira_training_size
            else:
                self.val_dataset_size = dataset_size // 10
            
            if smaller_set:
                self.val_dataset_size = self.val_dataset_size // 2
                
            self.dataset = wds.DataPipeline(
            wds.SimpleShardList(
                str((Path(self.model_dir) / "model_output.cbor").absolute())
                ),
                wds.cbors2_to_samples(),
                Segmenter(only_X, only_Y, weights).segment_dataset,
            )
            self.dataset.with_epoch(self.val_dataset_size)
        if batched:
            self.dataset = self.dataset.compose(
                wds.batched(self.batch_size, partial=False)
            )

    def train_dataloader(self: "GenericOutputDM") -> DataLoader:
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

        loader.length = (self.train_dataset_size * self.num_workers) // self.batch_size
        
        return loader

    def val_dataloader(self: "GenericOutputDM") -> DataLoader:
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

        loader.length = (self.val_dataset_size * self.num_workers) // self.batch_size

        return loader

    def teardown(self: "GenericOutputDM", stage: Optional[str] = None) -> None:
        """Teardown of data module."""
        pass
