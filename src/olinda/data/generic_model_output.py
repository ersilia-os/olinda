"""Generic datamodule for teacher model output and featurized inputs."""

from pathlib import Path
from typing import Any, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import webdataset as wds


class GenericOutputDM(pl.LightningDataModule):
    """Generic teacher model output datamodule."""

    def __init__(
        self: "GenericOutputDM",
        model_dir: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 2,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
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

    def setup(self: "GenericOutputDM", stage: Optional[str]) -> None:
        """Setup dataloaders.

        Args:
            stage (Optional[str]): Optional pipeline state
        """
        if stage == "train":
            self.train_dataset_size = 1999380
            shuffle = 5000
        elif stage == "val":
            self.val_dataset_size = 20000
            shuffle = None

        self.dataset = wds.DataPipeline(
            wds.SimpleShardList(
                str((Path(self.model_dir) / "model_output.cbor").absolute())
            ),
            wds.cbors2_to_samples(),
            wds.shuffle(shuffle),
            wds.batched(self.batch_size, partial=False),
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

        loader.length = self.train_dataset_size // self.batch_size

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

        loader.length = self.val_dataset_size // self.batch_size

        return loader

    def teardown(self: "GenericOutputDM", stage: Optional[str] = None) -> None:
        """Teardown of data module."""
        pass
