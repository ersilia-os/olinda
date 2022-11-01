"""Generic datamodule for teacher model output and featurized inputs."""

from pathlib import Path
from typing import Any, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import webdataset as wds

from olinda.utils import calculate_cbor_size


class Segmenter:
    def __init__(self, only_X: bool, only_Y: bool) -> None:
        self.only_X = only_X
        self.only_Y = only_Y

    def segment_dataset(self, iterator: Any) -> Any:
        """Segment dataset."""
        for sample in iterator:
            _, _, featurized_smile, output = sample
            if self.only_X:
                yield featurized_smile
            elif self.only_Y:
                yield output
            else:
                yield sample


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

    def setup(
        self: "GenericOutputDM",
        stage: Optional[str],
        only_X: bool = False,
        only_Y: bool = False,
        batched: bool = True,
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
            self.train_dataset_size = dataset_size
            shuffle = 5000
        elif stage == "val":
            self.val_dataset_size = dataset_size // 10
            shuffle = None

        self.dataset = wds.DataPipeline(
            wds.SimpleShardList(
                str((Path(self.model_dir) / "model_output.cbor").absolute())
            ),
            wds.cbors2_to_samples(),
            Segmenter(only_X, only_Y).segment_dataset,
            wds.shuffle(shuffle),
        )
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
