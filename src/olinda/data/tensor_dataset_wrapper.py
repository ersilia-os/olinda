"""A wapper to provide a tensorflow dataset interface."""

import tensorflow as tf
import numpy as np

from olinda.data import GenericOutputDM


class TensorflowDatasetWrapper:
    """A wapper to provide a tensorflow dataset interface."""

    def __init__(
        self,
        datamodule: GenericOutputDM,
        stage: str = "train",
        only_X: bool = True,
        only_Y: bool = False,
    ):
        if only_X:
            datamodule.setup(stage, only_X=True, batched=False)
            self.dataset = datamodule.dataset
            sample = next(iter(self.dataset))
            sample = np.array(sample)
            self.output_shape = (32, 32)
        elif only_Y:
            datamodule.setup(stage, only_Y=True, batched=False)
            self.dataset = datamodule.dataset
            sample = next(iter(self.dataset))
            sample = np.array(sample)
            self.output_shape = sample.shape
        else:
            raise Exception("Select either only X or only Y")
        if stage == "train":
            self.loader = datamodule.train_dataloader()
        elif stage == "val":
            self.loader = datamodule.val_dataloader()

    def __iter__(self):
        for sample in self.loader:
            yield np.array(sample).astype("float32")

    def __len__(self):
        return self.length

    def output_shapes(self):
        return (self.output_shape, (len(self.loader.length),))

    def output_types(self):
        return (tf.float32,)
