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
        only_Y: bool = True,
    ):
        self.only_X = only_X
        self.only_Y = only_Y

        datamodule.setup(stage, only_X, only_Y, batched=False)
        self.dataset = datamodule.dataset
        sample = next(iter(self.dataset))
        #sample = np.array(sample)

        if stage == "train":
            self.loader = datamodule.train_dataloader()
        elif stage == "val":
            self.loader = datamodule.val_dataloader()

        if only_X and only_Y:
            self.output_sign = (
                tf.TensorSpec(shape=np.array(sample[0]).shape, dtype=tf.float32),
                tf.TensorSpec(shape=np.array(sample[1]).shape, dtype=tf.float32),
            )
        else:
            self.output_sign = (tf.TensorSpec(shape=sample.shape, dtype=tf.float32),)

    def __iter__(self):
        for sample in self.loader:
            if self.only_X and self.only_Y:
                x, y = sample
                yield (np.array(x), np.array(y))
            else:
                yield np.array(sample)

    def __len__(self):
        return self.length

    def output_signature(self):
        return self.output_sign
