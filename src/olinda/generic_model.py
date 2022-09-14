"""A wrapper to standardize models."""

from typing import Any

import pytorch_lightning as pl
import torch.nn as nn

from olinda.models.base import DistillBaseModel


class GenericModel(DistillBaseModel):
    def __init__(self: "GenericModel", model: Any) -> None:
        """Init.

        Args:
            model (Any): Any ML model

        Raises:
            Exception : Unsupported Model
        """
        super().__init__()
        # Check type of model and convert to pytorch accordingly
        if issubclass(type(model), (pl.LightningModule, nn.Module)):
            self.nn = model
            self.name = type(model).__name__.lower()

        elif type(model) is str:
            # Download model from Ersilia hub and convert to pytorch
            pass
        else:
            raise Exception(f"Unsupported Model type: {type(model)}")

    def forward(self: "GenericModel", x: Any) -> Any:
        """Forward function.

        Args:
            x (Any): Input

        Returns:
            Any: Ouput
        """
        return self.nn(x)
