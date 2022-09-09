"""A wrapper to standardize models."""

from typing import Any, Union

import pytorch_lightning as pl
from torch.nn import Module

from olinda.models.base import DistillBaseModel


class GenericModel(DistillBaseModel):
    def __init__(self: "GenericModel", model: Any) -> None:
        """Init.

        Args:
            model (Any): Any ML model

        Raises:
            Exception : Unsupported Model
        """
        # Check type of model and convert to pytorch accordingly
        if type(model) is Union[pl.LightningModule, Module]:
            self._model = model
        elif type(model) is str:
            # Download model from Ersilia hub and convert to pytorch
            pass
        else:
            raise Exception("Unsupported Model")

    def forward(self: "GenericModel", x: Any) -> Any:
        """Forward function.

        Args:
            x (Any): Input

        Returns:
            Any: Ouput
        """
        return self._model(x)
