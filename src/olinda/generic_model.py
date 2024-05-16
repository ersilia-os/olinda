"""A wrapper to standardize models."""

from typing import Any

import pytorch_lightning as pl
import tensorflow as tf
import torch.nn as nn

import onnx
import os
import pickle

from olinda.models.base import DistillBaseModel
from olinda.utils import run_ersilia_api_in_context, run_onnx_runtime


class GenericModel(DistillBaseModel):
    def __init__(self: "GenericModel", model: Any) -> None:
        """Init.

        Args:
            model (Any): Any ML model

        Raises:
            Exception : Unsupported Model
        """
        super().__init__()
        # Check type of model and convert accordingly
        if issubclass(type(model), (pl.LightningModule, nn.Module)):
            self.nn = model
            self.type = "pytorch"
            self.name = type(model).__name__.lower()

        elif issubclass(type(model), (tf.keras.Model)):
            self.nn = model
            self.type = "tensorflow"
            self.name = type(model).__name__.lower()
        
        elif issubclass(type(model), (onnx.onnx_ml_pb2.ModelProto)):
            self.nn = run_onnx_runtime(model)
            self.type = "onnx"
            self.name = type(model).__name__.lower()
            self.model = model

        elif type(model) is str:
            self.nn = run_ersilia_api_in_context(model)
            self.type = "ersilia"
            self.name = type(model).__name__.lower()

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
    
    def save(self: "GenericModel", path: str) -> None:
        if self.type == "onnx":
            onnx.save(self.model, os.path.join(path))
        else:
            raise Exception(f"Cannot save non-ONNX model")
    
    def save_featurizer(self, featurizer: Any, path: str) -> None:
        with open(os.path.join(path), "wb") as file_out:
            pickle.dump(featurizer, file_out)
