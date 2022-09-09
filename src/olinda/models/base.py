"""Distillation model base."""

from abc import ABC
from typing import Any

import pytorch_lightning as pl


class DistillBaseModel(ABC, pl.LightningModule):
    """Distillation model base."""

    def gen_training_dataset() -> Any:
        pass

    def to_onnx() -> Any:
        pass

    def to_tflite() -> Any:
        pass
