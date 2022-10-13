"""Model Tuner."""

from abc import ABC, abstractmethod
from random import random

import autokeras as ak

from olinda.data import GenericOutputDM
from olinda.generic_model import GenericModel


class ModelTuner(ABC):
    """Automatic model tuner."""

    @abstractmethod
    def fit(self: "ModelTuner", dataset: GenericOutputDM) -> GenericModel:
        """Fit an optimal model using the given dataset.

        Args:
            dataset (GenericOutputDM): Dataset to fit an optimal model.

        Returns:
            GenericModel : Student model as wrapped in a generic model class.
        """
        pass


class AutoKerasTuner(ModelTuner):
    """AutoKeras based model tuner."""

    def __init__(self: "AutoKerasTuner", max_trials: int = 3) -> None:
        """Initialize model tuner.

        Args:
            max_trials (int): Maximum interations to perform.
        """
        self.max_trials = max_trials

    def fit(self: "AutoKerasTuner", dataset: GenericOutputDM) -> None:
        """Fit an optimal model using the given dataset.

        Args:
            dataset (GenericOutputDM): Dataset to fit an optimal model.

        Returns:
            GenericModel : Student model as wrapped in a generic model class.
        """
        self.mdl = ak.StructuredDataRegressor(
            overwrite=False,
            max_trials=self.max_trials,
            project_name=f"autokeras-{random()*1000}",
        )
        self.X = dataset.dataset[0]
        self.Y = dataset.dataset[1]
        self.mdl.fit(self.X, self.Y)
        return GenericModel(self.mdl.export_model())


class KerasTuner(ModelTuner):
    """Keras tuner based model tuner."""

    def __init__(self: "KerasTuner", max_trials: int = 3) -> None:
        """Initialize model tuner.

        Args:
            max_trials (int): Maximum interations to perform.
        """
        self.max_trials = max_trials

    def fit(self: "KerasTuner", dataset: GenericOutputDM) -> None:
        """Fit an optimal model using the given dataset.

        Args:
            dataset (GenericOutputDM): Dataset to fit an optimal model.

        Returns:
            GenericModel : Student model as wrapped in a generic model class.
        """
        pass
