from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class Featurizer(Protocol):
  name: str

  def transform(self, smiles: list[str]) -> np.ndarray: ...


@runtime_checkable
class Teacher(Protocol):
  task: str

  def predict(self, smiles: list[str]) -> np.ndarray: ...

  def predict_proba(self, smiles: list[str]) -> np.ndarray: ...
