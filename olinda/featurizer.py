"""Morgan count fingerprints — the one descriptor olinda uses, everywhere."""

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

# Errors as well as warnings: an unparseable SMILES is an ordinary, expected input here, and every
# caller already reports it — `transform` returns an all-zero row, which the artifact turns into NaN and
# summarises in a single warning naming the count. RDKit's own multi-line complaint per molecule is
# written straight from C++ to stderr, so it cannot be captured or counted, and on a bad input file it
# buries the result it is describing.
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")


def featurizer_from_meta(class_name: str | None, cfg: dict):
  """Rebuild a featurizer from the class name and config recorded in a model's metadata.

  Used when loading a saved model or an ONNX bundle, where the featurizer must be reconstructed
  exactly as it was at training time.

  Parameters
  ----------
  class_name : str or None
      Recorded featurizer class; ``None`` means the model carries no featurizer.
  cfg : dict
      The featurizer's ``to_dict()`` payload.

  Returns
  -------
  MorganCountFeaturizer or None
      ``None`` when ``class_name`` is falsy or unrecognised.
  """
  import warnings

  if not class_name:
    return None
  if class_name == "MorganCountFeaturizer":
    return MorganCountFeaturizer.from_dict(cfg)
  # stdlib warnings, not the loguru logger: this is on the inference path, which must stay
  # importable with only numpy / pandas / rdkit / onnxruntime installed.
  warnings.warn(
    f"unknown featurizer class: {class_name} (CLAMP models are no longer supported)",
    RuntimeWarning,
    stacklevel=2,
  )
  return None


@dataclass(frozen=True)
class MorganCountFeaturizer:
  """Folded Morgan **count** fingerprint via RDKit's ``MorganGenerator`` (counts clipped).

  Reproduces the Ersilia model ``eos5axz`` and ``scripts/compute_morgan_fingerprints.py`` exactly
  (defaults: ``radius=3``, ``fp_size=2048``, counts clipped at 255), so a model trained on
  ``erl0_morgan.h5`` can predict from raw SMILES consistently.
  """

  radius: int = 3
  fp_size: int = 2048
  count_clip: int = 255
  name: str = "morgan_count"

  def transform(self, smiles: list[str]) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.fp_size)
    out = np.zeros((len(smiles), self.fp_size), dtype=np.float32)
    for row, smi in enumerate(smiles):
      mol = Chem.MolFromSmiles(str(smi))
      if mol is None:
        continue
      for i, c in gen.GetCountFingerprint(mol).GetNonzeroElements().items():
        out[row, i] = float(self.count_clip) if c > self.count_clip else float(c)
    return out

  def to_dict(self) -> dict:
    return {
      "radius": int(self.radius),
      "fp_size": int(self.fp_size),
      "count_clip": int(self.count_clip),
      "name": self.name,
    }

  @classmethod
  def from_dict(cls, d: dict):
    return cls(
      radius=int(d.get("radius", 3)),
      fp_size=int(d.get("fp_size", 2048)),
      count_clip=int(d.get("count_clip", 255)),
      name=d.get("name", "morgan_count"),
    )
