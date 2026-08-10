"""Serve a fused ``model.onnx`` bundle: featurize (RDKit) then run the single self-describing ONNX graph.

The bundle (built by :func:`olinda.export.build_bundle`) is ONE ``model.onnx`` whose ``metadata_props`` carry
the Morgan featurizer config (+ RDKit version), whether a hard head is present, and the output names. This
module reads that metadata, rebuilds the featurizer, and runs the graph in a single ``session.run``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_NAME = "model.onnx"


class RDKitVersionMismatch(RuntimeError):
  """Raised when the installed RDKit differs from the build that produced ``model.onnx``."""


def _check_rdkit_version(meta: dict) -> None:
  """Verify the installed RDKit matches the one recorded in ``model.onnx`` metadata.

  Morgan fingerprints are only bit-for-bit reproducible with the exact RDKit build the model was fused
  against, so a mismatch would silently corrupt every prediction — we refuse rather than guess.
  """
  want = (meta.get("featurizer") or {}).get("rdkit_version")
  if not want:
    return
  import rdkit

  have = rdkit.__version__
  if have != want:
    raise RDKitVersionMismatch(
      f"model.onnx was built with RDKit {want}, but {have} is installed — Morgan fingerprints are only "
      f"reproducible with the exact build. Install rdkit=={want} (e.g. `pip install rdkit=={want}`)."
    )


class OnnxPipeline:
  """Single-session execution of a fused ``model.onnx`` (featurizer + one ONNX graph)."""

  def __init__(self, session, featurizer, meta: dict):
    self._session = session
    self.featurizer = featurizer
    self.meta = meta
    self._out_names = [o.name for o in session.get_outputs()]

  @staticmethod
  def is_bundle(model_dir: str | Path) -> bool:
    """True iff *model_dir* holds a fused ``model.onnx``."""
    return (Path(model_dir) / MODEL_NAME).exists()

  @classmethod
  def load(cls, model_dir: str | Path) -> "OnnxPipeline":
    """Open ``model.onnx`` and rebuild the featurizer from its embedded metadata."""
    import onnxruntime as ort

    from olinda.featurizer import featurizer_from_meta

    model_dir = Path(model_dir)
    session = ort.InferenceSession((model_dir / MODEL_NAME).read_bytes(), providers=["CPUExecutionProvider"])
    raw = session.get_modelmeta().custom_metadata_map.get("olinda")
    if not raw:
      raise ValueError(f"{model_dir / MODEL_NAME} has no 'olinda' metadata — rebuild with `olinda export`")
    meta = json.loads(raw)
    _check_rdkit_version(meta)
    featurizer = featurizer_from_meta(meta.get("featurizer_class"), meta.get("featurizer", {}))
    return cls(session, featurizer, meta)

  def predict_channels(self, smiles: list[str]) -> dict:
    """Featurize *smiles* and run the graph; return the model's named outputs as a dict of arrays."""
    fp = self.featurizer.transform([str(s) for s in smiles]).astype(np.float32)
    outs = self._session.run(None, {self._session.get_inputs()[0].name: fp})
    return {name: np.asarray(v).ravel() for name, v in zip(self._out_names, outs)}

  def predict(self, smiles: list[str]) -> np.ndarray:
    """Return the headline ``prediction`` for a list of SMILES."""
    return self.predict_channels(smiles)["prediction"]
