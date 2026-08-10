"""The public inference API: load a distilled ``model.onnx`` and predict from SMILES.

An olinda artifact is a single self-describing file. Everything needed to run it — the Morgan
featurizer configuration, the RDKit build it was fused against, the task names and their outputs —
travels inside the file's ``metadata_props``, so the ``.onnx`` is the only input required:

    from olinda import OnnxArtifact

    model = OnnxArtifact("model.onnx")
    model.columns          # ['abaumannii_inhibition_probability', ...]
    model.trained_at       # '2026-08-11T09:14:00+00:00'
    df = model.run(["CCO", "c1ccccc1"])   # DataFrame: smiles + one column per task
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from olinda.onnx_pipeline import MODEL_NAME, RDKitVersionMismatch as RDKitVersionMismatch
from olinda.onnx_pipeline import _check_rdkit_version

_BATCH = 4096


class OnnxArtifact:
  """A distilled olinda model, loaded from its ``model.onnx``.

  Parameters
  ----------
  path : str or Path
      The ``model.onnx`` itself, or a run directory containing one.
  check_rdkit : bool, optional
      Verify the installed RDKit matches the build the model was fused against (default ``True``).
      Morgan fingerprints only reproduce bit-for-bit on the exact build, so disabling this can
      silently change every prediction.

  Attributes
  ----------
  columns : list of str
      The task names this model predicts, in output order.
  path : Path
      Where the artifact was loaded from.
  """

  def __init__(self, path: str | Path, *, check_rdkit: bool = True) -> None:
    import onnxruntime as ort

    from olinda.featurizer import featurizer_from_meta

    path = Path(path)
    if path.is_dir():
      path = path / MODEL_NAME
    if not path.exists():
      raise FileNotFoundError(f"no olinda artifact at {path}")

    self.path = path
    options = ort.SessionOptions()
    options.log_severity_level = 3
    self._session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    raw = self._session.get_modelmeta().custom_metadata_map.get("olinda")
    if not raw:
      raise ValueError(
        f"{path} carries no olinda metadata — it may not be an olinda artifact, or it predates "
        "self-describing bundles. Rebuild it with `olinda export`."
      )
    self.metadata = json.loads(raw)
    if check_rdkit:
      _check_rdkit_version(self.metadata)

    self._featurizer = featurizer_from_meta(
      self.metadata.get("featurizer_class"), self.metadata.get("featurizer", {})
    )
    if self._featurizer is None:
      raise ValueError(f"{path} records an unsupported featurizer: {self.metadata.get('featurizer_class')}")
    self._input_name = self._session.get_inputs()[0].name
    self._output_names = [o.name for o in self._session.get_outputs()]

    # A single-task model is just the one-column case, so normalise here and let everything
    # downstream treat every artifact identically. The fallback covers bundles fused before
    # columns were recorded, which named their blended output "prediction".
    self._columns = self.metadata.get("columns") or [
      {
        "name": "prediction",
        "output": "prediction",
        "has_hard": bool(self.metadata.get("has_hard")),
      }
    ]

  # ── what the artifact says about itself ────────────────────────────────────

  @property
  def columns(self) -> list[str]:
    """The task names this model predicts, in output order."""
    return [c["name"] for c in self._columns]

  @property
  def n_columns(self) -> int:
    return len(self._columns)

  @property
  def trained_at(self) -> str | None:
    """UTC timestamp of the fuse that produced this artifact."""
    return self.metadata.get("trained_at")

  @property
  def olinda_version(self) -> str | None:
    return self.metadata.get("olinda_version")

  @property
  def rdkit_version(self) -> str | None:
    """The RDKit build the featurizer must match."""
    return (self.metadata.get("featurizer") or {}).get("rdkit_version")

  @property
  def has_ground_truth(self) -> bool:
    """True if any task blends in a hard-label head, so predictions use measured data."""
    return any(c.get("has_hard") for c in self._columns)

  @property
  def n_features(self) -> int:
    return int((self.metadata.get("featurizer") or {}).get("fp_size", 2048))

  def describe(self) -> dict:
    """A compact summary of the artifact, suitable for logging or display."""
    return {
      "path": str(self.path),
      "producer": self.metadata.get("producer", "olinda"),
      "olinda_version": self.olinda_version,
      "trained_at": self.trained_at,
      "rdkit_version": self.rdkit_version,
      "n_features": self.n_features,
      "columns": self.columns,
      "has_ground_truth": self.has_ground_truth,
    }

  # ── inference ──────────────────────────────────────────────────────────────

  def featurize(self, smiles) -> np.ndarray:
    """Morgan count fingerprints for *smiles*, exactly as the model was trained on."""
    return self._featurizer.transform([str(s) for s in smiles]).astype(np.float32)

  def run_channels(self, smiles, batch_size: int = _BATCH) -> dict:
    """Every named output of the graph, as a dict of 1-D arrays keyed by output name.

    Useful for inspecting the pieces behind a blended prediction (the surrogate, the calibrated
    ground truth, and the applicability weight). Most callers want :meth:`run`.
    """
    smiles = [str(s) for s in smiles]
    chunks: list[dict] = []
    for start in range(0, len(smiles), batch_size):
      fp = self.featurize(smiles[start : start + batch_size])
      outs = self._session.run(None, {self._input_name: fp})
      chunks.append({n: np.asarray(v).ravel() for n, v in zip(self._output_names, outs)})
    if not chunks:
      return {n: np.array([], dtype=np.float64) for n in self._output_names}
    return {n: np.concatenate([c[n] for c in chunks]) for n in self._output_names}

  def run(self, smiles, batch_size: int = _BATCH):
    """Predict for a list of SMILES.

    Parameters
    ----------
    smiles : sequence of str
        The molecules to score.
    batch_size : int, optional
        Rows per forward pass. Bounds memory on large inputs; does not change results.

    Returns
    -------
    pandas.DataFrame
        A ``smiles`` column followed by **one column per task** — the final blended prediction,
        which already folds in the applicability weighting. The intermediate channels behind it
        are available from :meth:`run_channels`.
    """
    import pandas as pd

    smiles = [str(s) for s in smiles]
    channels = self.run_channels(smiles, batch_size=batch_size)
    values = {c["name"]: channels[c["output"]] for c in self._columns}
    return pd.DataFrame({"smiles": smiles, **values})

  def __len__(self) -> int:
    return self.n_columns

  def __repr__(self) -> str:
    return f"OnnxArtifact({self.path.name!r}, columns={self.columns}, trained_at={self.trained_at!r})"
