"""The public inference API: load a distilled ``model.onnx`` and predict from SMILES.

An olinda artifact is a single self-describing file. Everything needed to run it — the Morgan
featurizer configuration, the RDKit build it was fused against, the task names and their outputs —
travels inside the file's ``metadata_props``, so the ``.onnx`` is the only input required:

    from olinda import OlindaArtifact

    model = OlindaArtifact("model.onnx")
    model.columns          # ['abaumannii_inhibition_probability', ...]
    model.trained_at       # '2026-08-11T09:14:00+00:00'
    df = model.run(["CCO", "c1ccccc1"])   # DataFrame: smiles + one column per task
"""

from __future__ import annotations

import json
import sys
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


_BATCH = 4096


def _as_smiles_list(smiles) -> list[str]:
  """Normalise user input to a list of SMILES strings, refusing the shapes that would mislead.

  A bare ``"CCO"`` is the trap worth guarding: a string is a perfectly good sequence of characters, so
  it would featurize as three one-atom molecules and return three rows without complaint. Being handed
  one molecule instead of a list is a natural mistake, and a wrong answer is a far worse outcome than
  an exception.
  """
  if isinstance(smiles, (str, bytes)):
    raise TypeError(
      f"expected a sequence of SMILES, got a single string {smiles!r:.40} — iterating it would score "
      f"each character as a molecule. Pass a list: [{smiles!r:.40}]"
    )
    # (a set would scramble the row order against the caller's input, so it is not accepted either)
  if isinstance(smiles, (dict, set, frozenset)):
    raise TypeError(f"expected an ordered sequence of SMILES, got {type(smiles).__name__} — order matters")
  try:
    return [str(s) for s in smiles]
  except TypeError as exc:
    raise TypeError(f"expected a sequence of SMILES, got {type(smiles).__name__}") from exc


class _Progress:
  """A minimal stderr progress bar.

  Hand-rolled rather than reaching for rich or tqdm: the inference install is deliberately limited to
  numpy, pandas, rdkit and onnxruntime, and a progress bar is not worth widening that. Silent unless
  stderr is a terminal, so redirected output and notebooks stay clean.
  """

  WIDTH = 28

  def __init__(self, total: int, label: str = "olinda", enabled: bool | None = None) -> None:
    self.total = max(1, int(total))
    self.label = label
    if enabled is None:
      enabled = sys.stderr.isatty() and total > _BATCH
    self.enabled = bool(enabled)
    self._done = 0

  def advance(self, n: int) -> None:
    if not self.enabled:
      return
    self._done = min(self.total, self._done + n)
    frac = self._done / self.total
    filled = int(round(frac * self.WIDTH))
    bar = "█" * filled + "░" * (self.WIDTH - filled)
    sys.stderr.write(f"\r  {self.label} ▕{bar}▏ {frac:>4.0%} · {self._done:,}/{self.total:,}")
    sys.stderr.flush()

  def close(self) -> None:
    if self.enabled:
      sys.stderr.write("\r" + " " * (self.WIDTH + 40) + "\r")
      sys.stderr.flush()

  def __enter__(self) -> _Progress:
    return self

  def __exit__(self, *exc) -> None:
    self.close()


class OlindaArtifact:
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

    # Fail here rather than with a bare KeyError deep inside run() if the embedded column list and
    # the graph disagree — which is what a hand-edited or mismatched artifact looks like.
    missing = [c["output"] for c in self._columns if c["output"] not in self._output_names]
    if missing:
      raise ValueError(
        f"{path} declares output(s) {missing} that the graph does not produce; "
        f"it exposes {self._output_names}. Rebuild it with `olinda export`."
      )

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

  def channels_for(self, task: str) -> dict:
    """``{role: output name}`` for one task's internal channels, empty if it has none.

    A blended column exposes ``surrogate``, ``ground_truth``, ``ground_truth_soft`` and
    ``applicability`` alongside its prediction. A soft-only column exposes none — its surrogate *is*
    its prediction. Artifacts fused before these were declared also return empty, which is why
    callers should ask rather than build the names themselves.
    """
    for c in self._columns:
      if c["name"] == task:
        return dict(c.get("channels") or {})
    raise KeyError(f"{task!r} is not a task of this model; it predicts {self.columns}")

  @property
  def has_ground_truth(self) -> bool:
    """True if any task blends in a hard-label head, so predictions use measured data."""
    return any(c.get("has_hard") for c in self._columns)

  @property
  def n_features(self) -> int:
    return int((self.metadata.get("featurizer") or {}).get("fp_size", 2048))

  def to_json(self, indent: int = 2) -> str:
    """Everything the file records about itself, as a JSON string.

    This is the raw metadata block, not the curated view :meth:`describe` returns — useful for
    inspecting or logging exactly what was embedded at fuse time.
    """
    return json.dumps(self.metadata, indent=indent, sort_keys=False, default=str)

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
    # The featurizer already allocates float32, so this is a view rather than a copy.
    return np.asarray(self._featurizer.transform([str(s) for s in smiles]), dtype=np.float32)

  def run_channels(self, smiles, batch_size: int = _BATCH, progress: bool | None = None) -> dict:
    """Every named output of the graph, as a dict of 1-D arrays keyed by output name.

    Most callers want :meth:`run`, which is this plus a DataFrame. A blended column also declares its
    intermediate channels as graph outputs — the surrogate, the calibrated ground truth and the
    applicability weight — so they appear here too, under the names :meth:`channels_for` reports.
    Ask for them by that route rather than assembling the names: a soft-only column has none, and so
    does an artifact fused before the channels were declared.

    ``progress`` shows a bar on stderr; the default shows one only for inputs larger than a single
    batch, and only when stderr is a terminal.

    Molecules RDKit cannot parse yield ``NaN`` in every channel rather than a number. Their
    fingerprint is all-zero, which the graph happily scores — the trees take every "bit absent"
    branch and the gate's first layer sees nothing but its own bias — so an unparseable input would
    otherwise come back looking like a confident prediction.
    """
    import warnings

    smiles = _as_smiles_list(smiles)
    batch_size = int(batch_size)
    if batch_size < 1:
      raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    chunks: list[dict] = []
    n_invalid = 0
    with _Progress(len(smiles), enabled=progress) as bar:
      for start in range(0, len(smiles), batch_size):
        batch = smiles[start : start + batch_size]
        fp = self.featurize(batch)
        invalid = fp.sum(axis=1) == 0
        n_invalid += int(invalid.sum())
        outs = self._session.run(None, {self._input_name: fp})
        block = {n: np.asarray(v, dtype=np.float64).ravel() for n, v in zip(self._output_names, outs)}
        if invalid.any():
          for values in block.values():
            values[invalid] = np.nan
        chunks.append(block)
        bar.advance(len(batch))

    if n_invalid:
      warnings.warn(
        f"{n_invalid} of {len(smiles)} input SMILES could not be parsed; their predictions are NaN",
        RuntimeWarning,
        stacklevel=2,
      )
    if not chunks:
      return {n: np.array([], dtype=np.float64) for n in self._output_names}
    return {n: np.concatenate([c[n] for c in chunks]) for n in self._output_names}

  def run(self, smiles, batch_size: int = _BATCH, progress: bool | None = None):
    """Predict for a list of SMILES.

    Parameters
    ----------
    smiles : sequence of str
        The molecules to score.
    batch_size : int, optional
        Rows per forward pass. Bounds memory on large inputs; does not change results.
    progress : bool, optional
        Show a progress bar on stderr. Defaults to showing one for multi-batch inputs on a terminal.

    Returns
    -------
    pandas.DataFrame
        A ``smiles`` column followed by **one column per task** — the final blended prediction,
        which already folds in the applicability weighting. The intermediate channels behind it
        are available from :meth:`run_channels`.
    """
    import pandas as pd

    smiles = _as_smiles_list(smiles)
    channels = self.run_channels(smiles, batch_size=batch_size, progress=progress)  # already str
    values = {c["name"]: channels[c["output"]] for c in self._columns}
    return pd.DataFrame({"smiles": smiles, **values})

  def __len__(self) -> int:
    return self.n_columns

  def __repr__(self) -> str:
    return f"OlindaArtifact({self.path.name!r}, columns={self.columns}, trained_at={self.trained_at!r})"
