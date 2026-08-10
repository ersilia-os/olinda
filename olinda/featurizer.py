import numpy as np
from dataclasses import dataclass
from multiprocessing import Pool

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.Chem.rdmolops import FastFindRings

RDLogger.DisableLog("rdApp.warning")

# Below this many molecules, a process pool's fork/teardown cost dwarfs the featurization
# work, so we go serial. This matters most when transform() is called repeatedly on tiny
# inputs inside a hot loop (e.g. robustness perturbation), where a per-call pool is pathological.
_MIN_PARALLEL = 512


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
  Fingerprint or MorganCountFeaturizer or None
      ``None`` when ``class_name`` is falsy or unrecognised.
  """
  from olinda.helpers import logger

  if not class_name:
    return None
  if class_name == "Fingerprint":
    return Fingerprint.from_dict(cfg)
  if class_name == "MorganCountFeaturizer":
    return MorganCountFeaturizer.from_dict(cfg)
  logger.warning(f"unknown featurizer class: {class_name} (CLAMP models are no longer supported)")
  return None


def _ebv_to_numpy(ebv):
  return np.frombuffer(ebv.ToBitString().encode("utf-8"), dtype=np.uint8) - ord("0")


def _counts_dict_to_folded_vector(counts, fp_size):
  v = np.zeros(fp_size, dtype=np.float32)
  for k, c in counts.items():
    v[int(k) % fp_size] += float(c)
  return v


def _mol_to_fp_vector(mol, which, fp_size, radius):
  w = which.lower()

  if w == "morgan":
    fp = AllChem.GetMorganFingerprintAsBitVect(
      mol, radius, nBits=fp_size, useFeatures=False, useChirality=True
    )
    return _ebv_to_numpy(fp).astype(np.float32)

  if w == "ecfp4":
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_size, useFeatures=False, useChirality=True)
    return _ebv_to_numpy(fp).astype(np.float32)

  if w == "rdk":
    fp = Chem.RDKFingerprint(mol, fpSize=fp_size, maxPath=6)
    return _ebv_to_numpy(fp).astype(np.float32)

  if w == "pattern":
    fp = Chem.PatternFingerprint(mol, fpSize=fp_size)
    return _ebv_to_numpy(fp).astype(np.float32)

  if w == "morganc":
    counts = AllChem.GetMorganFingerprint(
      mol, radius, useChirality=True, useBondTypes=True, useFeatures=True, useCounts=True
    ).GetNonzeroElements()
    return _counts_dict_to_folded_vector(counts, fp_size)

  if w == "rdkc":
    counts = AllChem.UnfoldedRDKFingerprintCountBased(mol, maxPath=6).GetNonzeroElements()
    return _counts_dict_to_folded_vector(counts, fp_size)

  raise ValueError(
    f"Unsupported which='{which}'. Supported: "
    f"'morgan', 'ecfp4', 'rdk', 'pattern', 'morganc', 'rdkc', and composites with '+' or '*'."
  )


def _smiles_to_fp(smi, fp_size, radius, is_smarts, which, sanitize):
  if is_smarts:
    mol = Chem.MolFromSmarts(str(smi), mergeHs=False)
  else:
    mol = Chem.MolFromSmiles(str(smi), sanitize=False)

  if mol is None:
    return np.zeros(fp_size, dtype=np.float32)

  if sanitize:
    Chem.SanitizeMol(mol, catchErrors=True)
    FastFindRings(mol)
  mol.UpdatePropertyCache(strict=False)

  if ("*" in which) or ("+" in which):
    concat = "*" in which
    split_sym = "*" if concat else "+"

    out = np.zeros(fp_size, dtype=np.float32)
    parts = which.split(split_sym)

    if concat:
      remaining = fp_size
      n_remaining = len(parts)
      cursor = 0
      for part in parts:
        part_size = remaining // n_remaining
        vec = _mol_to_fp_vector(mol, part, part_size, radius)
        out[cursor : cursor + len(vec)] += vec
        cursor += len(vec)
        remaining -= len(vec)
        n_remaining -= 1
    else:
      for part in parts:
        vec = _mol_to_fp_vector(mol, part, fp_size, radius)
        out[: len(vec)] += vec

    return np.log1p(out)

  return _mol_to_fp_vector(mol, which, fp_size, radius)


def smiles_to_fps(smiles, fp_size, which, radius, is_smarts, sanitize, njobs):
  from functools import partial

  xs = list(smiles)
  n = len(xs)
  if n == 0:
    return np.empty((0, fp_size), dtype=np.float32)
  out = np.empty((n, fp_size), dtype=np.float32)
  if njobs and njobs > 1 and n >= _MIN_PARALLEL:
    _fn = partial(
      _smiles_to_fp,
      fp_size=fp_size,
      radius=radius,
      is_smarts=is_smarts,
      which=which,
      sanitize=sanitize,
    )
    with Pool(processes=njobs) as pool:
      for i, fp in enumerate(pool.imap(_fn, xs, chunksize=max(1, n // (njobs * 4)))):
        out[i] = fp
  else:
    for i, s in enumerate(xs):
      out[i] = _smiles_to_fp(s, fp_size, radius, is_smarts, which, sanitize)
  return out


@dataclass(frozen=True)
class Fingerprint:
  which: str = "morgan"
  fp_size: int = 2048
  radius: int = 2
  is_smarts: bool = False
  sanitize: bool = True
  njobs: int = 8
  name: str = "fingerprint"

  def transform(self, smiles: list[str]) -> np.ndarray:
    return smiles_to_fps(
      smiles=smiles,
      fp_size=self.fp_size,
      which=self.which,
      radius=self.radius,
      is_smarts=self.is_smarts,
      sanitize=self.sanitize,
      njobs=self.njobs,
    )

  def to_dict(self) -> dict:
    return {
      "which": self.which,
      "fp_size": int(self.fp_size),
      "radius": int(self.radius),
      "is_smarts": bool(self.is_smarts),
      "sanitize": bool(self.sanitize),
      "njobs": int(self.njobs),
      "name": self.name,
    }

  @classmethod
  def from_dict(cls, d: dict):
    return cls(
      which=d.get("which", "morgan"),
      fp_size=int(d.get("fp_size", 2048)),
      radius=int(d.get("radius", 2)),
      is_smarts=bool(d.get("is_smarts", False)),
      sanitize=bool(d.get("sanitize", True)),
      njobs=int(d.get("njobs", 8)),
      name=d.get("name", "fingerprint"),
    )


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
