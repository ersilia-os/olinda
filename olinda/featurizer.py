import json, os, numpy as np, onnxruntime as ort
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import FastFindRings

RDLogger.DisableLog("rdApp.warning")


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
  if njobs and njobs > 1:
    _fn = partial(
      _smiles_to_fp, fp_size=fp_size, radius=radius,
      is_smarts=is_smarts, which=which, sanitize=sanitize,
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


class OnnxEncoder:
  def __init__(self, onnx_path: str | os.PathLike, base, providers=None, name: str = "onnx_encoder") -> None:
    self.onnx_path = str(onnx_path)
    self.base = base
    self.providers = providers or ["CPUExecutionProvider"]
    self.name = name

    self._sess = ort.InferenceSession(self.onnx_path, providers=self.providers)
    self._in = self._sess.get_inputs()[0].name
    self._out = self._sess.get_outputs()[0].name

  def transform(self, smiles: list[str]) -> np.ndarray:
    x = self.base.transform(smiles).astype(np.float32)
    y = self._sess.run([self._out], {self._in: x})[0]
    return np.asarray(y)

  def to_dict(self) -> dict:
    base = self.base.to_dict() if hasattr(self.base, "to_dict") else {"name": type(self.base).__name__}
    return {
      "onnx_path": self.onnx_path,
      "providers": list(self.providers),
      "name": self.name,
      "base": base,
    }

  @classmethod
  def from_dict(cls, d: dict, base_factory):
    base = base_factory(d.get("base", {}))
    return cls(
      onnx_path=d["onnx_path"],
      base=base,
      providers=d.get("providers") or ["CPUExecutionProvider"],
      name=d.get("name", "onnx_encoder"),
    )


def build_featurizer_from_hp(ckpt_dir: str | Path):
  ckpt_dir = Path(ckpt_dir)
  hp_path = ckpt_dir / "hp.json"
  onnx_path = ckpt_dir / "compound_encoder.onnx"

  with open(hp_path, "r") as f:
    hp = json.load(f)

  compound_mode = hp.get("compound_mode", "morgan")

  sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  shape = sess.get_inputs()[0].shape
  fp_size = (
    shape[1] if isinstance(shape, (list, tuple)) and len(shape) == 2 and isinstance(shape[1], int) else 8192
  )

  base = Fingerprint(which=compound_mode, fp_size=int(fp_size), radius=2, njobs=8, name=f"fp:{compound_mode}")
  return OnnxEncoder(
    onnx_path=str(onnx_path), base=base, providers=["CPUExecutionProvider"], name="compound_encoder"
  )
