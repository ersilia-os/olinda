from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from olinda.helpers import logger
from olinda.featurizer import Fingerprint
from olinda.validate import _mae, _rmse, _spearmanr


@dataclass(frozen=True)
class SplitResult:
  name: str
  train_idx: np.ndarray
  test_idx: np.ndarray


def _load_smiles_df(path: Path, smiles_col: str, y_col: str) -> tuple[list[str], np.ndarray]:
  if path.suffix.lower() in (".parquet", ".pq"):
    df = pd.read_parquet(str(path))
  else:
    df = pd.read_csv(str(path))

  if smiles_col not in df.columns:
    raise ValueError(f"missing smiles column: {smiles_col}")
  if y_col not in df.columns:
    raise ValueError(f"missing y column: {y_col}")

  smiles = df[smiles_col].astype(str).tolist()
  y = df[y_col].astype(np.float32).to_numpy()
  return smiles, y


def _random_split(n: int, test_frac: float, seed: int) -> SplitResult:
  rng = np.random.default_rng(seed)
  idx = rng.permutation(n)
  n_test = int(round(n * test_frac))
  test_idx = np.sort(idx[:n_test])
  train_idx = np.sort(idx[n_test:])
  return SplitResult(name="random", train_idx=train_idx, test_idx=test_idx)


def _scaffold_split(smiles: list[str], test_frac: float, seed: int) -> SplitResult:
  rng = np.random.default_rng(seed)
  scaffolds: dict[str, list[int]] = {}
  for i, smi in enumerate(smiles):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
      scaf = ""
    else:
      scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    scaffolds.setdefault(scaf, []).append(i)

  groups = list(scaffolds.values())
  rng.shuffle(groups)
  groups.sort(key=len, reverse=True)

  n = len(smiles)
  n_test = int(round(n * test_frac))
  test_idx: list[int] = []
  train_idx: list[int] = []

  for g in groups:
    if len(test_idx) + len(g) <= n_test:
      test_idx.extend(g)
    else:
      train_idx.extend(g)

  if len(test_idx) == 0:
    raise ValueError("scaffold split produced empty test set")

  return SplitResult(
    name="scaffold",
    train_idx=np.sort(np.asarray(train_idx, dtype=np.int64)),
    test_idx=np.sort(np.asarray(test_idx, dtype=np.int64)),
  )


def _morgan_bit_fps(smiles: list[str], fp_size: int, radius: int) -> list:
  fps = []
  for smi in smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
      fps.append(DataStructs.ExplicitBitVect(fp_size))
    else:
      fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size, useChirality=True))
  return fps


def _similarity_split(
  fps: list,
  test_frac: float,
  seed: int,
  max_sim: float = 0.4,
) -> SplitResult:
  rng = np.random.default_rng(seed)
  idx = rng.permutation(len(fps))
  n_test = int(round(len(fps) * test_frac))

  train_idx: list[int] = []
  test_idx: list[int] = []

  for i in idx:
    if len(train_idx) == 0:
      train_idx.append(int(i))
      continue
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in train_idx])
    max_s = max(sims) if sims else 0.0
    if max_s <= max_sim and len(test_idx) < n_test:
      test_idx.append(int(i))
    else:
      train_idx.append(int(i))

  if len(test_idx) < n_test:
    remaining = [i for i in train_idx]
    scores = []
    for i in remaining:
      sims = DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in train_idx if j != i])
      scores.append((i, max(sims) if sims else 0.0))
    scores.sort(key=lambda x: x[1])
    move = [i for i, _ in scores[: n_test - len(test_idx)]]
    test_idx.extend(move)
    train_idx = [i for i in train_idx if i not in set(move)]

  return SplitResult(
    name="similarity",
    train_idx=np.sort(np.asarray(train_idx, dtype=np.int64)),
    test_idx=np.sort(np.asarray(test_idx, dtype=np.int64)),
  )


def _train_val_split(idx: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
  if val_frac <= 0:
    return idx, np.asarray([], dtype=np.int64)
  rng = np.random.default_rng(seed)
  perm = rng.permutation(idx)
  n_val = int(round(len(idx) * val_frac))
  val_idx = np.sort(perm[:n_val])
  train_idx = np.sort(perm[n_val:])
  return train_idx, val_idx


def _fit_xgb(
  X_train: np.ndarray,
  y_train: np.ndarray,
  X_val: np.ndarray,
  y_val: np.ndarray,
  seed: int,
  num_boost_round: int,
  early_stopping_rounds: int,
) -> xgb.Booster:
  params = {
    "max_depth": 8,
    "eta": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "lambda": 1.0,
    "alpha": 0.0,
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "seed": int(seed),
  }

  dtrain = xgb.DMatrix(X_train, label=y_train)
  evals = []
  if len(X_val) > 0:
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dval, "val")]

  booster = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=int(num_boost_round),
    evals=evals,
    early_stopping_rounds=int(early_stopping_rounds) if len(X_val) > 0 else None,
    verbose_eval=False,
  )
  return booster


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
  return {
    "mae": _mae(y, p),
    "rmse": _rmse(y, p),
    "spearman": _spearmanr(y, p),
  }


def _randomize_smiles(smi: str, n: int, seed: int) -> list[str]:
  mol = Chem.MolFromSmiles(smi)
  if mol is None:
    return [smi]
  out = []
  for _ in range(n):
    out.append(Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=True))
  return out


def _tanimoto_nn(train_fps: list, test_fps: list) -> np.ndarray:
  sims = []
  for fp in test_fps:
    s = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
    sims.append(max(s) if s else 0.0)
  return np.asarray(sims, dtype=np.float32)


def _bin_metrics(y: np.ndarray, p: np.ndarray, sim: np.ndarray, bins: list[float]) -> list[dict]:
  out = []
  for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (sim >= lo) & (sim < hi)
    if not np.any(mask):
      out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0})
      continue
    out.append({
      "bin": f"{lo:.1f}-{hi:.1f}",
      "n": int(mask.sum()),
      **_metrics(y[mask], p[mask]),
    })
  return out


def robustness_eval_smiles(
  data_path: str | Path,
  out_dir: str | Path,
  smiles_col: str = "smiles",
  y_col: str = "y",
  split: str = "scaffold",
  test_frac: float = 0.2,
  similarity_threshold: float = 0.4,
  fp: str = "morgan",
  fp_size: int = 2048,
  radius: int = 2,
  njobs: int = 8,
  seed: int = 42,
  num_boost_round: int = 1000,
  early_stopping_rounds: int = 50,
  val_frac: float = 0.1,
  enum_max: int = 8,
  ensemble_size: int = 5,
) -> dict:
  data_path = Path(data_path)
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  smiles, y = _load_smiles_df(data_path, smiles_col=smiles_col, y_col=y_col)
  logger.info(f"Loaded dataset rows={len(smiles)}")

  featurizer = Fingerprint(which=fp, fp_size=int(fp_size), radius=int(radius), njobs=int(njobs))
  X = featurizer.transform(smiles).astype(np.float32)
  fps = _morgan_bit_fps(smiles, fp_size=int(fp_size), radius=int(radius))

  random_split = _random_split(len(smiles), test_frac=test_frac, seed=seed)
  if split == "scaffold":
    robust_split = _scaffold_split(smiles, test_frac=test_frac, seed=seed)
  elif split == "similarity":
    robust_split = _similarity_split(fps, test_frac=test_frac, seed=seed, max_sim=float(similarity_threshold))
  else:
    raise ValueError(f"unknown split: {split}")

  def train_eval(split_res: SplitResult, split_seed: int) -> dict:
    tr_idx, va_idx = _train_val_split(split_res.train_idx, val_frac=val_frac, seed=split_seed)
    booster = _fit_xgb(
      X[tr_idx],
      y[tr_idx],
      X[va_idx],
      y[va_idx],
      seed=split_seed,
      num_boost_round=num_boost_round,
      early_stopping_rounds=early_stopping_rounds,
    )
    dtest = xgb.DMatrix(X[split_res.test_idx])
    preds = booster.predict(dtest).astype(np.float32)
    return {
      "metrics": _metrics(y[split_res.test_idx], preds),
      "booster": booster,
      "test_idx": split_res.test_idx,
      "train_idx": split_res.train_idx,
    }

  logger.info("Training random split model")
  random_res = train_eval(random_split, split_seed=seed + 1)
  logger.info(
    "Random split metrics: "
    f"mae={random_res['metrics']['mae']:.6f} "
    f"rmse={random_res['metrics']['rmse']:.6f} "
    f"spearman={random_res['metrics']['spearman']:.6f}"
  )

  logger.info(f"Training {split} split model")
  robust_res = train_eval(robust_split, split_seed=seed + 2)
  logger.info(
    f"{split} split metrics: "
    f"mae={robust_res['metrics']['mae']:.6f} "
    f"rmse={robust_res['metrics']['rmse']:.6f} "
    f"spearman={robust_res['metrics']['spearman']:.6f}"
  )

  drop = {
    "mae": robust_res["metrics"]["mae"] - random_res["metrics"]["mae"],
    "rmse": robust_res["metrics"]["rmse"] - random_res["metrics"]["rmse"],
    "spearman": robust_res["metrics"]["spearman"] - random_res["metrics"]["spearman"],
  }
  logger.info(
    f"Drop vs random: mae={drop['mae']:.6f} rmse={drop['rmse']:.6f} spearman={drop['spearman']:.6f}"
  )

  enum_steps = [n for n in (1, 2, 4, 8, enum_max) if n <= enum_max]
  enum_steps = sorted(set(enum_steps))
  test_smiles = [smiles[i] for i in robust_res["test_idx"]]
  y_test = y[robust_res["test_idx"]]
  booster = robust_res["booster"]
  perturb_curve = []
  per_mol_std = []

  for n_enum in enum_steps:
    enum_preds = []
    for i, smi in enumerate(test_smiles):
      variants = _randomize_smiles(smi, n=n_enum, seed=seed + i)
      Xv = featurizer.transform(variants).astype(np.float32)
      pv = booster.predict(xgb.DMatrix(Xv)).astype(np.float32)
      enum_preds.append(pv)
    means = np.array([p.mean() for p in enum_preds], dtype=np.float32)
    stds = np.array([p.std() for p in enum_preds], dtype=np.float32)
    if n_enum == max(enum_steps):
      per_mol_std = stds.tolist()
    perturb_curve.append({"n_enum": n_enum, **_metrics(y_test, means)})

  train_fps = [fps[i] for i in robust_res["train_idx"]]
  test_fps = [fps[i] for i in robust_res["test_idx"]]
  nn_sim = _tanimoto_nn(train_fps, test_fps)
  bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]

  test_preds = booster.predict(xgb.DMatrix(X[robust_res["test_idx"]])).astype(np.float32)
  sim_bins = _bin_metrics(y_test, test_preds, nn_sim, bins=bins)

  ens_preds = []
  for i in range(int(ensemble_size)):
    tr_idx, va_idx = _train_val_split(robust_res["train_idx"], val_frac=val_frac, seed=seed + 10 + i)
    ens_booster = _fit_xgb(
      X[tr_idx],
      y[tr_idx],
      X[va_idx],
      y[va_idx],
      seed=seed + 100 + i,
      num_boost_round=num_boost_round,
      early_stopping_rounds=early_stopping_rounds,
    )
    ens_preds.append(ens_booster.predict(xgb.DMatrix(X[robust_res["test_idx"]])).astype(np.float32))
  ens_arr = np.stack(ens_preds, axis=0)
  mean_pred = ens_arr.mean(axis=0)
  std_pred = ens_arr.std(axis=0)
  z = 1.644854
  lower = mean_pred - z * std_pred
  upper = mean_pred + z * std_pred
  coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
  avg_width = float(np.mean(upper - lower))

  coverage_bins = []
  width_bins = []
  for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (nn_sim >= lo) & (nn_sim < hi)
    if not np.any(mask):
      coverage_bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0})
      width_bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0})
      continue
    cov = float(np.mean((y_test[mask] >= lower[mask]) & (y_test[mask] <= upper[mask])))
    width = float(np.mean(upper[mask] - lower[mask]))
    coverage_bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()), "coverage": cov})
    width_bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()), "avg_width": width})

  report = {
    "split": split,
    "random_metrics": random_res["metrics"],
    "robust_metrics": robust_res["metrics"],
    "drop_vs_random": drop,
    "perturbation_curve": perturb_curve,
    "per_molecule_std": per_mol_std,
    "nn_similarity": nn_sim.tolist(),
    "error_vs_similarity_bins": sim_bins,
    "interval_coverage": coverage,
    "interval_avg_width": avg_width,
    "coverage_vs_similarity_bins": coverage_bins,
    "width_vs_similarity_bins": width_bins,
  }

  out_path = out_dir / "robustness_report.json"
  with open(out_path, "w") as fp:
    json.dump(report, fp, indent=2)
  logger.success(f"Robustness report written to {out_path}")
  return report
