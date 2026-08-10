"""Benchmark XGBoost training configs on a real olinda split (train.h5 / val.h5).

Times a ladder of configurations so speed defaults can be chosen from evidence rather than guessed:
wall-clock, rounds-to-best (early stopping), and val R²/RMSE/top-decile per config. Intended for the
~1M × 2048 Morgan-count regime but works on any split. GPU is used automatically when available.

Example
-------
    conda run -n olinda python scripts/benchmark_xgb.py --in /path/to/split --limit 200000
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import xgboost as xgb

from olinda.console import echo, rule, summary_panel
from olinda.train.backend import XGBoostBackend

_xgb_backend = XGBoostBackend("cpu")
from olinda.train.xgb import detect_training_device

# Config ladder: (label, param overrides on top of a shared base). max_bin is applied to BOTH the
# QuantileDMatrix and the params (they must agree). objective=None → auto from the target shape.
CONFIGS = [
  (
    "legacy (max_bin128, colsample0.8, squarederror)",
    {"max_bin": 128, "colsample_bytree": 0.8, "objective": "squarederror"},
  ),
  ("+ max_bin 64", {"max_bin": 64, "colsample_bytree": 0.8, "objective": "squarederror"}),
  ("+ colsample 0.5", {"max_bin": 64, "colsample_bytree": 0.5, "objective": "squarederror"}),
  ("+ auto objective (new default)", {"max_bin": 64, "colsample_bytree": 0.5, "objective": None}),
]

_BASE = {
  "tree_method": "hist",
  "max_depth": 8,
  "eta": 0.3,
  "subsample": 0.8,
  "min_child_weight": 5.0,
  "lambda": 1.0,
  "seed": 42,
}


def _load(path, limit):
  with h5py.File(path, "r") as f:
    n = f["x"].shape[0] if limit is None else min(int(limit), f["x"].shape[0])
    x = np.asarray(f["x"][:n], dtype=np.float32)
    y = np.asarray(f["y"][:n], dtype=np.float32)
  return x, y


def _metrics(y, p):
  y, p = y.astype(np.float64), p.astype(np.float64)
  err = p - y
  rmse = float(np.sqrt((err**2).mean()))
  sst = float(((y - y.mean()) ** 2).sum())
  r2 = 1.0 - float((err**2).sum()) / sst if sst else float("nan")
  tail = y >= np.quantile(y, 0.9)
  tdr = float(np.sqrt((err[tail] ** 2).mean())) if tail.any() else float("nan")
  return rmse, r2, tdr


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--in", dest="in_dir", default=".", help="Directory with train.h5 / val.h5.")
  ap.add_argument("--limit", type=int, default=None, help="Use first N train rows (quick subset).")
  ap.add_argument("--num-boost-round", type=int, default=600, help="Common cap across configs.")
  ap.add_argument("--early-stopping", type=int, default=40)
  args = ap.parse_args()

  in_dir = Path(args.in_dir)
  rule("olinda · xgb benchmark", style="green", right=str(in_dir))
  device, reason = detect_training_device()
  echo(f"device={device} ({reason})", "run")

  xtr, ytr = _load(in_dir / "train.h5", args.limit)
  xva, yva = _load(in_dir / "val.h5", None)
  echo(f"train {xtr.shape[0]:,}×{xtr.shape[1]} · val {xva.shape[0]:,}", "run")
  dva_dense = xgb.DMatrix(xva)

  results = []
  for label, over in CONFIGS:
    p = dict(_BASE)
    max_bin = over["max_bin"]
    p["max_bin"] = max_bin
    p["colsample_bytree"] = over["colsample_bytree"]
    obj_native = _xgb_backend.objective_params()
    p.update(obj_native)
    obj_label = obj_native["objective"]
    if device == "cuda":
      p["device"] = "cuda"

    t0 = time.perf_counter()
    dtr = xgb.QuantileDMatrix(xtr, label=ytr, max_bin=max_bin)
    dva = xgb.QuantileDMatrix(xva, label=yva, ref=dtr, max_bin=max_bin)
    booster = xgb.train(
      p,
      dtr,
      num_boost_round=args.num_boost_round,
      evals=[(dva, "val")],
      early_stopping_rounds=args.early_stopping,
      verbose_eval=False,
    )
    dt = time.perf_counter() - t0
    best_it = int(booster.best_iteration)
    pv = booster[: best_it + 1].predict(dva_dense)
    rmse, r2, tdr = _metrics(yva, pv)
    results.append((label, obj_label, dt, best_it + 1, r2, rmse, tdr))
    echo(f"{label} · {obj_label} · {dt:.1f}s · {best_it + 1} trees · R²={r2:.4f}", "success")

  base_t = results[0][2]
  summary_panel(
    "olinda · xgb benchmark",
    [
      (
        r[0],
        f"[bold]{r[2]:.1f}s[/] ([bold]{base_t / r[2]:.2f}×[/])  [dim]· {r[3]} trees · {r[1]} "
        f"· R² {r[4]:.4f} · RMSE {r[5]:.4f} · tail {r[6]:.4f}[/]",
      )
      for r in results
    ],
    border_style="green",
    icon="✓",
  )


if __name__ == "__main__":
  main()
