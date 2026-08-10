"""Benchmark XGBoost-CPU vs LightGBM-CPU on a real olinda split (train.h5 / val.h5).

Trains BOTH engines with the SAME shape-chosen objective, reweighting, and canonical hyperparameters on a
subset, then reports wall-clock, rounds-to-best, and val R²/RMSE/top-decile — the honest CPU comparison
behind the auto-backend choice. (On GPU, XGBoost is the pick; this quantifies the CPU case.)

    conda run -n olinda python scripts/benchmark_backends.py --in . --limit 200000
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

from olinda.console import echo, rule, summary_panel
from olinda.data import apply_bin_weights, resolve_regression_weights
from olinda.train.backend import CANONICAL_DEFAULTS, get_backend


def _load(path, limit):
  with h5py.File(path, "r") as f:
    n = f["x"].shape[0] if limit is None else min(int(limit), f["x"].shape[0])
    return np.asarray(f["x"][:n], dtype=np.float32), np.asarray(f["y"][:n], dtype=np.float32)


def _metrics(y, p):
  y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
  err = p - y
  rmse = float(np.sqrt((err**2).mean()))
  sst = float(((y - y.mean()) ** 2).sum())
  r2 = 1.0 - float((err**2).sum()) / sst if sst else float("nan")
  tail = y >= np.quantile(y, 0.9)
  tdr = float(np.sqrt((err[tail] ** 2).mean())) if tail.any() else float("nan")
  return rmse, r2, tdr


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--in", dest="in_dir", default=".")
  ap.add_argument("--limit", type=int, default=200000, help="Train rows (subset).")
  ap.add_argument("--val-limit", type=int, default=40000)
  ap.add_argument("--num-boost-round", type=int, default=2000)
  ap.add_argument("--early-stopping", type=int, default=50)
  args = ap.parse_args()

  in_dir = Path(args.in_dir)
  rule("olinda · backend benchmark (CPU)", style="green", right=str(in_dir))
  xtr, ytr = _load(in_dir / "train.h5", args.limit)
  xva, yva = _load(in_dir / "val.h5", args.val_limit)
  echo(f"train {xtr.shape[0]:,}×{xtr.shape[1]} · val {xva.shape[0]:,}", "run")

  edges, weights, rw = resolve_regression_weights(ytr, mode="auto")
  echo(f"objective=squarederror · reweight={rw['strategy']}", "run")
  wtr = apply_bin_weights(ytr, edges, weights) if weights is not None else None
  wva = apply_bin_weights(yva, edges, weights) if weights is not None else None
  max_bin = CANONICAL_DEFAULTS["max_bin"]

  results = []
  for name in ("xgboost", "lightgbm"):
    be = get_backend(name, "cpu")
    native = be.params(CANONICAL_DEFAULTS)
    t0 = time.perf_counter()
    dtrain = be.dataset(xtr, ytr, wtr, max_bin)
    dval = be.dataset(xva, yva, wva, max_bin, reference=dtrain)
    res = be.train(
      dtrain,
      dval,
      native,
      num_boost_round=args.num_boost_round,
      early_stopping=args.early_stopping,
      train_weighted=wtr is not None,
      val_eval=(xva, yva),
    )
    dt = time.perf_counter() - t0
    rmse, r2, tdr = _metrics(yva, be.predict(res.model, xva))
    results.append((name, dt, res.n_trees, r2, rmse, tdr))
    echo(f"{name}: {dt:.1f}s · {res.n_trees} trees · R²={r2:.4f} · RMSE={rmse:.5f}", "success")

  base = results[0][1]
  summary_panel(
    "olinda · backend benchmark (CPU)",
    [
      (
        r[0],
        f"[bold]{r[1]:.1f}s[/] ([bold]{base / r[1]:.2f}×[/])  [dim]· {r[2]} trees · R² {r[3]:.4f} · "
        f"RMSE {r[4]:.5f} · tail {r[5]:.5f}[/]",
      )
      for r in results
    ],
    border_style="green",
    icon="✓",
  )


if __name__ == "__main__":
  main()
