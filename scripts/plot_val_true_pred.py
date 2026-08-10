"""Scatter of validation true vs predicted values for a trained olinda model.

Loads a model's booster, predicts on a validation ``val.h5`` (datasets ``x``/``y``), and plots
true (x) vs predicted (y) with the y=x reference line and regression metrics.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import xgboost as xgb

from olinda.console import echo, rule, summary_panel
from olinda.train.plots import save_true_vs_pred


def _metrics(y, p):
  err = p - y
  ss_res = float((err**2).sum())
  ss_tot = float(((y - y.mean()) ** 2).sum())
  r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
  yc, pc = y - y.mean(), p - p.mean()
  denom = float(np.sqrt((yc**2).sum()) * np.sqrt((pc**2).sum()))
  pear = float((yc * pc).sum() / denom) if denom else float("nan")
  return float(np.abs(err).mean()), float(np.sqrt((err**2).mean())), r2, pear


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model-dir", default="dev/model-test")
  ap.add_argument("--val", default="val.h5")
  ap.add_argument("--out", default="val_true_vs_pred.png")
  ap.add_argument("--max-points", type=int, default=40000, help="Subsample for the scatter.")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  rule("olinda · validation true vs pred", style="green")
  with h5py.File(args.val, "r") as f:
    x = np.asarray(f["x"][:], dtype=np.float32)
    y = np.asarray(f["y"][:], dtype=np.float32)
  booster = xgb.Booster()
  booster.load_model(str(Path(args.model_dir) / "xgb.json"))
  p = np.asarray(booster.predict(xgb.DMatrix(x)), dtype=np.float32)
  echo(f"Predicted {len(y):,} validation rows with {Path(args.model_dir).name}", "run")

  mae, rmse, r2, pear = _metrics(y.astype(np.float64), p.astype(np.float64))
  save_true_vs_pred(y, p, args.out, max_points=args.max_points, seed=args.seed)

  summary_panel(
    "olinda · validation true vs pred",
    [
      ("Val rows", f"{len(y):,}"),
      ("R²", f"[bold]{r2:.4f}[/]"),
      ("RMSE", f"[bold]{rmse:.5f}[/]"),
      ("MAE", f"{mae:.5f}  [dim]· Pearson {pear:.4f}[/]"),
      ("Figure", f"[dim]{args.out}[/]"),
    ],
    border_style="green",
    icon="✓",
  )


if __name__ == "__main__":
  main()
