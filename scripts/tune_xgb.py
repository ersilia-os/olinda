"""Thin CLI wrapper over :func:`olinda.train.tune.run_tuning` (same logic as ``olinda tune``).

Discovers good XGBoost hyperparameters (especially eta) for the H5 train path via a short, pruned Optuna
study on a subset of a reference split, and prints a paste-ready ``CANONICAL_DEFAULTS`` block.

Example
-------
    conda run -n olinda python scripts/tune_xgb.py --in /path/to/split -m runs/exp1 --trials 40
"""

import argparse
from pathlib import Path

from olinda.train.tune import run_tuning


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--in", dest="in_dir", default=".", help="Directory with train.h5 / val.h5.")
  ap.add_argument("-m", "--model-dir", required=True, help="Output dir (best_params.json is written here).")
  ap.add_argument(
    "--max-rows", type=int, default=100000, help="Cap on rows used for tuning (train+val, split proportion)."
  )
  ap.add_argument("--trials", type=int, default=100, help="Optuna trials (upper bound).")
  ap.add_argument(
    "--time-budget", type=int, default=900, help="Hard wall-clock cap (s); 0 = no cap. 15m default."
  )
  ap.add_argument("--tune-rounds", type=int, default=2000, help="Per-trial boosting cap (early-stopped).")
  ap.add_argument("--early-stopping", type=int, default=50)
  ap.add_argument(
    "--patience", type=int, default=12, help="Stop after N completed trials w/o improvement (0=off)."
  )
  ap.add_argument("--reweight", default="auto", help="auto|on|off")
  ap.add_argument("--seed", type=int, default=42)
  args = ap.parse_args()

  run_tuning(
    args.in_dir,
    max_rows=args.max_rows,
    trials=args.trials,
    time_budget=args.time_budget,
    tune_rounds=args.tune_rounds,
    early_stopping=args.early_stopping,
    patience=args.patience,
    reweight=args.reweight,
    seed=args.seed,
    out=Path(args.model_dir) / "best_params.json",
  )


if __name__ == "__main__":
  main()
