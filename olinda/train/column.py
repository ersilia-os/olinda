"""Train one column's surrogate.

The per-column training loop lives here rather than in the CLI so a run can be driven from Python
without going through Click callbacks. It reads the column's target and split from the run directory,
gathers its rows from the shared reference matrix, and writes the model, metrics, calibrator and
validation plot into that column's directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from olinda import run as runlib
from olinda.console import echo, live_region_taken, strategy_banner
from olinda.train.backend import CANONICAL_DEFAULTS


def train_one_column(model_dir, manifest, col, matrix, be, backend_name, num_boost_round) -> dict:
  """Train one column's surrogate into ``columns/<id>/``; returns its name, metrics and tree count."""
  from olinda.calibrate import IsotonicCalibrator
  from olinda.data import resolve_regression_weights
  from olinda.featurizer import MorganCountFeaturizer
  from olinda.metrics import regression_metrics
  from olinda.models import StudentModel
  from olinda.train.plots import save_true_vs_pred

  col_dir = runlib.column_dir(model_dir, col["id"])
  col_dir.mkdir(parents=True, exist_ok=True)
  # Early-stopping patience scales with the round budget (1%, floored at 50) — not a user knob.
  early_stopping = max(50, num_boost_round // 100)

  y = runlib.read_target(model_dir, col["id"])
  train_idx, val_idx = runlib.read_split(model_dir, col["id"])
  ytr = np.asarray(y[train_idx], dtype=np.float64)

  # Automatic target reweighting, decided independently per column from that column's own shape.
  bin_edges, bin_weights, reweight_info = resolve_regression_weights(ytr, mode="auto")
  weighted = bin_weights is not None
  reweight_info["used_in"] = {"training": weighted, "evaluation": weighted, "early_stopping": weighted}
  quiet = live_region_taken()  # a live table already reports this column's status
  if not quiet:
    strategy_banner(
      reweight_info["strategy"],
      reweight_info["reason"],
      weight_range=reweight_info.get("weight_range") if weighted else None,
    )

  tuned_params = resolve_tuned_params(model_dir, col_dir, backend_name)
  canonical = {**CANONICAL_DEFAULTS, **tuned_params}
  native_params = {**be.translate(canonical), **be.objective_params()}

  xval = matrix.gather(val_idx)
  yval = np.asarray(y[val_idx], dtype=np.float32)
  dtrain, dval = be.build_train_val_indexed(
    matrix, y, train_idx, val_idx, canonical["max_bin"], bin_edges, bin_weights
  )
  res = be.train(
    dtrain,
    dval,
    native_params,
    num_boost_round=num_boost_round,
    early_stopping=early_stopping,
    train_weighted=weighted,
    val_eval=(xval, yval),
  )

  x_dim = int(matrix.n_cols)
  featurizer = MorganCountFeaturizer(radius=int(manifest["features"].get("radius", 3)), fp_size=x_dim)
  student = StudentModel(
    model=res.model,
    backend=backend_name,
    featurizer=featurizer,
    metadata={
      "task": "regression",
      "column": col["name"],
      "x_dim": x_dim,
      "features": manifest["features"].get("features", "morgan"),
      "backend": backend_name,
      "objective": "squarederror",
      "reweight": reweight_info,
      "hyperparams": {"source": "tuned" if tuned_params else "defaults", "tuned": tuned_params or None},
    },
  )
  student.save(col_dir)

  pval = be.predict(res.model, xval)
  metrics = regression_metrics(yval, pval)
  with open(col_dir / "val_metrics.json", "w") as fp:
    json.dump(metrics, fp, indent=2)
  col["metrics"] = metrics

  # Fit the surrogate's isotonic correction here, where the validation predictions already exist.
  # Export then only loads calibrator.json instead of re-reading and re-predicting the whole split.
  if len(yval) >= 4 and np.isfinite(yval).any():
    IsotonicCalibrator().fit(raw=pval, target=np.asarray(yval, dtype=np.float64)).save(
      col_dir / "calibrator.json"
    )

  save_true_vs_pred(
    yval, pval, col_dir / "val_true_pred.png", title=f"{col['name']}  (R²={metrics['r2']:.3f})"
  )
  if not quiet:
    echo(
      f"R² [bold]{metrics['r2']:.4f}[/] · ρ {metrics['spearman']:.4f} · RMSE {metrics['rmse']:.5f} "
      f"· {res.n_trees:,} trees",
      "success",
    )
  del xval
  return {"name": col["name"], "metrics": metrics, "n_trees": res.n_trees}


def resolve_tuned_params(model_dir: Path, col_dir: Path, backend_name: str) -> dict:
  """Tuned hyperparameters for a column: its own ``best_params.json``, else the run-level one."""
  path = next((p for p in (col_dir / runlib.PARAMS_NAME, model_dir / runlib.PARAMS_NAME) if p.exists()), None)
  if path is None:
    echo("no best_params.json — using built-in defaults (`olinda tune -m` to tune)", "info")
    return {}

  with open(path) as fp:
    tuned = json.load(fp)
  tag = tuned.pop("backend", None)  # a tag, not a hyperparameter
  if tag and tag != backend_name:
    echo(
      f"best_params.json was tuned on {tag}; its canonical params still apply to {backend_name}", "warning"
    )
  unknown = [k for k in tuned if k not in CANONICAL_DEFAULTS]
  if unknown:
    echo(f"ignoring unrecognized best_params.json keys {unknown} — re-run `olinda tune`", "warning")
    for k in unknown:
      tuned.pop(k)
  shown = " · ".join(
    f"{k}={v}"
    for k, v in tuned.items()
    if k in ("learning_rate", "max_depth", "min_split_gain", "min_child_weight")
  )
  echo(f"tuned hyperparameters from {path.name} — {shown}", "run")
  return tuned
