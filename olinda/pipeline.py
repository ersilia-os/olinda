"""Fan out a multi-column calculations dataset into one distilled student per selected column.

A *calculations* file carries one or more teacher value columns over the reference library, all
row-aligned to the reference-library descriptor HDF5. This module distills each selected column into its own
model directory (``model_dir/<column>/``), composing the stable training sub-steps once per column:

    value-stratified split  →  (optional) Optuna tuning  →  backend-dispatched GBM fit  →  validate

The split is :func:`olinda.data.split_reference_to_h5`; tuning is :func:`olinda.train.tune.run_tuning`;
the fit reuses the engine-agnostic primitives in :mod:`olinda.train.backend` (``select_backend`` /
``get_backend`` / ``be.build_train_val`` / ``be.train``), exactly as ``olinda learn-soft`` does — so this
orchestrator adds no new training logic and does not touch the (in-flight) train CLI or backends.

Ground-truth fusion is a planned follow-up; :func:`distill_calculations` accepts a ``ground_truth``
argument as a stable seam but does not yet act on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from olinda.console import echo, rule, summary_panel
from olinda.helpers import logger
from olinda.metrics import regression_metrics

MANIFEST_NAME = "manifest.json"
MANIFEST_FORMAT = "olinda.multicolumn.v1"


def train_split(
  in_dir: str | Path,
  model_dir: str | Path,
  featurizer=None,
  *,
  features: str = "morgan",
  num_boost_round: int = 5000,
  early_stopping: int = 50,
  reweight: str = "auto",
  no_onnx: bool = False,
) -> dict:
  """Train one gradient-boosting student from ``in_dir/train.h5`` + ``in_dir/val.h5`` into ``model_dir``.

  This is a library-level re-composition of the ``olinda learn-soft`` command body: engine is auto-selected
  by device (XGBoost on CUDA, LightGBM on CPU), the loss is squared error, reweighting is auto-picked
  from the target's shape, tuned hyperparameters are used if ``model_dir/best_params.json`` exists, and
  validation metrics + a true-vs-pred scatter are written next to the model.

  Parameters
  ----------
  in_dir : str or Path
      Directory holding ``train.h5`` / ``val.h5`` (as produced by ``split_reference_to_h5``).
  model_dir : str or Path
      Output directory for the model artifacts (also read for ``best_params.json``).
  featurizer : optional
      Featurizer stored with the model so it can predict from SMILES; label-independent, so the
      caller builds it once and reuses it across columns.
  features : str
      Descriptor identity recorded in the model metadata (``"morgan"``).

  Returns
  -------
  dict
      ``{"metrics": <regression metrics>, "backend": ..., "n_trees": ..., "x_dim": ...}``.
  """
  import h5py

  from olinda.data import resolve_regression_weights
  from olinda.models import StudentModel
  from olinda.train.backend import CANONICAL_DEFAULTS, get_backend, select_backend
  from olinda.train.plots import save_true_vs_pred

  in_dir = Path(in_dir)
  model_dir = Path(model_dir)
  model_dir.mkdir(parents=True, exist_ok=True)
  train_h5 = in_dir / "train.h5"
  val_h5 = in_dir / "val.h5"
  for p in (train_h5, val_h5):
    if not p.exists():
      raise FileNotFoundError(f"missing {p.name} in {in_dir} — run the split step first")

  with h5py.File(train_h5, "r") as f:
    x_dim = int(f["x"].shape[1])
    ytr = np.asarray(f["y"][:], dtype=np.float64)

  backend_name, device, _ = select_backend()
  be = get_backend(backend_name, device)

  # Squared-error loss + automatic imbalance reweighting, exactly as `olinda learn-soft`.
  obj_native = be.objective_params()
  bin_edges, bin_weights, reweight_info = resolve_regression_weights(ytr, mode=reweight)
  weighted = bin_weights is not None
  reweight_info["used_in"] = {"training": weighted, "evaluation": weighted, "early_stopping": weighted}

  # Use tuned canonical hyperparameters if a prior tuning step left best_params.json in model_dir.
  tuned_params: dict = {}
  tuned_path = model_dir / "best_params.json"
  if tuned_path.exists():
    with open(tuned_path) as fp:
      tuned_params = json.load(fp)
    tuned_params.pop("backend", None)  # a tag, not a hyperparameter

  canonical = {**CANONICAL_DEFAULTS, **tuned_params}
  native_params = {**be.translate(canonical), **obj_native}
  dtrain, dval = be.build_train_val(train_h5, val_h5, canonical["max_bin"], bin_edges, bin_weights)
  res = be.train(
    dtrain,
    dval,
    native_params,
    num_boost_round=num_boost_round,
    early_stopping=early_stopping,
    train_weighted=weighted,
  )

  student = StudentModel(
    model=res.model,
    backend=backend_name,
    featurizer=featurizer,
    metadata={
      "task": "regression",
      "x_dim": x_dim,
      "features": features,
      "backend": backend_name,
      "objective": "squarederror",
      "reweight": reweight_info,
      "hyperparams": {"source": "tuned" if tuned_params else "defaults", "tuned": tuned_params or None},
    },
  )
  student.save(model_dir)
  if not no_onnx:
    be.to_onnx(res.model, model_dir / "student.onnx", x_dim)

  with h5py.File(val_h5, "r") as f:
    xval = np.asarray(f["x"][:], dtype=np.float32)
    yval = np.asarray(f["y"][:], dtype=np.float32)
  pval = be.predict(res.model, xval)
  metrics = regression_metrics(yval, pval)
  with open(model_dir / "val_metrics.json", "w") as fp:
    json.dump(metrics, fp, indent=2)
  save_true_vs_pred(
    yval, pval, model_dir / "val_true_pred.png", title=f"Validation  (R²={metrics['r2']:.3f})"
  )
  return {"metrics": metrics, "backend": backend_name, "n_trees": res.n_trees, "x_dim": x_dim}


def distill_column(
  descriptors_h5: str | Path,
  y,
  model_dir: str | Path,
  featurizer=None,
  *,
  features: str = "morgan",
  val_frac: float = 0.1,
  seed: int = 42,
  limit: int | None = None,
  tune: bool = False,
  tune_kwargs: dict | None = None,
  num_boost_round: int = 5000,
  early_stopping: int = 50,
  reweight: str = "auto",
  no_onnx: bool = False,
) -> dict:
  """Distill a single teacher-value vector into a student model directory.

  Splits the reference library by ``y`` into ``model_dir/{train,val}.h5``, optionally tunes
  hyperparameters (writing ``best_params.json`` there), then trains + validates via
  :func:`train_split`. Rows with non-finite ``y`` are dropped by the splitter.

  Returns the :func:`train_split` result dict.
  """
  from olinda.data import split_reference_to_h5

  model_dir = Path(model_dir)
  model_dir.mkdir(parents=True, exist_ok=True)
  split_reference_to_h5(
    descriptors_h5,
    y,
    out_dir=model_dir,
    val_frac=val_frac,
    seed=seed,
    limit=limit,
    feature_attrs={"features": features},
  )

  if tune:
    from olinda.train.tune import run_tuning

    run_tuning(
      in_dir=model_dir,
      seed=seed,
      reweight=reweight,
      out=model_dir / "best_params.json",
      **(tune_kwargs or {}),
    )

  return train_split(
    in_dir=model_dir,
    model_dir=model_dir,
    featurizer=featurizer,
    features=features,
    num_boost_round=num_boost_round,
    early_stopping=early_stopping,
    reweight=reweight,
    no_onnx=no_onnx,
  )


def distill_calculations(
  reference_calcs: str | Path,
  descriptors_h5: str | Path,
  model_dir: str | Path,
  featurizer=None,
  *,
  columns=None,
  ground_truth=None,
  features: str = "morgan",
  val_frac: float = 0.1,
  seed: int = 42,
  limit: int | None = None,
  tune: bool = False,
  tune_kwargs: dict | None = None,
  num_boost_round: int = 5000,
  early_stopping: int = 50,
  reweight: str = "auto",
  no_onnx: bool = False,
) -> dict:
  """Distill each selected value column of a calculations file into its own student model.

  For every column in ``columns`` (default: all value columns in the file) a student model is trained
  under ``model_dir/<column>/`` and a top-level ``manifest.json`` records what was produced. The
  featurizer is label-independent, so build it once and pass it in — it is shared across columns.

  Parameters
  ----------
  reference_calcs : str or Path
      Calculations file (SMILES + one or more value columns), row-aligned to ``descriptors_h5``.
  descriptors_h5 : str or Path
      Reference-library Morgan descriptor HDF5 (``data`` = features, ``input`` = SMILES order).
  model_dir : str or Path
      Output root; each column gets a ``<column>/`` subdirectory.
  featurizer : optional
      Featurizer saved with each model for SMILES inference (the ``MorganCountFeaturizer``).
  columns : list of str, optional
      Value columns to distill (default: all).
  ground_truth : str or Path, optional
      Reserved seam for per-column ground-truth fusion — **not yet implemented**. Passing a value
      raises ``NotImplementedError``.

  Returns
  -------
  dict
      The manifest written to ``model_dir/manifest.json``.
  """
  if ground_truth is not None:
    raise NotImplementedError(
      "ground-truth fusion is not wired yet; distill without --ground-truth for now "
      "(per-column fusion is a planned follow-up)"
    )

  model_dir = Path(model_dir)
  model_dir.mkdir(parents=True, exist_ok=True)

  from olinda.data import load_reference_calcs_frame

  all_cols, selected = load_reference_calcs_frame(reference_calcs, descriptors_h5, columns=columns)

  per_column: dict[str, dict] = {}
  for i, (col, y) in enumerate(selected.items(), start=1):
    rule(f"olinda · distill · {col}", style="green", right=f"column {i}/{len(selected)}")
    col_dir = model_dir / col
    result = distill_column(
      descriptors_h5,
      y,
      col_dir,
      featurizer=featurizer,
      features=features,
      val_frac=val_frac,
      seed=seed,
      limit=limit,
      tune=tune,
      tune_kwargs=tune_kwargs,
      num_boost_round=num_boost_round,
      early_stopping=early_stopping,
      reweight=reweight,
      no_onnx=no_onnx,
    )
    per_column[col] = {
      "dir": col,
      "backend": result["backend"],
      "n_trees": result["n_trees"],
      "combined": False,
      "metrics": result["metrics"],
    }
    m = result["metrics"]
    echo(f"{col}: R²={m['r2']:.4f} · RMSE={m['rmse']:.5f} · n={m['n']:,}", "success")

  manifest = {
    "format": MANIFEST_FORMAT,
    "calc_columns": all_cols,
    "distilled": list(selected.keys()),
    "ground_truth": False,
    "per_column": per_column,
  }
  with open(model_dir / MANIFEST_NAME, "w") as fp:
    json.dump(manifest, fp, indent=2)

  summary_panel(
    "olinda · distill",
    [
      ("Columns", f"{len(all_cols)} in file · [bold]{len(selected)}[/] distilled"),
      ("Models", "  ".join(f"[dim]{model_dir / c}[/]" for c in selected)),
      ("Manifest", f"[dim]{model_dir / MANIFEST_NAME}[/]"),
    ],
    border_style="green",
    icon="✓",
  )
  logger.success(f"Distilled {len(selected)} column(s) → {model_dir}")
  return manifest
