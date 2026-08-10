from __future__ import annotations

from pathlib import Path

import rich_click as click
import rich_click.rich_click as rc

# NOTE: heavy dependencies (numpy, pandas, xgboost, rdkit, matplotlib, ...) are imported
# lazily inside the command bodies below, not at module scope. This keeps CLI startup fast
# (e.g. `olinda --help`, `olinda setup`) — importing them all eagerly cost ~12 s per call.


click.rich_click.TEXT_MARKUP = "rich"
click.rich_click.SHOW_ARGUMENTS = True

rc.TEXT_MARKUP = "rich"
rc.SHOW_ARGUMENTS = True
rc.COLOR_SYSTEM = "truecolor"
rc.STYLE_OPTION = "bold magenta"
rc.STYLE_COMMAND = "bold green"
rc.STYLE_METAVAR = "italic yellow"
rc.STYLE_SWITCH = "underline cyan"
rc.STYLE_USAGE = "bold blue"
rc.STYLE_OPTION_DEFAULT = "dim italic"


def _align_command_columns(name_width: int = 11) -> None:
  """Pin the command-name column to a fixed width across every help panel.

  rich-click sizes each group's name column to *that group's* longest command, so the help text in the
  "Main commands" and "Fit pipeline commands" panels doesn't line up; its only knob is a *proportional*
  ratio, which either leaves a big gap or truncates names. A fixed width (just past the longest command,
  "learn-soft"/"learn-hard") keeps the help text tight to the names AND aligned across panels. rich-click exposes no
  config for this, so we defensively wrap the internal command-table builder — on any API drift it silently
  falls back to the default rendering rather than breaking the CLI.
  """
  try:
    from rich_click.rich_panel import RichCommandPanel
  except Exception:
    return

  _orig_get_table = RichCommandPanel.get_table

  def _get_table(self, *args, **kwargs):
    table = _orig_get_table(self, *args, **kwargs)
    try:
      # Fix the name column and make the help column absorb ALL slack, so the name column stays exactly
      # `name_width` even on a wide terminal (otherwise `expand` inflates both columns, per panel).
      name_col, help_col = table.columns[0], table.columns[1]
      name_col.width, name_col.ratio, name_col.no_wrap = name_width, None, True
      help_col.ratio, help_col.width = 1, None
    except Exception:
      pass
    return table

  RichCommandPanel.get_table = _get_table


_align_command_columns()

# Two-level help grouping (à la zairachem): top-level user commands, then the lower-level pipeline steps.
rc.COMMAND_GROUPS = {
  "olinda": [
    {"name": "Main commands", "commands": ["setup", "fit", "predict"]},
    {"name": "Fit pipeline commands", "commands": ["prepare", "tune", "learn-soft", "learn-hard", "export"]},
  ]
}


@click.group()
def cli():
  pass


@cli.command("setup")
@click.option(
  "--target-dir",
  default=None,
  help="Directory to download data into (default: ~/.olinda/).",
)
def setup_cmd(target_dir):
  """Download olinda data from public S3 — the reference-library Morgan fingerprints (``erl0_morgan.h5``).

  Anything already present is skipped, so re-running only fetches what is missing. The download is
  best-effort (a warning, not an error, if it isn't on S3 yet); you can generate it locally with
  ``scripts/compute_morgan_fingerprints.py``.
  """
  from olinda.console import rule
  from olinda.data.fetch import download_morgan_fingerprints

  rule("olinda · setup", style="cyan")
  download_morgan_fingerprints(target_dir)


@cli.command("fit")
@click.option(
  "--soft-labels",
  "-s",
  required=True,
  help="Teacher (soft) values over the reference library (col 0 = SMILES, col 1 = value, library order).",
)
@click.option(
  "--hard-labels",
  "-h",
  default=None,
  help="Optional CSV/TSV/Parquet of your own compounds (SMILES + one label column) → adds a hard head.",
)
@click.option(
  "--model-dir", "-m", required=True, help="Run directory — all prepared data and the model.onnx live here."
)
@click.option(
  "--task",
  type=click.Choice(["auto", "binary", "regression"]),
  default="auto",
  show_default=True,
  help="Hard-label type (auto-detected by default); only used with --hard-labels.",
)
@click.option(
  "--max-samples", default=None, type=int, help="Use only the first N reference compounds (dev subsampling)."
)
@click.option("--val-frac", default=0.1, type=float, show_default=True)
@click.option(
  "--num-boost-round",
  default=10000,
  type=int,
  show_default=True,
  help="learn-soft upper cap; early stopping decides.",
)
@click.option(
  "--tune/--no-tune",
  "do_tune",
  default=False,
  show_default=True,
  help="Run an Optuna tuning pass before learn-soft.",
)
@click.option("--trials", default=100, type=int, show_default=True, help="Optuna trials (only with --tune).")
def fit_cmd(
  soft_labels, hard_labels, model_dir, task, max_samples, val_frac, num_boost_round, do_tune, trials
):
  """Distill a teacher into a student end-to-end: prepare → (tune) → learn-soft → (learn-hard) → one model.onnx.

  Runs the fit-pipeline sub-commands in order, sharing one `--model-dir`. With `--hard-labels`, a hard head is
  learned and fused in; with `--tune`, an Optuna pass precedes `learn-soft`. The result is a single
  self-describing `model.onnx` that `olinda predict` runs.
  """
  from olinda.console import rule, summary_panel

  md = Path(model_dir)
  rule("olinda · fit", style="cyan", right=str(md))

  prepare_cmd.callback(
    soft_labels=soft_labels,
    hard_labels=hard_labels,
    model_dir=model_dir,
    task=task,
    max_samples=max_samples,
    val_frac=val_frac,
  )
  if do_tune:
    tune_cmd.callback(model_dir=model_dir, trials=trials, max_rows=100_000)
  learn_soft_cmd.callback(model_dir=model_dir, num_boost_round=num_boost_round)
  if hard_labels is not None:
    learn_hard_cmd.callback(model_dir=model_dir)

  pipeline = (
    "prepare → " + ("tune → " if do_tune else "") + "learn-soft" + (" → learn-hard" if hard_labels else "")
  )
  summary_panel(
    "olinda · fit",
    [
      ("Pipeline", pipeline),
      ("Head", "soft + hard" if hard_labels else "soft only"),
      ("Model", f"[dim]{md / 'model.onnx'}[/]"),
    ],
    border_style="green",
    icon="✓",
  )


@cli.command("prepare")
@click.option(
  "--soft-labels",
  "-s",
  required=True,
  help="Teacher (soft) values over the reference library (col 0 = SMILES, col 1 = value, same order as the library).",
)
@click.option(
  "--hard-labels",
  "-h",
  default=None,
  help="Optional CSV/TSV/Parquet of your own compounds (SMILES + one label column) for `learn-hard`.",
)
@click.option(
  "--model-dir", "-m", required=True, help="Run directory — all prepared data + models live here."
)
@click.option(
  "--task",
  type=click.Choice(["auto", "binary", "regression"]),
  default="auto",
  show_default=True,
  help="Hard-label type (auto-detected by default); only used with --hard-labels.",
)
@click.option(
  "--max-samples",
  default=None,
  type=int,
  help="Use only the first N reference compounds (development subsampling).",
)
@click.option("--val-frac", default=0.1, type=float, show_default=True)
def prepare_cmd(soft_labels, hard_labels, model_dir, task, max_samples, val_frac):
  """Prepare the reference (soft) split and, optionally, the hard-label set into one run directory.

  Writes the value-stratified, shuffled train/val split (train.h5 / val.h5) from --soft-labels, and —
  if --hard-labels is given — featurizes those compounds into hard.h5 for a later `learn-hard`.
  """
  import h5py
  import numpy as np

  from olinda.console import echo, rule, step, summary_panel
  from olinda.data import (
    OLINDA_HOME,
    MORGAN_FINGERPRINTS_FILENAME,
    load_reference_calcs,
    split_reference_to_h5,
  )

  rule("olinda · prepare", style="cyan", right=str(model_dir))
  n_steps = 3 if hard_labels is not None else 2

  descriptors = OLINDA_HOME / MORGAN_FINGERPRINTS_FILENAME
  if not descriptors.exists():
    echo(f"reference library missing · [dim]{descriptors}[/]", "error")
    raise click.ClickException(
      f"Morgan descriptors missing at {descriptors} — run `olinda setup` "
      "(or scripts/compute_morgan_fingerprints.py to generate them locally)."
    )

  # Stamp the split with its feature identity so `learn-soft` attaches the matching featurizer.
  feature_attrs = {"features": "morgan"}
  with h5py.File(descriptors, "r") as f:
    for k in ("radius", "nbits"):
      if k in f.attrs:
        feature_attrs[k] = int(f.attrs[k])

  step(1, n_steps, "splitting the reference library (train / val)")
  y = load_reference_calcs(soft_labels, descriptors)
  split_reference_to_h5(
    descriptors, y, out_dir=model_dir, val_frac=val_frac, limit=max_samples, feature_attrs=feature_attrs
  )

  # Persist the reference-aligned soft-label vector (row-for-row with erl0_morgan.h5) so `learn-hard` can
  # calibrate the hard model against the soft labels across the full reference library.
  step(2, n_steps, "saving reference-aligned soft labels")
  md = Path(model_dir)
  md.mkdir(parents=True, exist_ok=True)
  with h5py.File(md / "soft.h5", "w") as f:
    f.create_dataset("y", data=np.asarray(y, dtype=np.float32))
  echo(f"soft.h5 · [bold]{len(y):,}[/] rows", "run")

  hard_info = None
  if hard_labels is not None:
    from olinda.ground_truth import prepare_hard_labels

    step(3, n_steps, "featurizing hard labels")
    hard_info = prepare_hard_labels(hard_labels, model_dir, task=task)
    echo(f"hard.h5 · [bold]{hard_info['n']:,}[/] rows · task=[bold]{hard_info['task']}[/]", "run")

  rows = [
    ("Soft split", f"train.h5 / val.h5 · val_frac {val_frac}"),
    ("Soft labels", f"[bold]{len(y):,}[/] reference rows → soft.h5"),
  ]
  if hard_info is not None:
    rows.append(("Hard labels", f"[bold]{hard_info['n']:,}[/] rows · task [bold]{hard_info['task']}[/]"))
  rows.append(("Saved", f"[dim]{md}[/]"))
  summary_panel("olinda · prepare", rows, border_style="green", icon="✓")


@cli.command("learn-soft")
@click.option(
  "--model-dir", "-m", required=True, help="Run directory with train.h5 / val.h5 (from `prepare`)."
)
@click.option(
  "--num-boost-round", default=10000, type=int, show_default=True, help="Upper cap; early stopping decides."
)
def learn_soft_cmd(model_dir, num_boost_round):
  """Learn the surrogate: fast gradient-boosting regression on the prepared split (train.h5 / val.h5).

  The engine is auto-selected by device — XGBoost on a CUDA GPU, LightGBM on CPU (``OLINDA_BACKEND``
  overrides). The loss is squared error (well-conditioned, ONNX-safe); skew/imbalance is handled by
  reweighting, not the loss. If a prior ``olinda tune -m <model-dir>`` wrote ``best_params.json`` there,
  those hyperparameters are used; otherwise built-in defaults.

  Sample reweighting is **automatic**: olinda weights the target only when it is imbalanced (auto-picking
  KDE / bins), leaving balanced targets unweighted. When active it applies to train AND val together so
  early stopping matches the objective. Global and tail metrics (top-decile RMSE, Spearman) are always
  reported, and a single self-describing `model.onnx` bundle is fused at the end (soft-only here;
  `learn-hard` re-fuses it with the hard head).
  """
  import json

  import h5py
  import numpy as np

  from olinda.console import echo, rule, engine_banner, strategy_banner, summary_panel
  from olinda.data import resolve_regression_weights
  from olinda.models import StudentModel
  from olinda.train.backend import CANONICAL_DEFAULTS, get_backend, select_backend
  from olinda.train.plots import save_true_vs_pred

  model_dir = Path(model_dir)
  # Early-stopping patience scales with the round budget (1%, floored at 50) — not a user knob.
  early_stopping = max(50, num_boost_round // 100)
  train_h5 = model_dir / "train.h5"
  val_h5 = model_dir / "val.h5"
  for p in (train_h5, val_h5):
    if not p.exists():
      echo(f"missing {p.name} · [dim]{p}[/]", "error")
      raise click.ClickException(f"missing {p.name} in {model_dir} — run `olinda prepare` first")

  with h5py.File(train_h5, "r") as f:
    x_dim = int(f["x"].shape[1])
    features = str(f.attrs.get("features", "morgan"))
    feat_radius = int(f.attrs["radius"]) if "radius" in f.attrs else 3
    ytr = np.asarray(f["y"][:], dtype=np.float64)
  rule("olinda · train", style="green", right=f"{features} · {x_dim}-dim features")

  # Engine auto-selected by device (GPU→XGBoost, CPU→LightGBM); OLINDA_BACKEND overrides.
  backend_name, device, backend_reason = select_backend()
  be = get_backend(backend_name, device)
  engine_banner(backend_name, device, backend_reason)

  # Single objective: squared error (well-conditioned, ONNX-safe). Backend maps to its native loss.
  obj_native = be.objective_params()

  # Automatic target reweighting: 'auto' weights only when imbalanced (olinda picks none/KDE/bins from the
  # target's shape); 'on'/'off' force/disable. When active, weights apply to train AND val together.
  bin_edges, bin_weights, reweight_info = resolve_regression_weights(ytr, mode="auto")
  weighted = bin_weights is not None
  reweight_info["used_in"] = {"training": weighted, "evaluation": weighted, "early_stopping": weighted}
  strategy_banner(
    reweight_info["strategy"],
    reweight_info["reason"],
    weight_range=reweight_info.get("weight_range") if weighted else None,
  )

  # Use tuned (canonical) hyperparameters if a prior `olinda tune -m <model-dir>` left best_params.json here.
  tuned_params: dict = {}
  tuned_path = model_dir / "best_params.json"
  if tuned_path.exists():
    with open(tuned_path) as fp:
      tuned_params = json.load(fp)
    tag = tuned_params.pop("backend", None)  # a tag, not a hyperparameter
    if tag and tag != backend_name:
      echo(
        f"best_params.json was tuned on {tag}; its canonical params still apply to {backend_name}", "warning"
      )
    unknown = [k for k in tuned_params if k not in CANONICAL_DEFAULTS]
    if unknown:
      echo(
        f"ignoring unrecognized best_params.json keys {unknown} (stale format?) — re-run `olinda tune`",
        "warning",
      )
      for k in unknown:
        tuned_params.pop(k)
    shown = " · ".join(
      f"{k}={v}"
      for k, v in tuned_params.items()
      if k in ("learning_rate", "max_depth", "min_split_gain", "min_child_weight")
    )
    echo(f"Tuned hyperparameters from {tuned_path.name} — {shown}", "run")
  else:
    echo("No best_params.json in model-dir — using built-in defaults (run `olinda tune -m` to tune)", "info")

  canonical = {**CANONICAL_DEFAULTS, **tuned_params}
  native_params = {**be.translate(canonical), **obj_native}
  # Load val once — used for live R²/ρ during training AND the final metrics/plot below.
  with h5py.File(val_h5, "r") as f:
    xval = np.asarray(f["x"][:], dtype=np.float32)
    yval = np.asarray(f["y"][:], dtype=np.float32)
  dtrain, dval = be.build_train_val(train_h5, val_h5, canonical["max_bin"], bin_edges, bin_weights)
  res = be.train(
    dtrain,
    dval,
    native_params,
    num_boost_round=num_boost_round,
    early_stopping=early_stopping,
    train_weighted=weighted,
    val_eval=(xval, yval),
  )

  # Attach the Morgan featurizer so the saved model can predict directly from SMILES.
  from olinda.featurizer import MorganCountFeaturizer

  featurizer = MorganCountFeaturizer(radius=feat_radius, fp_size=x_dim)

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

  # Validation metrics (val already loaded above for live progress).
  from olinda.metrics import regression_metrics

  pval = be.predict(res.model, xval)
  metrics = regression_metrics(yval, pval)
  with open(model_dir / "val_metrics.json", "w") as fp:
    json.dump(metrics, fp, indent=2)

  # True-vs-pred scatter saved alongside the model (skipped with a warning if stylia is absent).
  plot_path = save_true_vs_pred(
    yval, pval, model_dir / "val_true_pred.png", title=f"Validation  (R²={metrics['r2']:.3f})"
  )

  rows = [
    (
      "Trees",
      f"[bold]{res.n_trees}[/]  [dim]· {backend_name} · {obj_native['objective']} · reweight: "
      f"{reweight_info['strategy']}{' (train+val)' if weighted else ''}[/]",
    ),
    ("Val rows", f"{metrics['n']:,}  [dim]· metrics below are unweighted / global[/]"),
    ("RMSE", f"[bold]{metrics['rmse']:.5f}[/]  [dim]· MAE {metrics['mae']:.5f}[/]"),
    (
      "R²",
      f"[bold]{metrics['r2']:.4f}[/]  ·  Pearson [bold]{metrics['pearson']:.4f}[/]  ·  Spearman [bold]{metrics['spearman']:.4f}[/]",
    ),
    ("Top-decile RMSE", f"[bold]{metrics['top_decile_rmse']:.5f}[/]  [dim](sparse high-value tail)[/]"),
  ]
  if plot_path is not None:
    rows.append(("Plot", f"[dim]{plot_path}[/]"))
  rows.append(("Model", f"[dim]{model_dir}[/]"))
  summary_panel("olinda · train", rows, border_style="green", icon="✓")

  # Fuse the served bundle: a single self-describing model.onnx (soft-only here; learn-hard re-fuses).
  from olinda.export import build_bundle

  build_bundle(model_dir)


@cli.command("tune")
@click.option(
  "--model-dir",
  "-m",
  required=True,
  help="Run directory with train.h5 / val.h5; best_params.json is written here.",
)
@click.option("--trials", default=100, type=int, show_default=True, help="How many Optuna trials to run.")
@click.option(
  "--max-rows",
  default=100000,
  type=int,
  show_default=True,
  help="Cap on the rows used for tuning (train+val together, subsampled in the split's proportion). "
  "Higher = more faithful to the full data, slower.",
)
def tune_cmd(model_dir, trials, max_rows):
  """Short, pruned Optuna study to discover good hyperparameters (esp. learning rate) for `learn-soft`.

  Reads the prepared split from ``--model-dir`` / ``-m`` and writes ``best_params.json`` back into it, on
  the **same auto-selected engine** `learn-soft` uses, then a subsequent `learn-soft -m <dir>` auto-reads
  it. It tunes on a random subsample of ``--max-rows`` rows (train+val kept in the prepared proportion) —
  the console prints exactly how many are used. This is a fast way to find a good hyperparameter *region*:
  the learning rate transfers reasonably (full training re-fits the round count with its own early
  stopping), so `tune` is optional — the built-in defaults are already good. Everything else (per-trial
  rounds, early-stopping, study patience, reweighting = auto like `learn-soft`, seed, a safety time cap)
  uses good internal defaults — see ``olinda.train.tune.run_tuning`` / ``scripts/tune_xgb.py`` to override.
  Requires the ``[train]`` extra (Optuna).
  """
  from olinda.train.tune import run_tuning

  run_tuning(
    model_dir,
    trials=trials,
    max_rows=max_rows,
    out=Path(model_dir) / "best_params.json",
  )


@cli.command("learn-hard")
@click.option(
  "--model-dir",
  "-m",
  required=True,
  help="Run directory holding hard.h5 (from `prepare --hard-labels`); artifacts go under <dir>/_ground_truth/.",
)
def learn_hard_cmd(model_dir):
  """Learn from hard (binary) labels and calibrate them onto the soft-label scale — four clear steps.

  Reads `hard.h5` (from `prepare --hard-labels`) and: (1) trains the hard-label classifier `G`; (2) scores
  `G` across the full reference library (`erl0_morgan.h5`, required); (3) calibrates `G`'s output onto the
  teacher's soft-label scale using the reference-aligned soft labels (`prepare` saved `soft.h5`), with a
  monotonic map whose direction is learned from the data; (4) learns an applicability gate — two Bernoulli
  Naive-Bayes classifiers bucketing queries as NOT SIMILAR / LOW / HIGH by proximity to the labeled set.
  Artifacts land under `<model-dir>/_ground_truth/`; `predict` then emits a blended `prediction` plus the
  surrogate / ground_truth_soft / ground_truth / applicability channels (the gate needs no similarity search
  and the blend favours the surrogate away from the labeled set).
  """
  from olinda.ground_truth import HARD_H5_NAME, train_ground_truth

  md = Path(model_dir)
  if not (md / HARD_H5_NAME).exists():
    raise click.ClickException(
      f"no {HARD_H5_NAME} in {model_dir!r} — run `olinda prepare --hard-labels <file> -m {model_dir}` first"
    )
  train_ground_truth(md)

  # Re-fuse the served bundle to include the hard head (blended model.onnx).
  from olinda.export import build_bundle

  build_bundle(md)


@cli.command("export")
@click.option(
  "--model-dir",
  "-m",
  required=True,
  help="A trained model dir (from `learn-soft`/`learn-hard`); (re)builds <dir>/model.onnx.",
)
def export_cmd(model_dir):
  """(Re)build the single self-describing `model.onnx` for a trained model dir.

  Fuses every pipeline stage into ONE ONNX graph — soft-only (`fp → prediction`) or, when a hard head is
  present, the full blend (`fp → prediction` + `surrogate`/`ground_truth`/`ground_truth_soft`/`applicability`).
  The Morgan featurizer config + provenance (RDKit version, reference library, hard-head summary) are embedded
  in the file's metadata, so it is self-describing. Gated on a numeric parity check. Featurization (RDKit)
  stays in Python — the graph consumes a 2048-count Morgan fingerprint.
  """
  from olinda.export import build_bundle

  md = Path(model_dir)
  if not (md / "train_meta.json").exists() and not (md / "xgb.json").exists():
    raise click.ClickException(
      f"{model_dir!r} is not a trained model dir — run `olinda learn-soft -m {model_dir}` first"
    )
  build_bundle(md)


@cli.command("predict", help="Run a model on SMILES via its model.onnx — emits prediction + channels.")
@click.option(
  "--model-dir", "-m", required=True, help="Model dir containing model.onnx (from learn-soft / learn-hard)."
)
@click.option("--input", "-i", "input_path", required=True, help="CSV/TSV/Parquet with a `smiles` column.")
@click.option("--output", "-o", "out_path", required=True, help="Output CSV for the predictions.")
def predict_cmd(model_dir, input_path, out_path):
  """Run the fused `model.onnx`: verify RDKit, featurize the `smiles` column (RDKit Morgan), run the graph.

  Emits a `prediction` column plus the model's channels — `surrogate`, and for hard-label models also
  `ground_truth_soft` / `ground_truth` / `applicability`. The featurizer + RDKit version are read from the
  model's embedded metadata (the file is self-describing).
  """
  from olinda.console import echo, rule
  from olinda.onnx_pipeline import OnnxPipeline

  model_dir = Path(model_dir)
  input_path = Path(input_path)
  rule("olinda · predict", style="cyan", right=str(model_dir))
  if not input_path.exists():
    echo(f"input not found · [dim]{input_path}[/]", "error")
    raise click.ClickException("input file does not exist")
  if not OnnxPipeline.is_bundle(model_dir):
    echo("no model.onnx in the model dir", "error")
    raise click.ClickException(
      f"{model_dir} has no model.onnx — run `olinda learn-soft` / `learn-hard` (or `olinda export`) first"
    )
  _predict_onnx(model_dir, input_path, out_path)


def _predict_onnx(model_dir, input_path, out_path):
  """Predict via the fused ``model.onnx`` (featurizer embedded in metadata); write the channel columns.

  Loads the graph first — which verifies the installed RDKit matches the build recorded in the metadata
  (fingerprints are only reproducible on the exact build) — then featurizes the SMILES and runs the graph.
  """
  import pandas as pd
  import rdkit

  from olinda.console import echo, success
  from olinda.onnx_pipeline import OnnxPipeline, RDKitVersionMismatch

  try:
    pipe = OnnxPipeline.load(model_dir)  # verifies RDKit version against the model's metadata
  except RDKitVersionMismatch as exc:
    echo(str(exc), "error")
    raise click.ClickException(str(exc))
  want = (pipe.meta.get("featurizer") or {}).get("rdkit_version", "?")
  echo(f"rdkit [bold]{rdkit.__version__}[/] · matches model ({want})", "info")

  smiles = _read_smiles(input_path)
  head = "blend" if pipe.meta.get("has_hard") else "soft"
  echo(f"model.onnx · [bold]{head}[/] · {len(smiles):,} SMILES", "run")
  ch = pipe.predict_channels(smiles)

  order = [
    k for k in ("prediction", "surrogate", "ground_truth_soft", "ground_truth", "applicability") if k in ch
  ]
  out_path = Path(out_path)
  pd.DataFrame({"smiles": smiles, **{k: ch[k] for k in order}}).to_csv(out_path, index=False)
  success(f"predictions ({' · '.join(order)}) → [dim]{out_path}[/]")


def _read_smiles(input_path: Path, smiles_col: str = "smiles") -> list[str]:
  """Read the ``smiles`` column from a CSV/TSV/Parquet input for prediction."""
  import pandas as pd

  from olinda.console import echo

  suffix = input_path.suffix.lower()
  if suffix in (".parquet", ".pq"):
    df = pd.read_parquet(str(input_path))
  elif suffix in (".csv", ".tsv"):
    df = pd.read_csv(str(input_path), sep="\t" if suffix == ".tsv" else ",")
  else:
    echo(f"unsupported input format · {suffix} (use .csv / .tsv / .parquet)", "error")
    raise click.ClickException("unsupported input format")
  if smiles_col not in df.columns:
    echo(f"no '{smiles_col}' column in [dim]{input_path.name}[/]", "error")
    raise click.ClickException(f"input needs a '{smiles_col}' column with SMILES")
  return df[smiles_col].astype(str).tolist()


if __name__ == "__main__":
  cli()
