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
  import time

  from olinda.console import (
    STEP_COLORS,
    elapsed,
    path as cpath,
    resources,
    rule,
    set_active_color,
    summary_panel,
  )

  md = Path(model_dir)
  started = time.time()
  set_active_color(STEP_COLORS["fit"])
  rule("olinda · fit", style=STEP_COLORS["fit"], right=resources())

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
      ("Model", f"[dim]{cpath(md / 'model.onnx')}[/]"),
      ("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]"),
    ],
    border_style=STEP_COLORS["fit"],
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
  """Prepare every teacher column in --soft-labels into one run directory.

  Each value column gets its own value-stratified train/val split, recorded as row indices rather
  than a copy of the descriptor matrix. With --hard-labels (a wide file: SMILES plus one column per
  assay, empty where untested), each hard column is matched onto its soft column by name and
  featurized into that column's hard.h5 for a later `learn-hard`.
  """
  import h5py

  from olinda import run as runlib
  import time

  from olinda.console import (
    STEP_COLORS,
    echo,
    elapsed,
    path as cpath,
    rule,
    set_active_color,
    step,
    summary_panel,
  )
  from olinda.data import (
    OLINDA_HOME,
    MORGAN_FINGERPRINTS_FILENAME,
    check_column_budget,
    load_reference_calcs_frame,
    match_hard_columns,
    split_reference_to_indices,
  )

  started = time.time()
  set_active_color(STEP_COLORS["prepare"])
  rule("olinda · prepare", style=STEP_COLORS["prepare"], right=cpath(model_dir))
  md = Path(model_dir)
  md.mkdir(parents=True, exist_ok=True)

  descriptors = OLINDA_HOME / MORGAN_FINGERPRINTS_FILENAME
  if not descriptors.exists():
    echo(f"reference library missing · [dim]{descriptors}[/]", "error")
    raise click.ClickException(
      f"Morgan descriptors missing at {descriptors} — run `olinda setup` "
      "(or scripts/compute_morgan_fingerprints.py to generate them locally)."
    )

  features = {"features": "morgan"}
  with h5py.File(descriptors, "r") as f:
    n_rows, dim = int(f["data"].shape[0]), int(f["data"].shape[1])
    for k in ("radius", "nbits"):
      if k in f.attrs:
        features[k] = int(f.attrs[k])

  n_steps = 3 if hard_labels is not None else 2

  # --- soft labels: every value column in the file, verified against the library once -------
  step(1, n_steps, "reading teacher columns")
  try:
    all_cols, targets = load_reference_calcs_frame(soft_labels, descriptors)
    check_column_budget(all_cols)
  except ValueError as exc:
    raise click.ClickException(str(exc)) from exc
  echo(f"{len(all_cols)} teacher column(s): [bold]{', '.join(all_cols)}[/]", "run")

  # --- hard labels: match them onto the soft columns before doing any work ------------------
  hard_map: dict = {}
  if hard_labels is not None:
    from olinda.data.reference import _read_table, resolve_smiles_frame

    _, hard_values = resolve_smiles_frame(_read_table(hard_labels))
    try:
      hard_map = match_hard_columns(all_cols, [str(c) for c in hard_values.columns])
    except ValueError as exc:
      raise click.ClickException(str(exc)) from exc
    for hard_col, soft_col in hard_map.items():
      note = "exact" if hard_col == soft_col else "suffix"
      echo(f"hard '{hard_col}' → '{soft_col}' ([dim]{note}[/])", "run")

  # --- per-column value-stratified splits; the descriptor matrix is never copied ------------
  step(2, n_steps, "planning per-column splits")
  manifest = runlib.new_manifest(
    soft_labels=soft_labels,
    hard_labels=hard_labels,
    reference={"name": MORGAN_FINGERPRINTS_FILENAME, "n_rows": n_rows, "dim": dim},
    features=features,
    val_frac=val_frac,
    seed=42,
    limit=max_samples,
  )
  by_id, splits, name_to_id = {}, {}, {}
  for name in all_cols:
    y = targets[name]
    train_idx, val_idx = split_reference_to_indices(y, val_frac=val_frac, limit=max_samples)[:2]
    entry = runlib.add_column(manifest, name=name, y=y, train_idx=train_idx, val_idx=val_idx)
    by_id[entry["id"]] = y
    splits[entry["id"]] = (train_idx, val_idx)
    name_to_id[name] = entry["id"]
    runlib.column_dir(md, entry["id"]).mkdir(parents=True, exist_ok=True)
    echo(f"{name}: [bold]{len(train_idx):,}[/] train · [bold]{len(val_idx):,}[/] val", "run")

  runlib.write_targets(md, by_id)
  runlib.write_splits(md, splits)

  # --- hard labels: featurize once, then slice per matched column ---------------------------
  hard_info: dict = {}
  if hard_map:
    from olinda.ground_truth import prepare_hard_labels_wide

    step(3, n_steps, "featurizing hard labels")
    try:
      hard_info = prepare_hard_labels_wide(
        hard_labels, md, {h: name_to_id[s] for h, s in hard_map.items()}, task=task
      )
    except (ValueError, NotImplementedError) as exc:
      raise click.ClickException(str(exc)) from exc
    for hard_col, info in hard_info.items():
      soft_col = hard_map[hard_col]
      col = runlib.find_column(manifest, soft_col)
      col["hard"] = {
        "source_column": hard_col,
        "match": "exact" if hard_col == soft_col else "suffix",
        **info,
      }
      pos = f" · {info['n_positive']} positive" if info["n_positive"] is not None else ""
      echo(f"{soft_col}: [bold]{info['n']:,}[/] hard rows · {info['task']}{pos}", "run")

  runlib.write_manifest(md, manifest)

  n_hard = sum(1 for c in manifest["columns"] if c.get("hard"))
  summary_panel(
    "olinda · prepare",
    [
      ("Columns", f"[bold]{len(manifest['columns'])}[/] · {n_hard} with hard labels"),
      ("Split", f"value-stratified per column · val_frac {val_frac}"),
      ("Targets", f"[dim]{cpath(md / runlib.TARGETS_NAME)}[/]"),
      ("Saved", f"[dim]{cpath(md)}[/]"),
      ("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]"),
    ],
    border_style=STEP_COLORS["prepare"],
    icon="✓",
  )


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
  from olinda import run as runlib
  import time

  from olinda.console import (
    STEP_COLORS,
    LiveTable,
    echo,
    elapsed,
    engine_banner,
    path as cpath,
    rule,
    set_active_color,
    summary_panel,
  )
  from olinda.data import OLINDA_HOME, MORGAN_FINGERPRINTS_FILENAME
  from olinda.data.matrix import ReferenceMatrix
  from olinda.train.backend import get_backend, select_backend
  from olinda.train.column import train_one_column

  model_dir = Path(model_dir)
  try:
    manifest = runlib.read_manifest(model_dir)
  except FileNotFoundError as exc:
    raise click.ClickException(str(exc)) from exc

  columns = manifest["columns"]
  features = manifest["features"].get("features", "morgan")
  dim = manifest["reference_library"]["dim"]
  started = time.time()
  set_active_color(STEP_COLORS["learn-soft"])
  rule(
    "olinda · learn-soft",
    style=STEP_COLORS["learn-soft"],
    right=f"{len(columns)} column(s) · {features} · {dim}-dim",
  )

  # Engine auto-selected by device (GPU→XGBoost, CPU→LightGBM); OLINDA_BACKEND overrides.
  backend_name, device, backend_reason = select_backend()
  be = get_backend(backend_name, device)
  engine_banner(backend_name, device, backend_reason)

  # The descriptor matrix is identical for every column, so it is read once and addressed by index.
  descriptors = OLINDA_HOME / MORGAN_FINGERPRINTS_FILENAME
  if not descriptors.exists():
    raise click.ClickException(f"reference library missing at {descriptors} — run `olinda setup`")
  echo(f"loading reference descriptors · [dim]{descriptors.name}[/]", "run")
  matrix = ReferenceMatrix.load(descriptors)
  try:
    matrix.assert_matches(manifest["reference_library"])
  except ValueError as exc:
    raise click.ClickException(str(exc)) from exc

  results = []
  with LiveTable(
    [c["name"] for c in columns],
    title="Distilling columns",
    fields=["R²", "ρ", "RMSE", "Trees", "Time"],
    item_label="Column",
    running_verb="training",
    color=STEP_COLORS["learn-soft"],
  ) as table:
    for col in columns:
      table.start(col["name"])
      col_started = time.time()
      result = train_one_column(model_dir, manifest, col, matrix, be, backend_name, num_boost_round)
      m = result["metrics"]
      table.finish(
        col["name"],
        **{
          "R²": f"{m['r2']:.4f}",
          "ρ": f"{m['spearman']:.4f}",
          "RMSE": f"{m['rmse']:.5f}",
          "Trees": f"{result['n_trees']:,}",
          "Time": elapsed(time.time() - col_started),
        },
      )
      results.append(result)
      col["status"]["soft_trained"] = True
      runlib.write_manifest(model_dir, manifest)

  del matrix  # release ~2.8 GB before fusing, which is itself memory-hungry

  from olinda.export import build_bundle

  try:
    build_bundle(model_dir)
  except (ValueError, FileNotFoundError) as exc:
    raise click.ClickException(str(exc)) from exc

  summary_panel(
    "olinda · learn-soft",
    [
      ("Columns", f"[bold]{len(results)}[/] trained · {backend_name}"),
      *[
        (
          r["name"],
          f"R² [bold]{r['metrics']['r2']:.4f}[/] · ρ {r['metrics']['spearman']:.4f} · {r['n_trees']:,} trees",
        )
        for r in results
      ],
      ("Model", f"[dim]{cpath(model_dir / 'model.onnx')}[/]"),
      ("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]"),
    ],
    border_style=STEP_COLORS["learn-soft"],
    icon="✓",
  )


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
  import time

  from olinda import run as runlib
  from olinda.console import (
    STEP_COLORS,
    echo,
    elapsed,
    path as cpath,
    rule,
    set_active_color,
    summary_panel,
  )
  from olinda.ground_truth import HARD_H5_NAME, train_ground_truth

  md = Path(model_dir)
  started = time.time()
  set_active_color(STEP_COLORS["learn-hard"])
  try:
    manifest = runlib.read_manifest(md)
  except FileNotFoundError as exc:
    raise click.ClickException(str(exc)) from exc

  with_hard = [c for c in manifest["columns"] if (runlib.column_dir(md, c["id"]) / HARD_H5_NAME).exists()]
  untrained = [
    c["name"]
    for c in manifest["columns"]
    if not (runlib.column_dir(md, c["id"]) / "train_meta.json").exists()
  ]
  if untrained:
    raise click.ClickException(
      f"{len(untrained)} column(s) have no surrogate yet ({', '.join(untrained)}) — "
      f"run `olinda learn-soft -m {model_dir}` first. learn-hard fuses at the end and would "
      "otherwise fail there, after training every ground-truth head."
    )

  if not with_hard:
    raise click.ClickException(
      f"no column in {model_dir!r} has hard labels — run "
      f"`olinda prepare -s <soft> -h <hard> -m {model_dir}` first"
    )

  # One resident copy of the library for the whole run: each column otherwise reopened and streamed
  # it twice (scoring G, then fitting the gate), so a 10-column run read 2.8 GB twenty times over.
  from olinda.data.fetch import MORGAN_FINGERPRINTS_FILENAME, OLINDA_HOME
  from olinda.data.matrix import ReferenceMatrix

  echo(f"loading reference descriptors · [dim]{MORGAN_FINGERPRINTS_FILENAME}[/]", "run")
  matrix = ReferenceMatrix.load(OLINDA_HOME / MORGAN_FINGERPRINTS_FILENAME)
  try:
    matrix.assert_matches(manifest["reference_library"])
  except ValueError as exc:
    raise click.ClickException(str(exc)) from exc

  for i, col in enumerate(with_hard, start=1):
    rule(
      f"STEP {i}/{len(with_hard)} · {col['name']}",
      style=STEP_COLORS["learn-hard"],
      right=f"column {i}/{len(with_hard)}",
    )
    train_ground_truth(
      runlib.column_dir(md, col["id"]), soft=runlib.read_target(md, col["id"]), matrix=matrix
    )
    col["status"]["hard_trained"] = True
    runlib.write_manifest(md, manifest)

  # Re-fuse the served bundle so every column's hard head is included.
  from olinda.export import build_bundle

  build_bundle(md)
  summary_panel(
    "olinda · learn-hard",
    [
      ("Columns", f"[bold]{len(with_hard)}[/] of {len(manifest['columns'])} have a hard head"),
      *[(c["name"], f"[dim]{(c.get('hard') or {}).get('n', '?')} labelled compounds[/]") for c in with_hard],
      ("Model", f"[dim]{cpath(md / 'model.onnx')}[/]"),
      ("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]"),
    ],
    border_style=STEP_COLORS["learn-hard"],
    icon="✓",
  )


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

  from olinda import run as runlib

  md = Path(model_dir)
  try:
    manifest = runlib.read_manifest(md)
  except FileNotFoundError as exc:
    raise click.ClickException(str(exc)) from exc

  # Models live per column, not at the run root — check every column has one before fusing.
  missing = [
    col["name"]
    for col in manifest["columns"]
    if not (runlib.column_dir(md, col["id"]) / "train_meta.json").exists()
  ]
  if missing:
    raise click.ClickException(
      f"{len(missing)} column(s) not trained yet ({', '.join(missing)}) — "
      f"run `olinda learn-soft -m {model_dir}` first"
    )
  try:
    build_bundle(md)
  except (ValueError, FileNotFoundError) as exc:
    raise click.ClickException(str(exc)) from exc


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
  from olinda.console import STEP_COLORS, echo, path as cpath, rule, set_active_color

  model_dir = Path(model_dir)
  input_path = Path(input_path)
  set_active_color(STEP_COLORS["predict"])
  rule("olinda · predict", style=STEP_COLORS["predict"], right=cpath(model_dir))
  if not input_path.exists():
    echo(f"input not found · [dim]{input_path}[/]", "error")
    raise click.ClickException("input file does not exist")
  from olinda.artifact import RDKitVersionMismatch
  from olinda.predict import predict_file

  if not (Path(model_dir) / "model.onnx").exists():
    echo("no model.onnx in the model dir", "error")
    raise click.ClickException(
      f"{model_dir} has no model.onnx — run `olinda learn-soft` / `learn-hard` (or `olinda export`) first"
    )
  try:
    predict_file(model_dir, input_path, out_path)
  except (RDKitVersionMismatch, ValueError, FileNotFoundError) as exc:
    raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
  cli()
