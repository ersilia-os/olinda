"""Short Optuna study to discover good gradient-boosting hyperparameters (especially the learning rate).

Runs a fast, pruned study on a SUBSET of a reference split, using the **auto-selected backend** (XGBoost on
GPU, LightGBM on CPU) and fixing the objective + reweighting exactly the way ``olinda learn-soft`` does. Reports
the best hyperparameters (learning-rate sensitivity breakdown), writes a backend-tagged ``best_params.json``
(canonical param names, portable across engines), and prints a paste-ready ``CANONICAL_DEFAULTS`` block.

Time-bounded and blazing fast: a hard ``time_budget`` (checked every round), study-level ``patience``
(stop once converged), warm-started from known-good configs, Hyperband pruning, live progress, graceful
Ctrl-C. Driven by the ``olinda tune`` CLI command.
"""

from __future__ import annotations

import contextlib
import json
import time
import warnings
from pathlib import Path

import numpy as np

from olinda.console import console, echo, engine_banner, live_status, rule, spinner, summary_panel
from olinda.data import apply_bin_weights, resolve_regression_weights
from olinda.train.backend import CANONICAL_DEFAULTS, get_backend, select_backend

_MAXIMIZE_METRICS = {"auc", "aucpr", "map", "ndcg"}
_MAX_BIN = int(CANONICAL_DEFAULTS["max_bin"])
# learning-rate buckets for the sensitivity breakdown
_LR_BUCKETS = [(0.03, 0.1), (0.1, 0.2), (0.2, 0.35), (0.35, 0.6001)]

# What the study searches — single source of truth (used by _suggest AND the console report so they can't
# drift). Only these two; every other knob comes fixed from CANONICAL_DEFAULTS.
_SEARCH = {"learning_rate": (0.05, 0.3), "min_split_gain": (0.0, 5.0)}

# Tune runs on a ~100k subsample, so its leaf-occupancy floor is set lighter than the ~1.3M full fit
# (CANONICAL_DEFAULTS uses min_child_weight=200) — per LightGBM, min_data_in_leaf scales up with N. This is
# a two-profile heuristic, NOT a runtime ratio: best_params.json carries only the searched params, so
# `learn-soft` applies the full-scale floor from CANONICAL_DEFAULTS automatically.
_TUNE_OVERRIDES = {"min_child_weight": 50.0}

# Warm-start seeds tried FIRST (as full trials) so a strong baseline exists from round one. Canonical
# names (portable across backends). Both sit in the benchmarked-good learning-rate region (~0.05–0.1):
# #1 = the default config; #2 = a slower-eta point. (max_depth is fixed at 8, so it is not seeded here.)
_WARM_START = [
  {"learning_rate": 0.1, "min_split_gain": 0.0},
  {"learning_rate": 0.05, "min_split_gain": 1.0},
]


def _suggest(trial) -> dict:
  """Search space (canonical names) — only the two highest-impact knobs, so the study converges fast.

  Per the 1.4M-compound gradient-boosting benchmark (J. Cheminformatics 2023), the learning rate and the
  minimum split gain matter most under a budget; ``colsample`` is least relevant. ``max_depth`` is **fixed
  at 8** (``CANONICAL_DEFAULTS``) — the final fit is always ~1.3M rows (deep is right), and for LightGBM
  ``num_leaves = min(2**depth-1, 255)`` saturates at depth 8, so there is nothing to gain from searching it.
  learning_rate range brackets the benchmarked sweet spot (~0.1); it needs ``tune_rounds`` high enough
  (≥~2000) that the low end converges, else the short trials bias the study toward high eta.
  """
  lr_lo, lr_hi = _SEARCH["learning_rate"]
  g_lo, g_hi = _SEARCH["min_split_gain"]
  return {
    "learning_rate": trial.suggest_float("learning_rate", lr_lo, lr_hi, log=True),
    "min_split_gain": trial.suggest_float("min_split_gain", g_lo, g_hi),
  }


def _load_subset(run_dir: Path, max_rows: int | None, seed: int):
  """Load a random train/val subsample for the run's single column; also returns the full split size.

  The full split is far larger than a study needs, so rows are drawn at random (rather than sliced)
  while preserving the split's train:val proportion.
  """
  from olinda import run as runlib
  from olinda.data.matrix import ReferenceMatrix
  from olinda.data.fetch import MORGAN_FINGERPRINTS_FILENAME, OLINDA_HOME

  manifest = runlib.read_manifest(run_dir)
  columns = manifest["columns"]
  if len(columns) != 1:
    raise RuntimeError(
      f"this run has {len(columns)} columns; `olinda tune` supports single-column runs only. "
      "Train the multi-column run with the built-in defaults, or prepare one column at a time."
    )
  col = columns[0]
  y = runlib.read_target(run_dir, col["id"])
  train_idx, val_idx = runlib.read_split(run_dir, col["id"])

  total = len(train_idx) + len(val_idx)
  frac = 1.0 if not max_rows or max_rows >= total else max_rows / total
  rng = np.random.default_rng(seed)

  def pick(idx):
    n = max(1, round(len(idx) * frac))
    return np.sort(rng.choice(idx, size=n, replace=False)) if n < len(idx) else idx

  sub_train, sub_val = pick(train_idx), pick(val_idx)
  matrix = ReferenceMatrix.load(OLINDA_HOME / MORGAN_FINGERPRINTS_FILENAME, limit=runlib.row_limit(manifest))
  # The same guard learn-soft and learn-hard apply, which tune was missing: the indices above are
  # positional, so a library regenerated since `prepare` pairs each row's features with another
  # molecule's label. The study would run and report plausible numbers for the wrong data.
  matrix.assert_matches(manifest["reference_library"])
  return (
    matrix.gather(sub_train),
    np.asarray(y[sub_train], dtype=np.float32),
    matrix.gather(sub_val),
    np.asarray(y[sub_val], dtype=np.float32),
    total,
  )


def run_tuning(
  in_dir: str | Path,
  *,
  max_rows: int | None = 100000,
  trials: int = 100,
  time_budget: int = 900,
  tune_rounds: int = 2000,
  early_stopping: int = 50,
  patience: int = 12,
  reweight: str = "auto",
  seed: int = 42,
  out: str | Path = "best_params.json",
) -> dict:
  """Run the short Optuna study on the auto-selected backend and report/persist the best hyperparameters.

  Returns the tuned canonical params (also written to ``out`` with a ``backend`` tag). Optuna is imported
  lazily; a missing ``[train]`` extra raises a friendly error.
  """
  try:
    import optuna
  except ImportError as e:  # pragma: no cover - environment-dependent
    raise RuntimeError("optuna is required — install the train extra:  pip install -e '.[train]'") from e
  optuna.logging.set_verbosity(optuna.logging.WARNING)

  in_dir = Path(in_dir)
  rule("olinda · tune", style="green", right=str(in_dir))
  backend_name, device, backend_reason = select_backend()
  be = get_backend(backend_name, device)
  engine_banner(backend_name, device, backend_reason)

  # Subsample train+val to ~max_rows total, preserving the split's train:val proportion so the tuning
  # val fraction matches what `prepare` produced.
  xtr, ytr, xva, yva, total = _load_subset(in_dir, max_rows, seed)

  # Single squared-error objective (like `olinda learn-soft`); reweighting resolved the same way.
  obj_native = be.objective_params()
  em = obj_native.get("eval_metric", obj_native.get("metric", "rmse"))
  metric = em[-1] if isinstance(em, (list, tuple)) else em
  direction = "maximize" if metric in _MAXIMIZE_METRICS else "minimize"
  best_of = max if direction == "maximize" else min
  edges, weights, rw_info = resolve_regression_weights(ytr, mode=reweight)

  used = xtr.shape[0] + xva.shape[0]
  echo(
    f"tuning on {xtr.shape[0]:,} train + {xva.shape[0]:,} val = {used:,} rows "
    f"({100 * used / total:.0f}% of the {total:,} prepared; --max-rows={max_rows or 'all'}) · "
    f"{xtr.shape[1]}-d · engine {backend_name} (same as learn-soft) · metric={metric} · reweight={rw_info['strategy']}",
    "run",
  )
  if not time_budget:
    budget_txt = "no time cap"
  elif time_budget >= 60:
    budget_txt = f"{time_budget // 60}m cap"
  else:
    budget_txt = f"{time_budget}s cap"
  lr_lo, lr_hi = _SEARCH["learning_rate"]
  g_lo, g_hi = _SEARCH["min_split_gain"]
  tune_mcw = _TUNE_OVERRIDES["min_child_weight"]
  cd = CANONICAL_DEFAULTS
  echo(f"optimizing: learning_rate [{lr_lo}–{lr_hi}] · min_split_gain [{g_lo:g}–{g_hi:g}]", "run")
  echo(
    f"fixed (tune @~{used // 1000}k rows): max_depth {cd['max_depth']} · min_child_weight {tune_mcw:g} · "
    f"subsample {cd['subsample']} · colsample {cd['colsample']} · reg_lambda {cd['reg_lambda']:g} · "
    f"reg_alpha {cd['reg_alpha']:g} · max_bin {cd['max_bin']}",
    "run",
  )
  echo(
    f"→ learn-soft (full ~{total // 1000}k rows) uses these too, with min_child_weight "
    f"{cd['min_child_weight']:g} (leaf floor scaled up with N, per LightGBM)",
    "info",
  )
  echo(
    f"study: warm-started · ≤{trials} trials · ≤{tune_rounds} rounds/trial · Hyperband · patience "
    f"{patience} · {budget_txt}. Ctrl-C keeps the best-so-far.",
    "run",
  )

  wtr = apply_bin_weights(ytr, edges, weights) if weights is not None else None
  wva = apply_bin_weights(yva, edges, weights) if weights is not None else None
  dtrain = be.dataset(xtr, ytr, wtr, _MAX_BIN)
  dval = be.dataset(xva, yva, wva, _MAX_BIN, reference=dtrain)

  start = time.perf_counter()
  progress = {"best": None, "stale": 0}
  live = [lambda _s: None]

  def _elapsed() -> str:
    s = int(time.perf_counter() - start)
    return f"{s}s" if s < 60 else f"{s // 60}m{s % 60:02d}s"

  def objective_fn(trial):
    canonical = {**CANONICAL_DEFAULTS, **_TUNE_OVERRIDES, **_suggest(trial)}
    native = {**be.translate(canonical), **obj_native}

    def _on_iter(iteration, score):
      if time_budget and time.perf_counter() - start > time_budget:
        raise optuna.TrialPruned()  # hard time cap (per round → ≤1 round overshoot)
      trial.report(score, iteration)
      if iteration % 25 == 0:
        b = progress["best"]
        btxt = f"{b:.4f}" if b is not None else "—"
        live[0](
          f"  [bold cyan]{spinner(iteration)} tuning[/] trial [bold]{trial.number + 1}[/][dim]/{trials}[/]"
          f" · round {iteration}/{tune_rounds} · [dim]{metric}[/] [bold cyan]{score:.4f}[/]"
          f" · best [bold]{btxt}[/] [dim]· {_elapsed()}[/]"
        )
      if trial.should_prune():
        raise optuna.TrialPruned()

    best_score, best_it, _m = be.train_trial(dtrain, dval, native, tune_rounds, early_stopping, _on_iter)
    trial.set_user_attr("best_iteration", int(best_it))
    return best_score

  def _after_trial(study, trial):
    prev = progress["best"]
    with contextlib.suppress(ValueError):
      progress["best"] = study.best_value
    if trial.state == optuna.trial.TrialState.COMPLETE:
      improved = prev is None or progress["best"] != prev
      progress["stale"] = 0 if improved else progress["stale"] + 1
    if not console.is_terminal:
      val = f"{trial.value:.5f}" if trial.value is not None else trial.state.name.lower()
      b = progress["best"]
      echo(
        f"trial {trial.number + 1}/{trials} · {val} · best {b:.5f} · {_elapsed()}"
        if b is not None
        else f"trial {trial.number + 1}/{trials} · {val} · {_elapsed()}",
        "info",
      )
    if patience and progress["stale"] >= patience:
      echo(f"converged — no improvement in {patience} trials; stopping early", "info")
      study.stop()
    elif time_budget and time.perf_counter() - start > time_budget:
      study.stop()

  # multivariate TPE (models param correlations); suppress ONLY its experimental notice, narrowly.
  try:
    with warnings.catch_warnings():
      warnings.filterwarnings(
        "ignore", message=".*multivariate.*", category=optuna.exceptions.ExperimentalWarning
      )
      sampler = optuna.samplers.TPESampler(multivariate=True, seed=seed)
  except TypeError:
    sampler = optuna.samplers.TPESampler(seed=seed)

  study = optuna.create_study(direction=direction, sampler=sampler, pruner=optuna.pruners.HyperbandPruner())
  for seed_params in _WARM_START:
    study.enqueue_trial(seed_params)
  with live_status() as update:
    live[0] = update
    try:
      study.optimize(objective_fn, n_trials=trials, callbacks=[_after_trial], gc_after_trial=True)
    except KeyboardInterrupt:
      echo("interrupted — reporting best of the completed trials", "warning")
  dt = time.perf_counter() - start

  completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
  pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
  if not completed:
    echo("no trials completed — nothing to report (try a larger --time-budget or fewer rounds)", "error")
    return {}
  best = study.best_trial
  best_iter = int(best.user_attrs.get("best_iteration", -1)) + 1

  lr_rows = []
  for lo, hi in _LR_BUCKETS:
    vals = [t.value for t in completed if lo <= t.params["learning_rate"] < hi]
    if vals:
      lr_rows.append((
        f"lr {lo:.2f}–{min(hi, 0.6):.2f}",
        f"{best_of(vals):.5f}  [dim]· {len(vals)} trials[/]",
      ))

  # Write only the SEARCHED params (canonical, backend-tagged); `learn-soft -m <dir>` merges them over
  # CANONICAL_DEFAULTS, so the fixed knobs (incl. the full-scale min_child_weight) come from there.
  tuned = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.params.items()}
  tuned["backend"] = backend_name
  out = Path(out)
  out.parent.mkdir(parents=True, exist_ok=True)
  with open(out, "w") as fp:
    json.dump(tuned, fp, indent=2)

  elapsed = f"{int(dt)}s" if dt < 60 else f"{int(dt) // 60}m{int(dt) % 60:02d}s"
  others = " · ".join(f"{k} {v:.3g}" for k, v in best.params.items() if k != "learning_rate")
  rows = [
    (f"Best {metric}", f"[bold]{best.value:.5f}[/]  [dim]· at ~{best_iter} rounds (of ≤{tune_rounds})[/]"),
    ("Best learning_rate", f"[bold]{best.params['learning_rate']:.4f}[/]"),
    ("Best params", f"[dim]{others}[/]"),
    (
      "Elapsed",
      f"[bold]{elapsed}[/]  [dim]· {len(completed)} trials + {len(pruned)} pruned · {backend_name}·{device}[/]",
    ),
    *lr_rows,
    ("Saved", f"[dim]{out}[/]"),
  ]
  summary_panel("olinda · tune", rows, border_style="green", icon="✓")

  # paste-ready CANONICAL_DEFAULTS block (only searched keys overridden)
  paste = dict(CANONICAL_DEFAULTS)
  for k, v in best.params.items():
    paste[k] = round(v, 4) if isinstance(v, float) else v
  echo(
    "Paste-ready CANONICAL_DEFAULTS (backend + objective are resolved automatically at train time):", "info"
  )
  body = ",\n".join(f"  {k!r}: {v!r}" for k, v in paste.items())
  print(f"CANONICAL_DEFAULTS = {{\n{body},\n}}")

  echo(
    "Tuned on a subsample; best_params.json holds only learning_rate + min_split_gain — `learn-soft` applies "
    "the full-scale fixed defaults (min_child_weight 200) and re-fits the round count on all the data.",
    "warning",
  )
  return tuned
