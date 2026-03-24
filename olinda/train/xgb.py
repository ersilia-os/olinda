from __future__ import annotations

import logging
import os
import time
from functools import lru_cache

import numpy as np

import xgboost as xgb
from xgboost.callback import TrainingCallback
from rich.progress import (
  Progress,
  SpinnerColumn,
  TextColumn,
  TaskProgressColumn,
  TimeElapsedColumn,
  TimeRemainingColumn,
  ProgressColumn,
)
from rich.progress_bar import ProgressBar
from olinda.helpers import logger


_TASK_SPECS = {
  "regression": {"objective": "reg:squarederror", "eval_metric": "mae"},
  "classification": {"objective": "binary:logistic", "eval_metric": "aucpr"},
}

_MAXIMIZE_METRICS = frozenset({"aucpr", "auc", "map", "ndcg"})

_BASE_PARAMS = {
  "max_depth": 8,
  "eta": 0.05,
  "subsample": 0.9,
  "colsample_bytree": 0.9,
  "lambda": 1.0,
  "alpha": 0.0,
  "tree_method": "hist",
}


def _truthy(value: object) -> bool:
  return str(value).strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def detect_training_device() -> tuple[str, str]:
  forced = os.environ.get("OLINDA_XGB_DEVICE")
  if forced:
    forced = forced.strip().lower()
    if forced in {"cpu", "cuda"}:
      return forced, f"forced by OLINDA_XGB_DEVICE={forced}"

  try:
    build_info = xgb.build_info()
  except Exception:
    build_info = {}

  if not _truthy(build_info.get("USE_CUDA")):
    return "cpu", "XGBoost build has no CUDA support"

  try:
    X = np.asarray([[0.0], [1.0]], dtype=np.float32)
    y = np.asarray([0.0, 1.0], dtype=np.float32)
    dtrain = xgb.DMatrix(X, label=y)
    booster = xgb.train(
      params={
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 1,
        "eta": 1.0,
      },
      dtrain=dtrain,
      num_boost_round=1,
      verbose_eval=False,
    )
    booster.predict(dtrain)
  except Exception as exc:
    return "cpu", f"CUDA runtime unavailable ({exc})"

  return "cuda", "detected CUDA-capable XGBoost runtime"


class _RichTrainCallback(TrainingCallback):
  def __init__(self, progress: Progress, task_id, total_rounds: int, eval_metric: str,
               higher_is_better: bool = False):
    self.progress = progress
    self.task_id = task_id
    self.total_rounds = total_rounds
    self.eval_metric = eval_metric
    self.higher_is_better = higher_is_better
    self.best_score = float("-inf") if higher_is_better else float("inf")
    self.best_round = 0
    self.history: list[dict] = []

  def _extract_metric(self, evals_log):
    if not evals_log:
      return None, None
    ds_name = "val" if "val" in evals_log else next(iter(evals_log))
    metrics = evals_log.get(ds_name, {})
    if not metrics:
      return None, None
    if self.eval_metric in metrics:
      metric_name = self.eval_metric
      vals = metrics[metric_name]
    else:
      metric_name, vals = next(iter(metrics.items()))
    if not vals:
      return None, None
    score = vals[-1]
    if isinstance(score, (list, tuple)):
      score = score[0]
    return float(score), f"{ds_name}:{metric_name}"

  def after_iteration(self, model, epoch, evals_log) -> bool:
    self.progress.update(self.task_id, completed=min(epoch + 1, self.total_rounds))

    score, label = self._extract_metric(evals_log)
    if score is not None:
      improved = (score > self.best_score) if self.higher_is_better else (score < self.best_score)
      if improved:
        self.best_score = score
        self.best_round = epoch
      self.history.append({
        "iteration": int(epoch + 1),
        "metric": label,
        "value": float(score),
        "best": float(self.best_score),
      })
      self.progress.update(
        self.task_id,
        metric=f"{label}={score:.6f}",
        best=f"best={self.best_score:.6f}",
      )
    else:
      self.progress.update(self.task_id, metric="", best="")
    return False

  def after_training(self, model):
    try:
      total = int(model.num_boosted_rounds())
    except Exception:
      total = None
    if total:
      self.progress.update(self.task_id, total=total, completed=total)
    return model


class _PulseBarColumn(ProgressColumn):
  def __init__(
    self,
    bar_width: int = 40,
    style: str = "bar.back",
    complete_style: str = "bar.complete",
    finished_style: str = "bar.finished",
    pulse_style: str = "bar.pulse",
  ) -> None:
    super().__init__()
    self.bar_width = int(bar_width)
    self.style = style
    self.complete_style = complete_style
    self.finished_style = finished_style
    self.pulse_style = pulse_style

  def render(self, task) -> ProgressBar:
    return ProgressBar(
      total=task.total,
      completed=task.completed,
      width=max(1, self.bar_width),
      pulse=not task.finished,
      animation_time=task.get_time(),
      style=self.style,
      complete_style=self.complete_style,
      finished_style=self.finished_style,
      pulse_style=self.pulse_style,
    )


class _OptunaPruningCallback(TrainingCallback):
  def __init__(self, trial, metric: str) -> None:
    self.trial = trial
    self.metric = metric

  def after_iteration(self, model, epoch, evals_log) -> bool:
    if not evals_log:
      return False
    metrics = evals_log.get("val", {})
    if not metrics:
      return False
    vals = metrics.get(self.metric)
    if not vals:
      return False
    score = vals[-1]
    if isinstance(score, (list, tuple)):
      score = score[0]
    self.trial.report(float(score), step=int(epoch))
    if self.trial.should_prune():
      import optuna

      raise optuna.TrialPruned()
    return False


def make_progress(transient: bool = True) -> Progress:
  return Progress(
    SpinnerColumn(),
    TextColumn("[bold cyan]{task.fields[desc]}[/]"),
    _PulseBarColumn(bar_width=38),
    TaskProgressColumn(),
    TextColumn("{task.completed}/{task.total}"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    transient=transient,
    console=logger.rich,
  )


class _LimitedDataIter(xgb.DataIter):
  def __init__(self, base: xgb.DataIter, max_rows: int) -> None:
    super().__init__()
    self.base = base
    self.max_rows = int(max_rows)
    self.remaining = int(max_rows)

  def reset(self) -> None:
    self.base.reset()
    self.remaining = int(self.max_rows)

  def next(self, input_data) -> bool:  # type: ignore[override]
    if self.remaining <= 0:
      return False

    def _input_data(data, label=None, weight=None):
      n = len(data)
      if n > self.remaining:
        data = data[: self.remaining]
        if label is not None:
          label = label[: self.remaining]
        if weight is not None:
          weight = weight[: self.remaining]
        self.remaining = 0
      else:
        self.remaining -= n
      input_data(data=data, label=label, weight=weight)

    return self.base.next(_input_data)


class XGBTrainer:
  def __init__(
    self,
    task: str = "regression",
    params: dict | None = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int | None = 50,
    batch_rows: int = 65536,
    seed: int = 42,
    max_bin: int = 256,
    scale_pos_weight: float | None = None,
  ) -> None:
    if task not in _TASK_SPECS:
      raise ValueError(f"task must be one of {list(_TASK_SPECS)}")
    self.task = task
    self.params = params or {}
    self.num_boost_round = int(num_boost_round)
    self.early_stopping_rounds = early_stopping_rounds
    self.batch_rows = int(batch_rows)
    self.seed = int(seed)
    self.max_bin = int(max_bin)
    self.scale_pos_weight = scale_pos_weight

  def resolved_params(self) -> dict:
    p = {}
    p.update(_BASE_PARAMS)
    p.update(_TASK_SPECS[self.task])
    p.update(self.params)
    p["seed"] = self.seed
    if self.scale_pos_weight is not None and self.task == "classification":
      p["scale_pos_weight"] = self.scale_pos_weight
      logger.info(f"Imbalance-aware: scale_pos_weight={self.scale_pos_weight:.6f}")
    device, device_reason = detect_training_device()
    if device == "cuda":
      p["device"] = "cuda"
    else:
      p.pop("device", None)
    logger.info(f"Training device: {device} ({device_reason})")
    return p

  def fit_external(
    self,
    train_iter: xgb.DataIter,
    val_iter: xgb.DataIter | None = None,
    time_budget: int | None = None,
    n_trials: int = 50,
  ) -> tuple[xgb.Booster, dict]:
    params = self.resolved_params()

    logger.info("Resolved training parameters")
    self._log_params(params)

    tune_info = None
    if time_budget is not None and time_budget > 0:
      if val_iter is None:
        raise ValueError("tuning requires val_iter")
      best, tune_info = self._tune_external(
        train_iter=train_iter,
        val_iter=val_iter,
        base_params=params,
        time_budget=int(time_budget),
        n_trials=int(n_trials),
      )
      params.update(best)
      logger.info("Applied Optuna best parameters")
      self._log_params(params)

    booster, train_info = self._train_external(
      train_iter=train_iter,
      val_iter=val_iter,
      params=params,
    )

    device = params.get("device", "cpu")
    meta = {
      "task": self.task,
      "params": params,
      "training_device": device,
    }
    meta.update(train_info)
    if tune_info:
      meta["tuning"] = tune_info

    self._log_run_summary(train_info, tune_info, eval_metric=params.get("eval_metric", "mae"))
    return booster, meta

  def _train_external(
    self,
    train_iter: xgb.DataIter,
    val_iter: xgb.DataIter | None,
    params: dict,
  ) -> tuple[xgb.Booster, dict]:
    logger.info("Building QuantileDMatrix (train)...")
    t0 = time.perf_counter()
    dtrain = xgb.QuantileDMatrix(train_iter, max_bin=self.max_bin)
    dt_build = time.perf_counter() - t0
    logger.info(f"QuantileDMatrix (train) built in {dt_build:.2f}s")

    evals = [(dtrain, "train")]
    if val_iter is not None:
      logger.info("Building QuantileDMatrix (val)...")
      t0 = time.perf_counter()
      dval = xgb.QuantileDMatrix(val_iter, ref=dtrain, max_bin=self.max_bin)
      dt_val = time.perf_counter() - t0
      logger.info(f"QuantileDMatrix (val) built in {dt_val:.2f}s")
      evals.append((dval, "val"))
    else:
      logger.warning("No validation iterator provided; training will not log validation loss")

    eval_metric = params.get("eval_metric", "mae")

    with Progress(
      SpinnerColumn(),
      TextColumn("[bold cyan]TRAIN[/bold cyan]"),
      _PulseBarColumn(bar_width=48),
      TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
      TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
      TextColumn("{task.fields[metric]}"),
      TextColumn("{task.fields[best]}"),
      TimeElapsedColumn(),
      TimeRemainingColumn(),
      transient=True,
    ) as prog:
      task_id = prog.add_task("boosting", total=self.num_boost_round, metric="", best="")
      higher = eval_metric in _MAXIMIZE_METRICS
      cb = _RichTrainCallback(prog, task_id, self.num_boost_round, eval_metric,
                              higher_is_better=higher)

      t0 = time.perf_counter()
      evals_result = {}
      booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=self.num_boost_round,
        evals=evals,
        early_stopping_rounds=self.early_stopping_rounds,
        verbose_eval=False,
        evals_result=evals_result,
        callbacks=[cb],
      )
      train_time = time.perf_counter() - t0

    logger.info("Training finished; summarizing results")
    best_score = getattr(booster, "best_score", cb.best_score)
    best_round = getattr(booster, "best_iteration", cb.best_round)
    n_trees = int(booster.num_boosted_rounds())

    info = {
      "n_trees": n_trees,
      "best_round": int(best_round),
      "best_val_score": float(best_score),
      "train_seconds": round(train_time, 2),
      "train_history": cb.history,
      "evals_result": evals_result,
    }

    return booster, info

  def _tune_external(
    self,
    train_iter: xgb.DataIter,
    val_iter: xgb.DataIter,
    base_params: dict,
    time_budget: int,
    n_trials: int,
  ) -> tuple[dict, dict]:
    import warnings
    import optuna

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    class _OptunaLogHandler(logging.Handler):
      def emit(self, record):
        msg = self.format(record)
        logger.info(msg)

    opt_logger = logging.getLogger("optuna")
    opt_logger.handlers = []
    handler = _OptunaLogHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[optuna] %(message)s")
    handler.setFormatter(formatter)
    opt_logger.addHandler(handler)
    opt_logger.propagate = False

    logger.info(f"Hyperparameter tuning: budget={time_budget}s  max_trials={n_trials}")
    start = time.time()

    def _count_rows(it) -> int | None:
      dset = getattr(it, "dataset", None)
      if dset is None:
        return None
      try:
        return int(dset.count_rows())
      except Exception:
        return None

    train_rows = _count_rows(train_iter)
    val_rows = _count_rows(val_iter)
    if train_rows is None:
      logger.warning("Unable to count training rows for subsampling; using full data in phase 1")

    def suggest(trial):
      p = dict(base_params)
      p["max_depth"] = trial.suggest_int("max_depth", 3, 12)
      p["eta"] = trial.suggest_float("eta", 1e-3, 0.3, log=True)
      p["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
      p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
      p["lambda"] = trial.suggest_float("lambda", 1e-9, 50.0, log=True)
      p["alpha"] = trial.suggest_float("alpha", 1e-9, 50.0, log=True)
      p["min_child_weight"] = trial.suggest_float("min_child_weight", 0.1, 20.0, log=True)
      return p

    cheap_rounds = max(50, int(self.num_boost_round * 0.3))
    cheap_early = None
    if self.early_stopping_rounds is not None:
      cheap_early = max(10, int(self.early_stopping_rounds * 0.5))

    # ── Build subsampled DMatrix ONCE for Phase 1 ──
    PHASE1_FRAC = 0.15
    if train_rows is not None:
      sub_tr = max(1, int(train_rows * PHASE1_FRAC))
      sub_va = max(1, int(val_rows * PHASE1_FRAC)) if val_rows is not None else None
      tr_it = _LimitedDataIter(train_iter, sub_tr)
      va_it = _LimitedDataIter(val_iter, sub_va) if sub_va is not None else val_iter
    else:
      tr_it, va_it = train_iter, val_iter

    logger.info(
      "tune config: "
      f"train_rows={train_rows} val_rows={val_rows} "
      f"phase1_frac={PHASE1_FRAC} cheap_rounds={cheap_rounds} cheap_early={cheap_early} "
      f"full_rounds={self.num_boost_round} full_early={self.early_stopping_rounds}"
    )
    logger.info("Building cached DMatrix for phase 1 (subsampled)...")
    t0 = time.perf_counter()
    dtrain_cheap = xgb.QuantileDMatrix(tr_it, max_bin=self.max_bin)
    dval_cheap = xgb.QuantileDMatrix(va_it, ref=dtrain_cheap, max_bin=self.max_bin)
    logger.info(f"Phase 1 DMatrix built in {time.perf_counter() - t0:.2f}s")

    eval_metric = base_params.get("eval_metric", "mae")
    higher = eval_metric in _MAXIMIZE_METRICS
    direction = "maximize" if higher else "minimize"

    phase1_pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
      direction=direction,
      pruner=phase1_pruner,
      sampler=optuna.samplers.TPESampler(multivariate=True, seed=self.seed),
    )
    trials_log: list[dict] = []

    logger.info("tune phase 1: cheap search with pruning")
    t_phase1 = time.perf_counter()

    with make_progress(transient=True) as prog:
      task_id = prog.add_task("tune", total=n_trials, desc="tune")

      def objective(trial):
        p = suggest(trial)
        em = p.get("eval_metric", "mae")
        evals_result: dict = {}
        booster = xgb.train(
          params=p,
          dtrain=dtrain_cheap,
          num_boost_round=cheap_rounds,
          evals=[(dval_cheap, "val")],
          early_stopping_rounds=cheap_early,
          verbose_eval=False,
          evals_result=evals_result,
          callbacks=[_OptunaPruningCallback(trial, em)],
        )
        fallback = float("-inf") if higher else float("inf")
        try:
          return float(booster.best_score)
        except AttributeError:
          vals = evals_result.get("val", {}).get(em, [])
          return float(vals[-1]) if vals else fallback

      def _optuna_cb(study, trial):
        trial_value = trial.value
        if trial_value is not None:
          trial_value = float(trial_value)
        trials_log.append({
          "trial": int(trial.number),
          "value": trial_value,
          "params": dict(trial.params),
        })
        try:
          best_value = float(study.best_value)
        except ValueError:
          best_value = None
        logger.info(
          "tune trial: "
          f"id={trial.number} value={trial_value} best={best_value} "
          f"params={trial.params}"
        )
        prog.update(task_id, advance=1, refresh=True)
        if time.time() - start > time_budget:
          study.stop()

      study.optimize(objective, n_trials=n_trials, callbacks=[_optuna_cb], gc_after_trial=True)
    logger.info(f"tune phase 1 complete in {time.perf_counter() - t_phase1:.2f}s")

    del dtrain_cheap, dval_cheap  # free phase 1 memory

    completed = [
      t
      for t in study.trials
      if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed:
      logger.warning("tune phase 1: all trials failed — falling back to default parameters")
      return {}, {"elapsed_seconds": round(time.time() - start, 2), "phase1": {"n_trials": len(study.trials), "all_failed": True}, "best_params": {}}

    _worst = float("-inf") if higher else float("inf")
    completed.sort(
      key=lambda t: float(t.value) if t.value is not None else _worst,
      reverse=higher,
    )
    top_k = min(10, len(completed))
    phase2_trials = completed[:top_k]

    # ── Build full DMatrix ONCE for Phase 2 & 3 ──
    logger.info("Building cached DMatrix for phase 2/3 (full data)...")
    t0 = time.perf_counter()
    dtrain_full = xgb.QuantileDMatrix(train_iter, max_bin=self.max_bin)
    dval_full = xgb.QuantileDMatrix(val_iter, ref=dtrain_full, max_bin=self.max_bin)
    logger.info(f"Full DMatrix built in {time.perf_counter() - t0:.2f}s")

    def _train_eval_cached(trial_params: dict, rounds: int, early_stop: int | None) -> float:
      params = dict(base_params)
      params.update(trial_params)
      evals_result: dict = {}
      booster = xgb.train(
        params=params,
        dtrain=dtrain_full,
        num_boost_round=rounds,
        evals=[(dval_full, "val")],
        early_stopping_rounds=early_stop,
        verbose_eval=False,
        evals_result=evals_result,
      )
      fallback = float("-inf") if higher else float("inf")
      try:
        return float(booster.best_score)
      except AttributeError:
        vals = evals_result.get("val", {}).get(eval_metric, [])
        return float(vals[-1]) if vals else fallback

    logger.info(f"tune phase 2: evaluating top {top_k} configs on full data (cached DMatrix)")
    t_phase2 = time.perf_counter()
    phase2_results = []
    with make_progress(transient=True) as prog:
      task_id = prog.add_task("tune phase2", total=top_k, desc="tune phase2")
      for t in phase2_trials:
        if time.time() - start > time_budget:
          logger.info("tune phase 2: time budget exceeded, stopping early")
          break
        value = _train_eval_cached(
          dict(t.params),
          rounds=self.num_boost_round,
          early_stop=self.early_stopping_rounds,
        )
        phase2_results.append({"params": dict(t.params), "value": float(value)})
        logger.info(f"tune phase2 trial: value={float(value)} params={t.params}")
        prog.update(task_id, advance=1, refresh=True)

    logger.info(f"tune phase 2 complete in {time.perf_counter() - t_phase2:.2f}s")

    phase2_results.sort(key=lambda r: r["value"], reverse=higher)
    top_m = min(3, len(phase2_results))
    phase3_candidates = phase2_results[:top_m]
    seeds = [self.seed, self.seed + 1, self.seed + 2]

    logger.info(f"tune phase 3: confirming top {top_m} configs with {len(seeds)} seeds")
    t_phase3 = time.perf_counter()
    phase3_results = []
    total_phase3 = top_m * len(seeds)
    with make_progress(transient=True) as prog:
      task_id = prog.add_task("tune phase3", total=total_phase3, desc="tune phase3")
      for cand in phase3_candidates:
        if time.time() - start > time_budget:
          logger.info("tune phase 3: time budget exceeded, stopping early")
          break
        vals = []
        for sd in seeds:
          params = dict(cand["params"])
          params["seed"] = int(sd)
          vals.append(
            _train_eval_cached(
              params,
              rounds=self.num_boost_round,
              early_stop=self.early_stopping_rounds,
            )
          )
          prog.update(task_id, advance=1, refresh=True)
        vals = np.asarray(vals, dtype=np.float64)
        phase3_results.append({
          "params": dict(cand["params"]),
          "mean": float(vals.mean()),
          "std": float(vals.std()),
          "seeds": list(seeds),
        })
        logger.info(
          f"tune phase3 candidate: mean={float(vals.mean())} std={float(vals.std())} params={cand['params']}"
        )

    logger.info(f"tune phase 3 complete in {time.perf_counter() - t_phase3:.2f}s")

    del dtrain_full, dval_full  # free full DMatrix memory

    phase3_results.sort(key=lambda r: r["mean"], reverse=higher)
    best = phase3_results[0] if phase3_results else None
    best_params = dict(best["params"]) if best else dict(study.best_trial.params)
    if best:
      logger.info(f"tune best: mean={best['mean']} std={best['std']} params={best_params}")
    else:
      logger.info(f"tune best (phase1): params={best_params}")

    tune_info = {
      "elapsed_seconds": round(time.time() - start, 2),
      "phase1": {
        "n_trials": int(len(study.trials)),
        "best_trial": int(study.best_trial.number),
        "best_value": float(study.best_trial.value) if study.best_trial.value is not None else None,
        "trials": trials_log,
      },
      "phase2": {
        "top_k": int(top_k),
        "results": phase2_results,
      },
      "phase3": {
        "top_m": int(top_m),
        "results": phase3_results,
      },
      "best_params": dict(best_params),
    }

    return dict(best_params), tune_info

  def _log_params(self, params: dict) -> None:
    for k, v in sorted(params.items()):
      logger.debug(f"param.{k}={v}")

  def _log_run_summary(self, train_info: dict, tune_info: dict | None, eval_metric: str) -> None:
    logger.info(
      "Run summary: "
      f"trees={train_info.get('n_trees')} "
      f"best_round={train_info.get('best_round')} "
      f"best_val_{eval_metric}={train_info.get('best_val_score')} "
      f"train_seconds={train_info.get('train_seconds')}"
    )
    if tune_info:
      logger.info(
        "Optuna summary: "
        f"best={tune_info.get('best_value')} "
        f"trials={tune_info.get('n_trials')} "
        f"time_seconds={tune_info.get('elapsed_seconds')}"
      )
