from __future__ import annotations

import time

import xgboost as xgb
from rich.progress import (
  Progress,
  SpinnerColumn,
  BarColumn,
  TextColumn,
  TimeElapsedColumn,
  TimeRemainingColumn,
)
from olinda.helpers import logger


_TASK_SPECS = {
  "regression": {"objective": "reg:squarederror", "eval_metric": "mae"},
  "classification": {"objective": "binary:logistic", "eval_metric": "logloss"},
}

_BASE_PARAMS = {
  "max_depth": 8,
  "eta": 0.05,
  "subsample": 0.9,
  "colsample_bytree": 0.9,
  "lambda": 1.0,
  "alpha": 0.0,
  "tree_method": "hist",
}


class _RichTrainCallback(xgb.callback.TrainingCallback):
  def __init__(self, progress: Progress, task_id, total_rounds: int, eval_metric: str):
    self.progress = progress
    self.task_id = task_id
    self.total_rounds = total_rounds
    self.eval_metric = eval_metric
    self.best_score = float("inf")
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
      if score < self.best_score:
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

  def resolved_params(self) -> dict:
    p = {}
    p.update(_BASE_PARAMS)
    p.update(_TASK_SPECS[self.task])
    p.update(self.params)
    p["seed"] = self.seed
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

    meta = {"task": self.task, "params": params}
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

    evals = []
    if val_iter is not None:
      logger.info("Building QuantileDMatrix (val)...")
      t0 = time.perf_counter()
      dval = xgb.QuantileDMatrix(val_iter, ref=dtrain, max_bin=self.max_bin)
      dt_val = time.perf_counter() - t0
      logger.info(f"QuantileDMatrix (val) built in {dt_val:.2f}s")
      evals = [(dval, "val")]
    else:
      logger.warning("No validation iterator provided; training will not log validation loss")

    eval_metric = params.get("eval_metric", "mae")

    with Progress(
      SpinnerColumn(),
      TextColumn("[bold cyan]TRAIN[/bold cyan]"),
      BarColumn(bar_width=48),
      TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
      TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
      TextColumn("{task.fields[metric]}"),
      TextColumn("{task.fields[best]}"),
      TimeElapsedColumn(),
      TimeRemainingColumn(),
      transient=True,
    ) as prog:
      task_id = prog.add_task("boosting", total=self.num_boost_round, metric="", best="")
      cb = _RichTrainCallback(prog, task_id, self.num_boost_round, eval_metric)

      t0 = time.perf_counter()
      booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=self.num_boost_round,
        evals=evals,
        early_stopping_rounds=self.early_stopping_rounds,
        verbose_eval=False,
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
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info(f"Hyperparameter tuning: budget={time_budget}s  max_trials={n_trials}")
    start = time.time()

    def suggest(trial):
      p = dict(base_params)
      p["max_depth"] = trial.suggest_int("max_depth", 4, 12)
      p["eta"] = trial.suggest_float("eta", 1e-3, 0.2, log=True)
      p["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
      p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
      p["lambda"] = trial.suggest_float("lambda", 1e-8, 10.0, log=True)
      p["alpha"] = trial.suggest_float("alpha", 1e-8, 10.0, log=True)
      p["min_child_weight"] = trial.suggest_float("min_child_weight", 0.5, 10.0, log=True)
      return p

    def objective(trial):
      p = suggest(trial)
      dtrain = xgb.QuantileDMatrix(train_iter, max_bin=self.max_bin)
      dval = xgb.QuantileDMatrix(val_iter, ref=dtrain, max_bin=self.max_bin)
      booster = xgb.train(
        params=p,
        dtrain=dtrain,
        num_boost_round=self.num_boost_round,
        evals=[(dval, "val")],
        early_stopping_rounds=self.early_stopping_rounds,
        verbose_eval=False,
      )
      return float(booster.best_score)

    study = optuna.create_study(direction="minimize")
    trials_log: list[dict] = []

    with Progress(
      SpinnerColumn(),
      TextColumn("[bold yellow]TUNE[/bold yellow]"),
      BarColumn(bar_width=40),
      TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
      TextColumn("{task.fields[best]}"),
      TextColumn("{task.fields[last]}"),
      TimeElapsedColumn(),
      TimeRemainingColumn(),
      transient=False,
    ) as prog:
      task_id = prog.add_task("trials", total=n_trials, best="best=?", last="last=?")

      def _optuna_cb(study, trial):
        trial_value = trial.value
        if trial_value is not None:
          trial_value = float(trial_value)
        trials_log.append({
          "trial": int(trial.number),
          "value": trial_value,
          "params": dict(trial.params),
        })
        best_value = getattr(study, "best_value", None)
        if best_value is not None:
          best_value = float(best_value)
        logger.debug(
          f"optuna.trial={trial.number} value={trial_value} best={best_value} params={trial.params}"
        )
        prog.update(
          task_id,
          advance=1,
          best=f"best={best_value:.6f}" if best_value is not None else "best=?",
          last=f"last={trial_value:.6f}" if trial_value is not None else "last=?",
        )
        if time.time() - start > time_budget:
          study.stop()

      study.optimize(objective, n_trials=n_trials, callbacks=[_optuna_cb], gc_after_trial=True)
      prog.update(task_id, total=len(study.trials), completed=len(study.trials))

    best = study.best_trial
    best_value = best.value
    if best_value is not None:
      best_value = float(best_value)
    tune_info = {
      "best_value": best_value,
      "best_trial": int(best.number),
      "n_trials": int(len(study.trials)),
      "elapsed_seconds": round(time.time() - start, 2),
      "best_params": dict(best.params),
      "trials": trials_log,
    }

    return dict(best.params), tune_info

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
