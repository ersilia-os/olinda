"""Fast XGBoost regression over the reference library.

The training loop for the XGBoost backend: a boosting run against the ``xgb.QuantileDMatrix`` pair
that :meth:`olinda.train.backend.Backend.build_train_val_indexed` gathers from the shared
:class:`~olinda.data.matrix.ReferenceMatrix`, with live progress and early stopping. QuantileDMatrix
bins each feature once into a compact histogram held in RAM, so training is fast and low-memory; the
validation matrix shares the training matrix's exact bin edges.
"""

from __future__ import annotations

import os
import time

import numpy as np
import xgboost as xgb

from olinda.console import console, echo, live_status, spinner
from olinda.train.xgb import detect_training_device

# The hyperparameters `train_regression` runs with are not chosen here: they live in canonical,
# engine-agnostic names in ``olinda.train.backend.CANONICAL_DEFAULTS``, and ``XGBoostBackend.translate``
# turns them into the native xgb params passed in below. `tree_method="hist"` is correct for BOTH CPU and
# GPU on XGBoost ≥2.0 (GPU via `device="cuda"`, not the deprecated `gpu_hist`).


def _val_stats(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
  """(val RMSE, R², Spearman ρ) — used for readable per-round progress."""
  err = p - y
  rmse = float(np.sqrt((err**2).mean()))
  sst = float(((y - y.mean()) ** 2).sum())
  r2 = 1.0 - float((err**2).sum()) / sst if sst else float("nan")
  ry, rp = np.argsort(np.argsort(y)).astype(np.float64), np.argsort(np.argsort(p)).astype(np.float64)
  ryc, rpc = ry - ry.mean(), rp - rp.mean()
  sd = float(np.sqrt((ryc**2).sum()) * np.sqrt((rpc**2).sum()))
  rho = float((ryc * rpc).sum() / sd) if sd else float("nan")
  return rmse, r2, rho


def _fmt_secs(s: float) -> str:
  """Compact elapsed time, e.g. '9s' or '2m04s'."""
  s = int(s)
  return f"{s}s" if s < 60 else f"{s // 60}m{s % 60:02d}s"


class _LiveProgress(xgb.callback.TrainingCallback):
  """Redraw one live status line per round: same-metric train/val loss + val R²/ρ + best + elapsed.

  Train and val loss are both read from XGBoost's eval log (the ``metric``), so they are the *same*
  quantity and directly comparable (both weighted, or both unweighted, per the matrices). The
  interpretable val R²/ρ are computed from ``predict`` on the raw ``y_val`` (always unweighted) but only
  every ``every`` rounds — the expensive part — and cached in between. On a non-TTY the live line is a
  no-op, so a plain milestone line is echoed every ``every`` rounds instead.
  """

  def __init__(self, update, dval, y_val, metric: str, every: int, total: int, t0: float) -> None:
    self.update = update
    self.dval = dval
    self.y_val = np.asarray(y_val, dtype=np.float64)
    self.metric = metric
    self.every = max(int(every), 1)
    self.total = total
    self.t0 = t0
    self.best_round, self.best_val = 0, float("inf")
    self.r2, self.rho = float("nan"), float("nan")

  def _loss(self, evals_log, split):
    try:
      return float(evals_log[split][self.metric][-1])
    except (KeyError, IndexError):
      return float("nan")

  def after_iteration(self, model, epoch, evals_log) -> bool:
    tr, va = self._loss(evals_log, "train"), self._loss(evals_log, "val")
    if va < self.best_val:
      self.best_val, self.best_round = va, epoch
    if epoch % self.every == 0 or epoch == self.total - 1:
      p = np.asarray(model.predict(self.dval), dtype=np.float64)
      _, self.r2, self.rho = _val_stats(self.y_val, p)
      if not console.is_terminal:
        echo(
          f"round {epoch:>4}/{self.total} · {self.metric} train {tr:.4f} · val {va:.4f} "
          f"· R² {self.r2:.3f} · ρ {self.rho:.3f}",
          "info",
        )
    from olinda.train.backend import _row_values

    self.update(
      f"  [bold cyan]{spinner(epoch)} training[/] [dim]round[/] [bold]{epoch}[/][dim]/{self.total}[/]  "
      f"[dim]·[/]  [dim]{self.metric}[/] train [bold]{tr:.4f}[/] · val [bold cyan]{va:.4f}[/]  "
      f"[dim]·[/]  R² [bold]{self.r2:.3f}[/] · ρ [bold]{self.rho:.3f}[/]  "
      f"[dim]· best@{self.best_round} · {_fmt_secs(time.perf_counter() - self.t0)}[/]",
      **_row_values(self.r2, self.rho, va, self.metric, epoch),
    )
    return False


def train_regression(
  dtrain: xgb.QuantileDMatrix,
  dval: xgb.QuantileDMatrix,
  *,
  params: dict | None = None,
  num_boost_round: int = 5000,
  early_stopping_rounds: int = 50,
  seed: int = 42,
  log_every: int = 100,
  train_weighted: bool = False,
):
  """Train a fast XGBoost regressor with early stopping on ``dval``.

  Returns ``(booster, evals_result, best_iteration)``. ``params`` are the full native XGBoost params
  (objective, eval_metric, tree/regularization knobs) — normally produced by
  ``olinda.train.backend.XGBoostBackend``; single fit, no hyperparameter search. ``train_weighted`` only
  annotates the header (the weights live in the matrices).
  """
  device, reason = detect_training_device()
  p = dict(params or {})
  p.setdefault("tree_method", "hist")
  p["seed"] = int(seed)
  p["nthread"] = os.cpu_count() or 0
  if device == "cuda":
    p["device"] = "cuda"
  # objective/eval_metric normally come from choose_objective via `params`; fall back to XGBoost's own
  # default (reg:squarederror / rmse) for a bare call. The early-stop metric is the LAST eval_metric.
  objective = p.get("objective", "reg:squarederror")
  em = p.get("eval_metric", "rmse")
  metric = em[-1] if isinstance(em, (list, tuple)) else em

  echo(
    f"Training · [bold]{objective}[/] · {metric} · eta={p['eta']} · max_depth={p['max_depth']} "
    f"· device={device}{' · loss weighted' if train_weighted else ''}",
    "run",
  )
  evals_result: dict = {}
  t0 = time.perf_counter()
  with live_status() as update:
    booster = xgb.train(
      p,
      dtrain,
      num_boost_round=num_boost_round,
      evals=[(dtrain, "train"), (dval, "val")],
      early_stopping_rounds=early_stopping_rounds,
      evals_result=evals_result,
      verbose_eval=False,
      callbacks=[_LiveProgress(update, dval, dval.get_label(), metric, log_every, num_boost_round, t0)],
    )
  dt = time.perf_counter() - t0
  best_it = int(booster.best_iteration)
  best_score = float(booster.best_score)
  # XGBoost's predict() uses ALL trees by default (not best_iteration), so trim the booster to the
  # best iteration — otherwise early stopping is moot and inference would use the overfit model.
  n_total = booster.num_boosted_rounds()
  if best_it + 1 < n_total:
    booster = booster[: best_it + 1]
  echo(
    f"Trained [bold]{booster.num_boosted_rounds()}[/] trees (best of {num_boost_round}) "
    f"· val {metric} [bold]{best_score:.5f}[/] · {_fmt_secs(dt)}",
    "success",
  )
  return booster, evals_result, best_it
