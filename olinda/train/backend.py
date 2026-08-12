"""Gradient-boosting backends — XGBoost (GPU-capable) and LightGBM (CPU-fast) behind one interface.

The engine is chosen automatically from the training device (:func:`select_backend`): a CUDA GPU → XGBoost
(mature GPU path, and the only one that accelerates our ~96%-zero sparse Morgan features), otherwise →
LightGBM (much faster / lower-memory on large CPU workloads; its GPU can't use sparse features, so it only
ever runs on CPU here). ``OLINDA_BACKEND=xgboost|lightgbm|auto`` overrides.

Hyperparameters are kept in **canonical, backend-agnostic names** (:data:`CANONICAL_DEFAULTS`) so that
``choose_objective``, ``tune``, ``best_params.json`` and the paste-ready block are portable; each backend
translates canonical → its native parameter names, and maps the canonical objective *kind*
(``squarederror|logistic|pseudohuber``) to its native objective + eval metric.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from olinda.console import echo
from olinda.train.xgb import detect_training_device

# Single source of truth for default hyperparameters, in canonical (engine-agnostic) names.
# learning_rate=0.1 (not 0.3): benchmarked on the real 200k Morgan split, 0.1 gave a large accuracy gain
# over 0.3 for BOTH engines (XGBoost R² 0.70→0.76, LightGBM 0.68→0.78) and still converges within the
# 5000-round cap; 0.05 barely helps and blows the round budget. Matches QSAR literature (eta 0.05–0.1).
# Structural defaults in engine-agnostic ("canonical") names — each backend's translate() maps them to its
# native params. Tuned for large-scale (~1.3M rows) sparse Morgan-count QSAR. Only learning_rate and
# min_split_gain are searched by `olinda tune` (see olinda/train/tune.py); everything else is fixed here.
CANONICAL_DEFAULTS = {
    # Histogram bins. Morgan COUNT fingerprints are small integers (mostly 0/1/2, clipped at 255), so a
    # feature has very few distinct values — 64 is already lossless for them and builds faster and leaner
    # than 128 (benchmarked: more ≈ noise here).
    "max_bin": 64,
    "max_depth": 8,  # deep trees for large N; LightGBM num_leaves = min(2^8-1, 255) saturates at this depth
    "learning_rate": 0.1,  # TUNED (0.05–0.3); 0.1 beat 0.3 by a wide margin on the real split
    "subsample": 0.8,  # row bagging: mild regularization + speed; standard for large-scale GBM
    "colsample": 0.5,  # 1024/2048 features per tree: decorrelation + speed on high-dim sparse (least-impactful knob)
    "min_child_weight": 200.0,  # ≈ min samples/leaf (via min_sum_hessian for L2); LightGBM: use hundreds at ~1M+ rows
    "min_split_gain": 0.0,  # TUNED (0–5); minimum loss reduction required to split
    "reg_lambda": 1.0,  # L2 penalty (XGBoost default)
    "reg_alpha": 0.0,  # L1 penalty off (safe default)
}


def select_backend() -> tuple[str, str, str]:
    """Resolve ``(backend_name, device, reason)``.

    ``OLINDA_BACKEND`` (``auto`` default) forces the engine; ``auto`` picks XGBoost on CUDA, LightGBM on CPU.
    Falls back to XGBoost if LightGBM is selected but not importable.
    """
    override = os.environ.get("OLINDA_BACKEND", "auto").lower()
    device, dreason = detect_training_device()
    if override in ("xgboost", "lightgbm"):
        name, reason = override, f"OLINDA_BACKEND={override}"
    elif override == "auto":
        name = "xgboost" if device == "cuda" else "lightgbm"
        reason = f"auto · device={device}"
    else:
        raise ValueError(
            f"OLINDA_BACKEND must be auto|xgboost|lightgbm, got {override!r}"
        )
    if name == "lightgbm":
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            echo(
                "lightgbm not installed — falling back to XGBoost on CPU (pip install lightgbm)",
                "warning",
            )
            name, reason = "xgboost", reason + " · lightgbm missing→xgboost"
    return name, device, reason


def get_backend(name: str, device: str):
    """Instantiate a backend by name."""
    if name == "xgboost":
        return XGBoostBackend(device)
    if name == "lightgbm":
        return LightGBMBackend(device)
    raise ValueError(f"unknown backend {name!r}")


@dataclass
class TrainResult:
    model: object
    best_iteration: int
    best_score: float
    metric: str
    n_trees: int


def _row_values(
    r2: float, rho: float, val_loss: float, metric: str, rounds: int
) -> dict:
    """The mid-training numbers, keyed to match ``learn-soft``'s live table columns.

    Both engines report the same quantities under different metric names — XGBoost's ``rmse`` is already
    a root, LightGBM's ``l2`` is not — so the RMSE column is squared-rooted here rather than showing two
    incomparable scales depending on which engine happened to be picked.
    """
    rmse = float(val_loss) ** 0.5 if metric in ("l2", "mse") else float(val_loss)
    fmt = lambda v, p: "[dim]—[/]" if v != v else f"{v:.{p}f}"  # noqa: E731 - NaN until the first val pass
    return {
        "R²": fmt(r2, 4),
        "ρ": fmt(rho, 4),
        "RMSE": fmt(rmse, 5),
        "Trees": f"{rounds:,}",
    }


# --------------------------------------------------------------------------------------------------
# XGBoost
# --------------------------------------------------------------------------------------------------
class XGBoostBackend:
    name = "xgboost"
    model_file = "xgb.json"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def objective_params(self) -> dict:
        # squared error only — well-conditioned, ONNX-exportable (identity link), safe for bounded targets
        return {"objective": "reg:squarederror", "eval_metric": "rmse"}

    def translate(self, canonical: dict) -> dict:
        c = {**CANONICAL_DEFAULTS, **canonical}
        return {
            "tree_method": "hist",
            "max_bin": int(c["max_bin"]),
            "max_depth": int(c["max_depth"]),
            "eta": float(c["learning_rate"]),
            "subsample": float(c["subsample"]),
            "colsample_bytree": float(c["colsample"]),
            "min_child_weight": float(c["min_child_weight"]),
            "gamma": float(c["min_split_gain"]),
            "lambda": float(c["reg_lambda"]),
            "alpha": float(c["reg_alpha"]),
        }

    def params(self, canonical: dict) -> dict:
        """Full native params = structural translation + squared-error objective."""
        return {**self.translate(canonical), **self.objective_params()}

    # -- multi-column path (one resident library, per-column indices) --
    def build_train_val_indexed(
        self, matrix, y, train_idx, val_idx, max_bin, bin_edges, bin_weights
    ):
        """Build train/val matrices for one column, reading the shared in-RAM library by index."""
        import xgboost as xgb

        from olinda.data.dataset import IndexDataIter

        def _qdm(idx, ref=None):
            it = IndexDataIter(
                matrix, y, idx, bin_edges=bin_edges, bin_weights=bin_weights
            )
            return xgb.QuantileDMatrix(it, max_bin=int(max_bin), ref=ref)

        dtrain = _qdm(train_idx)
        return dtrain, _qdm(val_idx, ref=dtrain)

    def train(
        self,
        dtrain,
        dval,
        native_params,
        num_boost_round,
        early_stopping,
        train_weighted,
        val_eval=None,
    ):
        from olinda.train.reference import train_regression

        booster, evals, best_it = train_regression(
            dtrain,
            dval,
            params=native_params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping,
            train_weighted=train_weighted,
        )
        em = native_params.get("eval_metric", "rmse")
        metric = em[-1] if isinstance(em, (list, tuple)) else em
        # booster was trimmed to best_iteration, which drops its `best_score` attr — read it from the log.
        val_hist = evals.get("val", {}).get(metric, [])
        best_score = (
            float(val_hist[best_it]) if best_it < len(val_hist) else float("nan")
        )
        return TrainResult(
            booster, best_it, best_score, metric, booster.num_boosted_rounds()
        )

    # -- tune path (in-RAM subset, per-round pruning hook) --
    def dataset(self, X, y, weight, max_bin, reference=None):
        import xgboost as xgb

        return xgb.QuantileDMatrix(
            X, label=y, weight=weight, max_bin=max_bin, ref=reference
        )

    def train_trial(
        self, dtrain, dval, native_params, num_boost_round, early_stopping, on_iteration
    ):
        import xgboost as xgb

        p = dict(native_params)
        p["nthread"] = os.cpu_count() or 0
        if self.device == "cuda":
            p["device"] = "cuda"
        em = p.get("eval_metric", "rmse")
        metric = em[-1] if isinstance(em, (list, tuple)) else em

        class _CB(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                try:
                    score = evals_log["val"][metric][-1]
                except (KeyError, IndexError):
                    return False
                on_iteration(epoch, float(score))  # may raise to prune/stop
                return False

        booster = xgb.train(
            p,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stopping,
            verbose_eval=False,
            callbacks=[_CB()],
        )
        return float(booster.best_score), int(booster.best_iteration), metric

    # -- persistence / inference --
    def predict(self, model, X):
        import xgboost as xgb

        return np.asarray(model.predict(xgb.DMatrix(X)))

    def save(self, model, model_dir):
        model.save_model(str(Path(model_dir) / self.model_file))

    def load(self, model_dir):
        import xgboost as xgb

        b = xgb.Booster()
        b.load_model(str(Path(model_dir) / self.model_file))
        return b

    def to_onnx(self, model, path, input_dim):
        from olinda.models.exporters import export_xgb_onnx

        export_xgb_onnx(model, path, input_dim)


# --------------------------------------------------------------------------------------------------
# LightGBM
# --------------------------------------------------------------------------------------------------
class LightGBMBackend:
    name = "lightgbm"
    model_file = "model.lgb"

    def __init__(self, device: str = "cpu") -> None:
        self.device = "cpu"  # LightGBM only ever runs on CPU here (its GPU can't use sparse features)

    def objective_params(self) -> dict:
        # squared error only (L2) — ONNX-exportable and well-conditioned; single metric for clean early stopping
        return {"objective": "regression", "metric": "l2"}

    def translate(self, canonical: dict) -> dict:
        c = {**CANONICAL_DEFAULTS, **canonical}
        md = int(c["max_depth"])
        return {
            "max_bin": int(c["max_bin"]),
            "max_depth": md,
            # Leaf-wise: cap leaves rather than rely on depth alone. (Benchmarked num_leaves 63/127/255 and
            # min_data_in_leaf 20/100/200 on the real 200k split — all within noise of each other; the real
            # accuracy lever is learning_rate, not these overfit knobs, so keep the simple mapping.)
            "num_leaves": min(2**md - 1, 255),
            "learning_rate": float(c["learning_rate"]),
            "bagging_fraction": float(c["subsample"]),
            "bagging_freq": 1,
            "feature_fraction": float(c["colsample"]),
            "min_sum_hessian_in_leaf": float(c["min_child_weight"]),
            "min_gain_to_split": float(c["min_split_gain"]),
            "lambda_l2": float(c["reg_lambda"]),
            "lambda_l1": float(c["reg_alpha"]),
            "verbosity": -1,
            "num_threads": os.cpu_count() or 0,
            "device_type": "cpu",
        }

    def params(self, canonical: dict) -> dict:
        return {**self.translate(canonical), **self.objective_params()}

    def _metric_name(self, native_params) -> str:
        m = native_params.get("metric", "l2")
        return m[-1] if isinstance(m, (list, tuple)) else m

    # -- multi-column path (one resident library, per-column indices) --
    def build_train_val_indexed(
        self, matrix, y, train_idx, val_idx, max_bin, bin_edges, bin_weights
    ):
        """Build train/val matrices for one column, reading the shared in-RAM library by index."""
        import lightgbm as lgb

        from olinda.data import apply_bin_weights
        from olinda.data.matrix import index_sequence

        def _ds(idx, reference=None):
            yy = np.asarray(y, dtype=np.float32)[idx]
            w = (
                apply_bin_weights(yy, bin_edges, bin_weights)
                if bin_edges is not None
                else None
            )
            return lgb.Dataset(
                index_sequence(matrix, idx),
                label=yy,
                weight=w,
                params={"max_bin": int(max_bin), "verbosity": -1},
                reference=reference,
                free_raw_data=False,
            )

        dtrain = _ds(train_idx)
        return dtrain, _ds(val_idx, reference=dtrain)

    def train(
        self,
        dtrain,
        dval,
        native_params,
        num_boost_round,
        early_stopping,
        train_weighted,
        val_eval=None,
    ):
        import time as _time

        import lightgbm as lgb

        from olinda.console import live_status, spinner
        from olinda.train.reference import _fmt_secs, _val_stats

        metric = self._metric_name(native_params)
        echo(
            f"Training · [bold]{native_params['objective']}[/] · {metric} · "
            f"lr={native_params['learning_rate']} · num_leaves={native_params['num_leaves']} · device=cpu"
            f"{' · loss weighted' if train_weighted else ''}",
            "run",
        )
        # Val features (if provided) let the live line show unweighted R²/ρ, recomputed every 100 rounds.
        xval = yval = None
        if val_eval is not None:
            xval, yval = val_eval[0], np.asarray(val_eval[1], dtype=np.float64)
        t0 = _time.perf_counter()
        st = {
            "best": float("inf"),
            "best_it": 0,
            "r2": float("nan"),
            "rho": float("nan"),
        }

        with live_status() as update:

            def _cb(env):
                it = env.iteration
                va = next(
                    (
                        float(v)
                        for n, m, v, _ in env.evaluation_result_list
                        if n == "val" and m == metric
                    ),
                    float("nan"),
                )
                if va < st["best"]:
                    st["best"], st["best_it"] = va, it
                if (
                    it % 100 == 0 and xval is not None
                ):  # R²/ρ need a val prediction pass — throttle it
                    _, st["r2"], st["rho"] = _val_stats(
                        yval, np.asarray(env.model.predict(xval), dtype=np.float64)
                    )
                if it % 25 == 0:
                    update(
                        f"  [bold cyan]{spinner(it)} training[/] [dim]round[/] [bold]{it}[/][dim]/{num_boost_round}[/]"
                        f"  [dim]·[/]  [dim]{metric}[/] val [bold cyan]{va:.4f}[/]  [dim]·[/]  R² [bold]{st['r2']:.4f}[/]"
                        f" · ρ [bold]{st['rho']:.4f}[/]  [dim]· best@{st['best_it']} · {_fmt_secs(_time.perf_counter() - t0)}[/]",
                        **_row_values(st["r2"], st["rho"], va, metric, it),
                    )

            booster = lgb.train(
                native_params,
                dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dval],
                valid_names=["val"],
                callbacks=[lgb.early_stopping(early_stopping, verbose=False), _cb],
            )
        best_it = int(booster.best_iteration)
        best_score = float(booster.best_score["val"][metric])
        n_trees = booster.num_trees()
        echo(
            f"Trained [bold]{best_it}[/] trees (best of {num_boost_round}) · val {metric} [bold]{best_score:.5f}[/]"
            f" · {_fmt_secs(_time.perf_counter() - t0)}",
            "success",
        )
        return TrainResult(booster, best_it, best_score, metric, n_trees)

    # -- tune path (in-RAM subset, per-round pruning hook) --
    def dataset(self, X, y, weight, max_bin, reference=None):
        import lightgbm as lgb

        return lgb.Dataset(
            X,
            label=y,
            weight=weight,
            params={"max_bin": int(max_bin), "verbosity": -1},
            reference=reference,
            free_raw_data=False,
        )

    def train_trial(
        self, dtrain, dval, native_params, num_boost_round, early_stopping, on_iteration
    ):
        import lightgbm as lgb

        metric = self._metric_name(native_params)

        def _cb(env):
            for name, mname, val, _ in env.evaluation_result_list:
                if name == "val" and mname == metric:
                    on_iteration(env.iteration, float(val))  # may raise to prune/stop

        booster = lgb.train(
            native_params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False), _cb],
        )
        return (
            float(booster.best_score["val"][metric]),
            int(booster.best_iteration),
            metric,
        )

    # -- persistence / inference --
    def predict(self, model, X):
        return np.asarray(model.predict(X))

    def save(self, model, model_dir):
        # Persist only up to best_iteration → the loaded model predicts with exactly the selected trees.
        model.save_model(
            str(Path(model_dir) / self.model_file), num_iteration=model.best_iteration
        )

    def load(self, model_dir):
        import lightgbm as lgb

        return lgb.Booster(model_file=str(Path(model_dir) / self.model_file))

    def to_onnx(self, model, path, input_dim):
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType

        # No `split`: it crashes when trees < split, and the plain conversion already matches the booster to
        # ~1e-7 (LightGBM's double vs ONNX float discrepancy is negligible in practice for our tree counts).
        onx = onnxmltools.convert_lightgbm(
            model, initial_types=[("input", FloatTensorType([None, int(input_dim)]))]
        )
        onnxmltools.utils.save_model(onx, str(path))
