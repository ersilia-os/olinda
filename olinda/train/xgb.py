from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import xgboost as xgb


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
