import json
from pathlib import Path

import numpy as np


class StudentModel:
    """A trained gradient-boosting student (XGBoost or LightGBM) + optional calibrator.

    The engine is recorded in ``metadata["backend"]`` and all model I/O (predict/save/load/onnx) is dispatched
    through :mod:`olinda.train.backend`, so a model round-trips regardless of which engine produced it.
    Legacy callers that pass an ``xgb.Booster`` as ``booster=`` default to the ``xgboost`` backend.
    """

    def __init__(
        self,
        booster=None,
        calibrator=None,
        metadata: dict | None = None,
        backend: str = "xgboost",
        model=None,
    ) -> None:
        self.model = model if model is not None else booster  # native GBM model
        self.booster = self.model  # backward-compat alias
        self.calibrator = calibrator
        self.metadata = metadata or {}
        self.backend = self.metadata.get("backend") or backend

    def _backend(self):
        from olinda.train.backend import get_backend

        return get_backend(self.backend, "cpu")

    def predict(
        self,
        X,
        batch_size: int = 65536,
        calibrate: bool = True,
    ) -> np.ndarray:
        """Predict for fingerprints *X* (decoded qmfp slot values, ``(N, slots)``).

        Featurizing is not the student's business: a model learns from fingerprints, and the fused
        ``model.onnx`` is what turns SMILES into predictions.

        Parameters
        ----------
        X : array-like
            Fingerprints.
        batch_size : int
            Rows per backend call, to bound peak memory on a large input.
        calibrate : bool
            Apply the fitted isotonic correction. Pass ``False`` for the raw booster output.

        Returns
        -------
        numpy.ndarray
            One prediction per input row.
        """
        be = self._backend()
        if len(X) > batch_size:
            preds = [
                be.predict(self.model, X[i : i + batch_size])
                for i in range(0, len(X), batch_size)
            ]
            raw = np.concatenate(preds)
        else:
            raw = np.asarray(be.predict(self.model, X))

        if calibrate and self.calibrator is not None:
            return self.calibrator.transform(raw)
        return raw

    def save(self, out_dir: str | Path) -> None:
        """Save the native model (backend-specific file) + training metadata. Never overwrites pack meta.json."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._backend().save(self.model, out_dir)

        meta = dict(self.metadata)
        meta["backend"] = self.backend

        with open(out_dir / "train_meta.json", "w") as fp:
            json.dump(meta, fp, indent=2)

    @classmethod
    def load(cls, out_dir: str | Path):
        """Load a bundle written by :meth:`save`, restoring its backend, metadata and calibrator.

        Accepts ``meta.json`` as well as ``train_meta.json`` so runs from before the rename still load.
        """
        from olinda.train.backend import get_backend

        out_dir = Path(out_dir)
        meta = {}
        for name in ("train_meta.json", "meta.json"):
            mp = out_dir / name
            if mp.exists():
                with open(mp, "r") as fp:
                    meta = json.load(fp)
                break
        backend = meta.get("backend", "xgboost")
        model = get_backend(backend, "cpu").load(out_dir)

        cal = None
        cal_path = out_dir / "calibrator.json"
        if cal_path.exists():
            from olinda.calibrate import IsotonicCalibrator

            cal = IsotonicCalibrator.load(cal_path)

        return cls(model=model, calibrator=cal, metadata=meta, backend=backend)
