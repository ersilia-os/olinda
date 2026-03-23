import json, numpy as np, xgboost as xgb
from pathlib import Path


class StudentModel:
  def __init__(self, booster: xgb.Booster, featurizer=None, metadata: dict | None = None) -> None:
    self.booster = booster
    self.featurizer = featurizer
    self.metadata = metadata or {}

  def predict(self, X=None, smiles: list[str] | None = None, batch_size: int = 65536) -> np.ndarray:
    if X is None:
      if self.featurizer is None or smiles is None:
        raise ValueError("provide X or (smiles + featurizer)")
      # Batch featurize + predict to keep memory bounded
      preds = []
      for i in range(0, len(smiles), batch_size):
        Xb = self.featurizer.transform(smiles[i : i + batch_size]).astype(np.float32)
        preds.append(self.booster.predict(xgb.DMatrix(Xb)))
      return np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32)
    if len(X) > batch_size:
      preds = []
      for i in range(0, len(X), batch_size):
        preds.append(self.booster.predict(xgb.DMatrix(X[i : i + batch_size])))
      return np.concatenate(preds)
    d = xgb.DMatrix(X)
    return np.asarray(self.booster.predict(d))

  def save(self, out_dir: str | Path) -> None:
    """Save booster + training metadata. Never overwrites pack meta.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    self.booster.save_model(str(out_dir / "xgb.json"))

    meta = dict(self.metadata)
    if self.featurizer is not None and hasattr(self.featurizer, "to_dict"):
      meta["featurizer"] = self.featurizer.to_dict()
      meta["featurizer_class"] = type(self.featurizer).__name__

    with open(out_dir / "train_meta.json", "w") as fp:
      json.dump(meta, fp, indent=2)

  @classmethod
  def load(cls, out_dir: str | Path, featurizer_factory=None):
    out_dir = Path(out_dir)
    booster = xgb.Booster()
    booster.load_model(str(out_dir / "xgb.json"))
    meta = {}
    for name in ("train_meta.json", "meta.json"):
      mp = out_dir / name
      if mp.exists():
        with open(mp, "r") as fp:
          meta = json.load(fp)
        break

    fz = None
    if featurizer_factory and "featurizer" in meta:
      fz = featurizer_factory(meta.get("featurizer_class"), meta["featurizer"])

    return cls(booster=booster, featurizer=fz, metadata=meta)
