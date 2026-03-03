from __future__ import annotations

import datetime
import json
from pathlib import Path

from olinda.helpers import logger
from olinda.data.pack import pack_distill_dataset
from olinda.data.dataset import ParquetDistillDataset, ParquetDataIter
from olinda.train.xgb import XGBTrainer
from olinda.models.bundle import StudentModel
from olinda.models.exporters import export_xgb_onnx


class Distiller:
  def __init__(
    self,
    trainer: XGBTrainer | None = None,
    batch_rows: int = 65536,
    seed: int = 42,
  ) -> None:
    self.trainer = trainer or XGBTrainer(task="regression")
    self.batch_rows = int(batch_rows)
    self.seed = int(seed)

  def pack(
    self,
    teacher_parquet: str | Path,
    out_dir: str | Path,
    x_col: str = "x",
    y_soft_col: str = "y",
    w_col: str | None = "w",
    split_col: str = "split",
    x_dim: int = 768,
    val_frac: float = 0.1,
    seed: int = 42,
    shard_rows: int = 200_000,
    compression: str = "zstd",
    hard_labels: str | Path | None = None,
    hard_smiles_col: str = "smiles",
    hard_y_col: str = "y",
    hard_weight: float = 1.0,
    featurizer=None,
    hard_batch_rows: int = 8192,
  ) -> Path:
    return pack_distill_dataset(
      teacher_parquet=teacher_parquet,
      out_dir=out_dir,
      x_col=x_col,
      y_soft_col=y_soft_col,
      w_col=w_col,
      split_col=split_col,
      x_dim=x_dim,
      val_frac=val_frac,
      seed=seed,
      shard_rows=shard_rows,
      compression=compression,
      hard_labels=hard_labels,
      hard_smiles_col=hard_smiles_col,
      hard_y_col=hard_y_col,
      hard_weight=hard_weight,
      featurizer=featurizer,
      hard_batch_rows=hard_batch_rows,
    )

  def train(
    self,
    packed_dir: str | Path,
    output_dir: str | Path | None = None,
    export_onnx: bool = True,
    time_budget: int | None = None,
    n_trials: int = 50,
    metadata: dict | None = None,
  ) -> StudentModel:
    packed_dir = Path(packed_dir)
    output_dir = Path(output_dir) if output_dir else packed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = ParquetDistillDataset(packed_dir)
    ntr, nva = ds.count()
    logger.info(f"Training dataset loaded: train={ntr} val={nva} x_dim={ds.x_dim}")

    train_iter = ParquetDataIter(
      ds.train,
      x_col=ds.x_col,
      y_col=ds.y_col,
      w_col=ds.w_col,
      x_dim=ds.x_dim,
      batch_rows=self.batch_rows,
      shuffle_row_groups=True,
      seed=self.seed,
    )
    val_iter = ParquetDataIter(
      ds.val,
      x_col=ds.x_col,
      y_col=ds.y_col,
      w_col=ds.w_col,
      x_dim=ds.x_dim,
      batch_rows=self.batch_rows,
      shuffle_row_groups=False,
      seed=self.seed,
    )

    booster, meta = self.trainer.fit_external(
      train_iter=train_iter,
      val_iter=val_iter,
      time_budget=time_budget,
      n_trials=n_trials,
    )

    m = {}
    m.update(meta or {})
    m.update(metadata or {})
    m.update({
      "trained_at": datetime.datetime.utcnow().isoformat(),
      "train_rows": int(ntr),
      "val_rows": int(nva),
      "x_dim": int(ds.x_dim),
    })

    student = StudentModel(booster=booster, featurizer=None, metadata=m)
    student.save(output_dir)

    if export_onnx:
      export_xgb_onnx(student.booster, output_dir / "student.onnx", input_dim=int(ds.x_dim))

    with open(output_dir / "train_meta.json", "w") as fp:
      json.dump(m, fp, indent=2)

    logger.success(f"Training artifacts written to {output_dir}")
    return student

  def distill(
    self,
    teacher_parquet: str | Path,
    out_dir: str | Path,
    task: str = "regression",
    time_budget: int | None = None,
    n_trials: int = 50,
    export_onnx: bool = True,
    x_col: str = "x",
    y_soft_col: str = "y",
    w_col: str | None = "w",
    split_col: str = "split",
    x_dim: int = 768,
    val_frac: float = 0.1,
    seed: int = 42,
    shard_rows: int = 200_000,
    compression: str = "zstd",
    hard_labels: str | Path | None = None,
    hard_smiles_col: str = "smiles",
    hard_y_col: str = "y",
    hard_weight: float = 1.0,
    featurizer=None,
    hard_batch_rows: int = 8192,
    metadata: dict | None = None,
  ) -> StudentModel:
    self.trainer.task = task

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.panel("PACK → TRAIN starting", title="DISTILL")

    packed = self.pack(
      teacher_parquet=teacher_parquet,
      out_dir=out_dir,
      x_col=x_col,
      y_soft_col=y_soft_col,
      w_col=w_col,
      split_col=split_col,
      x_dim=x_dim,
      val_frac=val_frac,
      seed=seed,
      shard_rows=shard_rows,
      compression=compression,
      hard_labels=hard_labels,
      hard_smiles_col=hard_smiles_col,
      hard_y_col=hard_y_col,
      hard_weight=hard_weight,
      featurizer=featurizer,
      hard_batch_rows=hard_batch_rows,
    )

    student = self.train(
      packed_dir=packed,
      output_dir=out_dir,
      export_onnx=export_onnx,
      time_budget=time_budget,
      n_trials=n_trials,
      metadata=metadata,
    )

    logger.success("PACK → TRAIN finished")
    return student
