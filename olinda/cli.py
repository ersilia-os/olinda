from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import rich_click as click
import rich_click.rich_click as rc

from olinda.helpers import logger
from olinda.data import pack_distill_dataset, pack_feature_table, ParquetDistillDataset, ParquetDataIter
from olinda.train import XGBTrainer
from olinda.models import StudentModel, export_xgb_onnx
from olinda.featurizer import Fingerprint, OnnxEncoder
from olinda.validate import validate_regression
from olinda.robustness import robustness_eval_smiles


click.rich_click.USE_RICH_MARKUP = True
click.rich_click.SHOW_ARGUMENTS = True

rc.USE_RICH_MARKUP = True
rc.SHOW_ARGUMENTS = True
rc.COLOR_SYSTEM = "truecolor"
rc.STYLE_OPTION = "bold magenta"
rc.STYLE_COMMAND = "bold green"
rc.STYLE_METAVAR = "italic yellow"
rc.STYLE_SWITCH = "underline cyan"
rc.STYLE_USAGE = "bold blue"
rc.STYLE_OPTION_DEFAULT = "dim italic"


def apply_opts(*opts):
  def _wrap(f):
    for opt in reversed(opts):
      f = opt(f)
    return f

  return _wrap


@click.group()
def cli():
  pass


opt_teacher_parquet = click.option(
  "--teacher-parquet", "-i", required=True, help="Parquet with x(768), y(soft), optional w/split"
)
opt_out = click.option("--out", "-o", required=True, help="Output directory")
opt_x_col = click.option("--x-col", default="x", show_default=True)
opt_y_col = click.option("--y-col", default="y", show_default=True)
opt_w_col = click.option("--w-col", default="w", show_default=True)
opt_split_col = click.option("--split-col", default="split", show_default=True)
opt_x_dim = click.option("--x-dim", default=768, type=int, show_default=True)
opt_val_frac = click.option("--val-frac", default=0.1, type=float, show_default=True)
opt_seed = click.option("--seed", default=42, type=int, show_default=True)
opt_shard_rows = click.option("--shard-rows", default=200000, type=int, show_default=True)
opt_compression = click.option("--compression", default="zstd", show_default=True)

opt_hard = click.option(
  "--hard-labels", required=False, default=None, help="CSV/Parquet with smiles,y (hard labels)"
)
opt_hard_smiles = click.option("--hard-smiles-col", default="smiles", show_default=True)
opt_hard_y = click.option("--hard-y-col", default="y", show_default=True)
opt_hard_weight = click.option("--hard-weight", default=1.0, type=float, show_default=True)

opt_task = click.option(
  "--task", type=click.Choice(["regression", "classification"]), default="regression", show_default=True
)
opt_time_budget = click.option(
  "--time-budget", required=False, default=None, type=int, help="Optuna time budget (seconds)"
)
opt_trials = click.option("--trials", default=50, type=int, show_default=True)
opt_no_onnx = click.option("--no-onnx", is_flag=True, default=False, help="Skip ONNX export")


@cli.command("pack", help="PACK: convert teacher Parquet into sharded distill dataset.")
@apply_opts(
  opt_teacher_parquet,
  opt_out,
  opt_x_col,
  opt_y_col,
  opt_w_col,
  opt_split_col,
  opt_x_dim,
  opt_val_frac,
  opt_seed,
  opt_shard_rows,
  opt_compression,
  opt_hard,
  opt_hard_smiles,
  opt_hard_y,
  opt_hard_weight,
)
def pack_cmd(
  teacher_parquet,
  out,
  x_col,
  y_col,
  w_col,
  split_col,
  x_dim,
  val_frac,
  seed,
  shard_rows,
  compression,
  hard_labels,
  hard_smiles_col,
  hard_y_col,
  hard_weight,
):
  out_dir = Path(out)
  pack_distill_dataset(
    teacher_parquet=teacher_parquet,
    out_dir=out_dir,
    x_col=x_col,
    y_soft_col=y_col,
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
    featurizer=None if hard_labels is None else _load_default_featurizer(),
  )


def _fit_impl(
  input_path,
  out_dir,
  y_col,
  split_col,
  w_col,
  split_frac,
  smiles_col,
  robust_split,
  similarity_threshold,
  fp,
  fp_size,
  radius,
  njobs,
  seed,
  num_boost_round,
  early_stopping,
  val_frac,
  enum_max,
  ensemble_size,
  task,
  time_budget,
  trials,
  no_onnx,
):
  input_path = Path(input_path)
  out_dir = Path(out_dir)

  if input_path.is_dir() and (input_path / "meta.json").exists():
    packed_dir = input_path
  else:
    packed_dir = out_dir / "packed"
    pack_feature_table(
      input_path=input_path,
      out_dir=packed_dir,
      y_col=y_col,
      split_col=split_col,
      w_col=w_col,
      val_frac=split_frac,
      seed=seed,
    )

  d = ParquetDistillDataset(packed_dir)
  ntr, nva = d.count()
  logger.info(f"Packed dataset rows: train={ntr} val={nva}")

  train_iter = ParquetDataIter(
    d.train,
    d.x_col,
    d.y_col,
    d.w_col,
    x_dim=d.x_dim,
    batch_rows=65536,
    shuffle_row_groups=True,
    seed=seed,
  )

  val_iter = None
  if nva > 0:
    val_iter = ParquetDataIter(
      d.val,
      d.x_col,
      d.y_col,
      d.w_col,
      x_dim=d.x_dim,
      batch_rows=65536,
      shuffle_row_groups=False,
      seed=seed,
    )

  trainer = XGBTrainer(task=task, num_boost_round=num_boost_round, early_stopping_rounds=early_stopping)
  booster, meta = trainer.fit_external(
    train_iter=train_iter, val_iter=val_iter, time_budget=time_budget, n_trials=trials
  )

  student = StudentModel(booster=booster, featurizer=None, metadata=meta)
  student.save(out_dir)

  if not no_onnx:
    export_xgb_onnx(student.booster, out_dir / "student.onnx", input_dim=int(d.x_dim))

  if nva > 0:
    validate_regression(
      packed_dir=packed_dir,
      model_dir=out_dir,
      out_dir=out_dir,
      batch_rows=65536,
      max_points=None,
      n_boot=200,
    )

    if input_path.is_file() and input_path.suffix.lower() in (".csv", ".tsv", ".parquet", ".pq"):
      try:
        robustness_eval_smiles(
          data_path=input_path,
          out_dir=out_dir,
          smiles_col=smiles_col,
          y_col=y_col,
          split=robust_split,
          test_frac=0.2,
          similarity_threshold=similarity_threshold,
          fp=fp,
          fp_size=fp_size,
          radius=radius,
          njobs=njobs,
          seed=seed,
          num_boost_round=num_boost_round,
          early_stopping_rounds=early_stopping,
          val_frac=val_frac,
          enum_max=enum_max,
          ensemble_size=ensemble_size,
        )
      except Exception as exc:
        logger.warning(f"Robustness evaluation skipped: {exc}")


@cli.command("fit", help="FIT: train XGBoost; run validation/robustness when val split exists.")
@click.option("--input", "input_path", required=True, help="Packed dir or CSV/Parquet file")
@click.option("--out", "out_dir", required=True, help="Output directory")
@click.option("--y-col", default="y", show_default=True)
@click.option("--split-col", default="split", show_default=True)
@click.option("--w-col", default=None)
@click.option(
  "--split", "split_frac", default=None, type=float, help="Validation fraction if no split column"
)
@click.option("--smiles-col", default="smiles", show_default=True)
@click.option("--robust-split", type=click.Choice(["scaffold", "similarity"]), default="scaffold")
@click.option("--similarity-threshold", default=0.4, type=float, show_default=True)
@click.option("--fp", default="morgan", show_default=True)
@click.option("--fp-size", default=2048, type=int, show_default=True)
@click.option("--radius", default=2, type=int, show_default=True)
@click.option("--njobs", default=8, type=int, show_default=True)
@click.option("--seed", default=42, type=int, show_default=True)
@click.option("--num-boost-round", default=1000, type=int, show_default=True)
@click.option("--early-stopping", default=50, type=int, show_default=True)
@click.option("--val-frac", default=0.1, type=float, show_default=True)
@click.option("--enum-max", default=8, type=int, show_default=True)
@click.option("--ensemble-size", default=5, type=int, show_default=True)
@apply_opts(opt_task, opt_time_budget, opt_trials, opt_no_onnx)
def fit_cmd(
  input_path,
  out_dir,
  y_col,
  split_col,
  w_col,
  split_frac,
  smiles_col,
  robust_split,
  similarity_threshold,
  fp,
  fp_size,
  radius,
  njobs,
  seed,
  num_boost_round,
  early_stopping,
  val_frac,
  enum_max,
  ensemble_size,
  task,
  time_budget,
  trials,
  no_onnx,
):
  _fit_impl(
    input_path=input_path,
    out_dir=out_dir,
    y_col=y_col,
    split_col=split_col,
    w_col=w_col,
    split_frac=split_frac,
    smiles_col=smiles_col,
    robust_split=robust_split,
    similarity_threshold=similarity_threshold,
    fp=fp,
    fp_size=fp_size,
    radius=radius,
    njobs=njobs,
    seed=seed,
    num_boost_round=num_boost_round,
    early_stopping=early_stopping,
    val_frac=val_frac,
    enum_max=enum_max,
    ensemble_size=ensemble_size,
    task=task,
    time_budget=time_budget,
    trials=trials,
    no_onnx=no_onnx,
  )


@cli.command("predict", help="PREDICT: run a trained student on an input matrix.")
@click.option("--model-dir", required=True, help="Directory with xgb.json (train_meta.json optional)")
@click.option("--input", "input_path", required=True, help="CSV/Parquet/NPY with features")
@click.option("--out", "out_path", default=None, help="Output CSV for predictions")
@click.option("--smiles-col", default="smiles", show_default=True, help="SMILES column name")
@click.option(
  "--columns",
  default=None,
  help="Comma-separated column list for CSV/Parquet (default: all numeric columns)",
)
@click.option("--fp", default="morgan", show_default=True, help="Fingerprint type")
@click.option("--fp-size", default=2048, type=int, show_default=True)
@click.option("--radius", default=2, type=int, show_default=True)
@click.option("--njobs", default=8, type=int, show_default=True)
@click.option("--smarts", is_flag=True, default=False, help="Treat input as SMARTS")
@click.option("--no-sanitize", is_flag=True, default=False, help="Disable RDKit sanitization")
def predict_cmd(
  model_dir,
  input_path,
  out_path,
  smiles_col,
  columns,
  fp,
  fp_size,
  radius,
  njobs,
  smarts,
  no_sanitize,
):
  model_dir = Path(model_dir)
  input_path = Path(input_path)
  if not input_path.exists():
    logger.error(f"Input not found: {input_path}")
    raise click.ClickException("input file does not exist")

  student = StudentModel.load(model_dir, featurizer_factory=_featurizer_factory)
  logger.info(f"Loaded student model from {model_dir}")

  expected_dim = None
  try:
    expected_dim = int(student.booster.num_features())
  except Exception:
    expected_dim = None

  if expected_dim is not None and expected_dim > 0 and student.featurizer is not None:
    fz_dim = getattr(student.featurizer, "fp_size", None)
    if fz_dim is None and hasattr(student.featurizer, "base"):
      fz_dim = getattr(student.featurizer.base, "fp_size", None)
    if fz_dim is not None and int(fz_dim) != expected_dim:
      logger.error(f"Featurizer dimension mismatch: model expects {expected_dim}, featurizer has {fz_dim}")
      raise click.ClickException("featurizer dimension mismatch")

  data = _load_predict_input(input_path, columns, smiles_col)
  if data["mode"] == "smiles":
    smiles = data["smiles"]
    if student.featurizer is None:
      if expected_dim is not None and expected_dim > 0 and int(fp_size) != expected_dim:
        logger.warning(f"Overriding fp_size {fp_size} -> {expected_dim} to match model feature size")
        fp_size = expected_dim
      student.featurizer = Fingerprint(
        which=fp,
        fp_size=int(fp_size),
        radius=int(radius),
        is_smarts=bool(smarts),
        sanitize=not bool(no_sanitize),
        njobs=int(njobs),
      )
      logger.info(
        "Using Fingerprint featurizer: "
        f"which={fp} fp_size={fp_size} radius={radius} smarts={smarts} sanitize={not no_sanitize} njobs={njobs}"
      )
    logger.info(f"Predicting smiles rows={len(smiles)}")
    y = student.predict(smiles=smiles)
  else:
    X = data["X"]
    if expected_dim is not None and expected_dim > 0 and int(X.shape[1]) != expected_dim:
      logger.error(f"Feature dimension mismatch: model expects {expected_dim}, got {X.shape[1]}")
      raise click.ClickException("feature dimension mismatch")
    logger.info(f"Predicting rows={X.shape[0]} cols={X.shape[1]}")
    y = student.predict(X=X)

  if out_path is None:
    out_path = input_path.with_suffix(".pred.csv")
  out_path = Path(out_path)

  df = pd.DataFrame({"prediction": y})
  df.to_csv(out_path, index=False)
  logger.success(f"Predictions written to {out_path}")


@cli.command("distill", help="Distill a teacher QSAR model into an XGBoost student (optionally export ONNX).")
@apply_opts(
  opt_teacher_parquet,
  opt_out,
  opt_task,
  opt_time_budget,
  opt_trials,
  opt_no_onnx,
  opt_x_col,
  opt_y_col,
  opt_w_col,
  opt_split_col,
  opt_x_dim,
  opt_val_frac,
  opt_seed,
  opt_shard_rows,
  opt_compression,
  opt_hard,
  opt_hard_smiles,
  opt_hard_y,
  opt_hard_weight,
)
def distill_cmd(
  teacher_parquet,
  out,
  task,
  time_budget,
  trials,
  no_onnx,
  x_col,
  y_col,
  w_col,
  split_col,
  x_dim,
  val_frac,
  seed,
  shard_rows,
  compression,
  hard_labels,
  hard_smiles_col,
  hard_y_col,
  hard_weight,
):
  out_dir = Path(out)
  pack_distill_dataset(
    teacher_parquet=teacher_parquet,
    out_dir=out_dir,
    x_col=x_col,
    y_soft_col=y_col,
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
    featurizer=None if hard_labels is None else _load_default_featurizer(),
  )
  _fit_impl(
    input_path=str(out_dir),
    out_dir=str(out_dir),
    y_col=y_col,
    split_col=split_col,
    w_col=w_col,
    split_frac=None,
    smiles_col=hard_smiles_col,
    robust_split="scaffold",
    similarity_threshold=0.4,
    fp="morgan",
    fp_size=2048,
    radius=2,
    njobs=8,
    seed=seed,
    num_boost_round=1000,
    early_stopping=50,
    val_frac=0.1,
    enum_max=8,
    ensemble_size=5,
    task=task,
    time_budget=time_budget,
    trials=trials,
    no_onnx=no_onnx,
  )


def _load_default_featurizer():
  from olinda.featurizer import Fingerprint

  logger.info(
    "Loading default featurizer for hard labels (Fingerprint morgan fp_size=768 is NOT valid; configure yours)"
  )
  return Fingerprint(which="morgan", fp_size=768, radius=2, njobs=1)


def _load_predict_input(input_path: Path, columns: str | None, smiles_col: str) -> dict:
  suffix = input_path.suffix.lower()
  if suffix == ".npy":
    X = np.load(str(input_path))
    if X.ndim != 2:
      logger.error("NPY input must be a 2D array")
      raise click.ClickException("npy input must be a 2D array")
    return {"mode": "matrix", "X": X.astype(np.float32), "columns": []}

  if suffix in (".parquet", ".pq"):
    df = pd.read_parquet(str(input_path))
  elif suffix in (".csv", ".tsv"):
    sep = "\t" if suffix == ".tsv" else ","
    df = pd.read_csv(str(input_path), sep=sep)
  else:
    logger.error(f"Unsupported input format: {suffix}")
    raise click.ClickException("unsupported input format")

  if columns:
    cols = [c.strip() for c in columns.split(",") if c.strip()]
    missing = [c for c in cols if c not in df.columns]
    if missing:
      logger.error(f"Missing columns: {missing}")
      raise click.ClickException("missing columns in input")
    feat_df = df[cols]
    X = feat_df.to_numpy().astype(np.float32)
    logger.debug(f"Using feature columns: {cols}")
    return {"mode": "matrix", "X": X, "columns": cols}

  if smiles_col in df.columns:
    smiles = df[smiles_col].astype(str).tolist()
    return {"mode": "smiles", "smiles": smiles, "columns": [smiles_col]}

  feat_df = df.select_dtypes(include=[np.number])
  cols = list(feat_df.columns)
  if not cols:
    logger.error("No numeric columns found; use --columns or provide a smiles column")
    raise click.ClickException("no numeric columns found")
  X = feat_df.to_numpy().astype(np.float32)
  logger.debug(f"Using feature columns: {cols}")
  return {"mode": "matrix", "X": X, "columns": cols}


def _featurizer_factory(class_name: str | None, cfg: dict):
  if not class_name:
    return None
  if class_name == "Fingerprint":
    return Fingerprint.from_dict(cfg)
  if class_name == "OnnxEncoder":

    def _base_factory(d):
      return Fingerprint.from_dict(d)

    return OnnxEncoder.from_dict(cfg, _base_factory)
  logger.warning(f"Unknown featurizer class: {class_name}")
  return None


if __name__ == "__main__":
  cli()
