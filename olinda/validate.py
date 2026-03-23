from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import xgboost as xgb

import matplotlib.pyplot as plt

from olinda.helpers import logger


def _as_array(col: pa.Array | pa.ChunkedArray) -> pa.Array:
  if isinstance(col, pa.ChunkedArray):
    return col.combine_chunks()
  return col


def _x_to_2d(x_col: pa.Array | pa.ChunkedArray, x_dim: int) -> np.ndarray:
  arr = _as_array(x_col)
  t = arr.type

  if pa.types.is_fixed_size_list(t):
    if t.list_size != x_dim:
      raise ValueError(f"x_dim mismatch: parquet={t.list_size} expected={x_dim}")
    values = arr.values
    v = np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float32)
    return v.reshape(len(arr), x_dim)

  if pa.types.is_list(t) or pa.types.is_large_list(t):
    py = arr.to_pylist()
    X = np.asarray(py, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != x_dim:
      raise ValueError(f"x shape mismatch from list: got {X.shape}, expected (*,{x_dim})")
    return X

  raise ValueError(f"unsupported x column type: {t}")


def _pearsonr(y, p) -> float:
  y = np.asarray(y, dtype=np.float64)
  p = np.asarray(p, dtype=np.float64)
  y = y - y.mean()
  p = p - p.mean()
  denom = np.sqrt((y * y).sum()) * np.sqrt((p * p).sum())
  if denom == 0:
    return float("nan")
  return float((y * p).sum() / denom)


def _spearmanr(y, p) -> float:
  y = np.asarray(y, dtype=np.float64)
  p = np.asarray(p, dtype=np.float64)
  ry = np.argsort(np.argsort(y))
  rp = np.argsort(np.argsort(p))
  return _pearsonr(ry, rp)


def _r2(y, p) -> float:
  y = np.asarray(y, dtype=np.float64)
  p = np.asarray(p, dtype=np.float64)
  ss_res = ((y - p) ** 2).sum()
  ss_tot = ((y - y.mean()) ** 2).sum()
  if ss_tot == 0:
    return float("nan")
  return float(1.0 - ss_res / ss_tot)


def _mae(y, p) -> float:
  y = np.asarray(y, dtype=np.float64)
  p = np.asarray(p, dtype=np.float64)
  return float(np.mean(np.abs(y - p)))


def _rmse(y, p) -> float:
  y = np.asarray(y, dtype=np.float64)
  p = np.asarray(p, dtype=np.float64)
  return float(np.sqrt(np.mean((y - p) ** 2)))


def _concordance_index(y, p) -> float:
  """Harrell's C-index for continuous outcomes."""
  y = np.asarray(y, dtype=np.float64)
  p = np.asarray(p, dtype=np.float64)
  n = len(y)
  if n < 2:
    return float("nan")

  concordant = 0.0
  permissible = 0.0
  for i in range(n):
    dy = y[i] - y[i + 1 :]
    dp = p[i] - p[i + 1 :]
    mask = dy != 0
    if not np.any(mask):
      continue
    dy = dy[mask]
    dp = dp[mask]
    permissible += len(dy)
    concordant += np.sum(np.sign(dy) == np.sign(dp)) + 0.5 * np.sum(dp == 0)

  if permissible == 0:
    return float("nan")
  return float(concordant / permissible)


def _coverage(y, p, std_scale: float = 1.0) -> float:
  """Percentage of points within ±std_scale * std(residuals)."""
  r = (p - y).astype(np.float64)
  if len(r) == 0:
    return float("nan")
  s = np.std(r)
  if s == 0:
    return 1.0
  return float(np.mean(np.abs(r) <= std_scale * s))


def _auc_roc(y, p, threshold: float = 0.5) -> float:
  """AUC-ROC: binarise *y* at *threshold*, score with *p*.

  Pure-NumPy implementation using the trapezoidal rule on the ROC curve.
  Returns NaN when only one class is present after binarisation.
  """
  y = np.asarray(y, dtype=np.float64)
  p = np.asarray(p, dtype=np.float64)
  labels = (y >= threshold).astype(np.int64)

  # AUC is undefined when only one class is present
  if labels.sum() == 0 or labels.sum() == len(labels):
    return float("nan")

  # Sort by descending predicted score
  order = np.argsort(-p)
  labels_sorted = labels[order]

  # Cumulative TP / FP counts
  tps = np.cumsum(labels_sorted)
  fps = np.cumsum(1 - labels_sorted)

  # Append origin (0, 0)
  tps = np.concatenate([[0], tps])
  fps = np.concatenate([[0], fps])

  # Normalise to rates
  tpr = tps / tps[-1]
  fpr = fps / fps[-1]

  # Trapezoidal AUC
  return float(np.trapezoid(tpr, fpr))


def _bootstrap_ci(y, p, metric_fn, n_boot: int = 200, seed: int = 0) -> tuple[float, float]:
  rng = np.random.default_rng(seed)
  n = len(y)
  if n == 0:
    return float("nan"), float("nan")
  vals = []
  for _ in range(n_boot):
    idx = rng.integers(0, n, size=n)
    vals.append(metric_fn(y[idx], p[idx]))
  lo = np.percentile(vals, 2.5)
  hi = np.percentile(vals, 97.5)
  return float(lo), float(hi)


def _load_meta(packed_dir: Path) -> dict:
  mp = packed_dir / "meta.json"
  if not mp.exists():
    raise ValueError(f"missing meta.json in {packed_dir}")
  with open(mp, "r") as fp:
    return json.load(fp)


def _iter_val_batches(packed_dir: Path, x_col: str, y_col: str, w_col: str | None, batch_rows: int):
  val_dir = packed_dir / "val"
  dset = ds.dataset(str(val_dir), format="parquet")
  cols = [x_col, y_col]
  if w_col:
    cols.append(w_col)
  scanner = dset.scanner(columns=cols, batch_size=batch_rows)
  for b in scanner.to_batches():
    yield pa.Table.from_batches([b])


def _collect_predictions(
  packed_dir: Path, booster: xgb.Booster, batch_rows: int, max_points: int | None = None
):
  meta = _load_meta(packed_dir)
  x_col = meta["x_col"]
  y_col = meta["y_col"]
  w_col = meta.get("w_col") or "w"
  x_dim = int(meta["x_dim"])

  ys, ps = [], []
  seen = 0

  for t in _iter_val_batches(packed_dir, x_col=x_col, y_col=y_col, w_col=w_col, batch_rows=batch_rows):
    X = _x_to_2d(t.column(x_col), x_dim)
    y = np.asarray(_as_array(t.column(y_col)).to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1)
    dmat = xgb.DMatrix(X)
    p = booster.predict(dmat).astype(np.float32).reshape(-1)

    ys.append(y)
    ps.append(p)
    seen += len(y)

    if max_points is not None and seen >= max_points:
      break

  if not ys:
    return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

  y = np.concatenate(ys)
  p = np.concatenate(ps)

  if max_points is not None and len(y) > max_points:
    y = y[:max_points]
    p = p[:max_points]

  return y, p


def _plot_pred_vs_true(y, p, out_png: Path, max_scatter: int = 200_000):
  n = len(y)
  if n == 0:
    return

  if n > max_scatter:
    idx = np.random.default_rng(0).choice(n, size=max_scatter, replace=False)
    yy = y[idx]
    pp = p[idx]
  else:
    yy, pp = y, p

  lo = float(min(yy.min(), pp.min()))
  hi = float(max(yy.max(), pp.max()))

  plt.figure()
  plt.scatter(yy, pp, s=6, alpha=0.35)
  plt.plot([lo, hi], [lo, hi])
  plt.xlabel("True (val)")
  plt.ylabel("Predicted (val)")
  plt.title("Predicted vs True")
  plt.tight_layout()
  plt.savefig(out_png, dpi=200)
  plt.close()


def _plot_residual_hist(y, p, out_png: Path):
  r = (p - y).astype(np.float64)
  plt.figure()
  plt.hist(r, bins=60)
  plt.xlabel("Residual (pred - true)")
  plt.ylabel("Count")
  plt.title("Residual Histogram")
  plt.tight_layout()
  plt.savefig(out_png, dpi=200)
  plt.close()


def _plot_residual_vs_pred(y, p, out_png: Path, max_scatter: int = 200_000):
  n = len(y)
  if n == 0:
    return

  r = (p - y).astype(np.float64)

  if n > max_scatter:
    idx = np.random.default_rng(1).choice(n, size=max_scatter, replace=False)
    pp = p[idx]
    rr = r[idx]
  else:
    pp, rr = p, r

  plt.figure()
  plt.scatter(pp, rr, s=6, alpha=0.35)
  plt.axhline(0.0)
  plt.xlabel("Predicted (val)")
  plt.ylabel("Residual (pred - true)")
  plt.title("Residual vs Predicted")
  plt.tight_layout()
  plt.savefig(out_png, dpi=200)
  plt.close()


def _plot_calibration_bins(y, p, out_png: Path, n_bins: int = 20):
  y = np.asarray(y, dtype=np.float64)
  p = np.asarray(p, dtype=np.float64)
  if len(y) == 0:
    return

  lo = float(min(y.min(), p.min()))
  hi = float(max(y.max(), p.max()))
  if lo == hi:
    return

  edges = np.linspace(lo, hi, n_bins + 1)
  bin_id = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)

  mean_p = np.zeros(n_bins, dtype=np.float64)
  mean_y = np.zeros(n_bins, dtype=np.float64)
  cnt = np.zeros(n_bins, dtype=np.int64)

  for b in range(n_bins):
    m = bin_id == b
    cnt[b] = int(m.sum())
    if cnt[b] > 0:
      mean_p[b] = float(p[m].mean())
      mean_y[b] = float(y[m].mean())
    else:
      mean_p[b] = np.nan
      mean_y[b] = np.nan

  plt.figure()
  plt.plot(mean_p, mean_y, marker="o")
  plt.plot([lo, hi], [lo, hi])
  plt.xlabel("Mean predicted (bin)")
  plt.ylabel("Mean true (bin)")
  plt.title("Calibration by Prediction Bins")
  plt.tight_layout()
  plt.savefig(out_png, dpi=200)
  plt.close()


def _erfinv(x: float) -> float:
  """Approximate inverse error function (Winitzki)."""
  a = 0.147
  s = 1 if x >= 0 else -1
  ln = math.log(1.0 - x * x)
  t = 2.0 / (math.pi * a) + ln / 2.0
  return s * math.sqrt(math.sqrt(t * t - ln / a) - t)


def _plot_roc_curve(y, p, out_png: Path, threshold: float = 0.5):
  """Plot the ROC curve (FPR vs TPR) with AUC annotated."""
  y = np.asarray(y, dtype=np.float64)
  p = np.asarray(p, dtype=np.float64)
  labels = (y >= threshold).astype(np.int64)

  n_pos = int(labels.sum())
  n_neg = int(len(labels) - n_pos)
  if n_pos == 0 or n_neg == 0:
    return  # ROC undefined with a single class

  order = np.argsort(-p)
  labels_sorted = labels[order]

  tps = np.cumsum(labels_sorted)
  fps = np.cumsum(1 - labels_sorted)

  tps = np.concatenate([[0], tps])
  fps = np.concatenate([[0], fps])

  tpr = tps / n_pos
  fpr = fps / n_neg

  auc = float(np.trapezoid(tpr, fpr))

  plt.figure()
  plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {auc:.4f})")
  plt.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Random")
  plt.xlim(-0.01, 1.01)
  plt.ylim(-0.01, 1.01)
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("ROC Curve")
  plt.legend(loc="lower right")
  plt.tight_layout()
  plt.savefig(out_png, dpi=200)
  plt.close()


def _plot_qq_residuals(y, p, out_png: Path):
  r = (p - y).astype(np.float64)
  if len(r) == 0:
    return
  r = (r - np.mean(r)) / (np.std(r) + 1e-12)
  r = np.sort(r)
  n = len(r)
  q = np.linspace(0.5 / n, 1 - 0.5 / n, n)
  z = np.array([math.sqrt(2) * _erfinv(2 * qi - 1) for qi in q])
  plt.figure()
  plt.scatter(z, r, s=6, alpha=0.35)
  lo = min(z.min(), r.min())
  hi = max(z.max(), r.max())
  plt.plot([lo, hi], [lo, hi])
  plt.xlabel("Theoretical quantiles")
  plt.ylabel("Standardized residuals")
  plt.title("QQ Plot (Residuals)")
  plt.tight_layout()
  plt.savefig(out_png, dpi=200)
  plt.close()


def validate_regression(
  packed_dir: str | Path,
  model_dir: str | Path | None = None,
  out_dir: str | Path | None = None,
  batch_rows: int = 65536,
  max_points: int | None = None,
  n_boot: int = 200,
):
  packed_dir = Path(packed_dir)
  model_dir = Path(model_dir) if model_dir else packed_dir
  out_dir = Path(out_dir) if out_dir else model_dir
  vdir = out_dir / "validation"
  vdir.mkdir(parents=True, exist_ok=True)

  booster = xgb.Booster()
  booster.load_model(str(model_dir / "xgb.json"))

  logger.info(f"Validation reading val split from {packed_dir / 'val'}")
  y, p = _collect_predictions(packed_dir, booster, batch_rows=batch_rows, max_points=max_points)

  metrics = {
    "val_rows": int(len(y)),
    "mae": _mae(y, p),
    "rmse": _rmse(y, p),
    "r2": _r2(y, p),
    "pearson": _pearsonr(y, p),
    "spearman": _spearmanr(y, p),
    "concordance": _concordance_index(y, p),
    "coverage_1std": _coverage(y, p, std_scale=1.0),
    "coverage_2std": _coverage(y, p, std_scale=2.0),
    "auc_roc": _auc_roc(y, p, threshold=0.5),
  }

  ci = {
    "mae_ci95": _bootstrap_ci(y, p, _mae, n_boot=n_boot, seed=1),
    "rmse_ci95": _bootstrap_ci(y, p, _rmse, n_boot=n_boot, seed=2),
    "r2_ci95": _bootstrap_ci(y, p, _r2, n_boot=n_boot, seed=3),
    "pearson_ci95": _bootstrap_ci(y, p, _pearsonr, n_boot=n_boot, seed=4),
    "spearman_ci95": _bootstrap_ci(y, p, _spearmanr, n_boot=n_boot, seed=5),
    "auc_roc_ci95": _bootstrap_ci(y, p, _auc_roc, n_boot=n_boot, seed=6),
  }

  logger.info(
    "Validation metrics: "
    f"val_rows={metrics['val_rows']} "
    f"mae={metrics['mae']:.6f} "
    f"rmse={metrics['rmse']:.6f} "
    f"r2={metrics['r2']:.6f} "
    f"pearson={metrics['pearson']:.6f} "
    f"spearman={metrics['spearman']:.6f} "
    f"concordance={metrics['concordance']:.6f} "
    f"coverage_1std={metrics['coverage_1std']:.6f} "
    f"coverage_2std={metrics['coverage_2std']:.6f} "
    f"auc_roc={metrics['auc_roc']:.6f}"
  )
  logger.debug(
    "Validation CI95: "
    f"mae_ci95={ci['mae_ci95']} "
    f"rmse_ci95={ci['rmse_ci95']} "
    f"r2_ci95={ci['r2_ci95']} "
    f"pearson_ci95={ci['pearson_ci95']} "
    f"spearman_ci95={ci['spearman_ci95']} "
    f"auc_roc_ci95={ci['auc_roc_ci95']}"
  )

  _plot_pred_vs_true(y, p, vdir / "pred_vs_true.png")
  _plot_residual_hist(y, p, vdir / "residual_hist.png")
  _plot_residual_vs_pred(y, p, vdir / "residual_vs_pred.png")
  _plot_calibration_bins(y, p, vdir / "calibration_bins.png")
  _plot_qq_residuals(y, p, vdir / "residuals_qq.png")
  _plot_roc_curve(y, p, vdir / "roc_curve.png")

  report = {"metrics": metrics, "ci": ci}

  with open(vdir / "validation_report.json", "w") as fp:
    json.dump(report, fp, indent=2)

  logger.success(f"Validation written to {vdir}")
  return report


if __name__ == "__main__":
  import argparse

  ap = argparse.ArgumentParser()
  ap.add_argument("--packed-dir", required=True)
  ap.add_argument("--model-dir", default=None)
  ap.add_argument("--out-dir", default=None)
  ap.add_argument("--batch-rows", type=int, default=65536)
  ap.add_argument("--max-points", type=int, default=None)
  ap.add_argument("--n-boot", type=int, default=200)
  args = ap.parse_args()

  validate_regression(
    packed_dir=args.packed_dir,
    model_dir=args.model_dir,
    out_dir=args.out_dir,
    batch_rows=args.batch_rows,
    max_points=args.max_points,
    n_boot=args.n_boot,
  )
