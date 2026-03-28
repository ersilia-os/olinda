from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import xgboost as xgb

import matplotlib.pyplot as plt

from stylia import label, create_figure, save_figure

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


def _concordance_index(y, p, max_pairs: int = 50_000) -> float:
  """Harrell's C-index for continuous outcomes (sampled for speed)."""
  y = np.asarray(y, dtype=np.float64)
  p = np.asarray(p, dtype=np.float64)
  n = len(y)
  if n < 2:
    return float("nan")

  # For large n the O(n²) pair loop is too slow; sample pairs instead.
  n_all_pairs = n * (n - 1) // 2
  if n_all_pairs <= max_pairs:
    # Small enough — exhaustive via broadcasting on blocks
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
  else:
    rng = np.random.default_rng(0)
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n, size=max_pairs)
    # Ensure i != j
    dup = i == j
    j[dup] = (j[dup] + 1) % n
    dy = y[i] - y[j]
    dp = p[i] - p[j]
    mask = dy != 0
    if not np.any(mask):
      return float("nan")
    dy = dy[mask]
    dp = dp[mask]
    permissible = float(len(dy))
    concordant = float(np.sum(np.sign(dy) == np.sign(dp)) + 0.5 * np.sum(dp == 0))

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

  # Fast density estimation via 2D histogram (binned), then look up per point
  GRID = 200
  yf = yy.astype(np.float64)
  pf = pp.astype(np.float64)
  hist, xedges, yedges = np.histogram2d(yf, pf, bins=GRID)
  # Map each point to its bin
  xi = np.clip(np.digitize(yf, xedges) - 1, 0, GRID - 1)
  yi = np.clip(np.digitize(pf, yedges) - 1, 0, GRID - 1)
  density = hist[xi, yi].astype(np.float64)
  dmin, dmax = density.min(), density.max()
  if dmax > dmin:
    density = (density - dmin) / (dmax - dmin)

  # Sort by density so densest points render on top
  order = np.argsort(density)

  fig, am = create_figure(nrows=1, ncols=1)
  ax = am[0]
  sc = ax.scatter(
    yy[order], pp[order],
    c=density[order], cmap="viridis", s=6, edgecolors="none",
  )
  ax.plot([lo, hi], [lo, hi], color="grey", linewidth=1, linestyle="--")
  cbar = plt.colorbar(sc, ax=ax)
  cbar.set_label("Density")
  label(ax, xlabel="True (val)", ylabel="Predicted (val)", title="Predicted vs True")
  save_figure(str(out_png))
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


def _plot_before_after_calibration(y, p_raw, p_cal, out_png: Path):
  """Side-by-side: raw vs calibrated predictions against true values."""
  n = len(y)
  if n == 0:
    return

  fig, axes = plt.subplots(1, 3, figsize=(18, 5))

  lo = float(min(y.min(), p_raw.min(), p_cal.min()))
  hi = float(max(y.max(), p_raw.max(), p_cal.max()))

  # Raw pred vs true
  axes[0].scatter(y, p_raw, s=6, alpha=0.35)
  axes[0].plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1)
  axes[0].set_xlabel("True (val)")
  axes[0].set_ylabel("Raw prediction")
  axes[0].set_title(f"Before calibration\nMAE={_mae(y, p_raw):.5f}  RMSE={_rmse(y, p_raw):.5f}")

  # Calibrated pred vs true
  axes[1].scatter(y, p_cal, s=6, alpha=0.35, color="darkorange")
  axes[1].plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1)
  axes[1].set_xlabel("True (val)")
  axes[1].set_ylabel("Calibrated prediction")
  axes[1].set_title(f"After calibration\nMAE={_mae(y, p_cal):.5f}  RMSE={_rmse(y, p_cal):.5f}")

  # Prediction range comparison
  axes[2].hist(p_raw, bins=60, alpha=0.6, label="Raw", color="steelblue", density=True)
  axes[2].hist(p_cal, bins=60, alpha=0.6, label="Calibrated", color="darkorange", density=True)
  axes[2].hist(y, bins=60, alpha=0.4, label="True", color="grey", density=True)
  axes[2].set_xlabel("Value")
  axes[2].set_ylabel("Density")
  axes[2].set_title("Distribution comparison")
  axes[2].legend()

  fig.suptitle("Post-hoc Isotonic Calibration", fontsize=13, fontweight="bold")
  fig.tight_layout()
  fig.savefig(out_png, dpi=150)
  plt.close()


def validate_regression(
  packed_dir: str | Path,
  model_dir: str | Path | None = None,
  out_dir: str | Path | None = None,
  batch_rows: int = 65536,
  max_points: int | None = None,
  n_boot: int = 200,
  calibrate: bool = True,
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
  }

  ci = {
    "mae_ci95": _bootstrap_ci(y, p, _mae, n_boot=n_boot, seed=1),
    "rmse_ci95": _bootstrap_ci(y, p, _rmse, n_boot=n_boot, seed=2),
    "r2_ci95": _bootstrap_ci(y, p, _r2, n_boot=n_boot, seed=3),
    "pearson_ci95": _bootstrap_ci(y, p, _pearsonr, n_boot=n_boot, seed=4),
    "spearman_ci95": _bootstrap_ci(y, p, _spearmanr, n_boot=n_boot, seed=5),
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
    f"coverage_2std={metrics['coverage_2std']:.6f}"
  )
  logger.debug(
    "Validation CI95: "
    f"mae_ci95={ci['mae_ci95']} "
    f"rmse_ci95={ci['rmse_ci95']} "
    f"r2_ci95={ci['r2_ci95']} "
    f"pearson_ci95={ci['pearson_ci95']} "
    f"spearman_ci95={ci['spearman_ci95']}"
  )

  _plot_pred_vs_true(y, p, vdir / "pred_vs_true.png")
  _plot_residual_hist(y, p, vdir / "residual_hist.png")
  _plot_residual_vs_pred(y, p, vdir / "residual_vs_pred.png")
  _plot_calibration_bins(y, p, vdir / "calibration_bins.png")
  _plot_qq_residuals(y, p, vdir / "residuals_qq.png")

  report = {"metrics": metrics, "ci": ci}

  # ── Post-hoc isotonic calibration ──
  if calibrate:
    from olinda.calibrate import IsotonicCalibrator

    calibrator = IsotonicCalibrator().fit(raw=p, target=y)
    p_cal = calibrator.transform(p)

    cal_metrics = {
      "val_rows": int(len(y)),
      "mae": _mae(y, p_cal),
      "rmse": _rmse(y, p_cal),
      "r2": _r2(y, p_cal),
      "pearson": _pearsonr(y, p_cal),
      "spearman": _spearmanr(y, p_cal),
      "concordance": _concordance_index(y, p_cal),
      "coverage_1std": _coverage(y, p_cal, std_scale=1.0),
      "coverage_2std": _coverage(y, p_cal, std_scale=2.0),
    }

    logger.info(
      "Calibrated metrics: "
      f"mae={cal_metrics['mae']:.6f} (was {metrics['mae']:.6f}) "
      f"rmse={cal_metrics['rmse']:.6f} (was {metrics['rmse']:.6f}) "
      f"r2={cal_metrics['r2']:.6f} (was {metrics['r2']:.6f}) "
      f"spearman={cal_metrics['spearman']:.6f} (was {metrics['spearman']:.6f})"
    )

    _plot_pred_vs_true(y, p_cal, vdir / "pred_vs_true_calibrated.png")
    _plot_calibration_bins(y, p_cal, vdir / "calibration_bins_calibrated.png")
    _plot_before_after_calibration(y, p, p_cal, vdir / "calibration_comparison.png")

    calibrator.save(model_dir / "calibrator.json")
    report["calibrated_metrics"] = cal_metrics
    report["calibrator_path"] = str(model_dir / "calibrator.json")
  else:
    logger.info("Calibration skipped (--no-calibrate)")

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
