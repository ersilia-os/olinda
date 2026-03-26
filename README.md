<div align="center">

# Olinda

Lightweight model distillation for chemistry workflows.

Train compact **XGBoost** student models from feature tables or SMILES-derived fingerprints, with built-in hyperparameter tuning, validation, and robustness evaluation.

<br/>

[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![XGBoost](https://img.shields.io/badge/XGBoost-%3E%3D3.2-017CEE?style=flat-square)](#)
[![License](https://img.shields.io/badge/License-MIT-2ea44f?style=flat-square)](#license)

<br/>

[Installation](#installation) &middot;
[Quick Start](#quick-start) &middot;
[CLI Reference](#cli-reference) &middot;
[Hyperparameter Tuning](#hyperparameter-tuning) &middot;
[Validation](#validation) &middot;
[License](#license)

</div>

---

## Overview

Olinda fits a student model from either packed feature tables or SMILES-derived fingerprints, runs validation and robustness checks when a validation split exists, and exports artifacts in both XGBoost and ONNX formats.

**Key capabilities:**

- **Automatic GPU detection** &mdash; seamlessly switches between CPU and CUDA at runtime
- **Efficient hyperparameter tuning** &mdash; two-phase Optuna search with DMatrix caching and median pruning
- **Streaming data pipeline** &mdash; `QuantileDMatrix` with Parquet-backed iterators for datasets that exceed RAM
- **Comprehensive evaluation** &mdash; regression metrics with bootstrap CIs, calibration plots, QQ diagnostics, and density scatter maps
- **Post-hoc isotonic calibration** &mdash; automatic monotonic calibrator fitted on validation predictions, ensuring student outputs stay within the teacher's range
- **Robustness analysis** &mdash; scaffold/similarity splits, SMILES perturbation invariance, and ensemble uncertainty
- **Combined soft + hard label training** &mdash; mix teacher soft labels with ground-truth hard labels via `--hard-labels`
- **One-command distillation** &mdash; pack, train, evaluate, and export in a single `olinda distill` call

---

## Installation

```bash
pip install -e .
```

Optional extras:

| Extra | Purpose | Install |
|-------|---------|---------|
| `train` | Hyperparameter tuning (Optuna) | `pip install -e ".[train]"` |
| `viz` | Validation plots (Matplotlib, stylia, scipy) | `pip install -e ".[viz]"` |
| `dev` | Linting and testing (Ruff, pytest) | `pip install -e ".[dev]"` |

Olinda ships with standard `xgboost`; GPU boosting activates automatically when a CUDA-enabled build and compatible GPU are detected.

---

## Quick Start

### Train from CSV

```bash
olinda fit \
  --input data.csv \
  --out runs/my_model \
  --y-col activity \
  --split 0.15
```

### Train with combined soft + hard labels

```bash
olinda fit \
  --input soft_labels.csv \
  --out runs/combined \
  --y-col clf \
  --smiles-col smiles \
  --split 0.15 \
  --hard-labels hard_labels.csv \
  --hard-smiles-col Drug \
  --hard-y-col Y \
  --hard-weight 1.0
```

### Predict

```bash
olinda predict \
  --model-dir runs/my_model \
  --input compounds.csv \
  --smiles-col smiles \
  --out predictions.csv
```

---

## Data Format

Olinda accepts CSV, TSV, or Parquet files with the following columns:

| Column | Required | Description |
|--------|----------|-------------|
| Feature columns | Yes | Numeric columns, or a SMILES column for fingerprint generation |
| `y` | Yes | Target value (configurable via `--y-col`) |
| `split` | No | `0` = train, `1` = validation. If absent, use `--split` to create one |
| `w` | No | Sample weight |

When a SMILES column is detected and fewer than 32 numeric feature columns are present, Olinda automatically generates Morgan fingerprints.

---

## CLI Reference

Olinda provides four commands: **pack**, **fit**, **predict**, and **distill**.

### `olinda fit`

Train an XGBoost student with optional hyperparameter tuning.

```bash
olinda fit \
  --input <packed_dir_or_file> \
  --out <output_dir> \
  --num-boost-round 1000 \
  --early-stopping 50 \
  --time-budget 600 \
  --trials 10 \
  --robustness
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | *required* | Packed directory or CSV/Parquet file |
| `--out` | *required* | Output directory |
| `--y-col` | `y` | Target column name |
| `--split` | *none* | Validation fraction (e.g. `0.15`) if no `split` column |
| `--num-boost-round` | `1000` | Maximum boosting rounds |
| `--early-stopping` | `50` | Early stopping patience |
| `--time-budget` | `600` | Optuna tuning budget in seconds |
| `--trials` | `10` | Maximum Optuna trials |
| `--robustness` | `false` | Run robustness evaluation |
| `--no-onnx` | `false` | Skip ONNX export |
| `--fp` | `morgan` | Fingerprint type for SMILES input |
| `--fp-size` | `2048` | Fingerprint bit length |
| `--hard-labels` | *none* | CSV/Parquet with hard labels (SMILES + target) |
| `--hard-smiles-col` | `smiles` | SMILES column in hard-labels file |
| `--hard-y-col` | `y` | Target column in hard-labels file |
| `--hard-weight` | `1.0` | Sample weight for hard-label rows |

### `olinda predict`

Run inference with a trained model.

```bash
olinda predict \
  --model-dir <model_dir> \
  --input <file> \
  --out predictions.csv
```

Accepts SMILES (auto-detected), feature matrix CSV/Parquet, or NumPy `.npy` files. When predicting from SMILES, fingerprint parameters are inferred from the model or can be overridden via `--fp`, `--fp-size`, and `--radius`.

### `olinda pack`

Convert a teacher Parquet into a sharded distillation dataset.

```bash
olinda pack \
  --teacher-parquet teacher_output.parquet \
  --out packed_dataset \
  --x-dim 768
```

### `olinda distill`

End-to-end pipeline: pack a teacher Parquet, then train, evaluate, and export.

```bash
olinda distill \
  --teacher-parquet teacher_output.parquet \
  --out runs/distilled \
  --time-budget 600
```

---

## Hyperparameter Tuning

When `--time-budget > 0`, Olinda runs a two-phase Optuna search:

| Phase | Data | Purpose |
|-------|------|---------|
| **1. Explore** | 30% subsample | Search the parameter space with median pruning. DMatrix is built **once** and reused across all trials. |
| **2. Validate** | 100% data | Re-evaluate the top 5 configurations on the full dataset using a single cached DMatrix. |

**Tuned parameters:** `max_depth`, `eta`, `gamma`, `subsample`, `colsample_bytree`, `lambda`, `alpha`, `min_child_weight`

The tuning pipeline caches `QuantileDMatrix` objects across trials, avoiding the expensive per-trial data reconstruction that dominates wall time on large datasets. A time budget is enforced across both phases.

---

## Validation

When a validation split exists, Olinda generates a full evaluation suite under `<out>/validation/`:

### Metrics

| Metric | Description |
|--------|-------------|
| MAE, RMSE | Error magnitude |
| R², Pearson, Spearman | Correlation |
| Concordance (Harrell's C) | Ranking accuracy |
| Coverage (1/2 std) | Calibration coverage |

All metrics include bootstrap 95% confidence intervals (200 replicates).

### Post-hoc Calibration

After computing raw metrics, Olinda automatically fits an **isotonic regression** calibrator (Pool Adjacent Violators) from raw predictions to teacher soft labels on the validation set. This produces:

- A `calibrator.json` saved alongside the model, applied transparently at prediction time
- Before/after metrics showing calibration improvement
- Calibrated predictions that stay within the teacher's output range and preserve rank ordering

### Plots

| File | Description |
|------|-------------|
| `pred_vs_true.png` | Predicted vs. true density scatter (viridis colormap) |
| `pred_vs_true_calibrated.png` | Same, after isotonic calibration |
| `calibration_comparison.png` | Side-by-side before/after calibration with distribution overlay |
| `residual_hist.png` | Residual distribution |
| `residual_vs_pred.png` | Residuals vs. predicted values |
| `calibration_bins.png` | Binned calibration curve (raw) |
| `calibration_bins_calibrated.png` | Binned calibration curve (calibrated) |
| `residuals_qq.png` | QQ plot of standardized residuals |
| `train_loss.png` | Training and validation loss curves |

---

## Robustness Evaluation

Enabled with `--robustness` when validation data and a SMILES column are present. Produces `robustness_report.json` with:

- **Scaffold / similarity split evaluation** &mdash; MAE, RMSE, and Spearman on structurally challenging splits
- **Drop vs. random split** &mdash; performance degradation relative to random baseline
- **SMILES perturbation invariance** &mdash; prediction stability across randomized SMILES representations
- **OOD awareness** &mdash; error analysis binned by nearest-neighbor Tanimoto similarity
- **Ensemble uncertainty** &mdash; prediction intervals from a multi-seed ensemble

---

## Outputs

A training run produces the following artifacts:

```
<out>/
  xgb.json                    # Trained XGBoost model
  calibrator.json             # Isotonic calibrator (applied at predict time)
  train_meta.json             # Training configuration and metrics
  student.onnx                # ONNX export (unless --no-onnx)
  train_loss.png              # Loss curve
  validation/
    validation_report.json    # Raw + calibrated metrics, bootstrap CIs
    pred_vs_true.png
    pred_vs_true_calibrated.png
    calibration_comparison.png
    residual_hist.png
    residual_vs_pred.png
    calibration_bins.png
    calibration_bins_calibrated.png
    residuals_qq.png
  robustness_report.json      # When --robustness is enabled
```

---

## Testing

```bash
pip install -e ".[dev,train,viz]"
pytest tests/ -v
```

---

## License

MIT
