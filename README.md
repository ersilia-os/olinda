<div align="center">

# Olinda

Lightweight model distillation for chemistry workflows. Olinda trains compact **XGBoost students** from feature tables or SMILES-derived fingerprints with a clean CLI for fitting, evaluation, and prediction.

<br/>

[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![License](https://img.shields.io/badge/license-MIT-2ea44f?style=flat-square)](#license)

<br/>

[Install](#install) ·
[Data format](#data-format) ·
[CLI](#cli) ·
[Outputs](#outputs) ·
[Robustness](#robustness-evaluation) ·
[License](#license)

</div>

---

## Why Olinda?
Olinda focuses on fast, repeatable distillation runs for chemistry datasets. It fits a student model from either packed feature tables or SMILES-derived fingerprints, and only runs validation plus robustness checks when a validation split exists. This keeps training lightweight while still providing evaluation when it matters.

You can use Olinda to:

- ⚡ **Train quickly** with minimal overhead
- 🧪 **Evaluate reliably** when validation data exists
- 🧩 **Predict flexibly** from SMILES or feature matrices
- 📦 **Export clean artifacts** (XGBoost + ONNX)

---

## Install

```bash
pip install -e .
```

Optional dependencies:

```bash
pip install -e .[train]
pip install -e .[viz]
pip install -e .[dev]
```

---

## Data format

For CSV or Parquet input, provide:

- Feature columns: numeric columns
- `y`: target value
- `split` (optional): 0 for train, 1 for validation
- `w` (optional): sample weight

If `split` is missing and `--split` is provided, Olinda creates a validation split. If no validation split is present, validation and robustness are skipped.

---

## CLI

### Fit

Fit from CSV with an automatic validation split:

```bash
olinda fit \
  --input /path/to/data.csv \
  --out /path/to/run \
  --y-col y \
  --split 0.15
```

Fit from a packed dataset directory:

```bash
olinda fit \
  --input /path/to/packed_dir \
  --out /path/to/run
```

### Predict

Predict from SMILES CSV:

```bash
olinda predict \
  --model-dir /path/to/run \
  --input /path/to/smiles.csv \
  --smiles-col smiles \
  --out /path/to/preds.csv
```

Predict from feature matrix CSV/Parquet:

```bash
olinda predict \
  --model-dir /path/to/run \
  --input /path/to/features.parquet \
  --columns feat_000,feat_001,feat_002 \
  --out /path/to/preds.csv
```

---

## Outputs

- `xgb.json`: trained model
- `train_meta.json`: training metadata
- `student.onnx`: ONNX export unless `--no-onnx`
- `validation/`: validation report and plots when a validation split exists
- `robustness_report.json`: robustness report when a validation split exists and the input is a file

---

## Robustness evaluation

When validation data is present, Olinda computes:

- Scaffold or similarity split evaluation with MAE/RMSE/Spearman
- Drop vs random split
- SMILES perturbation invariance via randomized SMILES
- OOD awareness using nearest-neighbor Tanimoto similarity and ensemble uncertainty

---

## License

MIT