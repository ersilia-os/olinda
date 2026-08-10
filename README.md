<div align="center">

# Model distillation for chemistry

**Olinda** trains compact gradient-boosting student models (XGBoost / LightGBM) from a large reference
library's descriptors or SMILES — with fast Optuna tuning, ground-truth models, and ONNX export.

<br/>

[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![XGBoost](https://img.shields.io/badge/XGBoost-%3E%3D3.2-017CEE?style=flat-square)](#)
[![License](https://img.shields.io/badge/License-MIT-2ea44f?style=flat-square)](#license)

<br/>

[Installation](#installation) &middot;
[Quick Start](#quick-start) &middot;
[CLI Reference](#cli-reference) &middot;
[License](#license)

</div>

---

## Overview

Olinda fits a gradient-boosting **student** from a reference library's precomputed descriptors, streaming
the data through XGBoost or LightGBM, and exports the model as both a native booster and ONNX. The student
carries its featurizer, so it predicts directly from SMILES.

**Key capabilities:**

- **Dual gradient-boosting engine** &mdash; XGBoost on GPU, LightGBM on CPU, auto-selected by hardware; hyperparameters are backend-agnostic and portable.
- **Streaming HDF5 pipeline** &mdash; `QuantileDMatrix` fed from disk for reference libraries that exceed RAM.
- **Squared-error loss, ONNX-safe** &mdash; with automatic inverse-density reweighting for skewed / rare-tail targets.
- **Fast, time-capped tuning** &mdash; a short, pruned Optuna study (`olinda tune`) over the highest-impact knobs.
- **Ground-truth models** &mdash; train a separate hard-label model plus a learned, index-free applicability weight (`olinda learn-hard`).
- **One portable artifact** &mdash; a single self-describing `model.onnx` carrying the featurizer, the calibration, and its provenance, alongside the native booster (`xgb.json` / `model.lgb`).

---

## Installation

**To run a distilled model**, the base install is all you need &mdash; four dependencies, no gradient-boosting stack:

```bash
pip install olinda          # numpy · pandas · rdkit · onnxruntime
```

**To distil a model**, add the training extra:

```bash
pip install "olinda[train]"     # + XGBoost, LightGBM, lazy-qsar, HDF5, Optuna, the CLI
```

| Extra | Purpose |
|-------|---------|
| `train` | Everything needed to distil a model, and the `olinda` CLI |
| `viz` | Validation plots (Matplotlib, stylia, scipy) |
| `dev` | Linting and testing (Ruff, pytest) |

The `olinda` command belongs to `[train]`; on a base install it exits with a message pointing at the extra rather than a traceback. GPU boosting activates automatically when a CUDA-enabled XGBoost build and a compatible GPU are detected; otherwise training runs on LightGBM (CPU).

---

## Running a distilled model

A trained olinda model is **one self-describing `model.onnx`**. The featurizer configuration, the RDKit build it was fused against, the task names and the provenance all travel inside the file, so the `.onnx` is the only input you need &mdash; no run directory, no config, no reference library:

```python
from olinda import OnnxArtifact

model = OnnxArtifact("model.onnx")
df = model.run(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])
```

`run()` returns a DataFrame with a `smiles` column plus **one column per task**, named after the task. A single-task model is simply the one-column case &mdash; there is no separate mode.

The artifact describes itself:

```python
model.columns  # task names, in output order
model.trained_at  # '2026-08-11T09:14:00+00:00'
model.olinda_version  # the version that produced it
model.rdkit_version  # the RDKit build the featurizer requires
model.has_ground_truth  # True if predictions blend in measured data
model.describe()  # all of the above, as a dict
```

The installed RDKit is checked against the recorded build on load and **refused on mismatch** &mdash; Morgan fingerprints only reproduce bit-for-bit on the exact version, so a silent mismatch would corrupt every prediction.

Predictions already fold in the applicability weighting; `prediction` is the final number. To inspect the pieces behind it &mdash; the raw surrogate, the calibrated ground truth, and the applicability weight &mdash; use `model.run_channels(smiles)`.

Large inputs are batched internally; pass `batch_size=` to change the default.

The file also sets the standard ONNX provenance fields (`producer_name`, `producer_version`, `doc_string`), so tools like Netron identify it as an olinda artifact without knowing anything about olinda.

---

## Data

Olinda's data lives on public S3 — no AWS credentials or extra tooling required. `olinda setup` downloads the reference-library Morgan count fingerprints (`erl0_morgan.h5`, ~2.8 GB), the one representation olinda uses:

```bash
olinda setup                          # → ~/.olinda/erl0_morgan.h5
olinda setup --target-dir /path/dir   # custom directory
```

Anything already present is skipped. The download is **best-effort** — if it isn't on S3 yet, setup warns (not errors) and you can generate it locally with `scripts/compute_morgan_fingerprints.py`. Source: `https://eosvc-public.s3.eu-central-1.amazonaws.com/olinda/data/`.

---

## Quick Start

Fetch the reference-library data once (`olinda setup`), then either run the whole pipeline in one shot:

```bash
olinda fit     -s teacher.csv -m runs/my_model                   # prepare → learn-soft → one model.onnx
olinda fit     -s teacher.csv -h gt.csv -m runs/my_model         # …→ learn-hard too (adds a hard head)
olinda predict -m runs/my_model -i compounds.csv -o predictions.csv   # SMILES → model.onnx
```

…or run the fit-pipeline steps individually (all sharing one `--model-dir`/`-m`):

```bash
olinda prepare -s teacher.csv -m runs/my_model                   # → runs/my_model/{train.h5, val.h5, soft.h5}
olinda tune       -m runs/my_model                               # optional: discover good hyperparameters
olinda learn-soft -m runs/my_model                               # fast GBM surrogate → model.onnx
olinda predict    -m runs/my_model -i compounds.csv -o predictions.csv
```

`olinda fit` chains `prepare → (tune, with --tune) → learn-soft → (learn-hard, if --hard-labels)`, producing a
single self-describing `model.onnx`. `teacher.csv` has two columns — `smiles` and the teacher value (any name)
— with **exactly the reference-library molecules, in the same order** (olinda verifies this).

### Ground truth (hard labels)

When you also have real (experimental) **binary** labels, name them up front in `prepare` via `--hard-labels`,
then learn from them and calibrate onto the soft-label scale:

```bash
olinda prepare -s teacher.csv -h gt.csv -m runs/my_model   # → {train.h5, val.h5, soft.h5, hard.h5}
olinda learn-hard -m runs/my_model                         # needs erl0_morgan.h5 (from `olinda setup`)
```

`learn-hard` runs four clear steps into `runs/my_model/_ground_truth/`:

1. **Train `G`** — a plain, portfolio-selected XGBoost classifier ([lazy-qsar](https://github.com/ersilia-os/lazy-qsar)'s
   `BaseXGBClassifier(calibrated=False)`, on olinda's Morgan count fingerprints) whose output is raw
   `predict_proba` — lazy-qsar's internal probability calibrator is off. *(Continuous hard labels are a
   not-yet-implemented placeholder.)*
2. **Score `G` across the full reference library** (`erl0_morgan.h5`) → a hard score per reference compound
   (`g_reference.h5`).
3. **Calibrate `G` onto the soft-label scale** — a monotonic (isotonic) map fit on the reference library,
   where both `G`'s score and the teacher's soft label (`soft.h5`, saved by `prepare`) exist. The direction
   is **learned from the data** (a low hard score may map to a *high* soft label). Saved as `g_to_soft.json`.
4. **Learn the applicability gate** — bucket every reference compound by its nearest-neighbour Tanimoto
   similarity to the labeled set (**NOT SIMILAR / LOW / HIGH**) and fit **two Bernoulli Naive-Bayes
   classifiers** on Morgan features (`applicability_nb.json`). At predict time these place a query in a
   bucket with **no similarity search** (the labeled compounds are folded in as HIGH positives).

Three diagnostic plots land in `_ground_truth/plots/`: the **calibration map** (`G` vs soft with the fitted
isotonic curve), the **score distributions** (`G` and soft over the reference), and **calibrated-vs-soft**.

`predict` then emits a headline **`prediction`** — the applicability-weighted blend
`(1−a)·surrogate + a·ground_truth_soft`, where the gate sets `a` (NOT→0, LOW→0.33, HIGH→0.66) — plus the raw
channels `surrogate`, `ground_truth_soft`, `ground_truth`, and `applicability`. The blend defaults to the
surrogate and leans on the hard signal only for queries close to the labeled chemistry.

### One fused `model.onnx`

`learn-soft` and `learn-hard` fuse the whole pipeline into a **single, self-describing `model.onnx`** at the
model-dir root (rebuild any time with `olinda export -m runs/my_model`):

- **soft-only** (no hard labels): `fp → prediction` (= the surrogate).
- **with hard labels**: one graph fusing `soft_model → [soft_correction]`, `hard_model → G score →
  hard_correction`, `applicability`, and the blend — outputs `prediction` plus `surrogate`,
  `ground_truth`, `ground_truth_soft`, `applicability`.

The Morgan featurizer config **and provenance travel inside the file's metadata** (`metadata_props`): the
featurizer dict + **RDKit version**, the reference-library id, `has_hard`, and (when hard) the training-set
size + task. So the ONNX is self-describing and runs entirely on **onnxruntime** — only featurization stays
in Python (RDKit has no ONNX op): the graph consumes a 2048-count Morgan fingerprint, not SMILES. The build
is gated on a numeric parity check against the Python pipeline.

`olinda predict -m <dir> -i <in.csv> -o <out.csv>` runs this `model.onnx`: it **verifies the installed RDKit
matches the version recorded in the model's metadata** (fingerprints only reproduce on the exact build, so a
mismatch is refused), reads the `smiles` column, featurizes, and writes `prediction` + the channels.

---

## `--soft-labels` format

A CSV/TSV/Parquet with two columns — column 0 is `smiles`, column 1 is the teacher value (any name):

```
smiles,value
CCCCCC=CCC=CCCCCCCCC(=O)O,0.83
NC(Cc1cnc[nH]1)C(=O)...,0.12
...
```

It must contain **exactly the reference-library molecules, in the same order** as the library (olinda verifies the SMILES row-by-row against `erl0_morgan.h5` and errors on any mismatch). Rows whose value is non-finite are dropped from training.

---

## Reference-library workflow (H5)

The pipeline streams the reference library through the gradient-boosting engine:

**Representation.** olinda uses one descriptor set everywhere: 2048-d Morgan **count** fingerprints (`erl0_morgan.h5`, from `olinda setup` or `scripts/compute_morgan_fingerprints.py`), reproducing Ersilia model `eos5axz`. The saved model bundles the Morgan featurizer, so it predicts straight from SMILES.

- **`prepare`** sorts by value, takes equally-spaced-by-rank validation (stratified), shuffles, and writes `train.h5`/`val.h5` (datasets `x` float32 `(m, 2048)`, `y` float32) into the model dir. With `--hard-labels`, it also writes `hard.h5` for `learn-hard`.
- **Auto-selected engine (XGBoost on GPU, LightGBM on CPU).** `learn-soft` picks the backend from the hardware: a CUDA **GPU → XGBoost** (mature GPU path, and the only one that accelerates the ~96%-zero sparse Morgan features), **CPU → LightGBM** (faster / lower-memory on large CPU workloads; its GPU can't use sparse features, so it runs CPU-only here). Override with `OLINDA_BACKEND=xgboost|lightgbm|auto`. Hyperparameters are canonical/backend-agnostic, so tuning and `best_params.json` are portable across engines. Both save a `StudentModel` (native `xgb.json` or `model.lgb`) **plus the fused `model.onnx`**, plus `val_metrics.json` (MAE/RMSE/R²/Pearson/Spearman/top-decile) and a true-vs-pred scatter `val_true_pred.png`. The default config is tuned for ~1.3M × 2048 Morgan counts (see the defaults table below).
- **Squared-error loss (ONNX-safe).** `learn-soft` uses squared error (`reg:squarederror` / L2) — well-conditioned and the only GBM objective that round-trips to ONNX correctly. Target skew/imbalance is handled by reweighting (below), not by the loss.
- **Tuning is optional and out-of-band (`olinda tune`).** To *discover* a better hyperparameter set (especially the learning rate) for your data, run `olinda tune -m <dir>` **before** `learn-soft`, on the **same auto-selected engine** `learn-soft` uses. `tune` writes `<dir>/best_params.json` (canonical, backend-tagged); a subsequent `learn-soft -m <dir>` **auto-reads it** (else uses built-in defaults). It's a **fast** Optuna study bounded by `--trials`, tuning on a `--max-rows` random subsample (train+val kept in the split's proportion; the console prints exactly how many rows are used), warm-started from known-good configs, searching the highest-impact knobs — `learning_rate` (0.05–0.3) and `min_split_gain` (0–5); `max_depth` is fixed at 8 — with Hyperband pruning; everything else stays at good defaults. It finds a good hyperparameter *region* on the subsample; `best_params.json` carries **only** `learning_rate` + `min_split_gain`, so `learn-soft` applies the full-scale fixed defaults and re-fits the round count on the full data. Requires the `[train]` extra (Optuna).

**What `tune` optimizes vs fixes** (printed every run). Only two knobs are searched; everything else is fixed at defaults vetted for large-scale sparse Morgan-count QSAR — the study can't wander far. The one size-dependent default (`min_child_weight`, ≈ min samples/leaf) uses a lighter floor while tuning on the ~100k subsample and the full value for the 1.3M fit (LightGBM: `min_data_in_leaf` scales up with N):

| knob | value | why |
|---|---|---|
| `learning_rate` | **searched** 0.05–0.3 | ~0.1 benchmarked best; shrinkage per round |
| `min_split_gain` | **searched** 0–5 | min loss reduction to split (regularization) |
| `max_depth` | 8 | deep for large N; LightGBM `num_leaves` saturates at 255 (=2⁸−1) |
| `min_child_weight` | 50 tune / **200** full | reliable leaf occupancy; hundreds for ≥1M rows |
| `subsample` | 0.8 | row bagging: mild regularization + speed |
| `colsample` | 0.5 | 1024/2048 features/tree; least-impactful knob (benchmark) |
| `max_bin` | 64 | histogram bins; speed/memory at large N |
| `reg_lambda` / `reg_alpha` | 1 / 0 | L2 on (XGBoost default), L1 off |
- **Automatic target reweighting.** olinda inspects the target and weights it **only when it is imbalanced** — skewed / bimodal / heavy-tailed targets get inverse-density weights, balanced targets are left alone (no flag: it's automatic). When it weights, it auto-picks the strategy: smooth **KDE inverse-density** for continuous targets, or **inverse-density bins** for discrete/tiny targets. Reweighting *reallocates* the model's attention from the dense bulk toward sparse value-regions — it typically **lowers the global R²** while **improving tail metrics**. When active, weights apply to **train and val together** so early stopping optimises the same objective. Both global and tail metrics (top-decile RMSE, Spearman) are always reported.

---

## CLI Reference

Main commands are **setup**, **fit**, and **predict**; the fit pipeline is **prepare**, **tune**, **learn-soft**, **learn-hard**, and **export**. Every fit-pipeline step shares one `--model-dir`/`-m`.

| Command | Description |
|---------|-------------|
| `setup` | Download the reference-library Morgan fingerprints (`erl0_morgan.h5`) from public S3 to `~/.olinda/` |
| `fit` | End-to-end: chains `prepare → (tune) → learn-soft → (learn-hard)` into one `model.onnx` |
| `predict` | Run a model's `model.onnx` on a `smiles` column (5 channels if a hard-label model is present) |
| `prepare` | Featurize inputs (Morgan counts) into the run dir: `-s/--soft-labels` → `train.h5`/`val.h5`/`soft.h5`; optional `-h/--hard-labels` → `hard.h5` |
| `tune` | Short, pruned Optuna study on a subset to discover good hyperparameters; writes `best_params.json` that `learn-soft` auto-reads |
| `learn-soft` | Learn the surrogate: fast gradient-boosting regression from `train.h5`/`val.h5` — engine auto-selected (XGBoost on GPU, LightGBM on CPU) |
| `learn-hard` | Learn a hard-label model + a learned applicability weight from `hard.h5`, and re-fuse the bundle |
| `export` | (Re)build the fused `model.onnx` for an already-trained model dir |

`olinda predict` reads a `.csv`/`.tsv`/`.parquet` file with a `smiles` column. The featurizer is reconstructed from the model's own metadata, and the installed RDKit is checked against the version recorded there.

---

## Outputs

`olinda prepare` writes `train.h5`/`val.h5`/`soft.h5` (and `hard.h5` with `--hard-labels`) into the run dir; `olinda learn-soft` then adds:

```
<model-dir>/
  train.h5, val.h5        # prepared soft split
  soft.h5                 # teacher labels aligned to the reference library
  hard.h5                 # prepared hard labels (only with prepare --hard-labels)
  model.onnx              # the served artifact: fused, self-describing
  xgb.json | model.lgb    # native booster (engine-dependent)
  train_meta.json         # backend + featurizer + hyperparameter metadata
  val_metrics.json        # MAE / RMSE / R² / Pearson / Spearman / top-decile RMSE
  val_true_pred.png       # validation true-vs-pred scatter
  best_params.json        # present only if `olinda tune` was run first
```

`olinda learn-hard` adds a `_ground_truth/` subdirectory and re-fuses `model.onnx` with the hard head:

```
<model-dir>/_ground_truth/
  gt/                     # the hard-label model (lazy-qsar)
  g_reference.h5          # G scored across the whole reference library
  g_to_soft.json          # isotonic map from G onto the soft-label scale
  applicability_nb.json   # the two Bernoulli Naive-Bayes gate classifiers
  ground_truth_meta.json  # task, featurizer, and artifact provenance
  gt_eval.json            # portfolio selection, calibration, and gate diagnostics
  plots/                  # calibration map, score distributions, calibrated-vs-soft
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
