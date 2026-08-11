# Model distillation for chemistry

Some models are too slow to run on a million compounds. Olinda makes a fast one that behaves like
them: score the slow model once over a reference library of ~1.4M molecules, train a compact
gradient-boosting student to reproduce it, and ship the result as a single ONNX file.

The student predicts in ~40 µs per molecule and needs none of the teacher's dependencies.

## Running a model

A distilled model is one self-describing file. The featurizer, the RDKit build it needs, the task
names — all of it travels inside the `.onnx`, so nothing else is required to run it.

```python
from olinda import OlindaArtifact

model = OlindaArtifact("model.onnx")
model.run(["CCO", "c1ccccc1"])  # DataFrame: smiles + one column per task
```

The file describes itself — `model.columns`, `model.trained_at`, `model.rdkit_version`,
`model.describe()`, or `model.to_json()` for everything at once.

Two things worth knowing. Fingerprints only reproduce bit-for-bit on the RDKit build the model was
built against, so loading under a different one is **refused** rather than silently wrong. And
molecules RDKit cannot parse come back as `NaN` with a warning, never as a number.

```bash
pip install olinda                 # inference only: numpy, pandas, rdkit, onnxruntime
pip install "olinda[train]"        # + the gradient-boosting stack and the CLI
```

## Distilling a model

Fetch the reference library once (~2.8 GB), then fit:

```bash
olinda setup
olinda fit -s teacher.csv -m runs/my_model
olinda predict -m runs/my_model -i compounds.csv -o predictions.csv
```

`teacher.csv` holds a `smiles` column plus one or more value columns, covering **exactly the
reference-library molecules in the same order** — olinda verifies this and refuses otherwise.

Up to **10 teacher columns** in one file. Each becomes an independent student, and all of them fuse
into one `model.onnx` with one output per task. A single column is just the one-column case; there is
no separate mode.

### Ground truth

If you also have real measurements, pass them with `-h`:

```bash
olinda fit -s teacher.csv -h measured.csv -m runs/my_model
```

`measured.csv` is a `smiles` column plus one column per assay, left empty where a compound was not
tested. Columns are matched to the teacher by name, allowing a suffix — `abaumannii_inhibition`
matches `abaumannii_inhibition_probability`. Anything ambiguous is an error, not a guess.

Olinda then trains a classifier on your labels, calibrates it onto the teacher's scale (learning the
direction from the data), and learns where to trust it: compounds are bucketed by similarity to your
labelled set, and the final prediction is

```
(1 − a) · surrogate  +  a · calibrated ground truth
```

with `a` rising only near chemistry you have actually measured. Far from it, the model falls back to
the distilled surrogate. That weighting is already inside the number `run()` returns.

## Commands

| | |
|---|---|
| `setup` | Download the reference-library fingerprints to `~/.olinda/` |
| `fit` | Everything below, in order |
| `predict` | Run a model over a file of SMILES |
| `prepare` | Read the teacher columns and plan each one's split |
| `tune` | Optional Optuna pass; single-column runs only |
| `learn-soft` | Train the surrogate for every column |
| `learn-hard` | Train and calibrate the ground-truth head |
| `export` | Rebuild `model.onnx` from a trained run |
| `clean` | Delete the working files, keeping only `model.onnx` |

Every step shares one `--model-dir`. The engine is picked from your hardware — XGBoost on a CUDA GPU,
LightGBM on CPU — and `OLINDA_BACKEND` overrides it.

## What a run directory holds

`fit` ends with `clean`, so it leaves exactly one file:

```
runs/my_model/
  model.onnx         all columns, fused — the only file you need to ship
```

Everything else was scaffolding. Run the steps by hand and it stays:

```
runs/my_model/
  manifest.json      what the run is: columns, splits, hard-label matches
  targets.h5         one teacher vector per column
  splits.h5          per-column train/val row indices
  columns/c0/        that column's model, metrics, calibrator, plots
  model.onnx
```

That is the reason to take the long path — the per-column metrics and plots only exist there. When
you are done looking, `olinda clean -m runs/my_model` collapses it to the artifact. It is one-way:
`export` and `learn-hard` read the manifest, so neither works afterwards.

The descriptor matrix is never copied into a run. Splits are stored as row indices into the shared
library, which is the difference between ~100 MB and ~1 TB for a ten-column run.
