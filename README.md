# Olinda, model distillation for chemistry

Some models are too slow to run on a million compounds. Olinda makes a fast one that behaves like
them: score the slow model once over a reference library of ~1.4M molecules, train a compact
gradient-boosting student to reproduce it, and ship the result as a single ONNX file.

## Installation

```bash
pip install olinda                 # inference only: numpy, pandas, rdkit, onnxruntime
pip install "olinda[train]"        # + the gradient-boosting stack and the CLI
```

## Running a model at inference time

A distilled model is one self-describing file. The featurizer, the RDKit build it needs, the task
names — all of it travels inside the `.onnx`, so nothing else is required to run it.

```python
from olinda import OlindaArtifact

model = OlindaArtifact("model.onnx")
model.run(["CCO", "c1ccccc1"])  # DataFrame: smiles + one column per task
```

## Distilling a model (soft labels)

Fetch the reference library once (~2.8 GB), then fit:

```bash
olinda setup
olinda fit -s teacher.csv -m runs/my_model.onnx
olinda predict -m runs/my_model.onnx -i compounds.csv -o predictions.csv
```

`-m` names the artifact you want. `fit` builds the run in `runs/my_model/` beside it and deletes that
folder once everything has fused, so what you keep is one file.

`teacher.csv` holds a `smiles` column plus one or more value columns, covering **exactly the
reference-library molecules in the same order** — olinda verifies this and refuses otherwise.

Up to **10 teacher columns** in one file. Each becomes an independent student, and all of them fuse
into one `model.onnx` with one output per task. A single column is just the one-column case; there is
no separate mode.

### Ground truth (hard labels)

If you also have real measurements, pass them with `-h`:

```bash
olinda fit -s teacher.csv -h measured.csv -m runs/my_model.onnx
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

The three you normally need:

| | |
|---|---|
| `setup` | Download the reference-library fingerprints to `~/.olinda/` |
| `fit` | Distil a teacher into one `model.onnx` |
| `predict` | Run a model over a file of SMILES |

`fit` chains the steps below. Each is also a command in its own right — run them one at a time when you
want the per-column boosters, metrics and plots that `fit` discards:

| | |
|---|---|
| `prepare` | Read the teacher columns and plan each one's split |
| `tune` | Optional Optuna pass; single-column runs only |
| `learn-soft` | Train the surrogate for every column |
| `learn-hard` | Train and calibrate the ground-truth head |
| `export` | Rebuild `model.onnx` from a trained run |
| `clean` | Move the model out and delete the run folder |

`fit`, `predict` and `clean` take `-m` as a path to the **`.onnx`**; the pipeline steps between them
take `-m` as the **run folder** they share. The engine is picked from your hardware — XGBoost on a
CUDA GPU, LightGBM on CPU — and `OLINDA_BACKEND` overrides it.

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization with the mission to equip laboratories, universities, and clinics in the Global South with AI/ML tools for infectious disease research. We work on the principles of open science, decolonized research, and egalitarian access to knowledge and research outputs. You can support Ersilia by clicking [here](https://www.ersilia.io/donate).
