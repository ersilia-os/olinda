# Olinda, model distillation for chemistry

Some ML-for-chemistry models are too slow to run on a million compounds. Olinda makes a fast one that behaves like
them: score the slow model once over a reference library of ~1.4M molecules, train a compact
gradient-boosting student to reproduce it, and ship the result as a single ONNX file.

## Installation

Olinda is pip-installable:

```bash
pip install olinda                 # run models from Python: numpy, onnxruntime, wasmtime
pip install "olinda[cli]"          # + the `olinda` command: click, rich-click, rich, loguru, tqdm
pip install "olinda[report]"       # + `olinda validate` and its figures: pandas, pyarrow, matplotlib, onnx, stylia
pip install "olinda[train]"        # + distillation: XGBoost, LightGBM, lazy-qsar, Optuna, onnxmltools, h5py
```

Each extra includes the one above it, so `olinda[train]` installs everything. No install needs RDKit:
Olinda featurizes molecules with RDKit compiled to WebAssembly, which ships inside the package.

Check that the command-line tool is installed (it needs `olinda[cli]` or above):

```bash
olinda --help
```

Before distilling, download the pre-calculated fingerprints of the reference library (1.4M compounds; 2.8GB):

```bash
olinda setup
```

## Running a model at inference time

A distilled model is one self-describing ONNX file. You can load and run the model in Python:

```python
from olinda import OlindaArtifact

model = OlindaArtifact("model.onnx")
model.run(["CCO", "c1ccccc1"])  # one row per molecule, one column per task (model.columns)
```

Alternatively, you can use the CLI:

```bash
olinda predict --model-onnx model.onnx --input my_smiles.csv --output my_output.csv
```

Or, with nothing installed, in the browser at [ersilia-os.github.io/olinda](https://ersilia-os.github.io/olinda/):
upload the `model.onnx` (or try the built-in example model), paste SMILES, and get the predictions as a table. It all runs locally in the browser;
the model and the molecules never leave your machine.

## Distilling a model

### Getting your soft labels

Olinda relies on a reference library that is [maintained](https://github.com/ersilia-os/ersilia-model-hub-maintained-inputs) by the Ersilia team.

Use the following command to obtain a 1-column file with the SMILES structures of the reference library:

```bash
olinda library -o ersilia_reference_library.csv
```

Use this file to make predictions with your model of choice. Feel free to explore models from the [Ersilia Model Hub](https://catalog.ersilia.io), or models trained with [ZairaChem](https://github.com/ersilia-os/zaira-chem). We refer to the result of this predictions as **soft labels**.

### Training a distilled model

Assuming you have your soft labels calculated, you can simply get a distilled (surrogate model) as follows:

```bash
olinda fit --soft-labels my_soft_labels.csv --model-onnx my_model.onnx
```

Optionally, you can provide **hard labels** (ground truth) if you have them. For now, only binary (1/0) labels are allowed as hard labels.

```bash
olinda fit --soft-labels my_soft_labels.csv --hard-labels my_hard_labels.csv --model-onnx my_model.onnx
```

### Evaluating a distilled model

The easiest way to evaluate a distilled model is to pass an additional set of soft-labels (and optionally hard labels):

```bash
olinda validate --soft-labels my_validation_soft_labels.csv --model-onnx my_model.onnx --output-dir my_report_folder
```

## Commands

The `olinda` command comes with `olinda[cli]`; `validate` also needs `olinda[report]`, and the distilling
commands need `olinda[train]`.

| Command | What it does |
|---|---|
| `setup` | Download the reference library's Morgan fingerprints |
| `library` | Write the reference library's SMILES to a CSV |
| `fit` | Distil a teacher into a student end-to-end |
| `predict` | Run a distilled `model.onnx` on SMILES |
| `validate` | Score a model against labelled data and write a report |

`fit` drives the step-by-step pipeline below; run the steps yourself to inspect or redo one of them.

| Command | What it does |
|---|---|
| `prepare` | Set up a run directory from the soft (and hard) labels |
| `tune` | Search hyperparameters for `learn-soft` with Optuna |
| `learn-soft` | Train the surrogate on the teacher's soft labels |
| `learn-hard` | Train on hard labels and calibrate onto the soft-label scale |
| `export` | Rebuild `model.onnx` for a trained run |
| `clean` | Move the model out of the run and delete the working folder |

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Plum.png)
