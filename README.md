# Olinda, model distillation for chemistry

Some ML-for-chemistry models are too slow to run on a million compounds. Olinda makes a fast one that behaves like
them: score the slow model once over a reference library of ~1.4M molecules, train a compact
gradient-boosting student to reproduce it, and ship the result as a single ONNX file.

## Installation

Olinda is pip-installable:

```bash
pip install olinda                 # use the base install if you only need to run Olinda at inference type
pip install "olinda[report]"       # add plotting functionalities in case you need a report on validation data
pip install "olinda[train]"        # install training capabilities (e.g. XGBoost)
```

Check that Olinda is correctly installed:

```bash
olinda --help
```

Before getting started, you need to download the pre-calculated Morgan fingerprints for a reference library (1.4M compounds; 2.8GB):

```bash
olinda setup
```

## Running a model at inference time

A distilled model is one self-describing ONNX file. You can load and run the model in Python:

```python
from olinda import OlindaArtifact

model = OlindaArtifact("model.onnx")
model.run(["CCO", "c1ccccc1"])
```

Alternatively, you can use the CLI:

```bash
olinda predict --model-onnx model.onnx --input my_smiles.csv --output my_output.csv
```

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
olinda report --soft-labels my_validation_soft_labels.csv --model-onnx my_model.onnx --output-dir my_report_folder
```

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery. You can support Ersilia by clicking [here](https://www.ersilia.io/donate).
