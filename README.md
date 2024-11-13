# Olinda

Olinda is a generic cheminformatics model distillation library.
It can automatically distill models from Pytorch, Tensorflow, ONNX amd ZairaChem formats.

## Getting Started

### Pytorch, Tensorflow, ONNX models
Create a conda environment

```bash
conda create -n olinda python=3.10
conda activate olinda
```

Clone the Olinda repository and install it

```bash
git clone https://github.com/ersilia-os/olinda.git
cd olinda
python -m pip install -e .
```

### ZairaChem models
To distill ZairaChem models, install the [ZairaChem](https://github.com/JHlozek/zaira-chem.git) pipeline which installs Olinda into the same environment


## Usage
Within the conda environment, models can be distilled quickly with a single function:

```
olinda distill -m path_to_model -o path_to_save.onnx
```

Alternatively, you can run the distillation in Python code:

```python
from olinda import Distiller

d = Distiller()
student_model = d.distill("path_to_model")
student_model.save("path_to_save.onnx")
```

Use the 'num_data' parameter (or -t flag in the cli) to specify a smaller training dataset to test the pipeline.

```python
student_model = distill("path_to_model", num_data=1000)
```

## Run ONNX inference with Python api
Model inference can be run from the cli with the predict command by specifying an input csv file with a 'smiles' column.

```
olinda predict -i input_file.csv -m path_to_model -o output_file.csv
```

A lite version for running model predictions can be found by installing the onnx_runner package.
```bash
cd onnx_runner
python -m pip install -e .
```

Then run predictions in Python with:
```
import onnx_runner
model = onnx_runner.ONNX_Runner("path/to/model.onnx")
model.predict(["CCC", "CCO"])
```


### How the distillation works?

The distillation function first downloads a reference SMILES dataset if it is not already present. It then generates featurized inputs using the reference SMILES dataset for training the student model. Next it uses the provided teacher model to generate input-output pairs. The input-output pairs together with the featurized inputs constitute the training dataset for the student model. Finally a suitable architecture for the student model is selected using heuristics and the selected model is trained using the training dataset.

```mermaid
  graph TD;
  A[Generate reference SMILES dataset] --> B[Generate a training dataset using the given teacher model] --> C[Join the original ZairaChem training data to the Olinda reference dataset] --> D[Search a suitable architecture for student model] --> E[Train student model]
```

During the distillation process, helpful messages and progress bars are printed to keep the user informed. In the case of a crash or process interruption the distillation process can be resumed automatically. It caches all the intermediate results in a local directory (`xdg_home() / olinda`).

The student model is trained on a library of 100k molecules from ChEMBL 29, where ZairaChem descriptors have been pre-calculated in folds of 50k molecules and stored in S3 buckets. At runtime, Olinda will download the required number of folds, if not already present on the system.

Distilled models are returned in ONNX format for cross-platform use.

### Pipeline parameters

The Olinda pipeline has been implemented with the following technical decisions below. These parameters are not exposed through an API but are detailed here for more advanced users.

- ZairaChem Prediction Changes: When distilling ZairaChem models, Olinda fetches the predictions for the original ZairaChem training set in addition to using predictions for a reference library of public compounds. The ZairaChem predictions are first checked for compounds that have been incorrectly predicted, which are then changed by 1 - pred_score for the incorrectly predicted compounds before using the score to train the Olinda model. The relevant code snippet can be found within ```olinda/distillation.py``` in ```gen_model_output()```:
```
# correct wrong zairachem predictions before training Olinda
training_output = model.get_training_preds()
for i, row in enumerate(training_output.iterrows()):
    if row[1]["pred"] >= 0.0 and row[1]["pred"] <= 0.5 and row[1]["true"] == 1.0:
        training_output.at[i, "pred"] = 1.0 - row[1]["pred"]
    elif row[1]["pred"] >= 0.5 and row[1]["pred"] <= 1.0 and row[1]["true"] == 0:
        training_output.at[i, "pred"] = 1.0 - row[1]["pred"]
```
-  KerasTuner 30 epochs: The Olinda surrogate model is trained with KerasTuner using a max_epochs of 30 and 2-4 additional hidden layers. These can be changed in ```olinda/tuner.py``` within the ```KerasTuner()``` class:
```
class KerasTuner(ModelTuner):
    """Keras tuner based model tuner."""

    def __init__(
        self: "KerasTuner", layers_range: List = [2, 4], max_epochs: int = 30
    ) -> None:
```
-  Reference library size: 100k compounds are used as the default number of literature reference compounds. This can be reduced by adjusting the ```num_data``` parameter of the ```Distiller()``` class in ```olinda/distillation.py```. The current maximum precalculated descriptors available in the S3 bucket is for this 100k compounds. Additional compounds will be precalculated in future.
-  
-  Class weighting scheme: Olinda assigns weights to compounds to address class imbalance based on the prediction scores. The pipeline counts the number of predicted actives/inactives, based on a prediction score threshhold of 0.5, and wieghts compounds according to the inverse proportion of the two classes. This can be disabled by assigning ```active_weight=1``` in ```olinda/distillation.py```: 
```
# inverse of ratio of predicted active to inactive 
y_bin_train = [1 if val > 0.5 else 0 for val in training_output["pred"]]
y_bin_ref = [1 if val > 0.5 else 0 for val in output["pred"]]
active_weight = (y_bin_train.count(0) + y_bin_ref.count(0)) / (y_bin_train.count(1) + y_bin_ref.count(1))
```


## License
This project is licensed under GNU AFFERO GENERAL PUBLIC LICENSE Version 3.

## About
The [Ersilia Open Source Initiative](https://ersilia.io) is a Non Profit Organization (1192266) with the mission is to equip labs, universities and clinics in LMIC with AI/ML tools for infectious disease research.

[Help us](https://ersilia.io/model-hub) achieve our mission!




### TODO

#### Poetry install fails on m1 macs

```bash
CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=OFF -DProtobuf_USE_STATIC_LIBS=ON" poetry install
```

- [ ] ErsiliaModel compatibility (Currently hard-coded for eos97yu. Ersilia models require an output adapter to standardise prediction formatting)
- [ ] Multi-output models
