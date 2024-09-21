## Distillation customization

The distillation API is very flexible and covers a wide varietry of use cases. User can easily customize the distillation behavior by passing parameters to the `distill` function.

```python
def distill(
    model: Any,
    featurizer: Optional[Featurizer],
    working_dir: Path,
    clean: bool = False,
    tuner: ModelTuner = AutoKerasTuner(),
    reference_smiles_dm: Optional[ReferenceSmilesDM] = None,
    featurized_smiles_dm: Optional[FeaturizedSmilesDM] = None,
    generic_output_dm: Optional[GenericOutputDM] = Nonei,
    test: bool = False,
    num_data: int = 1999380,
) -> pl.LightningModule:
    """Distill models.

    Args:
        model (Any): Teacher Model.
        featurizer (Optional[Featurizer]): Featurizer to use.
        working_dir (Path): Path to model workspace directory.
        clean (bool): Clean workspace before starting.
        tuner (ModelTuner): Tuner to use for selecting and optimizing student model.
        reference_smiles_dm (Optional[ReferenceSmilesDM]): Reference SMILES datamodules.
        featurized_smiles_dm (Optional[FeaturizedSmilesDM]): Reference Featurized SMILES datamodules.
        generic_output_dm (Optional[GenericOutputDM]): Precalculated training dataset for student model.
        test (bool): Run a test distillation on a smaller fraction of the dataset.
        num_data: (int) : Set the number of ChEMBL training points to use (up to 1999380)

    Returns:
        pl.LightningModule: Student Model.
    """

```

### Custom SMILES reference dataset

This will skip generating a featurized input dataset and use the provided one.

```python
from olinda import distill
from olinda.data import ReferenceSmilesDM

# Wrap your dataset in a datamodule class from pytorch lightning
# or use the provided `ReferenceSmilesDM` class
custom_reference_dm = ReferenceSmilesDM()
student_model = distill(your_model, reference_smiles_dm=custom_reference_dm)
```

### Custom featurizer for student model inputs

```python
from olinda import distill
from olinda.featurizer import Featurizer, MorganFeaturizer

# Implement your own featurizer by inheriting the `Featurizer` abstract class
# or use one of the provided Featurizers (see below for more info)
student_model = distill(your_model, featurizer=MorganFeaturizer())
```

### Custom featurized input dataset

This will skip generating a featurized input dataset and use the provided one.

```python
from olinda import distill
from olinda.data import FeaturizedSmilesDM

# Wrap your dataset in a datamodule class from pytorch lightning
# or use the provided `FeaturizedSmilesDM` class
custom_reference_dm = FeaturizedSmilesDM()
student_model = distill(your_model, featurized_smiles_dm=custom_reference_dm)
```

### Custom student model training dataset

This will skip generating a student model training dataset and use the provided one.

```python
from olinda import distill
from olinda.data import GenericOutputDM

# Wrap your dataset in a datamodule class from pytorch lightning
# or use the provided `GenericOutputDM` class
custom_student_training_dm = GenericOutputDM()
student_model = distill(your_model, generic_output_dm=custom_student_training_dm)
```

### Custom Tuners for student model tuning

Olinda provides multiple Tuners out of the box. Custom tuners can also be implemented using the `ModelTuner` interface. See below for more information

```python
from olinda import distill
from olinda.tuner import KerasTuner


student_model = distill(your_model, tuner=KerasTuner())
```

## DataModules

## Featurizers
Currently we support only one Featurizer class, MorganFeaturizer. 
This featurizer converts SMILES strings into Morgan Fingerprints of 1024 bits and radius 3.

A related Featurizer is available but is no longer actively maintained for Olinda, which subsequently transforms each Morgan Fingerprint vector into a 32x32 grid using the Ersilia package [Griddify](https://github.com/ersilia-os/griddify).

## Tuners
We provide two Tuners for the student model training based on the fantastic Keras library:
* Autokeras
* KerasTuner