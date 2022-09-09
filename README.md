# Olinda

A model distillation library

model : Pytorch, Tensorflow, ONNX, ErsiliaModel ID
prepare training dataset: WebDataset
    Caching results to protect from progress loss
        - A consistenst workspace directory - XDG data home by default
        - Save the dataloader object using joblib
        - separate folders for models+distill params combinations
          - These are always generated locally
        - separate folders for reference+featurizer combinations
          - These can either be generated locally or can be downloaded
    input_type: Can be multiple types(this is an ENUM)
    featurizing: smiles -> griddify(input, input_type) -> images
        Can be multiple types but output is always same dimensions
Distillation: model, dataset, config
    Model Selection:
        Model architecture
        Model Hyperparameters
            - architecture
            - featurizer
Model_I/O:
    ONNX
    TFLITE
    Upload
Model Wrapper
    to normalize models for inputs
Model Deployment:
    AWS lambda server
    Maybe Other Public Cloud

Link to chemxor
