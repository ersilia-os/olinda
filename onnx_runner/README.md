# olinda_api

This is a lite wrapper library for running onnx models produced in the Olinda pipeline with a Morgan Fingerprint featurizer.

### Installation

```bash
python -m pip install -e .
```

### Example usage

```
import onnx_runner
model = onnx_runner.ONNX_Runner("path/to/model.onnx")
model.predict(["CCC", "CCO"])
```
