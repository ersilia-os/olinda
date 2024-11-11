# olinda_api

This is a wrapper library for running onnx models produced in the Olinda pipeline with a Morgan Fingerprint featurizer.

### Example usage

```
import onnx_runner
model = onnx_runner.onnx_runner("path/to/model.onnx")
model.predict(["CCC", "CCO"])
```
