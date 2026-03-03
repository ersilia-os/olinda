from pathlib import Path

import xgboost as xgb


def export_xgb_onnx(booster: xgb.Booster, out_path: str | Path, input_dim: int) -> None:
  import onnxmltools
  from onnxmltools.convert.common.data_types import FloatTensorType

  out_path = Path(out_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  model = onnxmltools.convert_xgboost(
    booster,
    initial_types=[("input", FloatTensorType([None, int(input_dim)]))],
  )
  onnxmltools.utils.save_model(model, str(out_path))
