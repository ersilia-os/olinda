from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import xgboost as xgb


def export_xgb_onnx(booster: xgb.Booster, out_path: str | Path, input_dim: int) -> None:
  """Convert a trained XGBoost booster to ONNX at ``out_path``.

  xgboost is intentionally not imported at module scope: ``olinda.models`` re-exports this function
  eagerly, and importing xgboost on every ``import olinda.models`` would slow CLI startup.
  """
  import onnxmltools
  from onnxmltools.convert.common.data_types import FloatTensorType

  out_path = Path(out_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  model = onnxmltools.convert_xgboost(
    booster,
    initial_types=[("input", FloatTensorType([None, int(input_dim)]))],
  )
  onnxmltools.utils.save_model(model, str(out_path))
