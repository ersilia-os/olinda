"""The base install must be enough to run a distilled model.

``pip install olinda`` promises numpy, pandas, rdkit and onnxruntime — nothing else. One module-scope
``import h5py`` anywhere in ``olinda/__init__ → artifact → featurizer`` would break that, and the rest
of the suite would never notice, because the dev environment has the training stack installed.

So this file must stay importable and passable *without* the training extras: no h5py, no xgboost,
no click at module scope, and every check that inspects ``sys.modules`` runs in a subprocess — under
pytest the other test modules have already imported the world by collection time.
"""

from __future__ import annotations

import subprocess
import sys

# What a base install does not have. Reaching for any of these on the inference path is the bug.
_TRAINING_ONLY = (
  "lightgbm",
  "xgboost",
  "h5py",
  "lazyqsar",
  "onnx",
  "onnxmltools",
  "optuna",
  "click",
  "rich_click",
  "rich",
  "tqdm",
  "loguru",
  "stylia",
  "matplotlib",
)


def _python(code: str) -> str:
  out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
  assert out.returncode == 0, out.stderr
  return out.stdout.strip()


def test_importing_the_public_api_pulls_in_nothing_heavy():
  code = (
    "import sys;"
    "from olinda import OlindaArtifact, RDKitVersionMismatch;"
    f"bad=[m for m in {_TRAINING_ONLY!r} if m in sys.modules];"
    "print(','.join(bad))"
  )
  assert _python(code) == "", "importing olinda reached for a training-only module"


def test_featurizing_needs_only_rdkit_and_numpy():
  """Featurization is the one heavy thing inference does in Python — it must stay on the light path."""
  code = (
    "import sys;"
    "from olinda.featurizer import MorganCountFeaturizer;"
    "x = MorganCountFeaturizer().transform(['CCO', 'c1ccccc1']);"
    f"bad=[m for m in {_TRAINING_ONLY!r} if m in sys.modules];"
    "print(f'{x.shape[0]}x{x.shape[1]}|' + ','.join(bad))"
  )
  assert _python(code) == "2x2048|"


def test_the_cli_entry_point_explains_itself_when_training_deps_are_missing():
  """On a base install `olinda ...` must say what to install, not dump an ImportError traceback.

  Poisoning ``sys.modules['olinda.cli']`` makes the import raise exactly as a missing rich-click
  would, without needing an environment that actually lacks it.
  """
  out = subprocess.run(
    [
      sys.executable,
      "-c",
      "import sys; sys.modules['olinda.cli'] = None; from olinda._entry import main; main()",
    ],
    capture_output=True,
    text=True,
  )
  assert out.returncode != 0
  assert "pip install 'olinda[train]'" in out.stderr
