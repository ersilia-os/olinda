"""One tiny synthetic run, shared by every test that needs a finished model.

A real olinda run wants a 1.36M-compound reference library and minutes of boosting. Everything the
suite asserts about the pipeline — that columns stay independent, that the fuse is numerically
faithful, that `clean` does not change predictions — is just as true of a few hundred rows, so the
fixtures here build that instead.

:func:`build_tiny_model` is also called from CI (the ``report-install`` job in
``.github/workflows/test.yml``) to produce a model with the training stack that is then scored with
only the reporting extra installed. It therefore has to stay importable and runnable outside pytest —
hence plain functions here, and no ``import pytest`` at module scope.
"""

from __future__ import annotations

import os
from pathlib import Path

# Real molecules, repeated to fill the library. Enough *distinct* ones matters more than the row
# count: the teacher targets below are a function of the fingerprint, so a library of ten molecules
# has ten distinct target values however many rows it has, and a booster fitted on six of them learns
# nothing (measured: R² 0.00, one tree, and a degenerate isotonic fit downstream).
SMILES = [
  "CCO",
  "CCN",
  "CCC",
  "c1ccccc1",
  "CC(=O)O",
  "CCOC(=O)C",
  "Clc1ccccc1",
  "COc1ccccc1",
  "CCOCC",
  "OCc1ccccc1",
  "CCCCO",
  "CCCCCC",
  "c1ccc2ccccc2c1",
  "CC(C)Cc1ccccc1",
  "NCCc1ccc(O)cc1",
  "CC(=O)Nc1ccccc1",
  "OC(=O)c1ccccc1O",
  "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
  "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
  "CN1CCC[C@H]1c1cccnc1",
  "Oc1ccc(CCN)cc1O",
  "CSCC[C@H](N)C(=O)O",
  "c1ccc(-c2ccccc2)cc1",
  "ClCCl",
]

LIBRARY_ROWS = 240

# The teacher's column carries the `_probability` suffix an Ersilia model writes; the measurement file
# carries the bare endpoint a lab writes. That relationship is the real-world case and is what
# exercises the suffix matching in `match_hard_columns`.
SUFFIX = "_probability"


def write_library(home: Path, n_rows: int = LIBRARY_ROWS):
  """Write a synthetic ``erl0_morgan.h5`` into *home*; return ``(smiles, fingerprints)``.

  The layout is exactly what `olinda setup` downloads: a ``data`` matrix of uint8 Morgan counts and a
  row-aligned ``input`` dataset of SMILES, which is the file's source of truth for order.
  """
  import h5py
  import numpy as np

  from olinda.featurizer import MorganCountFeaturizer

  home.mkdir(parents=True, exist_ok=True)
  smiles = (SMILES * ((n_rows // len(SMILES)) + 1))[:n_rows]
  x = MorganCountFeaturizer().transform(smiles).astype(np.uint8)
  with h5py.File(home / "erl0_morgan.h5", "w") as f:
    f.create_dataset("data", data=x)
    f.create_dataset("input", data=np.array([s.encode() for s in smiles]))
  return smiles, x


def _target(x, seed: int):
  """A learnable teacher value in [0, 1] for every row of *x*.

  A fixed random projection of the whole fingerprint, not a slice of it: Morgan bits are sparse, so
  any given window is all zeros for small molecules and the "target" comes out constant. Projecting
  every bit guarantees the value varies across distinct molecules, and a different seed per column
  makes the columns genuinely different functions.
  """
  import numpy as np

  weights = np.random.default_rng(seed).normal(size=x.shape[1])
  raw = x.astype(np.float64) @ weights
  return (raw - raw.min()) / (np.ptp(raw) + 1e-9)


def write_soft(out_dir: Path, smiles, x, columns=("activity", "solubility")) -> Path:
  """Write a teacher file covering the whole library, one value column per name in *columns*."""
  import pandas as pd

  out_dir.mkdir(parents=True, exist_ok=True)
  frame = {"smiles": smiles}
  for i, name in enumerate(columns):
    frame[f"{name}{SUFFIX}"] = _target(x, i)
  path = out_dir / "soft.csv"
  pd.DataFrame(frame).to_csv(path, index=False)
  return path


def write_hard(out_dir: Path, smiles, x, column: str = "activity", rows=None) -> Path:
  """Write a sparse measurement file: the bare endpoint name, 0/1 labels, a subset of the library.

  *rows* bounds which library rows may be labelled — the ``--max-samples`` tests need the labelled
  compounds to sit inside the truncated view, or the gate has no neighbours to learn from.
  """
  import numpy as np
  import pandas as pd

  out_dir.mkdir(parents=True, exist_ok=True)
  n = len(smiles) if rows is None else int(rows)
  take = np.arange(0, n, 3)
  values = _target(x, 0)[take]
  path = out_dir / "hard.csv"
  pd.DataFrame({
    "smiles": [smiles[i] for i in take],
    column: (values > np.median(values)).astype(int),
  }).to_csv(path, index=False)
  return path


def run_cli(args):
  """Invoke the real CLI in-process and return the click result.

  Through ``CliRunner`` rather than a subprocess so a failure surfaces its traceback in the test
  output, and so monkeypatched paths (``OLINDA_HOME``) are still in effect.
  """
  from click.testing import CliRunner

  from olinda.cli import cli

  return CliRunner().invoke(cli, [str(a) for a in args])


def plain(result) -> str:
  """A command's output as one flat line: no ANSI, no box drawing, no wrapping.

  Rich sizes itself to the terminal, so the same message is one line in a wide shell and several on
  CI at 80 columns — and a message inside an error panel is wrapped *around* the border, so the raw
  text reads ``is left │ │ over from another run``. Stripping the frame makes these assertions about
  what was said, not how it happened to be laid out.
  """
  import re

  text = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", result.output)
  text = re.sub(r"[│┃╭╮╰╯─━┏┓┗┛]", " ", text)
  return re.sub(r"\s+", " ", text)


def build_tiny_model(dest, *, backend: str = "xgboost", with_hard: bool = True) -> Path:
  """Fit a complete model into *dest* and return the path to its ``model.onnx``.

  Leaves ``soft.csv`` and ``hard.csv`` beside it, so a caller can score the model against the same
  labels it was trained on. Sets ``OLINDA_HOME`` and ``OLINDA_BACKEND`` in the environment for the
  duration — which is why the pytest fixtures below use monkeypatch instead, and CI, having its own
  process, can just call this.
  """
  dest = Path(dest)
  home = dest / "home"
  smiles, x = write_library(home)
  soft = write_soft(dest, smiles, x)
  hard = write_hard(dest, smiles, x)

  previous = {k: os.environ.get(k) for k in ("OLINDA_HOME", "OLINDA_BACKEND")}
  os.environ["OLINDA_HOME"] = str(home)
  os.environ["OLINDA_BACKEND"] = backend
  try:
    import olinda.data as data_pkg
    import olinda.data.fetch as fetch

    fetch.OLINDA_HOME = home
    data_pkg.OLINDA_HOME = home

    model = dest / "model.onnx"
    args = ["fit", "-s", soft, "-m", model]
    if with_hard:
      args += ["-h", hard]
    result = run_cli(args)
    if result.exit_code != 0:
      raise RuntimeError(f"olinda fit failed:\n{result.output}\n{result.exception!r}")
  finally:
    for key, value in previous.items():
      if value is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = value
  return model


def build_artifact(run_dir: Path, names=("assay_probability",), *, backend: str = "xgboost") -> Path:
  """Fuse a minimal soft-only artifact directly, without going through the CLI.

  The inference API does not care how a model was trained, only what the file says, so these tests
  skip `fit` entirely: one short booster per column, written into a hand-built run directory and
  fused. Whole thing is a fraction of a second, where a real `fit` is seconds — which matters because
  every test here needs an artifact.
  """
  import numpy as np

  from olinda import run as runlib
  from olinda.export import build_bundle
  from olinda.featurizer import MorganCountFeaturizer
  from olinda.models import StudentModel
  from olinda.train.backend import get_backend, select_backend

  previous = os.environ.get("OLINDA_BACKEND")
  os.environ["OLINDA_BACKEND"] = backend
  try:
    featurizer = MorganCountFeaturizer()
    smiles = (SMILES * 4)[:96]
    x = featurizer.transform(smiles).astype(np.float32)

    manifest = runlib.new_manifest(
      soft_labels="soft.csv",
      hard_labels=None,
      reference={"name": "erl0_morgan.h5", "n_rows": len(smiles), "dim": int(x.shape[1])},
      features={"features": "morgan", "radius": 3},
      val_frac=0.2,
      seed=42,
      limit=None,
    )
    name, device, _ = select_backend()
    be = get_backend(name, device)
    targets = {}
    for i, col_name in enumerate(names):
      y = _target(x, i).astype(np.float32)
      idx = np.arange(len(y))
      entry = runlib.add_column(manifest, name=col_name, y=y, train_idx=idx, val_idx=idx)
      targets[entry["id"]] = y
      dtrain = be.dataset(x, y, None, 64)
      dval = be.dataset(x, y, None, 64, reference=dtrain)
      res = be.train(dtrain, dval, be.params({"max_bin": 64}), 20, 10, False)
      StudentModel(
        model=res.model,
        backend=name,
        featurizer=featurizer,
        metadata={
          "task": "regression",
          "column": col_name,
          "x_dim": int(x.shape[1]),
          "features": "morgan",
          "backend": name,
          "featurizer": featurizer.to_dict(),
          "featurizer_class": "MorganCountFeaturizer",
        },
      ).save(runlib.column_dir(run_dir, entry["id"]))

    runlib.write_targets(run_dir, targets)
    runlib.write_splits(run_dir, {cid: (np.arange(len(smiles)), np.arange(len(smiles))) for cid in targets})
    runlib.write_manifest(run_dir, manifest)
    build_bundle(run_dir)
  finally:
    if previous is None:
      os.environ.pop("OLINDA_BACKEND", None)
    else:
      os.environ["OLINDA_BACKEND"] = previous
  return run_dir / "model.onnx"


# ── pytest fixtures ──────────────────────────────────────────────────────────
# Imported lazily so this module stays usable from CI, which has no pytest.

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def artifact(tmp_path_factory) -> Path:
  """One soft-only ``model.onnx``, fused once for the whole session.

  Session-scoped because the inference tests only read it. Anything that corrupts an artifact to check
  the load-time guards must copy it first — see :func:`corrupt_metadata`.
  """
  return build_artifact(tmp_path_factory.mktemp("artifact"))


def peek_internals(model_onnx: Path, smiles, names) -> dict:
  """Run *model_onnx* with internal tensors promoted to outputs, and return every output by name.

  The shipped graph declares one output per column and keeps ``S``, ``H_S`` and the weight ``a`` to
  itself — deliberately, so no caller can build a dependency on wiring that is free to change. That
  leaves the blend arithmetic unverifiable from outside, which is exactly the thing most worth
  verifying: a cross-wired branch produces plausible numbers.

  So promote them on a throwaway copy. The artifact on disk is untouched, the test sees the parts, and
  the contract stays closed.
  """
  import numpy as np
  import onnx
  import onnxruntime as ort
  from onnx import TensorProto, helper

  from olinda.artifact import OlindaArtifact

  model = onnx.load(str(model_onnx))
  declared = {o.name for o in model.graph.output}
  for name in names:
    if name not in declared:
      model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.DOUBLE, ["B"]))

  options = ort.SessionOptions()
  options.log_severity_level = 3
  session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
  fingerprints = OlindaArtifact(model_onnx).featurize(smiles)
  outputs = [o.name for o in session.get_outputs()]
  values = session.run(None, {session.get_inputs()[0].name: fingerprints})
  return {n: np.asarray(v, dtype=np.float64).ravel() for n, v in zip(outputs, values)}


def corrupt_metadata(source: Path, dest: Path, mutate=None) -> Path:
  """Copy *source* to *dest*, editing its olinda metadata block with *mutate* (or removing it).

  ``mutate`` receives the parsed dict and edits it in place. Pass no mutator to strip the block
  altogether. This is how the load-time guards get exercised — there is no other way to produce a file
  that lies about itself.
  """
  import json as _json

  import onnx

  model = onnx.load(str(source))
  for prop in list(model.metadata_props):
    if prop.key != "olinda":
      continue
    if mutate is None:
      model.metadata_props.remove(prop)
    else:
      meta = _json.loads(prop.value)
      mutate(meta)
      prop.value = _json.dumps(meta)
  onnx.save(model, str(dest))
  return dest


@pytest.fixture(params=["xgboost", "lightgbm"])
def backend(request, monkeypatch):
  """Run the test once per engine.

  Both are real shipping paths — LightGBM is what `select_backend` picks on any CPU machine, which is
  most users and all of CI, and XGBoost is the CUDA path. Pinning one engine "for determinism" is how
  the other went untested.
  """
  pytest.importorskip(request.param)
  monkeypatch.setenv("OLINDA_BACKEND", request.param)
  return request.param


@pytest.fixture
def library(tmp_path, monkeypatch):
  """A staged reference library and its label files: ``(home, smiles, x)``, with paths patched.

  ``OLINDA_HOME`` is patched on both modules that read it — ``olinda.data.fetch`` owns the constant
  and ``olinda.data`` re-exports it, so patching one leaves the other pointing at the real
  ``~/.olinda`` and the test would train against whatever the developer happens to have downloaded.
  """
  import olinda.data as data_pkg
  import olinda.data.fetch as fetch

  home = tmp_path / "home"
  smiles, x = write_library(home)
  monkeypatch.setattr(fetch, "OLINDA_HOME", home)
  monkeypatch.setattr(data_pkg, "OLINDA_HOME", home)
  monkeypatch.setenv("OLINDA_BACKEND", "xgboost")
  return home, smiles, x


def peek_internals(artifact_path, smiles, names) -> dict:
  """Run the fused graph with *names* promoted to outputs, on a copy — the file is not modified.

  The artifact declares one output per column and nothing else: `S`, `H_S` and the blend weight `a`
  are internal tensors, deliberately, so callers cannot come to depend on a wiring we still change.
  A test is the exception that proves the rule — checking that the published prediction really is
  ``(1-a)*S + a*H_S`` means reading the parts, and that assertion is the one that would catch a
  cross-wired blend. Tests may reach inside; shipping the same reach is what we are avoiding.

  Names are the graph's internal ones, prefixed per column: ``c0_s``, ``c0_h_s``, ``c0_a``.
  """
  import numpy as np
  import onnx
  import onnxruntime as ort
  from onnx import TensorProto, helper

  from olinda.featurizer import MorganCountFeaturizer

  artifact_path = Path(artifact_path)
  model = onnx.load(str(artifact_path))
  have = {o.name for o in model.graph.output}
  for name in names:
    if name not in have:
      model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.DOUBLE, ["B"]))
  tmp = artifact_path.parent / "_peek.onnx"
  onnx.save(model, str(tmp))

  options = ort.SessionOptions()
  options.log_severity_level = 3
  sess = ort.InferenceSession(str(tmp), options, providers=["CPUExecutionProvider"])
  fp = MorganCountFeaturizer().transform([str(s) for s in smiles]).astype(np.float32)
  outs = sess.run(None, {"input": fp})
  return {o.name: np.asarray(v).ravel() for o, v in zip(sess.get_outputs(), outs)}
