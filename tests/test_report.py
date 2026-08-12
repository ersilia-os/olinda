"""Reading a finished artifact back: its calibration curves and its size."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("xgboost")

from olinda.report.internals import describe_graph  # noqa: E402

_SM = ["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCOC(=O)C", "Clc1ccccc1", "COc1ccccc1"]


def _build_calibrated_artifact(tmp_path, monkeypatch):
  """A one-column run whose surrogate carries a known isotonic correction; returns (path, calibrator)."""
  from olinda import run as runlib
  from olinda.calibrate import IsotonicCalibrator
  from olinda.export import build_bundle
  from olinda.featurizer import MorganCountFeaturizer
  from olinda.models import StudentModel
  from olinda.train.backend import get_backend, select_backend

  monkeypatch.setenv("OLINDA_BACKEND", "xgboost")
  featurizer = MorganCountFeaturizer()
  smiles = (_SM * 12)[:96]
  x = featurizer.transform(smiles).astype(np.float32)
  # A target the booster can actually resolve: a flat one gives a single-leaf tree, and a calibrator
  # fitted on two distinct raw values would pass an interpolation check for trivial reasons.
  y = (x[:, :600].sum(1) / 20.0).astype(np.float32)

  manifest = runlib.new_manifest(
    soft_labels="soft.csv",
    hard_labels=None,
    reference={"name": "erl0_morgan.h5", "n_rows": len(smiles), "dim": int(x.shape[1])},
    features={"features": "morgan", "radius": 3},
    val_frac=0.2,
    seed=42,
    limit=None,
  )
  idx = np.arange(len(y))
  entry = runlib.add_column(manifest, name="assay_probability", y=y, train_idx=idx, val_idx=idx)

  name, device, _ = select_backend()
  be = get_backend(name, device)
  dtrain = be.dataset(x, y, None, 64)
  res = be.train(
    dtrain, be.dataset(x, y, None, 64, reference=dtrain), be.params({"max_bin": 64}), 20, 10, False
  )

  # The calibrator is fitted on a synthetic raw→target relation rather than on this toy booster's
  # own predictions: 8 distinct molecules and one tree collapse to a SINGLE distinct output, which
  # yields a one-anchor constant curve that any round-trip check passes for free. What is under test
  # is the anchors surviving the trip through the graph, so give it a curve with real structure.
  raw = np.linspace(-2.0, 3.0, 400)
  calibrator = IsotonicCalibrator().fit(raw=raw, target=1.0 / (1.0 + np.exp(-raw)))

  col_dir = runlib.column_dir(tmp_path, entry["id"])
  StudentModel(
    model=res.model,
    backend=name,
    featurizer=featurizer,
    calibrator=calibrator,
    metadata={
      "task": "regression",
      "column": "assay_probability",
      "x_dim": int(x.shape[1]),
      "features": "morgan",
      "backend": name,
      "featurizer": featurizer.to_dict(),
      "featurizer_class": "MorganCountFeaturizer",
    },
  ).save(col_dir)
  # StudentModel.save writes the booster and its metadata but not the calibrator — learn-soft persists
  # that separately (olinda/train/column.py:98), and StudentModel.load is what reads it back.
  calibrator.save(col_dir / "calibrator.json")

  runlib.write_targets(tmp_path, {entry["id"]: y})
  runlib.write_splits(tmp_path, {entry["id"]: (idx, idx)})
  runlib.write_manifest(tmp_path, manifest)
  build_bundle(tmp_path)
  return tmp_path / "model.onnx", calibrator, raw


def test_recovered_calibration_reproduces_the_calibrator(tmp_path, monkeypatch):
  """The whole point: anchors read from the graph must give back the map that was fitted.

  A curve that merely looks monotone and plausible would sail through a weaker check while being
  shifted, flipped, or unsorted — so compare values against IsotonicCalibrator.transform itself.

  The match is to within ``_ISOTONIC_TOL``, not exact: the fuse thins the knots to that tolerance so
  the graph stays small (``export._thin_isotonic``). Asserting equality would be asserting that the
  thinning does not happen.
  """
  from olinda.export import _ISOTONIC_TOL

  path, calibrator, raw = _build_calibrated_artifact(tmp_path, monkeypatch)
  curve = describe_graph(path)["assay_probability"]["soft_calibration"]
  assert curve is not None, "the fused graph should carry the surrogate correction"
  # Guard the guard: a one-anchor curve is constant, and comparing two constants proves nothing.
  assert curve["n_anchors"] > 10, f"degenerate curve ({curve['n_anchors']} anchors) — test is vacuous"

  probe = np.linspace(raw.min(), raw.max(), 200)
  recovered = np.interp(probe, curve["x"], curve["y"])
  np.testing.assert_allclose(recovered, calibrator.transform(probe), atol=_ISOTONIC_TOL)
  assert np.all(np.diff(recovered) >= -1e-12), "a calibration map must stay monotone"


def test_recovered_calibration_clamps_outside_its_range(tmp_path, monkeypatch):
  """Out-of-range inputs hold the end values — the same clamp the graph applies."""
  from olinda.export import _ISOTONIC_TOL

  path, calibrator, raw = _build_calibrated_artifact(tmp_path, monkeypatch)
  curve = describe_graph(path)["assay_probability"]["soft_calibration"]
  far = np.array([raw.min() - 10.0, raw.max() + 10.0])
  np.testing.assert_allclose(
    np.interp(far, curve["x"], curve["y"]), calibrator.transform(far), atol=_ISOTONIC_TOL
  )


def test_graph_description_counts_trees_and_names_columns(tmp_path, monkeypatch):
  path, _, _ = _build_calibrated_artifact(tmp_path, monkeypatch)
  info = describe_graph(path)
  assert list(info) == ["assay_probability"]
  entry = info["assay_probability"]
  assert entry["id"] == "c0"
  assert entry["n_trees"] > 0 and entry["n_nodes"] >= entry["n_trees"]
  assert entry["hard_calibration"] is None  # soft-only model


def test_a_non_olinda_onnx_is_refused(tmp_path, monkeypatch):
  import onnx

  path, _, _ = _build_calibrated_artifact(tmp_path, monkeypatch)
  m = onnx.load(str(path))
  del m.metadata_props[:]
  onnx.save(m, str(path))
  with pytest.raises(ValueError, match="no olinda metadata"):
    describe_graph(path)


# ── the figure style contract ────────────────────────────────────────────────


def test_every_figure_the_report_draws_has_a_title_and_caption():
  """The page reads its captions from this registry, so a new plot without an entry ships bare.

  Guards the pairing rather than the wording: `render` names files `<task>_<figure>` and both
  halves contain underscores, so the lookup matches the longest key the name ends with.
  """
  from olinda.report import plots

  for key, (title, text) in plots.FIGURES.items():
    assert title and text, f"{key} has an empty title or caption"
    assert plots.caption(f"some_task_name_{key}") == (title, text), f"{key} is not reachable"

  # the ambiguous pair the longest-match rule exists for
  assert plots.caption("t_soft_calibration")[0] == "S · surrogate correction"
  assert plots.caption("t_calibration")[0] == "Calibration"


def test_the_page_and_the_figures_take_their_colours_from_one_table():
  """`html.py` paints its colour key from `style.hexcol`, which must agree with what plots draw."""
  import stylia

  from olinda import style

  for role, name in style.ROLES.items():
    from_plot = stylia.ArticleColors().hex[name]
    assert style.hexcol(role).lower() == from_plot.lower(), f"{role} disagrees with stylia"
    # the offline copy is what the page uses when the report extra is absent
    assert style._PAPER_FALLBACK[name].lower() == from_plot.lower(), f"{name} fallback is stale"
