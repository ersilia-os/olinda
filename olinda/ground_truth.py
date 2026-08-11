"""Learn from hard (experimental) labels and calibrate them onto the teacher's soft-label scale.

olinda's surrogate ``S(x)`` distills a teacher's *soft* labels. When real experimental (*hard*) labels of
the same endpoint are available, ``learn-hard`` runs four steps, each printed clearly:

1. **Train ``G``** — a binary hard-label classifier: a plain, portfolio-selected XGBoost booster via `lazy-qsar
   <https://github.com/ersilia-os/lazy-qsar>`_'s ``BaseXGBClassifier(calibrated=False)`` on olinda's Morgan
   count fingerprints (:class:`~olinda.featurizer.MorganCountFeaturizer`). Output is raw ``predict_proba`` —
   lazy-qsar's internal probability calibrator is off. (Continuous hard labels are a raises-for-now
   placeholder — see :func:`_new_gt_model`.)
2. **Score ``G`` across the full reference library** (``erl0_morgan.h5``) → one hard score per reference
   compound, saved to ``g_reference.h5``.
3. **Calibrate** ``G`` onto the soft-label scale — a monotonic isotonic map fit on the reference library
   (where both ``G``'s score and the teacher's soft label exist), with the **direction learned from the
   data** (a low hard score may map to a high soft label). Saved as ``g_to_soft.json``.
4. **Learn the applicability gate** — bucket every reference compound by its 1-NN Tanimoto similarity to the
   labeled set (NOT SIMILAR / LOW / HIGH) and fit two Bernoulli Naive-Bayes classifiers on Morgan features
   (saved as ``applicability_nb.json``). At predict time these place a query in a bucket with no similarity
   search — see :mod:`olinda.applicability`.

The end goal is to predict the soft-label distribution informed by the hard labels. Artifacts land under
``<model_dir>/_ground_truth/``. The gate decides *where* to trust ``G``: the blend
``prediction = (1-a)·S + a·G_soft`` leans on the hard signal only near the labeled chemistry. All stages are
fused into a single ``model.onnx`` (see :mod:`olinda.export`) and served by
:class:`~olinda.onnx_pipeline.OnnxPipeline`.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from olinda.console import echo, rule, summary_panel
from olinda.featurizer import MorganCountFeaturizer

# Layout under <model_dir>/
HARD_H5_NAME = (
  "hard.h5"  # featurized hard labels written by `prepare --hard-labels`, consumed by `learn-hard`
)
GT_DIRNAME = "_ground_truth"
GT_MODEL_SUBDIR = "gt"
G_REFERENCE_NAME = "g_reference.h5"  # G's score for every reference-library compound (aligned to erl0_morgan)
CALIBRATOR_NAME = "g_to_soft.json"  # isotonic map from G's output onto the soft-label scale
APPLICABILITY_NAME = "applicability_nb.json"  # two Bernoulli-NB classifiers gating the hard signal
GT_META_NAME = "ground_truth_meta.json"
GT_EVAL_NAME = "gt_eval.json"


def has_hard_head(model_dir: str | Path) -> bool:
  """True iff *model_dir* has a hard-label head (``learn-hard`` was run) — i.e. ``_ground_truth/gt/`` exists."""
  return (Path(model_dir) / GT_DIRNAME / GT_MODEL_SUBDIR / "xgboost.json").exists()


def _detect_task(y: np.ndarray) -> str:
  """Infer the task from the label values: binary iff every finite label is 0 or 1, else regression."""
  finite = y[np.isfinite(y)]
  if finite.size and set(np.unique(finite).tolist()) <= {0.0, 1.0}:
    return "binary"
  return "regression"


def _featurize(smiles, featurizer):
  """Featurize SMILES, dropping rows RDKit could not parse (all-zero fingerprint)."""
  X = featurizer.transform([str(s) for s in smiles]).astype(np.float32)
  valid = X.sum(axis=1) > 0
  return X, valid


def _new_gt_model(task: str):
  """The hard-label model G. Binary → a plain, portfolio-selected XGBoost classifier (regression raises).

  We use lazy-qsar only for its portfolio/preset selection and booster training — ``calibrated=False`` turns
  OFF lazy-qsar's internal probability calibrator, so ``G`` outputs raw ``predict_proba``. This keeps the
  reference scoring, prediction, and the exported ONNX all on the same raw probability (no hidden calibration
  step), which is what makes ``G``'s ONNX faithful. The teacher-scale mapping is our own separate ``g_to_soft``
  isotonic calibrator (step 3). Continuous hard labels would need a regressor — a planned placeholder.
  """
  from lazyqsar.base.xgboost import BaseXGBClassifier

  if task == "binary":
    return BaseXGBClassifier(calibrated=False)
  raise NotImplementedError(
    f"hard-label task {task!r} is not supported yet — only binary (lazy-qsar XGBoost classifier). "
    "A regressor path is a planned placeholder."
  )


def _selection_report(gt_model) -> dict:
  """Read lazy-qsar's own model-selection diagnostics off a fitted estimator.

  ``BaseXGBClassifier`` / ``BaseXGBRegressor`` pick their config against an internal validation split
  during ``fit`` and expose the winning preset, its boosting rounds, and the portfolio scores.
  """
  scores = getattr(gt_model, "portfolio_scores_", {}) or {}
  preset = getattr(gt_model, "preset_name_", None)
  return {
    "preset": preset,
    "best_iteration": int(getattr(gt_model, "best_iteration_", -1)),
    "portfolio_scores": {k: float(v) for k, v in scores.items()},
    "selected_score": float(scores[preset]) if preset in scores else None,
  }


def _reference_path() -> Path:
  """Path to the reference-library Morgan HDF5; raise a friendly error if it isn't present."""
  from olinda.data.fetch import MORGAN_FINGERPRINTS_FILENAME, OLINDA_HOME

  path = Path(OLINDA_HOME) / MORGAN_FINGERPRINTS_FILENAME
  if not path.exists():
    raise FileNotFoundError(
      f"reference library {path} not found — run `olinda setup` (or scripts/compute_morgan_fingerprints.py). "
      "learn-hard needs it to score G across the reference and calibrate to the soft labels."
    )
  return path


def _score_reference(model, task: str, n_features: int, matrix, chunk: int = 50_000):
  """Score G over every reference compound, aligned to erl0_morgan row order.

  Reads from the shared in-RAM matrix rather than reopening the library, so a multi-column run pays
  for one load instead of one per scan per column.
  """
  # 50k rows x 2048 float32 is ~410 MB per chunk; 200k was 1.64 GB for the same throughput.
  n, dim = matrix.n_rows, matrix.n_cols
  if dim != n_features:
    raise ValueError(f"reference library has {dim}-d features but G expects {n_features}-d")
  out = []
  for start in range(0, n, chunk):
    xb = np.asarray(matrix.x[start : start + chunk], dtype=np.float32)
    g = np.asarray(model.predict_proba(xb))[:, 1] if task == "binary" else np.asarray(model.predict(xb))
    out.append(np.asarray(g, dtype=np.float32).ravel())
    echo(f"  scored {min(start + chunk, n):,}/{n:,} reference compounds", "info")
  return np.concatenate(out)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  ac, bc = a - a.mean(), b - b.mean()
  d = float(np.sqrt((ac**2).sum()) * np.sqrt((bc**2).sum()))
  return float("nan") if d == 0 else float((ac * bc).sum() / d)


def _fit_applicability(gt_bits, n_features, matrix, sim_lo: float, sim_hi: float, chunk: int = 50_000):
  """Learn the applicability gate: bucket the reference by Tanimoto-NN to the labeled set, fit two NB models.

  Streams the reference library once, computing each compound's 1-NN Tanimoto similarity to the labeled set
  (``gt_bits``) and accumulating the Bernoulli-NB sufficient statistics for two binary targets — ``sim >=
  sim_lo`` (at-least-LOW) and ``sim >= sim_hi`` (HIGH). The labeled compounds themselves are HIGH by
  definition, so they are folded in as guaranteed positives (this also relieves the HIGH-class imbalance).

  Returns
  -------
  tuple
      ``(clf, counts)`` where ``clf`` is an :class:`~olinda.applicability.ApplicabilityClassifier` and
      ``counts`` is ``{"n_ref", "n_high", "n_low", "n_not", "n_gt"}`` for reporting.
  """

  from olinda.applicability import (
    A_HIGH,
    A_LOW,
    ApplicabilityClassifier,
    BernoulliNB,
    prepare_gt_bits,
    tanimoto_nn,
  )

  d = int(n_features)
  # Bernoulli-NB sufficient statistics per target: rows-per-class (2,) and feature-on counts (2, d).
  low_n = np.zeros(2, dtype=np.float64)
  low_on = np.zeros((2, d), dtype=np.float64)
  high_n = np.zeros(2, dtype=np.float64)
  high_on = np.zeros((2, d), dtype=np.float64)
  n_high = n_low = 0

  def _accumulate(bits, y, class_n, feat_on, col_total):
    """Add one chunk's Bernoulli-NB sufficient statistics for a binary target.

    The per-class feature counts come from a matvec rather than ``bits[mask].sum(0)``: masking
    materialises a copy of up to the whole chunk, four times per chunk. Both classes are exact —
    the features are 0/1, so the sums are integers well inside float32's exact range.
    """
    n_pos = int(y.sum())
    on_pos = y.astype(np.float32) @ bits
    class_n[1] += n_pos
    class_n[0] += int(len(y) - n_pos)
    feat_on[1] += on_pos
    feat_on[0] += col_total - on_pos

  gt_prepared = prepare_gt_bits(gt_bits)  # built once, reused for every chunk
  n = matrix.n_rows
  for start in range(0, n, chunk):
    bits = (matrix.x[start : start + chunk] > 0).astype(np.float32)
    sim = tanimoto_nn(bits, prepared=gt_prepared)
    y_low = (sim >= sim_lo).astype(np.int64)
    y_high = (sim >= sim_hi).astype(np.int64)
    col_total = bits.sum(axis=0)  # shared by both targets' class-0 counts
    _accumulate(bits, y_low, low_n, low_on, col_total)
    _accumulate(bits, y_high, high_n, high_on, col_total)
    n_low += int(y_low.sum())
    n_high += int(y_high.sum())
    echo(f"  scanned {min(start + chunk, n):,}/{n:,} reference compounds", "info")

  # Fold the labeled compounds in as guaranteed positives (HIGH ⇒ also ≥ LOW).
  gb = (np.asarray(gt_bits) > 0).astype(np.float32)
  n_gt = int(gb.shape[0])
  low_n[1] += n_gt
  low_on[1] += gb.sum(axis=0)
  high_n[1] += n_gt
  high_on[1] += gb.sum(axis=0)

  clf = ApplicabilityClassifier(
    BernoulliNB.from_counts(low_n, low_on),
    BernoulliNB.from_counts(high_n, high_on),
    a_low=A_LOW,
    a_high=A_HIGH,
    sim_lo=sim_lo,
    sim_hi=sim_hi,
  )
  counts = {
    "n_ref": n,
    "n_high": n_high,
    "n_low": n_low - n_high,  # exclusively LOW (HIGH ⊂ ≥LOW)
    "n_not": n - n_low,
    "n_gt": n_gt,
  }
  return clf, counts


def prepare_hard_labels(input_csv: str | Path, out_dir: str | Path, *, task: str = "auto") -> dict:
  """Featurize a hard-label file into ``<out_dir>/hard.h5`` for a later ``learn-hard`` step.

  Reads a SMILES column + one label column, featurizes with :class:`MorganCountFeaturizer` (dropping
  unparseable rows), auto-detects the task (binary iff labels ⊆ {0,1}, unless overridden), and writes
  datasets ``x`` ``(n, 2048)`` float32 / ``y`` ``(n,)`` float32 with attrs ``task``, ``features``, and
  the featurizer config (JSON). The hard labels are the user's own compounds — not aligned to the
  reference library.

  Parameters
  ----------
  input_csv : str or Path
      CSV/TSV/Parquet with a SMILES column and one label column.
  out_dir : str or Path
      Directory the ``hard.h5`` is written into (the run's model dir).
  task : {"auto", "binary", "regression"}
      ``"auto"`` (default) infers the task from the labels.

  Returns
  -------
  dict
      ``{"task", "n", "n_dropped", "path"}``.
  """
  import h5py

  from olinda.data.reference import _read_table, resolve_smiles_value

  smiles, y_raw = resolve_smiles_value(_read_table(input_csv))
  y_raw = np.asarray(y_raw, dtype=np.float64)
  featurizer = MorganCountFeaturizer()
  X, valid = _featurize(smiles, featurizer)
  n_dropped = int((~valid).sum())
  if n_dropped:
    echo(f"dropping {n_dropped} unparseable SMILES", "warning")
  X, y = X[valid], y_raw[valid]
  finite = np.isfinite(y)
  X, y = X[finite], y[finite]
  if len(y) < 4:
    raise ValueError(f"need at least 4 usable hard-label rows, got {len(y)}")

  resolved_task = _detect_task(y) if task == "auto" else task
  if resolved_task not in ("binary", "regression"):
    raise ValueError(f"unknown task {task!r}")
  if resolved_task == "binary":
    y = y.astype(int).astype(np.float64)

  # Shuffle (deterministically) so the on-disk order carries no bias from how the compounds were listed —
  # e.g. actives-first or value-sorted input. learn-hard's internal train/val split then sees mixed rows,
  # mirroring the soft path (split_reference_to_h5 already shuffles).
  perm = np.random.RandomState(42).permutation(len(y))
  X, y = X[perm], y[perm]

  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  out_path = out_dir / HARD_H5_NAME
  with h5py.File(out_path, "w") as f:
    f.create_dataset("x", data=X.astype(np.float32))
    f.create_dataset("y", data=y.astype(np.float32))
    f.attrs["task"] = resolved_task
    f.attrs["features"] = "morgan_count"
    f.attrs["featurizer"] = json.dumps(featurizer.to_dict())
    f.attrs["n_dropped"] = n_dropped
  return {"task": resolved_task, "n": int(len(y)), "n_dropped": n_dropped, "path": str(out_path)}


MIN_HARD_ROWS = 4
MIN_PER_CLASS = 2  # below this a class-stratified train/val split is impossible


def prepare_hard_labels_wide(
  input_path: str | Path, run_dir: str | Path, mapping: dict, *, task: str = "auto"
) -> dict:
  """Featurize a wide hard-label file once and write one ``hard.h5`` per matched column.

  The file carries a SMILES column plus one column per assay, empty where a compound was not tested.
  RDKit featurization is the expensive part and does not depend on the column, so it happens once for
  the whole file; each column then selects the rows where its own value is present.

  Parameters
  ----------
  input_path : str or Path
      CSV/TSV/Parquet: a SMILES column followed by one value column per assay.
  run_dir : str or Path
      The run directory; each column's file lands in ``columns/<id>/hard.h5``.
  mapping : dict
      ``{hard_column: column_id}`` — which hard column feeds which prepared column.
  task : {"auto", "binary", "regression"}
      ``"auto"`` resolves the task per column.

  Returns
  -------
  dict
      ``{hard_column: {"task", "n", "n_positive", "n_dropped", "path"}}``.

  Raises
  ------
  ValueError
      If a column has too few usable rows, or — for a binary column — too few of either class to
      support a stratified split. Raised here rather than hours later inside ``learn-hard``.
  """
  import h5py

  from olinda.data.reference import _read_table, resolve_smiles_frame
  from olinda.run import column_dir

  smiles, values = resolve_smiles_frame(_read_table(input_path))
  featurizer = MorganCountFeaturizer()
  X, parsed = _featurize(smiles, featurizer)
  if (~parsed).any():
    echo(f"dropping {int((~parsed).sum())} unparseable SMILES", "warning")

  out: dict[str, dict] = {}
  for hard_col, col_id in mapping.items():
    y_raw = np.asarray(values[hard_col].to_numpy(), dtype=np.float64)
    keep = parsed & np.isfinite(y_raw)
    xc, yc = X[keep], y_raw[keep]
    if len(yc) < MIN_HARD_ROWS:
      raise ValueError(
        f"hard column '{hard_col}' has only {len(yc)} usable row(s); at least {MIN_HARD_ROWS} needed"
      )

    resolved = _detect_task(yc) if task == "auto" else task
    if resolved not in ("binary", "regression"):
      raise ValueError(f"unknown task {task!r}")
    n_positive = None
    if resolved == "binary":
      yc = yc.astype(int).astype(np.float64)
      n_positive = int(yc.sum())
      n_negative = int(len(yc) - n_positive)
      if min(n_positive, n_negative) < MIN_PER_CLASS:
        raise ValueError(
          f"hard column '{hard_col}' has {n_positive} positive and {n_negative} negative row(s); "
          f"at least {MIN_PER_CLASS} of each are needed for a class-stratified split. "
          "Drop the column or supply more labels."
        )
    else:
      raise NotImplementedError(
        f"hard column '{hard_col}' looks continuous; only binary ground truth is supported today"
      )

    perm = np.random.RandomState(42).permutation(len(yc))
    xc, yc = xc[perm], yc[perm]
    out_dir = column_dir(run_dir, col_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / HARD_H5_NAME
    with h5py.File(out_path, "w") as f:
      f.create_dataset("x", data=xc.astype(np.float32))
      f.create_dataset("y", data=yc.astype(np.float32))
      f.attrs["task"] = resolved
      f.attrs["features"] = "morgan_count"
      f.attrs["featurizer"] = json.dumps(featurizer.to_dict())
      f.attrs["n_dropped"] = int((~keep).sum())
    out[hard_col] = {
      "task": resolved,
      "n": int(len(yc)),
      "n_positive": n_positive,
      "n_dropped": int((~keep).sum()),
      "path": str(out_path),
    }
  return out


def train_ground_truth(model_dir: str | Path, soft=None, matrix=None) -> dict:
  """Train the hard-label model ``G`` and calibrate it onto the soft-label scale.

  Reads ``<model_dir>/hard.h5`` (written by :func:`prepare_hard_labels`) and ``<model_dir>/soft.h5``
  (reference-aligned soft labels from ``prepare``), then writes ``G`` (ONNX), its scores over the reference
  library, the ``G``→soft calibrator, and the applicability gate (two Bernoulli-NB classifiers) under
  ``<model_dir>/_ground_truth/``. The gate is learned here from the reference but *applied* at predict time
  (no similarity search) — see :mod:`olinda.applicability`.

  Parameters
  ----------
  model_dir : str or Path
      A run directory containing ``hard.h5`` and ``soft.h5``.

  Returns
  -------
  dict
      ``{"task", "n", "gt_dir", "selection", "calibration", "applicability"}``.
  """
  import h5py

  model_dir = Path(model_dir)
  hard_path = model_dir / HARD_H5_NAME
  if not hard_path.exists():
    raise FileNotFoundError(
      f"no {HARD_H5_NAME} in {model_dir} — run `olinda prepare --hard-labels <file> -m {model_dir}` first"
    )
  if soft is None:
    raise ValueError(
      "train_ground_truth needs this column's reference-aligned soft labels; the caller reads them "
      "from the run's targets.h5"
    )
  soft = np.asarray(soft, dtype=np.float64)
  if matrix is None:
    # Standalone call: load the library ourselves. `learn-hard` passes one in so a multi-column run
    # loads it once for the whole run instead of twice per column.
    from olinda.data.matrix import ReferenceMatrix

    matrix = ReferenceMatrix.load(_reference_path())

  gt_root = model_dir / GT_DIRNAME
  gt_dir = gt_root / GT_MODEL_SUBDIR
  gt_root.mkdir(parents=True, exist_ok=True)
  rule("olinda · learn-hard", style="green", right=str(model_dir))

  # --- load the prepared hard labels ---------------------------------------
  with h5py.File(hard_path, "r") as f:
    X = np.asarray(f["x"][:], dtype=np.float32)
    y = np.asarray(f["y"][:], dtype=np.float64)
    resolved_task = str(f.attrs.get("task", "binary"))
    featurizer_json = f.attrs.get("featurizer")
  featurizer_dict = json.loads(featurizer_json) if featurizer_json else MorganCountFeaturizer().to_dict()
  gt_model = _new_gt_model(resolved_task)  # placeholder gate: raises NotImplementedError for regression
  y = y.astype(int)  # binary labels (past the gate)

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # hush lazy-qsar's sklearn version-drift FutureWarnings

    # === Step 1/4 — train the hard-label classifier G ======================
    echo(
      f"Step 1/4 · training hard-label classifier G on {len(y):,} compounds ({int(y.sum())} positive)", "run"
    )
    gt_model.fit(X, y)
    gt_model.save(str(gt_dir))
    selection = _selection_report(gt_model)
    echo(f"  G ready · preset={selection['preset']} · {selection['best_iteration']} trees", "run")

    # === Step 2/4 — score G across the full reference library ==============
    echo("Step 2/4 · scoring G across the reference library", "run")
    g_ref = _score_reference(gt_model, "binary", X.shape[1], matrix)
    with h5py.File(gt_root / G_REFERENCE_NAME, "w") as f:
      f.create_dataset("g", data=g_ref.astype(np.float32))
    echo(f"  saved G scores for {len(g_ref):,} reference compounds → {G_REFERENCE_NAME}", "run")

    # === Step 3/4 — calibrate G onto the soft-label scale ==================
    from olinda.calibrate import IsotonicCalibrator, _spearman_sign

    echo("Step 3/4 · calibrating G → soft-label scale", "run")
    m = min(len(g_ref), len(soft))
    gv, sv = g_ref[:m].astype(np.float64), soft[:m]
    mask = np.isfinite(gv) & np.isfinite(sv)
    gv, sv = gv[mask], sv[mask]
    calibrator = IsotonicCalibrator().fit(gv, sv, increasing="auto")
    calibrator.save(gt_root / CALIBRATOR_NAME)
    direction = "increasing" if calibrator._sign > 0 else "decreasing"
    spearman = _spearman_sign(gv, sv)  # magnitude+sign of the rank correlation
    pearson_after = _pearson(calibrator.transform(gv), sv)
    echo(
      f"  calibrated ({direction}) on {len(gv):,} reference compounds · "
      f"Spearman(G,soft)={spearman:+.3f} · Pearson(calibrated,soft)={pearson_after:.3f}",
      "run",
    )

    # diagnostic plots (stylia; skipped with a warning if stylia is absent)
    from olinda.train.plots import save_ground_truth_plots

    plots = save_ground_truth_plots(
      gv, sv, calibrator, gt_root / "plots", direction=direction, pearson_after=pearson_after
    )
    for p in plots:
      echo(f"  plot → {p.relative_to(gt_root)}", "run")

    # === Step 4/4 — learn the applicability gate (two NB classifiers) ======
    from olinda.applicability import A_HIGH, A_LOW, SIM_HI, SIM_LO

    echo("Step 4/4 · learning applicability classifiers (NOT / LOW / HIGH) from the reference", "run")
    gt_bits = (X > 0).astype(np.float32)
    ad_clf, ad_counts = _fit_applicability(gt_bits, X.shape[1], matrix, SIM_LO, SIM_HI)
    ad_clf.save(gt_root / APPLICABILITY_NAME)
    echo(
      f"  buckets over {ad_counts['n_ref']:,} reference compounds · "
      f"HIGH {ad_counts['n_high']:,} · LOW {ad_counts['n_low']:,} · NOT {ad_counts['n_not']:,} "
      f"(+{ad_counts['n_gt']:,} labeled compounds folded in as HIGH)",
      "run",
    )
    echo(
      f"  fitted 2 Bernoulli-NB (sim_lo={SIM_LO}, sim_hi={SIM_HI} · weights LOW={A_LOW}, HIGH={A_HIGH}) "
      f"→ {APPLICABILITY_NAME}",
      "run",
    )

  # --- persist metadata + eval ---------------------------------------------
  import lazyqsar as _lq

  calibration = {
    "direction": direction,
    "spearman_g_soft": spearman,
    "pearson_calibrated_soft": pearson_after,
    "n_reference": int(len(gv)),
  }
  applicability = {
    "signal": "bernoulli_nb_2clf",
    "sim_lo": SIM_LO,
    "sim_hi": SIM_HI,
    "a_low": A_LOW,
    "a_high": A_HIGH,
    "n_gt": ad_counts["n_gt"],
    "n_high": ad_counts["n_high"],
    "n_low": ad_counts["n_low"],
    "n_not": ad_counts["n_not"],
    "artifact": APPLICABILITY_NAME,
  }
  meta = {
    "task": "binary",
    "n": int(len(y)),
    "features": "morgan_count",
    "featurizer": featurizer_dict,
    "featurizer_class": "MorganCountFeaturizer",
    "gt_dir": GT_MODEL_SUBDIR,
    "g_reference": G_REFERENCE_NAME,
    "calibrator": CALIBRATOR_NAME,
    # Applicability gate is learned here and applied at predict time (two NB classifiers, no similarity
    # search). Blend: prediction = (1-a)*surrogate + a*ground_truth_soft.
    "applicability": applicability,
    "lazyqsar_version": getattr(_lq, "__version__", "unknown"),
  }
  with open(gt_root / GT_META_NAME, "w") as fp:
    json.dump(meta, fp, indent=2)
  with open(gt_root / GT_EVAL_NAME, "w") as fp:
    json.dump(
      {"selection": selection, "calibration": calibration, "applicability": applicability}, fp, indent=2
    )

  _mark_surrogate_combined(model_dir)

  # --- report ---------------------------------------------------------------
  rows = [
    ("Hard model G", f"[bold]{selection['preset']}[/] · {selection['best_iteration']} trees"),
    ("Reference scored", f"[bold]{len(gv):,}[/] compounds"),
    (
      "G → soft calibration",
      f"[bold]{direction}[/] · Spearman {spearman:+.3f} · Pearson(cal) {pearson_after:.3f}",
    ),
    (
      "Applicability",
      f"HIGH [bold]{ad_counts['n_high']:,}[/] · LOW [bold]{ad_counts['n_low']:,}[/] · "
      f"NOT [bold]{ad_counts['n_not']:,}[/] → [dim]{APPLICABILITY_NAME}[/]",
    ),
    ("Saved", f"[dim]{gt_root}[/]"),
  ]
  if plots:
    rows.append(("Plots", f"[bold]{len(plots)}[/] → [dim]{gt_root / 'plots'}[/]"))
  summary_panel("olinda · learn-hard", rows, border_style="green", icon="✓")

  return {
    "task": "binary",
    "n": int(len(y)),
    "gt_dir": str(gt_dir),
    "selection": selection,
    "calibration": calibration,
    "applicability": applicability,
  }


def _mark_surrogate_combined(model_dir: Path) -> None:
  """Flip ``ground_truth: true`` on the surrogate's ``train_meta.json`` if present (the ``_ground_truth/``
  directory is the authoritative marker; this is just a convenience flag)."""
  path = model_dir / "train_meta.json"
  if not path.exists():
    return
  try:
    with open(path) as fp:
      data = json.load(fp)
    data["ground_truth"] = True
    with open(path, "w") as fp:
      json.dump(data, fp, indent=2)
  except (json.JSONDecodeError, OSError):  # pragma: no cover - non-fatal
    pass
