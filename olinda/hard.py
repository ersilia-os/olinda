"""Learn from hard (experimental) labels and calibrate them onto the teacher's soft-label scale.

olinda's surrogate ``S(x)`` distills a teacher's *soft* labels. When real experimental (*hard*) labels of
the same endpoint are available, ``learn-hard`` runs four steps, each printed clearly:

1. **Train ``H``** — a binary hard-label classifier: a plain, portfolio-selected XGBoost booster via `lazy-qsar
   <https://github.com/ersilia-os/lazy-qsar>`_'s ``BaseXGBClassifier(calibrated=False)`` on olinda's Morgan
   count fingerprints (:class:`~olinda.featurizer.MorganCountFeaturizer`). Output is raw ``predict_proba`` —
   lazy-qsar's internal probability calibrator is off. (Continuous hard labels are a raises-for-now
   placeholder — see :func:`_new_hard_model`.)
2. **Score ``H`` across the reference library** (``erl0_morgan.h5``) → one hard score per reference
   compound, saved to ``h_reference.h5``.
3. **Calibrate** ``H`` onto the soft-label scale — a monotonic isotonic map fit on the reference library
   (where both ``H``'s score and the teacher's soft label exist), with the **direction learned from the
   data** (a low hard score may map to a high soft label). Saved as ``h_to_s.json``.
4. **Learn T** — label every reference compound with its exact 1-NN Tanimoto similarity
   to the labeled set, then fit a small MLP that predicts that number from the fingerprint alone (saved
   under ``tanimoto/`` as ``t.onnx`` + ``t_meta.json``). At predict time two matrix multiplies
   estimate the similarity and a linear ramp turns it into ``a``, so nothing searches the labeled set and
   the labeled fingerprints never leave the run — see :mod:`olinda.tanimoto`.

The end goal is to predict the soft-label distribution informed by the hard labels. Artifacts land under
``<model_dir>/_hard/``. The gate decides *where* to trust ``H``: the blend
``prediction = (1-a)·S + a·G_soft`` leans on the hard signal only near the labeled chemistry, and how far it
can ever lean is capped by ``a_max``, which the head earns from how well its calibrated output reproduces
the teacher's scale (:func:`_blend_ceiling`) — a head that loses to the surrogate earns zero and the model
ships soft-only. All stages are fused into a single ``model.onnx`` (see :mod:`olinda.export`) and served by
:class:`~olinda.artifact.OlindaArtifact`.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from olinda.console import echo, step, summary_panel, sweep_progress
from olinda.featurizer import MorganCountFeaturizer

# Layout under <model_dir>/
HARD_H5_NAME = (
  "hard.h5"  # featurized hard labels written by `prepare --hard-labels`, consumed by `learn-hard`
)
HARD_DIRNAME = "_hard"
HARD_MODEL_SUBDIR = "h"
H_REFERENCE_NAME = "h_reference.h5"  # H's score for every reference-library compound (row-aligned)
H_TO_S_NAME = "h_to_s.json"  # the isotonic map carrying H onto S's scale, producing H_S
TANIMOTO_DIRNAME = "tanimoto"  # the similarity regressor gating the hard signal
HARD_META_NAME = "hard_meta.json"
HARD_EVAL_NAME = "hard_eval.json"


def has_hard_head(model_dir: str | Path) -> bool:
  """True iff *model_dir* has a **complete** hard-label head.

  ``learn-hard`` writes `H` first and its metadata last, with several minutes of reference scoring in
  between, so the presence of the model says only that the step *started*. Interrupt it and the
  column would claim a head with no calibrator and no gate — which then fails the fuse with a missing
  file rather than simply being treated as soft-only. The metadata is written last, so it is the
  completion marker.
  """
  return (Path(model_dir) / HARD_DIRNAME / HARD_META_NAME).exists()


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


def _new_hard_model(task: str):
  """The hard-label model H. Binary → a plain, portfolio-selected XGBoost classifier (regression raises).

  We use lazy-qsar only for its portfolio/preset selection and booster training — ``calibrated=False`` turns
  OFF lazy-qsar's internal probability calibrator, so ``H`` outputs raw ``predict_proba``. This keeps the
  reference scoring, prediction, and the exported ONNX all on the same raw probability (no hidden calibration
  step), which is what makes ``H``'s ONNX faithful. The teacher-scale mapping is our own separate ``h_to_s``
  isotonic calibrator (step 3). Continuous hard labels would need a regressor — a planned placeholder.
  """
  from lazyqsar.base.xgboost import BaseXGBClassifier

  if task == "binary":
    return BaseXGBClassifier(calibrated=False)
  raise NotImplementedError(
    f"hard-label task {task!r} is not supported yet — only binary (lazy-qsar XGBoost classifier). "
    "A regressor path is a planned placeholder."
  )


def _selection_report(hard_model) -> dict:
  """Read lazy-qsar's own model-selection diagnostics off a fitted estimator.

  ``BaseXGBClassifier`` / ``BaseXGBRegressor`` pick their config against an internal validation split
  during ``fit`` and expose the winning preset, its boosting rounds, and the portfolio scores.
  """
  scores = getattr(hard_model, "portfolio_scores_", {}) or {}
  preset = getattr(hard_model, "preset_name_", None)
  return {
    "preset": preset,
    "best_iteration": int(getattr(hard_model, "best_iteration_", -1)),
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
      "learn-hard needs it to score H across the reference and calibrate to the soft labels."
    )
  return path


def _score_reference(model, task: str, n_features: int, matrix, chunk: int = 50_000):
  """Score H over every reference compound, aligned to erl0_morgan row order.

  Reads from the shared in-RAM matrix rather than reopening the library, so a multi-column run pays
  for one load instead of one per scan per column.
  """
  # 50k rows x 2048 float32 is ~410 MB per chunk; 200k was 1.64 GB for the same throughput.
  n, dim = matrix.n_rows, matrix.n_cols
  if dim != n_features:
    raise ValueError(f"reference library has {dim}-d features but H expects {n_features}-d")
  out = []
  with sweep_progress("scoring", n) as tick:
    for start in range(0, n, chunk):
      xb = np.asarray(matrix.x[start : start + chunk], dtype=np.float32)
      g = np.asarray(model.predict_proba(xb))[:, 1] if task == "binary" else np.asarray(model.predict(xb))
      out.append(np.asarray(g, dtype=np.float32).ravel())
      tick(min(start + chunk, n))
  return np.concatenate(out)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  ac, bc = a - a.mean(), b - b.mean()
  d = float(np.sqrt((ac**2).sum()) * np.sqrt((bc**2).sum()))
  return float("nan") if d == 0 else float((ac * bc).sum() / d)


def _blend_ceiling(alignment_r2: float) -> float:
  """Cap the blend weight by how closely the calibrated hard signal reproduces the teacher's scale.

  Takes **R²**, not a correlation. Pearson is invariant to affine transformation, so a signal that is
  systematically shifted or rescaled would score 1.0 while dragging the blend off-centre; R² is
  computed against the identity line, so bias and scale error both reduce it. That makes it a measure
  of 1-to-1 agreement rather than of covariation, and ``r² - R²`` is exactly the misalignment penalty
  (``R² <= r²`` always, with equality iff the signal is already its own best affine rescaling).

  Lin's concordance is the textbook agreement statistic and is deliberately *not* used: an MSE-optimal
  predictor is always under-dispersed relative to its target — a property of conditional means — and
  CCC penalises precisely that shrinkage, so it would mark down a correctly calibrated head and reward
  an over-confident one.

  A spurious ``H`` leaves isotonic regression nothing to fit, so the map comes out nearly flat, the
  calibrated signal nearly constant, and R² collapses — switching the blend off, which is the property
  worth having. Below zero the signal is worse than predicting the teacher's mean, so the blend is
  disabled rather than inverted. Non-finite (a constant teacher column, a hard set too small for the
  fit to mean anything) is treated the same way.

  Anything under :data:`~olinda.tanimoto.A_MIN` is also dropped to zero: a few percent of a weakly
  aligned signal cannot shift a prediction enough to be worth the chance of shifting it the wrong way,
  so such a model ships soft-only instead of carrying a token hard branch.

  Two limits, both general rather than dataset-specific:

  * R² here is measured on the reference library, which is the data the isotonic map was fitted to —
    an in-sample estimate, so the ceiling errs high.
  * Agreement with the *teacher* is necessary but not sufficient. It cannot separate a head that is
    genuinely better than the teacher (and so disagrees) from one that is simply worse, nor catch a
    head that tracks the teacher acceptably while losing to the surrogate at predicting real labels.
    Comparing the head out-of-fold against the surrogate is what answers that. Until then this only
    ever lowers the ceiling, never raises it.
  """
  from olinda.tanimoto import A_CEILING, A_MIN

  r2 = float(alignment_r2)
  if not np.isfinite(r2):
    return 0.0
  ceiling = min(A_CEILING, max(0.0, r2))
  return float(ceiling) if ceiling >= A_MIN else 0.0


def _fit_tanimoto(
  hard_bits, n_features, matrix, sim_lo: float, sim_hi: float, alignment_r2: float, chunk: int = 50_000
):
  """Learn T: regress each reference compound's 1-NN Tanimoto to the labelled set.

  Streams the library once, computing the exact similarity per chunk (:func:`~olinda.tanimoto.tanimoto_nn`)
  and keeping it as a continuous target, then fits a small MLP on the same binarised Morgan features the
  rest of the pipeline uses. The regressor stands in for a nearest-neighbour search at
  predict time: approximate, but it keeps the labelled fingerprints out of the shipped artifact.

  Labelled compounds that are themselves in the library land at similarity 1.0 naturally; those outside
  it are not seen by the regressor, which is why its quality is reported against a held-out library slice
  rather than assumed.

  Parameters
  ----------
  alignment_r2 : float
      ``R²(calibrated H_S, soft)`` over the reference library, which caps the blend weight — see
      :func:`_blend_ceiling`.

  Returns
  -------
  tuple
      ``(regressor, stats)`` where ``regressor`` is a :class:`~olinda.tanimoto.TanimotoRegressor`
      and ``stats`` reports the target distribution and the fit's held-out quality.
  """
  from olinda.tanimoto import (
    T_BATCH,
    TanimotoRegressor,
    prepare_hard_bits,
    ramp,
    tanimoto_nn,
  )
  from olinda.metrics import regression_metrics

  hard_prepared = prepare_hard_bits(hard_bits)  # built once, reused for every chunk
  n = matrix.n_rows
  sim = np.empty(n, dtype=np.float32)
  with sweep_progress("scanning", n) as tick:
    for start in range(0, n, chunk):
      stop = min(start + chunk, n)
      bits = (matrix.x[start:stop] > 0).astype(np.float32)
      sim[start:stop] = tanimoto_nn(bits, prepared=hard_prepared)
      tick(stop)

  n_gt = int(np.asarray(hard_bits).shape[0])

  # Hold a slice out for early stopping and for the reported numbers; the net trains on everything
  # else, streamed a batch at a time straight off the resident uint8 library.
  rng = np.random.default_rng(42)
  order = rng.permutation(n)
  n_val = min(50_000, max(1, n // 10))
  val_idx, train_idx = np.sort(order[:n_val]), order[n_val:]

  def batches(shuffler):
    idx = train_idx.copy()
    shuffler.shuffle(idx)
    for start in range(0, len(idx), T_BATCH):
      take = np.sort(idx[start : start + T_BATCH])
      yield matrix.gather(take), sim[take]

  # A ceiling is only meaningful if the gate can ever legitimately open. If nothing in the view reaches
  # the ramp's lower knee, the labelled set has no neighbours here: the target is flat near zero, the
  # gate stays shut everywhere, and a positive ceiling is a number with nothing behind it. That same
  # combination — a_max > 0 with T never opening on any probe molecule — is what
  # `export.build_bundle` refuses to fuse, so leaving it ungated turns a degenerate gate into a failed
  # run. Reachable is the norm on the full library; a subsampled view is where this bites.
  reachable = float(sim.max()) >= sim_lo
  regressor = TanimotoRegressor.fit(
    batches,
    matrix.n_cols,
    (matrix.gather(val_idx), sim[val_idx]),
    echo=echo,
    a_max=_blend_ceiling(alignment_r2) if reachable else 0.0,
    sim_lo=sim_lo,
    sim_hi=sim_hi,
  )

  xval = matrix.gather(val_idx)
  predicted = regressor.predict_tanimoto(xval)
  del xval
  truth = sim[val_idx].astype(np.float64)
  metrics = regression_metrics(truth, predicted)
  # R² on the similarity says how well the net learned its target; what the blend actually depends on
  # is whether the gate *opens* for the compounds that deserve weight, which R² can hide entirely — a
  # gate that is simply always shut still scores respectably. So report the reach as well.
  deserved, opened = ramp(truth) > 0, ramp(predicted) > 0
  hit = int((deserved & opened).sum())
  regressor.metrics = {
    **{k: metrics[k] for k in ("n", "r2", "spearman", "rmse")},
    "recall": float(hit / deserved.sum()) if deserved.any() else float("nan"),
    "precision": float(hit / opened.sum()) if opened.any() else float("nan"),
  }

  stats = {
    "a_max": regressor.a_max,
    "alignment_r2": float(alignment_r2),
    "n_ref": int(n),
    "n_gt": n_gt,
    "n_train": int(len(train_idx)),
    "sim_median": float(np.median(sim)),
    "sim_p99": float(np.quantile(sim, 0.99)),
    "sim_max": float(sim.max()),
    "frac_above_lo": float((sim >= sim_lo).mean()),
    "frac_above_hi": float((sim >= sim_hi).mean()),
    **{f"fit_{k}": v for k, v in regressor.metrics.items()},
  }
  return regressor, stats


MIN_HARD_ROWS = 4
MIN_PER_CLASS = 2  # below this a class-stratified train/val split is impossible


def prepare_hard_labels_wide(
  input_path: str | Path,
  run_dir: str | Path,
  mapping: dict,
  *,
  task: str = "auto",
  smiles_column: str | None = None,
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
  smiles_column : str, optional
      Name of the SMILES column (default: ``smiles``/``input``, else the first column). Which value
      columns are used is already decided by *mapping*.

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

  smiles, values = resolve_smiles_frame(_read_table(input_path), smiles_column=smiles_column)
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
      off_scale = set(np.unique(yc).tolist()) - {0.0, 1.0}
      if off_scale:
        raise ValueError(
          f"hard column '{hard_col}' was forced to --task binary but holds non-binary values "
          f"(e.g. {sorted(off_scale)[:3]}). Casting them would floor everything below 1.0 to a "
          "negative. Threshold the column yourself, or drop --task."
        )
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
        f"hard column '{hard_col}' looks continuous; only binary hard labels is supported today"
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


def train_hard(model_dir: str | Path, soft=None, matrix=None) -> dict:
  """Train the hard-label model ``H`` and calibrate it onto the soft-label scale.

  Reads that column's ``hard.h5`` (written by :func:`prepare_hard_labels_wide`), then writes ``H``
  (ONNX), its scores over the reference library, the ``H``→``S`` calibrator, and ``T``
  (a similarity-regressing MLP) under ``<col_dir>/_hard/``. The gate is learned here from the
  reference but *applied* at predict time (no similarity search) — see :mod:`olinda.tanimoto`.

  Parameters
  ----------
  model_dir : str or Path
      The **column** directory (``columns/<id>/``), which holds that column's ``hard.h5``.
  soft : array-like
      The column's reference-aligned soft labels, read by the caller from the run's ``targets.h5``.
  matrix : ReferenceMatrix, optional
      The shared library. Passed in by ``learn-hard`` so a multi-column run loads it once; loaded here
      if absent.

  Returns
  -------
  dict
      ``{"task", "n", "hard_dir", "selection", "calibration", "tanimoto"}``.
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
      "train_hard needs this column's reference-aligned soft labels; the caller reads them "
      "from the run's targets.h5"
    )
  soft = np.asarray(soft, dtype=np.float64)
  if matrix is None:
    # Standalone call: load the library ourselves. `learn-hard` passes one in so a multi-column run
    # loads it once for the whole run instead of twice per column.
    from olinda.data.matrix import ReferenceMatrix

    matrix = ReferenceMatrix.load(_reference_path())

  hard_root = model_dir / HARD_DIRNAME
  hard_dir = hard_root / HARD_MODEL_SUBDIR
  hard_root.mkdir(parents=True, exist_ok=True)

  # --- load the prepared hard labels ---------------------------------------
  with h5py.File(hard_path, "r") as f:
    X = np.asarray(f["x"][:], dtype=np.float32)
    y = np.asarray(f["y"][:], dtype=np.float64)
    resolved_task = str(f.attrs.get("task", "binary"))
    featurizer_json = f.attrs.get("featurizer")
  featurizer_dict = json.loads(featurizer_json) if featurizer_json else MorganCountFeaturizer().to_dict()
  hard_model = _new_hard_model(resolved_task)  # placeholder gate: raises NotImplementedError for regression
  y = y.astype(int)  # binary labels (past the gate)

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # hush lazy-qsar's sklearn version-drift FutureWarnings

    # === Step 1/4 — train the hard-label classifier H ======================
    step(1, 4, "training the hard-label model H")
    echo(f"  {len(y):,} labelled compounds · {int(y.sum()):,} positive ({y.mean():.1%})", "info")
    hard_model.fit(X, y)
    hard_model.save(str(hard_dir))
    selection = _selection_report(hard_model)
    echo(f"  ready · {selection['preset']} preset · {selection['best_iteration']} trees", "info")

    # === Step 2/4 — score H across the reference library ===================
    step(2, 4, "scoring H across the reference library")
    g_ref = _score_reference(hard_model, "binary", X.shape[1], matrix)
    with h5py.File(hard_root / H_REFERENCE_NAME, "w") as f:
      f.create_dataset("g", data=g_ref.astype(np.float32))
    echo(f"  saved → {H_REFERENCE_NAME}", "info")

    # === Step 3/4 — calibrate H onto the soft-label scale ==================
    from olinda.calibrate import IsotonicCalibrator, _spearman_sign

    step(3, 4, "calibrating H onto the soft-label scale")
    # `g_ref` covers the loaded view of the library while `soft` is always the full reference-aligned
    # target, so under `--max-samples` the two differ in length. Pairing the head of each is correct
    # because the limit truncates the head — but state that, rather than leaving a `min()` to absorb
    # any future change to what the flag selects, which would silently mispair rows with labels.
    if len(g_ref) > len(soft):
      raise RuntimeError(
        f"scored {len(g_ref):,} reference rows but the target vector has {len(soft):,} — "
        "the run directory does not match this library"
      )
    gv, sv = g_ref.astype(np.float64), soft[: len(g_ref)]
    mask = np.isfinite(gv) & np.isfinite(sv)
    gv, sv = gv[mask], sv[mask]
    calibrator = IsotonicCalibrator().fit(gv, sv, increasing="auto")
    calibrator.save(hard_root / H_TO_S_NAME)
    direction = "increasing" if calibrator._sign > 0 else "decreasing"
    # fit() already ranked both arrays to choose the direction; reuse that rather than paying for a
    # second full pass over the reference library just to report the magnitude.
    spearman = calibrator.rank_correlation
    if spearman is None:  # only when a caller fixed the direction instead of detecting it
      spearman = _spearman_sign(gv, sv)
    from olinda.metrics import _r2

    calibrated = calibrator.transform(gv)
    pearson_after = _pearson(calibrated, sv)
    # R², not the correlation, is what bounds the blend — see _blend_ceiling. The gap between r² and
    # R² is the part of the disagreement that a shift or a rescaling would explain.
    alignment_r2 = _r2(sv, calibrated)
    echo(f"  {direction} fit · Spearman(H, soft) {spearman:+.3f}", "info")
    # R² is the one that matters downstream — it is what caps the blend — so it is named as such
    # rather than left as the third number in a row of three.
    echo(
      f"  agreement after calibration · Pearson {pearson_after:.3f} · "
      f"[bold]R² {alignment_r2:.3f}[/] [dim]· R² caps the blend weight[/]",
      "info",
    )

    # No plots here. `olinda validate` draws the calibration map from the fused graph itself, so a
    # training-time copy would only ever be a second rendering of the same numbers — and drawing it
    # pulled matplotlib into every training run, which is what made the CLI pause on its font cache.

    # === Step 4/4 — train T, the Tanimoto regressor ========================
    from olinda.tanimoto import A_CEILING, T_HI, T_LO

    step(4, 4, "learning T · Tanimoto to the labelled set")
    hard_bits = (X > 0).astype(np.float32)
    ad_clf, ad_counts = _fit_tanimoto(hard_bits, X.shape[1], matrix, T_LO, T_HI, alignment_r2)
    ad_clf.save(hard_root / TANIMOTO_DIRNAME)
    echo(
      f"  true similarity · median {ad_counts['sim_median']:.3f} · "
      f"{ad_counts['frac_above_lo']:.1%} above {T_LO} · {ad_counts['frac_above_hi']:.1%} above {T_HI}",
      "info",
    )
    echo(
      f"  T fit · R² [bold]{ad_counts['fit_r2']:.3f}[/] · ρ {ad_counts['fit_spearman']:.3f} "
      f"[dim]on {ad_counts['fit_n']:,} held out[/]",
      "info",
    )
    echo(
      f"  T reach · opens for [bold]{ad_counts['fit_recall']:.0%}[/] of the compounds that qualify · "
      f"{ad_counts['fit_precision']:.0%} of those it opens for deserve it",
      "info",
    )
    if ad_counts["a_max"] <= 0.0:
      reason = (
        f"nothing in the scored reference reaches similarity {T_LO} — the labelled compounds have "
        "no neighbours here, so the gate could never open"
        if ad_counts["sim_max"] < T_LO
        else f"R²(calibrated H_S, soft)={alignment_r2:.3f} — the hard head does not reproduce the "
        "teacher's scale well enough to be worth mixing in"
      )
      echo(f"  blend DISABLED · {reason}", "warning")
    else:
      echo(
        f"  blend weight ramps 0 → [bold]{ad_counts['a_max']:.3f}[/] across similarity {T_LO} → "
        f"{T_HI} · ceiling from R²={alignment_r2:.3f}, capped at {A_CEILING}",
        "run",
      )

  # --- persist metadata + eval ---------------------------------------------
  import lazyqsar as _lq

  calibration = {
    "direction": direction,
    "spearman_g_soft": spearman,
    "pearson_calibrated_soft": pearson_after,
    "r2_calibrated_soft": alignment_r2,
    # r² - R² is the share of the disagreement a shift or rescaling would account for; ~0 means the
    # calibrated signal is already 1-to-1 with the teacher rather than merely correlated with it.
    "misalignment_gap": float(pearson_after**2 - alignment_r2)
    if np.isfinite(pearson_after) and np.isfinite(alignment_r2)
    else None,
    "n_reference": int(len(gv)),
  }
  tanimoto = {
    "signal": "similarity_regressor",
    "sim_lo": T_LO,
    "sim_hi": T_HI,
    "artifact": TANIMOTO_DIRNAME,
    "a_max_ceiling": A_CEILING,
    **ad_counts,
  }
  meta = {
    "task": "binary",
    "n": int(len(y)),
    "features": "morgan_count",
    "featurizer": featurizer_dict,
    "featurizer_class": "MorganCountFeaturizer",
    "hard_dir": HARD_MODEL_SUBDIR,
    "h_reference": H_REFERENCE_NAME,
    "calibrator": H_TO_S_NAME,
    # Applicability gate is learned here and applied at predict time (a similarity regressor, no
    # similarity search). Blend: prediction = (1-a)*surrogate + a*h_s.
    "tanimoto": tanimoto,
    "lazyqsar_version": getattr(_lq, "__version__", "unknown"),
  }
  with open(hard_root / HARD_META_NAME, "w") as fp:
    json.dump(meta, fp, indent=2)
  with open(hard_root / HARD_EVAL_NAME, "w") as fp:
    json.dump({"selection": selection, "calibration": calibration, "tanimoto": tanimoto}, fp, indent=2)

  _mark_surrogate_combined(model_dir)

  # --- report ---------------------------------------------------------------
  rows = [
    ("Hard model H", f"[bold]{selection['preset']}[/] · {selection['best_iteration']} trees"),
    ("Reference scored", f"[bold]{len(gv):,}[/] compounds"),
    (
      "H → S calibration",
      f"[bold]{direction}[/] · Spearman {spearman:+.3f} · Pearson(cal) {pearson_after:.3f}",
    ),
    (
      "Applicability",
      f"similarity R² [bold]{ad_counts['fit_r2']:.3f}[/] · "
      f"{ad_counts['frac_above_lo']:.1%} of the library above {T_LO} → [dim]{TANIMOTO_DIRNAME}/[/]",
    ),
    ("Saved", f"[dim]{hard_root}[/]"),
  ]
  summary_panel("olinda · learn-hard", rows, border_style="green", icon="✓")

  return {
    "task": "binary",
    "n": int(len(y)),
    "hard_dir": str(hard_dir),
    "selection": selection,
    "calibration": calibration,
    "tanimoto": tanimoto,
  }


def _mark_surrogate_combined(model_dir: Path) -> None:
  """Flip ``hard: true`` on the surrogate's ``train_meta.json`` if present (the ``_hard/``
  directory is the authoritative marker; this is just a convenience flag)."""
  path = model_dir / "train_meta.json"
  if not path.exists():
    return
  try:
    with open(path) as fp:
      data = json.load(fp)
    data["hard"] = True
    with open(path, "w") as fp:
      json.dump(data, fp, indent=2)
  except (json.JSONDecodeError, OSError):  # pragma: no cover - non-fatal
    pass
