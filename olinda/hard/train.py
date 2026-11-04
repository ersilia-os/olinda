"""The four steps of ``learn-hard``: train H, score it over the library, calibrate it, gate it.

See :mod:`olinda.hard` for what each step is for and why. This module is the orchestration; the gate
itself lives in :mod:`olinda.hard.gate` and the label preparation in :mod:`olinda.hard.labels`.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from olinda.console import echo, step, sweep_progress
from olinda.featurizer import default_spec
from olinda.hard.gate import _fit_tanimoto, _pearson
from olinda.hard.layout import (
    H_REFERENCE_NAME,
    H_TO_S_NAME,
    HARD_DIRNAME,
    HARD_H5_NAME,
    HARD_META_NAME,
    HARD_MODEL_SUBDIR,
    TANIMOTO_DIRNAME,
)
from olinda.hard.regression import (
    _check_regressor_export,
    _label_scale,
)
from olinda.hard.summary import (
    _calibration_record,
    _echo_label_summary,
    _echo_tanimoto,
    _print_summary,
    _selection_report,
    _tanimoto_record,
    _write_hard_meta,
)


def _new_hard_model(task: str):
    """The hard-label model H: a plain, portfolio-selected XGBoost classifier (binary) or regressor (continuous).

    We use lazy-qsar only for its portfolio/preset selection and booster training — ``calibrated=False`` turns
    OFF lazy-qsar's internal probability calibrator, so ``H`` outputs raw ``predict_proba``. This keeps the
    reference scoring, prediction, and the exported ONNX all on the same raw probability (no hidden calibration
    step), which is what makes ``H``'s ONNX faithful. The teacher-scale mapping is our own separate ``h_to_s``
    isotonic calibrator (step 3). A continuous head is lazy-qsar's ``BaseXGBRegressor``, which has no
    calibrator to switch off: its output is the booster's prediction in the labels' own units.
    """
    from lazyqsar.base.xgboost import BaseXGBClassifier, BaseXGBRegressor

    if task == "binary":
        return BaseXGBClassifier(calibrated=False)
    if task == "regression":
        return BaseXGBRegressor()
    raise ValueError(f"unknown hard-label task {task!r}")


def _reference_path() -> Path:
    """Path to the reference-library Morgan HDF5; raise a friendly error if it isn't present."""
    from olinda.data.fetch import MORGAN_FINGERPRINTS_FILENAME, OLINDA_HOME

    path = Path(OLINDA_HOME) / MORGAN_FINGERPRINTS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"reference library {path} not found — run `olinda setup` (or scripts/compute_qmfp_library.py). "
            "learn-hard needs it to score H across the reference and calibrate to the soft labels."
        )
    return path


def _score_reference(
    model, task: str, n_features: int, matrix, chunk: int = 50_000, artifact=None
):
    """Score H over every reference compound, aligned to the reference library's row order.

    Reads from the shared in-RAM matrix rather than reopening the library, so a multi-column run pays
    for one load instead of one per scan per column.
    """
    # 50k rows x 2048 float32 is ~410 MB per chunk; 200k was 1.64 GB for the same throughput.
    n, dim = matrix.n_rows, matrix.n_cols
    if dim != n_features:
        raise ValueError(
            f"reference library has {dim}-d features but H expects {n_features}-d"
        )
    # lazy-qsar's predict_proba builds an xgb.DMatrix per call, which costs ~3x the prediction itself.
    # With calibration off it is exactly the booster up to best_iteration_, so ask the booster directly.
    booster = getattr(model, "booster_", None)
    direct = (
        task == "binary"
        and booster is not None
        and not hasattr(model, "calibrator_")
        and hasattr(model, "best_iteration_")
    )
    out = []
    with sweep_progress("scoring", n) as tick:
        for start in range(0, n, chunk):
            xb = np.asarray(matrix.x[start : start + chunk], dtype=np.float32)
            g = _score_chunk(model, task, xb, artifact, booster if direct else None)
            out.append(np.asarray(g, dtype=np.float32).ravel())
            tick(min(start + chunk, n))
    return np.concatenate(out)


def _score_chunk(model, task: str, xb: np.ndarray, artifact, booster):
    """Score one chunk of the library: through the shipped ONNX, the raw booster, or lazy-qsar's own predict.

    Guard clauses in order of preference; ``booster`` is passed only when the direct route is safe (see
    :func:`_score_reference`).
    """
    if artifact is not None:
        return artifact.run(xb)
    if booster is not None:
        return booster.inplace_predict(
            xb, iteration_range=(0, int(model.best_iteration_) + 1)
        )
    if task == "binary":
        return np.asarray(model.predict_proba(xb))[:, 1]
    return np.asarray(model.predict(xb))


def _load_hard_h5(hard_path: Path):
    """Read a column's prepared ``hard.h5``: features, raw labels, the task, and the fingerprint spec."""
    import h5py

    with h5py.File(hard_path, "r") as f:
        X = np.asarray(f["x"][:], dtype=np.float32)
        y = np.asarray(f["y"][:], dtype=np.float64)
        resolved_task = str(f.attrs.get("task", "binary"))
        fingerprint_json = f.attrs.get("fingerprint")
    fingerprint = json.loads(fingerprint_json) if fingerprint_json else default_spec()
    return X, y, resolved_task, fingerprint


def _calibrate_to_soft(g_ref: np.ndarray, soft: np.ndarray, hard_root: Path):
    """Fit and save the isotonic ``H -> S`` map, and report how well the calibrated head tracks the teacher.

    Returns ``(direction, spearman, pearson_after, alignment_r2, n_calibrated)``. R², not the correlation,
    is what later bounds the blend, so it is the number carried forward.
    """
    from olinda.calibrate import IsotonicCalibrator, _spearman_sign
    from olinda.metrics import _r2

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
    if (
        spearman is None
    ):  # only when a caller fixed the direction instead of detecting it
        spearman = _spearman_sign(gv, sv)

    calibrated = calibrator.transform(gv)
    pearson_after = _pearson(calibrated, sv)
    # R², not the correlation, is what bounds the blend — see _blend_ceiling. The gap between r² and
    # R² is the part of the disagreement that a shift or a rescaling would explain.
    alignment_r2 = _r2(sv, calibrated)
    echo(f"{direction} fit · Spearman(H, soft) {spearman:+.3f}", "info", sub=True)
    # R² is the one that matters downstream — it is what caps the blend — so it is named as such
    # rather than left as the third number in a row of three.
    echo(
        f"agreement after calibration · Pearson {pearson_after:.3f} · "
        f"[bold]R² {alignment_r2:.3f}[/] [dim]· R² caps the blend weight[/]",
        "info",
        sub=True,
    )
    return direction, spearman, pearson_after, alignment_r2, int(len(gv))


def _check_inputs(model_dir: Path, hard_path: Path, soft, matrix):
    """Refuse a column that is not ready for ``learn-hard``, and resolve the soft labels and the library.

    Returns ``(soft, matrix)``: the soft labels as float64, and the shared library, loaded here only when
    the caller did not pass one.
    """
    if not hard_path.exists():
        raise FileNotFoundError(
            f"no {HARD_H5_NAME} in {model_dir} — run `olinda prepare -s <teacher> --hard-labels <file> -m <run dir>` first"
        )
    # hard_meta.json is written last and is what says "this head is complete", so an interrupted re-run
    # must not leave the previous one behind next to half-new files.
    (model_dir / HARD_DIRNAME / HARD_META_NAME).unlink(missing_ok=True)
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
    return soft, matrix


def _fit_head(hard_model, task: str, X, y, y_original, hard_dir: Path):
    """Step 1: fit and save H; for a continuous head also load its ONNX back and check it follows the booster.

    Returns ``(artifact, selection)``; ``artifact`` is ``None`` for a binary head, which is scored directly
    off the booster.
    """
    step(1, 4, "training the hard-label model H")
    _echo_label_summary(task, y, y_original)
    hard_model.fit(X, y)
    hard_model.save(str(hard_dir))
    artifact = None
    if task == "regression":
        from lazyqsar.base.xgboost import BaseXGBArtifact

        artifact = BaseXGBArtifact.load(str(hard_dir))
        _check_regressor_export(hard_model, artifact, X)
    selection = _selection_report(hard_model)
    echo(
        f"ready · {selection['preset']} preset · {selection['best_iteration'] + 1:,} trees",
        "info",
        sub=True,
    )
    return artifact, selection


def _fit_gate(X, matrix, hard_root: Path, task: str, alignment_r2: float) -> dict:
    """Step 4: learn T, save it, and report it. Returns the fit statistics."""
    from olinda.tanimoto import A_CEILING_REGRESSION, T_HI, T_LO

    step(4, 4, "learning T · Tanimoto to the labelled set")
    hard_bits = (X > 0).astype(np.float32)
    ad_clf, ad_counts = _fit_tanimoto(
        hard_bits,
        X.shape[1],
        matrix,
        T_LO,
        T_HI,
        alignment_r2,
        ceiling=A_CEILING_REGRESSION if task == "regression" else None,
    )
    ad_clf.save(hard_root / TANIMOTO_DIRNAME)
    _echo_tanimoto(ad_counts, alignment_r2, task)
    return ad_counts


def _scale_labels(task: str, y: np.ndarray):
    """Prepare labels for training: ints for binary, scaled to order one for continuous.

    Returns ``(y, y_original, label_scale)``; ``y_original`` is ``None`` for a binary head.
    """
    if task == "binary":
        return y.astype(int), None, 1.0
    label_scale = _label_scale(y)
    return y / label_scale, y, label_scale


def _score_and_save(
    hard_model, task: str, n_features: int, matrix, artifact, hard_root: Path
):
    """Step 2: score H across the reference library and keep the scores next to the head."""
    import h5py

    step(2, 4, "scoring H across the reference library")
    g_ref = _score_reference(hard_model, task, n_features, matrix, artifact=artifact)
    with h5py.File(hard_root / H_REFERENCE_NAME, "w") as f:
        f.create_dataset("g", data=g_ref.astype(np.float32))
    echo(f"saved → {H_REFERENCE_NAME}", "info", sub=True)
    return g_ref


def _persist(hard_root, head, selection, calibrated, ad_counts):
    """Build the calibration and gate records and write ``hard_meta.json`` and ``hard_eval.json``.

    Returns ``(calibration, tanimoto)`` for the caller to hand back.
    """
    calibration = _calibration_record(*calibrated)
    tanimoto = _tanimoto_record(ad_counts, head["task"])
    _write_hard_meta(hard_root, head, selection, calibration, tanimoto)
    return calibration, tanimoto


def _run_steps(hard_model, task, X, y, y_original, soft, matrix, hard_root):
    """Run the four steps of ``learn-hard`` in order.

    Returns ``(selection, calibrated, ad_counts)`` where ``calibrated`` is the tuple
    :func:`_calibrate_to_soft` returns. The warnings filter lives here, around all four, because
    lazy-qsar's version-drift warnings can come from any of them.
    """
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )  # hush lazy-qsar's sklearn version-drift FutureWarnings

        # === Step 1/4 — train the hard-label classifier H ======================
        artifact, selection = _fit_head(
            hard_model, task, X, y, y_original, hard_root / HARD_MODEL_SUBDIR
        )

        # === Step 2/4 — score H across the reference library ===================
        g_ref = _score_and_save(
            hard_model, task, X.shape[1], matrix, artifact, hard_root
        )

        # === Step 3/4 — calibrate H onto the soft-label scale ==================
        step(3, 4, "calibrating H onto the soft-label scale")
        calibrated = _calibrate_to_soft(g_ref, soft, hard_root)
        alignment_r2 = calibrated[3]

        # No plots here. `olinda validate` draws the calibration map from the fused graph itself, so a
        # training-time copy would only ever be a second rendering of the same numbers — and drawing it
        # pulled matplotlib into every training run, which is what made the CLI pause on its font cache.

        # === Step 4/4 — train T, the Tanimoto regressor ========================
        ad_counts = _fit_gate(X, matrix, hard_root, task, alignment_r2)
    return selection, calibrated, ad_counts


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
    model_dir = Path(model_dir)
    hard_path = model_dir / HARD_H5_NAME
    soft, matrix = _check_inputs(model_dir, hard_path, soft, matrix)

    hard_root = model_dir / HARD_DIRNAME
    hard_dir = hard_root / HARD_MODEL_SUBDIR
    hard_root.mkdir(parents=True, exist_ok=True)

    X, y, resolved_task, fingerprint = _load_hard_h5(hard_path)
    hard_model = _new_hard_model(resolved_task)
    y, y_original, label_scale = _scale_labels(resolved_task, y)

    selection, calibrated, ad_counts = _run_steps(
        hard_model, resolved_task, X, y, y_original, soft, matrix, hard_root
    )

    calibration, tanimoto = _persist(
        hard_root,
        {
            "task": resolved_task,
            "n": int(len(y)),
            "label_scale": label_scale,
            "fingerprint": fingerprint,
        },
        selection,
        calibrated,
        ad_counts,
    )

    _mark_surrogate_combined(model_dir)

    _print_summary(hard_root, selection, calibration, ad_counts)

    return {
        "task": resolved_task,
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
