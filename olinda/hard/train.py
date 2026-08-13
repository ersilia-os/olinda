"""The four steps of ``learn-hard``: train H, score it over the library, calibrate it, gate it.

See :mod:`olinda.hard` for what each step is for and why. This module is the orchestration; the gate
itself lives in :mod:`olinda.hard.gate` and the label preparation in :mod:`olinda.hard.labels`.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from olinda.console import echo, step, summary_panel, sweep_progress
from olinda.console import path as cpath
from olinda.featurizer import MorganCountFeaturizer
from olinda.hard.gate import _fit_tanimoto, _pearson
from olinda.hard.layout import (
    H_REFERENCE_NAME,
    H_TO_S_NAME,
    HARD_DIRNAME,
    HARD_EVAL_NAME,
    HARD_H5_NAME,
    HARD_META_NAME,
    HARD_MODEL_SUBDIR,
    TANIMOTO_DIRNAME,
)


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
        raise ValueError(
            f"reference library has {dim}-d features but H expects {n_features}-d"
        )
    out = []
    with sweep_progress("scoring", n) as tick:
        for start in range(0, n, chunk):
            xb = np.asarray(matrix.x[start : start + chunk], dtype=np.float32)
            g = (
                np.asarray(model.predict_proba(xb))[:, 1]
                if task == "binary"
                else np.asarray(model.predict(xb))
            )
            out.append(np.asarray(g, dtype=np.float32).ravel())
            tick(min(start + chunk, n))
    return np.concatenate(out)


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
    featurizer_dict = (
        json.loads(featurizer_json)
        if featurizer_json
        else MorganCountFeaturizer().to_dict()
    )
    hard_model = _new_hard_model(
        resolved_task
    )  # placeholder gate: raises NotImplementedError for regression
    y = y.astype(int)  # binary labels (past the gate)

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )  # hush lazy-qsar's sklearn version-drift FutureWarnings

        # === Step 1/4 — train the hard-label classifier H ======================
        step(1, 4, "training the hard-label model H")
        echo(
            f"{len(y):,} labelled compounds · {int(y.sum()):,} positive ({y.mean():.1%})",
            "info",
            sub=True,
        )
        hard_model.fit(X, y)
        hard_model.save(str(hard_dir))
        selection = _selection_report(hard_model)
        echo(
            f"ready · {selection['preset']} preset · {selection['best_iteration']:,} trees",
            "info",
            sub=True,
        )

        # === Step 2/4 — score H across the reference library ===================
        step(2, 4, "scoring H across the reference library")
        g_ref = _score_reference(hard_model, "binary", X.shape[1], matrix)
        with h5py.File(hard_root / H_REFERENCE_NAME, "w") as f:
            f.create_dataset("g", data=g_ref.astype(np.float32))
        echo(f"saved → {H_REFERENCE_NAME}", "info", sub=True)

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
        if (
            spearman is None
        ):  # only when a caller fixed the direction instead of detecting it
            spearman = _spearman_sign(gv, sv)
        from olinda.metrics import _r2

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

        # No plots here. `olinda validate` draws the calibration map from the fused graph itself, so a
        # training-time copy would only ever be a second rendering of the same numbers — and drawing it
        # pulled matplotlib into every training run, which is what made the CLI pause on its font cache.

        # === Step 4/4 — train T, the Tanimoto regressor ========================
        from olinda.tanimoto import A_CEILING, T_HI, T_LO

        step(4, 4, "learning T · Tanimoto to the labelled set")
        hard_bits = (X > 0).astype(np.float32)
        ad_clf, ad_counts = _fit_tanimoto(
            hard_bits, X.shape[1], matrix, T_LO, T_HI, alignment_r2
        )
        ad_clf.save(hard_root / TANIMOTO_DIRNAME)
        echo(
            f"true similarity · median {ad_counts['sim_median']:.3f} · "
            f"{ad_counts['frac_above_lo']:.1%} above {T_LO} · {ad_counts['frac_above_hi']:.1%} above {T_HI}",
            "info",
            sub=True,
        )
        echo(
            f"T fit · R² [bold]{ad_counts['fit_r2']:.3f}[/] · ρ {ad_counts['fit_spearman']:.3f} "
            f"[dim]on {ad_counts['fit_n']:,} held out[/]",
            "info",
            sub=True,
        )
        echo(
            f"T reach · opens for [bold]{ad_counts['fit_recall']:.0%}[/] of the compounds that qualify · "
            f"{ad_counts['fit_precision']:.0%} of those it opens for deserve it",
            "info",
            sub=True,
        )
        if ad_counts["a_max"] <= 0.0:
            reason = (
                f"nothing in the scored reference reaches similarity {T_LO} — the labelled compounds have "
                "no neighbours here, so the gate could never open"
                if ad_counts["sim_max"] < T_LO
                else f"R²(calibrated H_S, soft)={alignment_r2:.3f} — the hard head does not reproduce the "
                "teacher's scale well enough to be worth mixing in"
            )
            # No leading indent on either of these: they are the step's conclusion, not one of its
            # details, and `echo` supplies the glyph. Indenting them while keeping the ▪/⚠ that mark a
            # top-level line is what made this read as a step nudged out of column.
            echo(f"blend DISABLED · {reason}", "warning")
        else:
            echo(
                f"blend weight ramps 0 → [bold]{ad_counts['a_max']:.3f}[/] across similarity {T_LO} → "
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
        json.dump(
            {"selection": selection, "calibration": calibration, "tanimoto": tanimoto},
            fp,
            indent=2,
        )

    _mark_surrogate_combined(model_dir)

    # --- report ---------------------------------------------------------------
    rows = [
        (
            "Hard model H",
            f"[bold]{selection['preset']}[/] · {selection['best_iteration']:,} trees",
        ),
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
        ("Saved", f"[dim]{cpath(hard_root)}[/]"),
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
