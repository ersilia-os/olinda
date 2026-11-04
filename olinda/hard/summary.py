"""What ``learn-hard`` tells the user and records: console summaries and the ``hard_meta.json`` payloads."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from olinda.console import echo, summary_panel
from olinda.console import path as cpath
from olinda.hard.layout import (
    H_REFERENCE_NAME,
    H_TO_S_NAME,
    HARD_EVAL_NAME,
    HARD_META_NAME,
    HARD_MODEL_SUBDIR,
    TANIMOTO_DIRNAME,
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


def _echo_label_summary(task: str, y: np.ndarray, y_original) -> None:
    """Say what H is about to learn from: how many labels, and their balance or range."""
    if task == "binary":
        echo(
            f"{len(y):,} labelled compounds · {int(y.sum()):,} positive ({y.mean():.1%})",
            "info",
            sub=True,
        )
    else:
        echo(
            f"{len(y):,} labelled compounds · values {y_original.min():.4g} to {y_original.max():.4g} "
            f"(median {np.median(y_original):.4g})",
            "info",
            sub=True,
        )


def _echo_tanimoto(ad_counts: dict, alignment_r2: float, task: str) -> None:
    """Report T's similarity target, its held-out fit and reach, and what blend weight it earned."""
    from olinda.tanimoto import A_CEILING, A_CEILING_REGRESSION, T_HI, T_LO

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
            f"{T_HI} · ceiling from R²={alignment_r2:.3f}, capped at "
            f"{A_CEILING_REGRESSION if task == 'regression' else A_CEILING}",
            "run",
        )


def _tanimoto_record(ad_counts: dict, task: str) -> dict:
    """The gate's entry in ``hard_meta.json`` and ``hard_eval.json``: its ramp, ceiling and fit statistics."""
    from olinda.tanimoto import A_CEILING, A_CEILING_REGRESSION, T_HI, T_LO

    return {
        "signal": "similarity_regressor",
        "sim_lo": T_LO,
        "sim_hi": T_HI,
        "artifact": TANIMOTO_DIRNAME,
        "a_max_ceiling": A_CEILING_REGRESSION if task == "regression" else A_CEILING,
        **ad_counts,
    }


def _write_hard_meta(
    hard_root: Path, head: dict, selection: dict, calibration: dict, tanimoto: dict
) -> None:
    """Write ``hard_meta.json`` (last, because it is what says the head is complete) and ``hard_eval.json``."""
    import lazyqsar as _lq

    meta = {
        **{k: head[k] for k in ("task", "n", "label_scale")},
        "features": "qmfp",
        "fingerprint": head["fingerprint"],
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


def _print_summary(
    hard_root: Path,
    selection: dict,
    calibration: dict,
    ad_counts: dict,
) -> None:
    """The closing panel of ``learn-hard``: what was trained, calibrated, gated and where it was saved."""
    from olinda.tanimoto import T_LO

    rows = [
        (
            "Hard model H",
            f"[bold]{selection['preset']}[/] · {selection['best_iteration'] + 1:,} trees",
        ),
        ("Reference scored", f"[bold]{calibration['n_reference']:,}[/] compounds"),
        (
            "H → S calibration",
            f"[bold]{calibration['direction']}[/] · Spearman {calibration['spearman_g_soft']:+.3f} · Pearson(cal) {calibration['pearson_calibrated_soft']:.3f}",
        ),
        (
            "Applicability",
            f"similarity R² [bold]{ad_counts['fit_r2']:.3f}[/] · "
            f"{ad_counts['frac_above_lo']:.1%} of the library above {T_LO} → [dim]{TANIMOTO_DIRNAME}/[/]",
        ),
        ("Saved", f"[dim]{cpath(hard_root)}[/]"),
    ]
    summary_panel("olinda · learn-hard", rows, border_style="green", icon="✓")


def _calibration_record(
    direction: str,
    spearman: float,
    pearson_after: float,
    alignment_r2: float,
    n_calibrated: int,
) -> dict:
    """The calibration's entry in ``hard_eval.json``: how H lines up with the teacher once mapped onto its scale."""
    return {
        "direction": direction,
        "spearman_g_soft": spearman,
        "pearson_calibrated_soft": pearson_after,
        "r2_calibrated_soft": alignment_r2,
        # r² - R² is the share of the disagreement a shift or rescaling would account for; ~0 means the
        # calibrated signal is already 1-to-1 with the teacher rather than merely correlated with it.
        "misalignment_gap": float(pearson_after**2 - alignment_r2)
        if np.isfinite(pearson_after) and np.isfinite(alignment_r2)
        else None,
        "n_reference": n_calibrated,
    }
