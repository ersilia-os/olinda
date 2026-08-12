"""T, the gate: how near your labelled chemistry a query is, and how far the head may lean.

Two decisions live here. :func:`_fit_tanimoto` fits ``T``, the small network predicting a compound's
1-NN Tanimoto to the labelled set from its fingerprint alone — which is what keeps your compounds out
of the shipped model. :func:`_blend_ceiling` decides ``a_max``, the most weight the hard head is ever
allowed, earned from how well its calibrated output tracks the teacher.
"""

from __future__ import annotations

import numpy as np

from olinda.console import epoch_progress, sweep_progress


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
    hard_bits,
    n_features,
    matrix,
    sim_lo: float,
    sim_hi: float,
    alignment_r2: float,
    chunk: int = 50_000,
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
    from olinda.metrics import regression_metrics
    from olinda.tanimoto import (
        T_BATCH,
        T_MAX_EPOCHS,
        TanimotoRegressor,
        prepare_hard_bits,
        ramp,
        tanimoto_nn,
    )

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
    with epoch_progress("learning T", T_MAX_EPOCHS) as report:
        regressor = TanimotoRegressor.fit(
            batches,
            matrix.n_cols,
            (matrix.gather(val_idx), sim[val_idx]),
            progress=report,
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
