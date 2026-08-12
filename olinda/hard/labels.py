"""Turn a file of your own measurements into the featurized ``hard.h5`` that ``learn-hard`` reads.

This is the `prepare --hard-labels` half of the hard-label path: match the columns, featurize the
SMILES, drop what RDKit could not parse, and refuse early if what is left cannot be split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from olinda.console import echo
from olinda.featurizer import MorganCountFeaturizer
from olinda.hard.layout import HARD_H5_NAME


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

    smiles, values = resolve_smiles_frame(
        _read_table(input_path), smiles_column=smiles_column
    )
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
