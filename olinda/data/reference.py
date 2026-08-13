"""Load teacher calculations aligned to the reference library.

``--soft-labels`` is a file with the teacher model's value for every molecule in the reference
library, in the same order as ``erl0_morgan.h5``: a SMILES column — named ``smiles`` or ``input``,
else column 0 — followed by the value columns, whatever they are called. This module reads that
file, verifies the SMILES line up with the HDF5 row order, and returns one value vector per
column. Callers that know better can name the columns outright rather than rely on the
convention; :func:`smiles_column_index` and :func:`select_value_columns` are where that is decided.
"""

from pathlib import Path

import numpy as np

from olinda.console import echo
from olinda.utils.logging import logger

_VERIFY_CHUNK = 100_000
_SMILES_NAMES = ("smiles", "input")

MAX_COLUMNS = 10
"""Most value columns a soft-label file may carry.

Each column becomes an independent student fused into one ``model.onnx``. At ~60 MB per column that
keeps the bundle around 600 MB, well under ONNX's 2 GB protobuf ceiling (~35 columns).
"""

_HARD_SEPARATORS = ("_", "-")


def check_column_budget(columns) -> None:
    """Raise if a soft-label file carries more value columns than :data:`MAX_COLUMNS`."""
    if len(columns) > MAX_COLUMNS:
        raise ValueError(
            f"soft-label file has {len(columns)} value columns; the maximum is {MAX_COLUMNS}. "
            f"Drop columns before distilling. Found: {list(columns)}"
        )


def match_hard_columns(soft_columns, hard_columns) -> dict:
    """Map each hard-label column onto the soft column it provides hard labels for.

    A hard column matches a soft column by exact name, or — because teacher outputs are often the same
    name with a suffix (``abaumannii_inhibition`` → ``abaumannii_inhibition_probability``) — by being a
    prefix of exactly one soft column, up to a ``_`` or ``-`` separator. Requiring the separator stops
    ``tox`` from silently matching ``toxicity_probability``.

    Parameters
    ----------
    soft_columns : sequence of str
        Value columns of the soft-label file.
    hard_columns : sequence of str
        Value columns of the hard-label file.

    Returns
    -------
    dict
        ``{hard_column: soft_column}``. Soft columns with no hard counterpart are simply absent, and
        stay soft-only.

    Raises
    ------
    ValueError
        If a hard column matches no soft column, matches more than one, or if two hard columns claim
        the same soft column.
    """
    soft = [str(c) for c in soft_columns]
    mapping: dict[str, str] = {}
    problems: list[str] = []

    for raw in hard_columns:
        hard = str(raw)
        if hard in soft:
            mapping[hard] = hard
            continue
        candidates = [
            s for s in soft if any(s.startswith(hard + sep) for sep in _HARD_SEPARATORS)
        ]
        if len(candidates) == 1:
            mapping[hard] = candidates[0]
        elif not candidates:
            problems.append(f"hard column '{hard}' matches no soft column")
        else:
            problems.append(f"hard column '{hard}' is ambiguous — matches {candidates}")

    claimed: dict[str, str] = {}
    for hard, target in mapping.items():
        if target in claimed:
            problems.append(
                f"soft column '{target}' claimed by both '{claimed[target]}' and '{hard}'"
            )
        claimed[target] = hard

    if problems:
        raise ValueError(
            "could not align hard labels with soft labels:\n  - "
            + "\n  - ".join(problems)
            + f"\nsoft columns: {soft}\nhard columns: {[str(c) for c in hard_columns]}"
        )
    return mapping


def smiles_column_index(df, name: str | None = None) -> int | None:
    """Index of the SMILES column, or ``None`` when there is nothing to go on.

    The one place that decides what a SMILES column is called. Ersilia and Isaura write
    ``(key, input, …)``, everything else tends to write ``smiles``, and both are accepted
    everywhere olinda reads molecules — see :data:`_SMILES_NAMES`. Matching is case-insensitive and
    ignores surrounding whitespace.

    Parameters
    ----------
    df : pandas.DataFrame
        The table to inspect.
    name : str, optional
        Take this column as the SMILES instead of guessing. Naming a column that isn't there is an
        error, not a fallback — the caller asked for a specific column and silently using another
        would train or score on the wrong molecules.

    Returns
    -------
    int or None
        Position of the SMILES column. ``None`` only when *name* is ``None`` and no column carries a
        recognised name; callers decide what that means (value-bearing files fall back to column 0,
        while `predict` asks for a named column outright).
    """
    lower = [str(c).strip().lower() for c in df.columns]
    if name is not None:
        wanted = str(name).strip().lower()
        if wanted not in lower:
            raise ValueError(
                f"no column named '{name}'; available={[str(c) for c in df.columns]}"
            )
        return lower.index(wanted)
    return next((i for i, name_ in enumerate(lower) if name_ in _SMILES_NAMES), None)


def select_value_columns(values, columns=None, *, what: str = "value"):
    """Narrow a frame of value columns to *columns*, in the order asked for.

    Parameters
    ----------
    values : pandas.DataFrame
        Candidate value columns, as returned by :func:`resolve_smiles_frame`.
    columns : sequence of str, optional
        Keep only these, in this order. ``None`` keeps everything.
    what : str, default "value"
        Noun used in the error message, so it names the file the user actually passed.

    Returns
    -------
    pandas.DataFrame
        The selected columns. Asking for a column that isn't there raises rather than quietly
        yielding a narrower model than requested.
    """
    values.columns = [str(c) for c in values.columns]
    if columns is None:
        return values
    wanted = [str(c) for c in columns]
    missing = [c for c in wanted if c not in values.columns]
    if missing:
        raise ValueError(
            f"{what} column(s) not found: {missing}; available={list(values.columns)}"
        )
    return values[wanted]


def resolve_smiles_frame(df, *, smiles_column: str | None = None, label_columns=None):
    """Return ``(smiles, values)`` where ``values`` holds the file's value columns.

    Handles the plain ``(smiles, value, …)`` layout and the Ersilia/Isaura ``(key, smiles|input, …)``
    layout: if a column named ``smiles``/``input`` (case-insensitive) exists, we start there — that
    column is the SMILES and **all** columns after it are value columns, so any preceding ``key`` is
    discarded. Otherwise we fall back to positional column 0 = SMILES, columns 1.. = values.

    Parameters
    ----------
    df : pandas.DataFrame
        The table to split.
    smiles_column : str, optional
        Name the SMILES column explicitly instead of relying on the conventions above.
    label_columns : sequence of str, optional
        Keep only these value columns, in this order. Columns are then taken by name from the whole
        table, so a value column sitting *before* the SMILES column is still reachable.

    Returns
    -------
    (np.ndarray, pandas.DataFrame)
        The SMILES vector and a frame of the value columns (in file order unless *label_columns*
        reorders them).
    """
    si = smiles_column_index(df, smiles_column)
    if si is None:
        if df.shape[1] < 2:
            raise ValueError(
                f"expected >=2 columns (smiles, value); got {list(df.columns)}"
            )
        si = 0

    if label_columns is not None:
        # Selecting by name means position no longer matters, so offer every column but the SMILES one.
        candidates = df.drop(df.columns[si], axis=1)
        return df.iloc[:, si].astype(str).to_numpy(), select_value_columns(
            candidates.copy(), label_columns
        )

    if si + 1 >= df.shape[1]:
        raise ValueError(
            f"found SMILES column '{df.columns[si]}' but no value column after it; columns={list(df.columns)}"
        )
    if si > 0:
        echo(
            f"discarding leading column(s) {list(df.columns[:si])}; using '{df.columns[si]}' as SMILES",
            "info",
        )
    smiles = df.iloc[:, si].astype(str).to_numpy()
    values = df.iloc[:, si + 1 :].copy()
    return smiles, values


def _read_table(path: str | Path):
    """Read a CSV/TSV/Parquet file into a pandas DataFrame (format inferred from the suffix)."""
    import pandas as pd

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    sep = "\t" if suffix == ".tsv" else ","
    return pd.read_csv(path, sep=sep)


def _verify_smiles_alignment(smiles, descriptors_h5: str | Path) -> int:
    """Verify ``smiles`` line up row-for-row with the ``input`` dataset of ``descriptors_h5``.

    Enforces the "exactly the same molecules, in the same order" contract: any length or order
    mismatch raises. Returns the reference-library row count on success.
    """
    import h5py

    smiles = np.asarray(smiles).astype("U")
    with h5py.File(str(descriptors_h5), "r") as f:
        ref = f[
            "input"
        ].asstr()  # bulk-decoded str view of the variable-length UTF-8 dataset
        n = int(f["input"].shape[0])
        if len(smiles) != n:
            raise ValueError(
                f"calculations file has {len(smiles)} rows but the descriptors library has {n}. "
                "It must contain exactly the same molecules, in the same order."
            )
        # Vectorized chunked comparison (no per-row Python loop over 1.35M rows).
        for start in range(0, n, _VERIFY_CHUNK):
            end = min(start + _VERIFY_CHUNK, n)
            block = np.asarray(ref[start:end], dtype="U")
            mism = np.flatnonzero(block != smiles[start:end])
            if mism.size:
                k = start + int(mism[0])
                raise ValueError(
                    f"SMILES mismatch at row {k}: calculations file has '{smiles[k]}' "
                    f"but the library has '{block[int(mism[0])]}'. The files must be row-aligned."
                )
    logger.debug(f"Calculations verified against {n} library molecules")
    return n


def load_reference_calcs_frame(
    path: str | Path,
    descriptors_h5: str | Path,
    columns=None,
    *,
    smiles_column: str | None = None,
) -> tuple[list[str], dict]:
    """Read a multi-column calculations file, verify row-alignment once, return selected columns.

    A *calculations* file carries one or more teacher value columns over the reference library, all
    row-aligned to ``descriptors_h5`` (the alignment is checked a single time, not per column).

    Parameters
    ----------
    path : str or Path
        CSV/TSV/Parquet: an optional leading ``key`` column, a ``smiles``/``input`` column, then one
        or more value columns.
    descriptors_h5 : str or Path
        Reference-library HDF5 (its ``input`` dataset is the source of truth for row order).
    columns : list of str, optional
        Value columns to return (default: all of them). Order is preserved.
    smiles_column : str, optional
        Name of the SMILES column (default: ``smiles``/``input``, else the first column).

    Returns
    -------
    (list of str, dict of str -> np.ndarray)
        ``(all_value_columns, selected)`` — the full ordered list of value-column names in the file
        (for positional hard-label matching later) and a mapping of each *selected* column to its
        float32 value vector.
    """
    smiles, values = resolve_smiles_frame(
        _read_table(path), smiles_column=smiles_column
    )
    all_cols = [str(c) for c in values.columns]
    values.columns = all_cols
    if columns is None:
        selected = list(all_cols)
    else:
        selected = [str(c) for c in columns]
        missing = [c for c in selected if c not in all_cols]
        if missing:
            raise ValueError(
                f"requested columns not found in calculations file: {missing}; available={all_cols}"
            )
    logger.debug(
        f"Calculations: {len(all_cols)} value column(s) {all_cols}; distilling {selected}"
    )
    _verify_smiles_alignment(smiles, descriptors_h5)
    return all_cols, {c: values[c].to_numpy(dtype=np.float32) for c in selected}
