"""Run a distilled model over a file of SMILES.

Reading the input and writing the predictions live here rather than in the CLI so the same path is
available from Python. The model itself is loaded through :class:`~olinda.artifact.OlindaArtifact`,
so a CLI run and a library call produce identical output.

CSV and TSV are read and written with the standard library: predicting needs nothing the model does
not, so pandas belongs to the reporting extra. Parquet is read through pyarrow when it is installed.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


def predict_file(
    model_onnx,
    input_path,
    out_path,
    *,
    smiles_column: str | None = None,
    workers: int | None = 1,
    strict: bool = False,
):
    """Predict via the fused ``model.onnx`` and write a ``smiles`` column plus one column per task.

    Goes through the same :class:`~olinda.artifact.OlindaArtifact` the library exposes, so the CLI and a
    Python caller produce identical output. Loading re-scores the model's golden vectors; a mismatch
    warns, or raises :class:`~olinda.artifact.GoldenMismatch` with *strict*. ``model_onnx`` may be the
    artifact itself or a directory holding one. Pass *smiles_column* to name the input's SMILES column
    instead of auto-detecting it, and *workers* to featurize in parallel (see
    :meth:`~olinda.artifact.OlindaArtifact.run`). Unparseable SMILES are written as empty cells.
    """
    import warnings

    from olinda.artifact import GoldenMismatchWarning, OlindaArtifact
    from olinda.console import echo, success

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = OlindaArtifact(model_onnx, strict=strict)
    for w in caught:  # status belongs to the console, not to a Python warning on stderr
        echo(str(w.message), "warning")
    golden = model.describe()["golden"]
    echo(
        f"RDKit [bold]{model.featurizer_rdkit_version}[/] (WebAssembly) · model featurized with "
        f"{model.rdkit_version} · golden check [bold]{golden}[/]",
        "warning"
        if any(issubclass(w.category, GoldenMismatchWarning) for w in caught)
        else "info",
    )

    smiles = read_smiles(Path(input_path), smiles_column)
    head = "blend" if model.has_hard else "soft"
    echo(
        f"model.onnx · [bold]{model.n_columns}[/] column(s) · {head} · {len(smiles):,} SMILES",
        "run",
    )
    values = model.run(smiles, workers=workers)
    out_path = Path(out_path)
    with open(out_path, "w", newline="") as fp:
        # \n line endings, as pandas wrote them before; csv's default is \r\n
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow(["smiles", *model.columns])
        for smi, row in zip(smiles, values):
            writer.writerow(
                [smi, *("" if math.isnan(v) else repr(float(v)) for v in row)]
            )
    success(f"predictions ({' · '.join(model.columns)}) → [dim]{out_path}[/]")


def _read_rows(input_path: Path) -> tuple[list[str], list[list[str]]]:
    """Header and rows of a CSV/TSV/Parquet file, as strings."""
    suffix = input_path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ValueError(
                'reading Parquet needs pyarrow — `pip install "olinda[report]"`, or convert to CSV'
            ) from exc
        table = pq.read_table(str(input_path))
        header = list(table.column_names)
        cols = [table.column(i).to_pylist() for i in range(len(header))]
        return header, [[str(v) for v in row] for row in zip(*cols)]
    if suffix in (".csv", ".tsv"):
        with open(input_path, newline="") as fp:
            reader = csv.reader(fp, delimiter="\t" if suffix == ".tsv" else ",")
            header = next(reader, [])
            return header, list(reader)
    raise ValueError(f"unsupported input format: {suffix} (use .csv / .tsv / .parquet)")


def read_smiles(input_path: Path, smiles_col: str | None = None) -> list[str]:
    """Read the SMILES column from a CSV/TSV/Parquet input for prediction.

    The column may be called ``smiles`` or ``input`` (case-insensitive) — the latter is what Ersilia
    and Isaura write, alongside a leading ``key`` column that is simply ignored. Pass *smiles_col* to
    name the column explicitly instead.
    """
    from olinda.console import echo
    from olinda.data.reference import _SMILES_NAMES, smiles_column_index

    input_path = Path(input_path)
    try:
        header, rows = _read_rows(input_path)
    except ValueError as exc:
        echo(str(exc), "error")
        raise

    if smiles_col is not None:
        if smiles_col not in header:
            echo(f"no '{smiles_col}' column in [dim]{input_path.name}[/]", "error")
            raise ValueError(f"input needs a '{smiles_col}' column with SMILES")
        index = header.index(smiles_col)
    else:
        index = smiles_column_index(header)
        if index is None:
            accepted = " / ".join(f"'{name}'" for name in _SMILES_NAMES)
            echo(f"no {accepted} column in [dim]{input_path.name}[/]", "error")
            raise ValueError(
                f"input needs a {accepted} column with SMILES; got {header}"
            )
    return [row[index] if index < len(row) else "" for row in rows]
