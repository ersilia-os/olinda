"""Run a distilled model over a file of SMILES.

Reading the input and writing the predictions live here rather than in the CLI so the same path is
available from Python. The model itself is loaded through :class:`~olinda.artifact.OlindaArtifact`,
so a CLI run and a library call produce identical output.
"""

from __future__ import annotations

from pathlib import Path


def predict_file(model_onnx, input_path, out_path, *, smiles_column: str | None = None):
    """Predict via the fused ``model.onnx`` and write one column per task.

    Goes through the same :class:`~olinda.artifact.OlindaArtifact` the library exposes, so the CLI and a
    Python caller produce identical output. Loading verifies the installed RDKit against the build
    recorded in the model's metadata. ``model_onnx`` may be the artifact itself or a directory holding
    one. Pass *smiles_column* to name the input's SMILES column instead of auto-detecting it.
    """
    import rdkit

    from olinda.artifact import OlindaArtifact, RDKitVersionMismatch
    from olinda.console import echo, success

    try:
        model = OlindaArtifact(model_onnx)
    except RDKitVersionMismatch as exc:
        echo(str(exc), "error")
        raise
    echo(
        f"rdkit [bold]{rdkit.__version__}[/] · matches model ({model.rdkit_version})",
        "info",
    )

    smiles = read_smiles(input_path, smiles_column)
    head = "blend" if model.has_hard else "soft"
    echo(
        f"model.onnx · [bold]{model.n_columns}[/] column(s) · {head} · {len(smiles):,} SMILES",
        "run",
    )
    df = model.run(smiles)
    out_path = Path(out_path)
    df.to_csv(out_path, index=False)
    success(f"predictions ({' · '.join(model.columns)}) → [dim]{out_path}[/]")


def read_smiles(input_path: Path, smiles_col: str | None = None) -> list[str]:
    """Read the SMILES column from a CSV/TSV/Parquet input for prediction.

    The column may be called ``smiles`` or ``input`` (case-insensitive) — the latter is what Ersilia
    and Isaura write, alongside a leading ``key`` column that is simply ignored. Pass *smiles_col* to
    name the column explicitly instead.
    """
    import pandas as pd

    from olinda.console import echo
    from olinda.data.reference import _SMILES_NAMES, smiles_column_index

    suffix = input_path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        df = pd.read_parquet(str(input_path))
    elif suffix in (".csv", ".tsv"):
        df = pd.read_csv(str(input_path), sep="\t" if suffix == ".tsv" else ",")
    else:
        echo(
            f"unsupported input format · {suffix} (use .csv / .tsv / .parquet)", "error"
        )
        raise ValueError(
            f"unsupported input format: {suffix} (use .csv / .tsv / .parquet)"
        )

    if smiles_col is not None:
        if smiles_col not in df.columns:
            echo(f"no '{smiles_col}' column in [dim]{input_path.name}[/]", "error")
            raise ValueError(f"input needs a '{smiles_col}' column with SMILES")
        return df[smiles_col].astype(str).tolist()

    index = smiles_column_index(df)
    if index is None:
        accepted = " / ".join(f"'{name}'" for name in _SMILES_NAMES)
        echo(f"no {accepted} column in [dim]{input_path.name}[/]", "error")
        raise ValueError(
            f"input needs a {accepted} column with SMILES; got {list(df.columns)}"
        )
    return df.iloc[:, index].astype(str).tolist()
