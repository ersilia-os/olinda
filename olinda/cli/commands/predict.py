from __future__ import annotations

from pathlib import Path

import rich_click as click


@click.command(
    "predict",
    help="Run a model on SMILES via its model.onnx — one output column per task.",
)
@click.option(
    "--model-onnx",
    "-m",
    "model_onnx",
    required=True,
    help="The distilled model.onnx (a run directory containing one also works).",
)
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    help="CSV/TSV/Parquet with a `smiles` (or `input`) column.",
)
@click.option(
    "--output", "-o", "out_path", required=True, help="Output CSV for the predictions."
)
@click.option(
    "--smiles-column",
    default=None,
    help="Name of the SMILES column (default: `smiles`/`input`, whichever the file has).",
)
def predict_cmd(model_onnx, input_path, out_path, smiles_column):
    """Run the fused `model.onnx`: verify RDKit, featurize the SMILES column (RDKit Morgan), run the graph.

    The input's SMILES column may be called `smiles` or `input` — the latter is what Ersilia writes,
    and a leading `key` column is ignored — or name it outright with --smiles-column.

    Writes a `smiles` column followed by one column per task, named after the teacher column it distils.
    For a model with a hard-label head that number is already the blend — the weight `a`
    happens inside the graph. The featurizer and the RDKit build it needs are read from the model's
    embedded metadata, so the `.onnx` is the only input. Unparseable SMILES come back as empty cells.
    """
    from olinda.artifact import MODEL_NAME
    from olinda.console import STEP_COLORS, echo, rule, set_active_color
    from olinda.console import path as cpath

    model_onnx = Path(model_onnx)
    input_path = Path(input_path)
    set_active_color(STEP_COLORS["predict"])
    rule("olinda · predict", style=STEP_COLORS["predict"], right=cpath(model_onnx))
    if not input_path.exists():
        echo(f"input not found · [dim]{input_path}[/]", "error")
        raise click.ClickException("input file does not exist")

    # A run directory is accepted too: `fit` leaves exactly one artifact in it, so pointing at either
    # is unambiguous, and it keeps the pipeline commands' `-m <run dir>` habit working here.
    if model_onnx.is_dir():
        model_onnx = model_onnx / MODEL_NAME
    if not model_onnx.exists():
        echo(f"no artifact at [dim]{model_onnx}[/]", "error")
        raise click.ClickException(
            f"{model_onnx} does not exist — run `olinda fit` (or `olinda export -m <run dir>`) first"
        )

    from olinda.artifact import RDKitVersionMismatch
    from olinda.predict import predict_file

    try:
        predict_file(model_onnx, input_path, out_path, smiles_column=smiles_column)
    except (RDKitVersionMismatch, ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
