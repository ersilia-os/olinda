from __future__ import annotations

from pathlib import Path

import rich_click as click

from olinda.cli import options
from olinda.cli.options import parse_label_columns


@click.command(
    "validate",
    help="Score a model against labelled data — correlation, AUROC, a report.",
)
@click.option(
    "--model-onnx",
    "-m",
    "model_onnx",
    required=True,
    help="The model to score (a run directory works too).",
)
@click.option(
    "--soft-labels",
    "-s",
    default=None,
    help="Held-out teacher values (SMILES + value columns). Any size, any order.",
)
@click.option(
    "--hard-labels",
    "-h",
    default=None,
    help="Held-out measurements (SMILES + binary columns).",
)
@click.option(
    "--output-dir",
    "-o",
    "out_dir",
    default="report",
    show_default=True,
    help="Report directory.",
)
@options.soft_smiles_column()
@click.option(
    "--soft-label-columns",
    default=None,
    help="Comma-separated columns of --soft-labels to score (default: every column matching a task).",
)
@options.hard_smiles_column()
@click.option(
    "--hard-label-columns",
    default=None,
    help="Comma-separated columns of --hard-labels to score (default: every column matching a task).",
)
def validate_cmd(
    model_onnx,
    soft_labels,
    hard_labels,
    out_dir,
    soft_smiles_column,
    soft_label_columns,
    hard_smiles_column,
    hard_label_columns,
):
    """Measure a finished `model.onnx` on data of your choosing and write a report.

    Unlike the teacher file `prepare` takes, these labels have **no size or ordering restriction** — any
    SMILES with values, matched to the model's tasks by name (allowing a suffix). Held-out data is the
    point: the surrogate's isotonic correction was fitted on the run's own validation rows, so only new
    data measures the calibrated model honestly.

    Columns are found by convention — a `smiles`/`input` column, then the values after it. Use the
    `--*-column(s)` flags for a file that doesn't follow it, or to score one column of a wide file.

    `--soft-labels` gives correlation and residual diagnostics; `--hard-labels` gives ROC, precision–recall
    and enrichment — of the model's **blended** output, which is what `predict` emits, not the hard-label
    head alone. With neither, you still get the artifact's own calibration curves, read straight from the
    graph. Requires the reporting extra: `pip install "olinda[report]"`.
    """
    from olinda.artifact import MODEL_NAME, RDKitVersionMismatch
    from olinda.console import STEP_COLORS, echo, rule, set_active_color, summary_panel
    from olinda.console import path as cpath

    model_onnx = Path(model_onnx)
    if model_onnx.is_dir():
        model_onnx = model_onnx / MODEL_NAME
    set_active_color(STEP_COLORS["validate"])
    rule("olinda · validate", style=STEP_COLORS["validate"], right=cpath(model_onnx))
    if not model_onnx.exists():
        echo(f"no artifact at [dim]{model_onnx}[/]", "error")
        raise click.ClickException(f"{model_onnx} does not exist")
    if soft_labels is None and hard_labels is None:
        echo("no labels given — reporting the model's internals only", "warning")

    from olinda.report import REPORT_NAME, validate_model

    try:
        report = validate_model(
            model_onnx,
            soft_labels=soft_labels,
            hard_labels=hard_labels,
            out_dir=out_dir,
            soft_smiles_column=soft_smiles_column,
            soft_label_columns=parse_label_columns(soft_label_columns),
            hard_smiles_column=hard_smiles_column,
            hard_label_columns=parse_label_columns(hard_label_columns),
        )
    except (RDKitVersionMismatch, RuntimeError, ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc

    for note in report.get("notes", []):
        echo(note, "warning")
    n_figures = sum(len(v) for v in report["figures"].values())
    summary_panel(
        "olinda · validate",
        [
            ("Model", f"[dim]{cpath(model_onnx)}[/]"),
            ("Tasks", ", ".join(report["model"]["columns"])),
            *[
                (
                    f"{kind} labels",
                    f"{report[kind]['n']:,} compounds · {', '.join(report[kind]['metrics'])}",
                )
                for kind in ("soft", "hard")
                if report.get(kind)
            ],
            ("Figures", f"[bold]{n_figures}[/] written"),
            ("Report", f"[dim]{cpath(Path(out_dir) / REPORT_NAME)}[/]"),
        ],
        border_style=STEP_COLORS["validate"],
        icon="✓",
    )
