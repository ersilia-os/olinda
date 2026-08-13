from __future__ import annotations

from pathlib import Path

import rich_click as click

from olinda.cli import options
from olinda.cli.commands.clean import clean_cmd
from olinda.cli.commands.learn_hard import learn_hard_cmd
from olinda.cli.commands.learn_soft import learn_soft_cmd

# `fit` runs the pipeline by calling these commands' callbacks directly, so they are
# dependencies rather than neighbours.
from olinda.cli.commands.prepare import prepare_cmd
from olinda.cli.commands.tune import tune_cmd
from olinda.cli.guards import require_train


@click.command("fit")
@click.option(
    "--soft-labels",
    "-s",
    required=True,
    help="Teacher (soft) values over the reference library (SMILES + one or more value columns, library order).",
)
@click.option(
    "--hard-labels",
    "-h",
    default=None,
    help="Optional CSV/TSV/Parquet of your own compounds (SMILES + one label column) → adds a hard head.",
)
@click.option(
    "--model-onnx",
    "-m",
    "model_onnx",
    required=True,
    help="Where to write the distilled model (must end in .onnx). Working files go in a folder of the "
    "same name beside it, and are deleted when the run finishes.",
)
@options.task()
@click.option(
    "--max-samples",
    default=None,
    type=int,
    help="Use only the first N reference compounds (dev subsampling).",
)
@options.val_frac()
@click.option(
    "--num-boost-round",
    default=10000,
    type=int,
    show_default=True,
    help="learn-soft upper cap; early stopping decides.",
)
@click.option(
    "--tune/--no-tune",
    "do_tune",
    default=False,
    show_default=True,
    help="Run an Optuna tuning pass before learn-soft.",
)
@click.option(
    "--trials",
    default=100,
    type=int,
    show_default=True,
    help="Optuna trials (only with --tune).",
)
@options.soft_smiles_column()
@click.option(
    "--soft-label-columns",
    default=None,
    help="Comma-separated teacher columns to distil (default: every value column in the file).",
)
@options.hard_smiles_column()
@click.option(
    "--hard-label-columns",
    default=None,
    help="Comma-separated measurement columns to use (default: every column matching a teacher task).",
)
def fit_cmd(
    soft_labels,
    hard_labels,
    model_onnx,
    task,
    max_samples,
    val_frac,
    num_boost_round,
    do_tune,
    trials,
    soft_smiles_column,
    soft_label_columns,
    hard_smiles_column,
    hard_label_columns,
):
    """Distill a teacher into a student end-to-end: prepare → (tune) → learn-soft → (learn-hard) → clean.

    Runs the fit-pipeline sub-commands in order. `--model-onnx runs/foo.onnx` builds the run in `runs/foo/`
    and, once everything has fused, moves the model to `runs/foo.onnx` and deletes that folder — so the
    only thing left is the self-describing artifact, which is all `olinda predict` needs. With
    `--hard-labels` a hard head is learned and fused in; with `--tune` an Optuna pass precedes `learn-soft`.

    Drive the steps individually against `-m runs/foo` if you want to keep the per-column boosters, metrics
    and plots.
    """
    require_train()
    import time

    from olinda import run as runlib
    from olinda.console import (
        STEP_COLORS,
        echo,
        elapsed,
        resources,
        rule,
        set_active_color,
        summary_panel,
    )
    from olinda.console import (
        path as cpath,
    )

    started = time.time()
    set_active_color(STEP_COLORS["fit"])
    try:
        work = runlib.work_dir_for(model_onnx)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    rule("olinda · fit", style=STEP_COLORS["fit"], right=resources())
    # A *prepared* folder is a different run: its column directories are bound positionally to whatever
    # teacher produced them, so mixing the two would train under one name and fuse another. Keyed on the
    # manifest rather than on the folder existing, because a fit that failed early — bad input file,
    # mismatched library — leaves an empty one behind, and that must not block the retry.
    if runlib.is_run_dir(work):
        echo(f"a prepared run is already there · [dim]{cpath(work)}[/]", "error")
        raise click.ClickException(
            f"{work} holds another run — delete it, or resume it with "
            f"`olinda learn-soft -m {work}` and finish with `olinda clean -m {model_onnx}`"
        )
    echo(
        f"building in [dim]{cpath(work)}[/] → [bold]{cpath(Path(model_onnx))}[/]", "run"
    )

    model_dir = str(work)
    prepare_cmd.callback(
        soft_labels=soft_labels,
        hard_labels=hard_labels,
        model_dir=model_dir,
        task=task,
        max_samples=max_samples,
        val_frac=val_frac,
        soft_smiles_column=soft_smiles_column,
        soft_label_columns=soft_label_columns,
        hard_smiles_column=hard_smiles_column,
        hard_label_columns=hard_label_columns,
    )
    if do_tune:
        tune_cmd.callback(model_dir=model_dir, trials=trials, max_rows=100_000)
    learn_soft_cmd.callback(model_dir=model_dir, num_boost_round=num_boost_round)
    if hard_labels is not None:
        learn_hard_cmd.callback(model_dir=model_dir)
    # Everything is fused, so the working folder has no consumer left.
    clean_cmd.callback(model_onnx=model_onnx)

    pipeline = (
        "prepare → "
        + ("tune → " if do_tune else "")
        + "learn-soft"
        + (" → learn-hard" if hard_labels else "")
        + " → clean"
    )
    summary_panel(
        "olinda · fit",
        [
            ("Pipeline", pipeline),
            ("Head", "soft + hard" if hard_labels else "soft only"),
            ("Model", f"[dim]{cpath(Path(model_onnx))}[/]"),
            ("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]"),
        ],
        border_style=STEP_COLORS["fit"],
        icon="✓",
    )
