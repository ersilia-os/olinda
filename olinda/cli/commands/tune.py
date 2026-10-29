from __future__ import annotations

import rich_click as click

from olinda.cli.guards import require_train


@click.command("tune")
@click.option(
    "--model-dir",
    "-m",
    required=True,
    help="A prepared run directory; each column's best_params.json is written into its own folder.",
)
@click.option(
    "--column",
    default=None,
    help="Tune only this column (name or id). Default: every column of the run, one study each.",
)
@click.option(
    "--trials",
    default=100,
    type=int,
    show_default=True,
    help="How many Optuna trials to run.",
)
@click.option(
    "--max-rows",
    default=100000,
    type=int,
    show_default=True,
    help="Cap on the rows used for tuning (train+val together, subsampled in the split's proportion). "
    "Higher = more faithful to the full data, slower.",
)
def tune_cmd(model_dir, trials, max_rows, column=None):
    """Short, pruned Optuna study to discover good hyperparameters (esp. learning rate) for `learn-soft`.

    Reads the prepared split from ``--model-dir`` / ``-m`` and runs one study per soft column (columns train
    independently), writing each column's ``best_params.json`` into its own folder, on the **same
    auto-selected engine** `learn-soft` uses; a subsequent `learn-soft -m <dir>` auto-reads them. Each study
    takes up to 15 minutes, so N columns cost up to N times that. It tunes on a random subsample of ``--max-rows`` rows (train+val kept in the prepared proportion) —
    the console prints exactly how many are used. This is a fast way to find a good hyperparameter *region*:
    the learning rate transfers reasonably (full training re-fits the round count with its own early
    stopping), so `tune` is optional — the built-in defaults are already good. Everything else (per-trial
    rounds, early-stopping, study patience, reweighting = auto like `learn-soft`, seed, a safety time cap)
    uses good internal defaults — call ``olinda.train.tune.run_tuning`` directly to override them.
    Requires the ``[train]`` extra (Optuna).
    """
    require_train()
    from olinda import run as runlib
    from olinda.train.tune import run_tuning

    try:
        manifest = runlib.read_manifest(model_dir)
        names = [column] if column else [c["name"] for c in manifest["columns"]]
        for name in names:
            run_tuning(model_dir, name, trials=trials, max_rows=max_rows)
    # An unknown column or a missing Optuna; both messages explain themselves.
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
