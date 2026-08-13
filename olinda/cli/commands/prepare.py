from __future__ import annotations

from pathlib import Path

import rich_click as click

from olinda.cli import options
from olinda.cli.guards import require_train
from olinda.cli.options import parse_label_columns


@click.command("prepare")
@click.option(
    "--soft-labels",
    "-s",
    required=True,
    help="Teacher (soft) values over the reference library (SMILES + value columns, same order as the library).",
)
@click.option(
    "--hard-labels",
    "-h",
    default=None,
    help="Optional CSV/TSV/Parquet of your own compounds (SMILES + one label column) for `learn-hard`.",
)
@click.option(
    "--model-dir",
    "-m",
    required=True,
    help="Run directory — all prepared data + models live here.",
)
@options.task()
@click.option(
    "--max-samples",
    default=None,
    type=int,
    help="Use only the first N reference compounds (development subsampling).",
)
@options.val_frac()
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
def prepare_cmd(
    soft_labels,
    hard_labels,
    model_dir,
    task,
    max_samples,
    val_frac,
    soft_smiles_column,
    soft_label_columns,
    hard_smiles_column,
    hard_label_columns,
):
    """Prepare every teacher column in --soft-labels into one run directory.

    Each value column gets its own value-stratified train/val split, recorded as row indices rather
    than a copy of the descriptor matrix. With --hard-labels (a wide file: SMILES plus one column per
    assay, empty where untested), each hard column is matched onto its soft column by name and
    featurized into that column's hard.h5 for a later `learn-hard`.

    Columns are found by convention — a `smiles`/`input` column, then the value columns after it — so
    the four `--*-column(s)` flags are only needed for files that don't follow it, or to distil a
    subset of a wide teacher file. Naming a column that isn't in the file is an error.
    """
    require_train()
    import time

    import h5py

    from olinda import run as runlib
    from olinda.console import (
        STEP_COLORS,
        echo,
        elapsed,
        rule,
        set_active_color,
        step,
        summary_panel,
    )
    from olinda.console import (
        path as cpath,
    )
    from olinda.data import (
        MORGAN_FINGERPRINTS_FILENAME,
        OLINDA_HOME,
        check_column_budget,
        load_reference_calcs_frame,
        match_hard_columns,
        split_reference_to_indices,
    )

    started = time.time()
    set_active_color(STEP_COLORS["prepare"])
    rule("olinda · prepare", style=STEP_COLORS["prepare"], right=cpath(model_dir))
    md = Path(model_dir)
    md.mkdir(parents=True, exist_ok=True)

    descriptors = OLINDA_HOME / MORGAN_FINGERPRINTS_FILENAME
    if not descriptors.exists():
        echo(f"reference library missing · [dim]{descriptors}[/]", "error")
        raise click.ClickException(
            f"Morgan descriptors missing at {descriptors} — run `olinda setup` "
            "(or scripts/compute_morgan_fingerprints.py to generate them locally)."
        )

    features = {"features": "morgan"}
    with h5py.File(descriptors, "r") as f:
        n_rows, dim = int(f["data"].shape[0]), int(f["data"].shape[1])
        for k in ("radius", "nbits"):
            if k in f.attrs:
                features[k] = int(f.attrs[k])

    n_steps = 3 if hard_labels is not None else 2

    # --- soft labels: the selected value columns, verified against the library once -----------
    step(1, n_steps, "reading teacher columns")
    wanted_soft = parse_label_columns(soft_label_columns)
    try:
        all_cols, targets = load_reference_calcs_frame(
            soft_labels,
            descriptors,
            columns=wanted_soft,
            smiles_column=soft_smiles_column,
        )
        # The budget bounds what actually gets distilled, so it counts the selection, not the file.
        selected_cols = list(targets)
        check_column_budget(selected_cols)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if wanted_soft is not None and len(selected_cols) < len(all_cols):
        skipped = [c for c in all_cols if c not in targets]
        echo(
            f"distilling {len(selected_cols)} of {len(all_cols)} column(s) · [dim]skipping {skipped}[/]",
            "info",
        )
    echo(
        f"{len(selected_cols)} teacher column(s): [bold]{', '.join(selected_cols)}[/]",
        "run",
    )

    # --- hard labels: match them onto the soft columns before doing any work ------------------
    hard_map: dict = {}
    if hard_labels is not None:
        from olinda.data.reference import _read_table, resolve_smiles_frame

        try:
            _, hard_values = resolve_smiles_frame(
                _read_table(hard_labels),
                smiles_column=hard_smiles_column,
                label_columns=parse_label_columns(hard_label_columns),
            )
            hard_map = match_hard_columns(
                selected_cols, [str(c) for c in hard_values.columns]
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        for hard_col, soft_col in hard_map.items():
            note = "exact" if hard_col == soft_col else "suffix"
            echo(f"hard '{hard_col}' → '{soft_col}' ([dim]{note}[/])", "run")

    # --- per-column value-stratified splits; the descriptor matrix is never copied ------------
    step(2, n_steps, "planning per-column splits")
    manifest = runlib.new_manifest(
        soft_labels=soft_labels,
        hard_labels=hard_labels,
        reference={"name": MORGAN_FINGERPRINTS_FILENAME, "n_rows": n_rows, "dim": dim},
        features=features,
        val_frac=val_frac,
        seed=42,
        limit=max_samples,
    )
    by_id, splits, name_to_id = {}, {}, {}
    for name in selected_cols:
        y = targets[name]
        train_idx, val_idx = split_reference_to_indices(
            y, val_frac=val_frac, limit=max_samples
        )[:2]
        entry = runlib.add_column(
            manifest, name=name, y=y, train_idx=train_idx, val_idx=val_idx
        )
        by_id[entry["id"]] = y
        splits[entry["id"]] = (train_idx, val_idx)
        name_to_id[name] = entry["id"]
        runlib.column_dir(md, entry["id"]).mkdir(parents=True, exist_ok=True)
        echo(
            f"{name}: [bold]{len(train_idx):,}[/] train · [bold]{len(val_idx):,}[/] val",
            "run",
        )

    runlib.write_targets(md, by_id)
    runlib.write_splits(md, splits)

    # --- hard labels: featurize once, then slice per matched column ---------------------------
    hard_info: dict = {}
    if hard_map:
        from olinda.hard import prepare_hard_labels_wide

        step(3, n_steps, "featurizing hard labels")
        try:
            hard_info = prepare_hard_labels_wide(
                hard_labels,
                md,
                {h: name_to_id[s] for h, s in hard_map.items()},
                task=task,
                smiles_column=hard_smiles_column,
            )
        except (ValueError, NotImplementedError) as exc:
            raise click.ClickException(str(exc)) from exc
        for hard_col, info in hard_info.items():
            soft_col = hard_map[hard_col]
            col = runlib.find_column(manifest, soft_col)
            col["hard"] = {
                "source_column": hard_col,
                "match": "exact" if hard_col == soft_col else "suffix",
                **info,
            }
            pos = (
                f" · {info['n_positive']} positive"
                if info["n_positive"] is not None
                else ""
            )
            echo(
                f"{soft_col}: [bold]{info['n']:,}[/] hard rows · {info['task']}{pos}",
                "run",
            )

    runlib.write_manifest(md, manifest)

    n_hard = sum(1 for c in manifest["columns"] if c.get("hard"))
    summary_panel(
        "olinda · prepare",
        [
            (
                "Columns",
                f"[bold]{len(manifest['columns'])}[/] · {n_hard} with hard labels",
            ),
            ("Split", f"value-stratified per column · val_frac {val_frac}"),
            ("Targets", f"[dim]{cpath(md / runlib.TARGETS_NAME)}[/]"),
            ("Saved", f"[dim]{cpath(md)}[/]"),
            ("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]"),
        ],
        border_style=STEP_COLORS["prepare"],
        icon="✓",
    )
