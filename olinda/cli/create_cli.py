from __future__ import annotations

from pathlib import Path

import rich_click as click

from olinda.cli import options
from olinda.cli.guards import require_train
from olinda.cli.options import parse_label_columns
from olinda.cli.rendering import configure_help

# NOTE: heavy dependencies (numpy, pandas, xgboost, rdkit, matplotlib, ...) are imported
# lazily inside the command bodies below, not at module scope. This keeps CLI startup fast
# (e.g. `olinda --help`, `olinda setup`) — importing them all eagerly cost ~12 s per call.


@click.group()
def cli():
    pass


@cli.command("setup")
@click.option(
    "--target-dir",
    default=None,
    help="Directory to download data into (default: ~/.olinda/).",
)
def setup_cmd(target_dir):
    """Download olinda data from public S3 — the reference-library Morgan fingerprints (``erl0_morgan.h5``).

    Anything already present is skipped, so re-running only fetches what is missing. The download is
    best-effort (a warning, not an error, if it isn't on S3 yet); you can generate it locally with
    ``scripts/compute_morgan_fingerprints.py``.
    """
    from olinda.console import rule
    from olinda.data.fetch import download_morgan_fingerprints

    rule("olinda · setup", style="cyan")
    download_morgan_fingerprints(target_dir)


@cli.command("fit")
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


@cli.command("prepare")
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


@cli.command("learn-soft")
@click.option(
    "--model-dir",
    "-m",
    required=True,
    help="A prepared run directory (from `prepare`).",
)
@click.option(
    "--num-boost-round",
    default=10000,
    type=int,
    show_default=True,
    help="Upper cap; early stopping decides.",
)
def learn_soft_cmd(model_dir, num_boost_round):
    """Learn the surrogate: fast gradient-boosting regression on each column's prepared split.

    The engine is auto-selected by device — XGBoost on a CUDA GPU, LightGBM on CPU (``OLINDA_BACKEND``
    overrides). The loss is squared error (well-conditioned, ONNX-safe); skew/imbalance is handled by
    reweighting, not the loss. If a prior ``olinda tune -m <model-dir>`` wrote ``best_params.json`` there,
    those hyperparameters are used; otherwise built-in defaults.

    Sample reweighting is **automatic**: olinda weights the target only when it is imbalanced (auto-picking
    KDE / bins), leaving balanced targets unweighted. When active it applies to train AND val together so
    early stopping matches the objective. Global and tail metrics (top-decile RMSE, Spearman) are always
    reported, and a single self-describing `model.onnx` bundle is fused at the end (soft-only here;
    `learn-hard` re-fuses it with the hard head).
    """
    require_train()
    import time

    from olinda import run as runlib
    from olinda.console import (
        STEP_COLORS,
        LiveTable,
        echo,
        elapsed,
        engine_banner,
        rule,
        set_active_color,
        summary_panel,
    )
    from olinda.console import (
        path as cpath,
    )
    from olinda.data import MORGAN_FINGERPRINTS_FILENAME, OLINDA_HOME
    from olinda.data.matrix import ReferenceMatrix
    from olinda.train.backend import get_backend, select_backend
    from olinda.train.column import train_one_column

    model_dir = Path(model_dir)
    try:
        manifest = runlib.read_manifest(model_dir)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    columns = manifest["columns"]
    features = manifest["features"].get("features", "morgan")
    dim = manifest["reference_library"]["dim"]
    started = time.time()
    set_active_color(STEP_COLORS["learn-soft"])
    rule(
        "olinda · learn-soft",
        style=STEP_COLORS["learn-soft"],
        right=f"{len(columns)} column(s) · {features} · {dim}-dim",
    )

    # Engine auto-selected by device (GPU→XGBoost, CPU→LightGBM); OLINDA_BACKEND overrides.
    backend_name, device, backend_reason = select_backend()
    be = get_backend(backend_name, device)
    engine_banner(backend_name, device, backend_reason)

    # The descriptor matrix is identical for every column, so it is read once and addressed by index.
    descriptors = OLINDA_HOME / MORGAN_FINGERPRINTS_FILENAME
    if not descriptors.exists():
        raise click.ClickException(
            f"reference library missing at {descriptors} — run `olinda setup`"
        )
    limit = runlib.row_limit(manifest)
    scope = f" · [dim]first {limit:,} rows[/]" if limit else ""
    echo(f"loading reference descriptors · [dim]{descriptors.name}[/]{scope}", "run")
    matrix = ReferenceMatrix.load(descriptors, limit=limit)
    try:
        matrix.assert_matches(manifest["reference_library"])
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    results = []
    with LiveTable(
        [c["name"] for c in columns],
        title="Distilling columns",
        fields=["R²", "ρ", "RMSE", "Trees", "Time"],
        item_label="Column",
        running_verb="training",
        color=STEP_COLORS["learn-soft"],
    ) as table:
        for col in columns:
            table.start(col["name"])
            col_started = time.time()
            result = train_one_column(
                model_dir, manifest, col, matrix, be, backend_name, num_boost_round
            )
            m = result["metrics"]
            table.finish(
                col["name"],
                **{
                    "R²": f"{m['r2']:.4f}",
                    "ρ": f"{m['spearman']:.4f}",
                    "RMSE": f"{m['rmse']:.5f}",
                    "Trees": f"{result['n_trees']:,}",
                    "Time": elapsed(time.time() - col_started),
                },
            )
            results.append(result)
            col["status"]["soft_trained"] = True
            runlib.write_manifest(model_dir, manifest)

    del matrix  # release ~2.8 GB before fusing, which is itself memory-hungry

    from olinda.export import build_bundle

    try:
        build_bundle(model_dir)
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc

    summary_panel(
        "olinda · learn-soft",
        [
            ("Columns", f"[bold]{len(results)}[/] trained · {backend_name}"),
            *[
                (
                    r["name"],
                    f"R² [bold]{r['metrics']['r2']:.4f}[/] · ρ {r['metrics']['spearman']:.4f} · {r['n_trees']:,} trees",
                )
                for r in results
            ],
            ("Model", f"[dim]{cpath(model_dir / 'model.onnx')}[/]"),
            ("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]"),
        ],
        border_style=STEP_COLORS["learn-soft"],
        icon="✓",
    )


@cli.command("tune")
@click.option(
    "--model-dir",
    "-m",
    required=True,
    help="A prepared run directory; best_params.json is written here.",
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
def tune_cmd(model_dir, trials, max_rows):
    """Short, pruned Optuna study to discover good hyperparameters (esp. learning rate) for `learn-soft`.

    Reads the prepared split from ``--model-dir`` / ``-m`` and writes ``best_params.json`` back into it, on
    the **same auto-selected engine** `learn-soft` uses, then a subsequent `learn-soft -m <dir>` auto-reads
    it. It tunes on a random subsample of ``--max-rows`` rows (train+val kept in the prepared proportion) —
    the console prints exactly how many are used. This is a fast way to find a good hyperparameter *region*:
    the learning rate transfers reasonably (full training re-fits the round count with its own early
    stopping), so `tune` is optional — the built-in defaults are already good. Everything else (per-trial
    rounds, early-stopping, study patience, reweighting = auto like `learn-soft`, seed, a safety time cap)
    uses good internal defaults — call ``olinda.train.tune.run_tuning`` directly to override them.
    Requires the ``[train]`` extra (Optuna).
    """
    require_train()
    from olinda.run import PARAMS_NAME
    from olinda.train.tune import run_tuning

    run_tuning(
        model_dir,
        trials=trials,
        max_rows=max_rows,
        out=Path(model_dir) / PARAMS_NAME,
    )


@cli.command("learn-hard")
@click.option(
    "--model-dir",
    "-m",
    required=True,
    help="A run prepared with --hard-labels; artifacts go under columns/<id>/_hard/.",
)
def learn_hard_cmd(model_dir):
    """Learn from hard (binary) labels and calibrate them onto the soft-label scale — four clear steps.

    Runs once per matched column, reading that column's `hard.h5` (from `prepare --hard-labels`): (1) trains
    `H`, the hard-label model; (2) scores `H` across the full reference library (`erl0_morgan.h5`,
    required); (3) calibrates `H` onto `S`'s scale — that is `H_S` — against the column's reference-aligned
    targets, with a monotonic map whose direction is learned from the data; (4) trains `T`, a small MLP
    predicting a compound's 1-NN Tanimoto to the labeled set from its fingerprint alone. Artifacts land
    under `columns/<id>/_hard/`, and the model is re-fused so each blended column keeps its prediction as
    the named output (`T` needs no similarity search at predict time, and the blend favours `S` away from
    the labeled set).
    """
    require_train()
    import time

    from olinda import run as runlib
    from olinda.console import (
        STEP_COLORS,
        echo,
        elapsed,
        rule,
        set_active_color,
        summary_panel,
    )
    from olinda.console import (
        path as cpath,
    )
    from olinda.hard import HARD_H5_NAME, train_hard

    md = Path(model_dir)
    started = time.time()
    set_active_color(STEP_COLORS["learn-hard"])
    try:
        manifest = runlib.read_manifest(md)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    with_hard = [
        c
        for c in manifest["columns"]
        if (runlib.column_dir(md, c["id"]) / HARD_H5_NAME).exists()
    ]
    untrained = [
        c["name"]
        for c in manifest["columns"]
        if not (runlib.column_dir(md, c["id"]) / "train_meta.json").exists()
    ]
    if untrained:
        raise click.ClickException(
            f"{len(untrained)} column(s) have no surrogate yet ({', '.join(untrained)}) — "
            f"run `olinda learn-soft -m {model_dir}` first. learn-hard fuses at the end and would "
            "otherwise fail there, after training every hard-label head."
        )

    if not with_hard:
        raise click.ClickException(
            f"no column in {model_dir!r} has hard labels — run "
            f"`olinda prepare -s <soft> -h <hard> -m {model_dir}` first"
        )

    rule(
        "olinda · learn-hard",
        style=STEP_COLORS["learn-hard"],
        right=f"{len(with_hard)} column(s) with hard labels · {cpath(md)}",
    )

    # One resident copy of the library for the whole run: each column otherwise reopened and streamed
    # it twice (scoring H, then fitting T), so a 10-column run read 2.8 GB twenty times over.
    from olinda.data.fetch import MORGAN_FINGERPRINTS_FILENAME, OLINDA_HOME
    from olinda.data.matrix import ReferenceMatrix

    limit = runlib.row_limit(manifest)
    scope = f" · [dim]first {limit:,} rows[/]" if limit else ""
    echo(
        f"loading reference descriptors · [dim]{MORGAN_FINGERPRINTS_FILENAME}[/]{scope}",
        "run",
    )
    matrix = ReferenceMatrix.load(
        OLINDA_HOME / MORGAN_FINGERPRINTS_FILENAME, limit=limit
    )
    try:
        matrix.assert_matches(manifest["reference_library"])
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    for i, col in enumerate(with_hard, start=1):
        # The command banner is printed once, above; a per-column rule only earns its line when there is
        # more than one column to tell apart.
        if len(with_hard) > 1:
            rule(
                f"column {i}/{len(with_hard)} · {col['name']}",
                style=STEP_COLORS["learn-hard"],
            )
        train_hard(
            runlib.column_dir(md, col["id"]),
            soft=runlib.read_target(md, col["id"]),
            matrix=matrix,
        )
        col["status"]["hard_trained"] = True
        runlib.write_manifest(md, manifest)

    # Re-fuse the served bundle so every column's hard head is included.
    from olinda.export import build_bundle

    build_bundle(md)
    summary_panel(
        "olinda · learn-hard",
        [
            (
                "Columns",
                f"[bold]{len(with_hard)}[/] of {len(manifest['columns'])} have a hard head",
            ),
            *[
                (
                    c["name"],
                    f"[dim]{(c.get('hard') or {}).get('n', '?')} labelled compounds[/]",
                )
                for c in with_hard
            ],
            ("Model", f"[dim]{cpath(md / 'model.onnx')}[/]"),
            ("Elapsed", f"[dim]{elapsed(time.time() - started)}[/]"),
        ],
        border_style=STEP_COLORS["learn-hard"],
        icon="✓",
    )


@cli.command("export")
@click.option(
    "--model-dir",
    "-m",
    required=True,
    help="A trained model dir (from `learn-soft`/`learn-hard`); (re)builds <dir>/model.onnx.",
)
def export_cmd(model_dir):
    """(Re)build the single self-describing `model.onnx` for a trained model dir.

    Fuses every column and every stage into ONE graph with one output per task — the surrogate alone, or,
    where a hard head exists, the blend `(1-a)·S + a·H_S`.
    The channels behind a blend are declared as outputs too. The Morgan featurizer config + provenance (RDKit
    version, reference library, hard-head summary) are embedded in the file's metadata, so it is
    self-describing. Gated on a numeric parity check. Featurization (RDKit) stays in Python — the graph
    consumes a 2048-count Morgan fingerprint.
    """
    require_train()
    from olinda import run as runlib
    from olinda.export import build_bundle

    md = Path(model_dir)
    try:
        manifest = runlib.read_manifest(md)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    # Models live per column, not at the run root — check every column has one before fusing.
    missing = [
        col["name"]
        for col in manifest["columns"]
        if not (runlib.column_dir(md, col["id"]) / "train_meta.json").exists()
    ]
    if missing:
        raise click.ClickException(
            f"{len(missing)} column(s) not trained yet ({', '.join(missing)}) — "
            f"run `olinda learn-soft -m {model_dir}` first"
        )
    try:
        build_bundle(md)
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("clean")
@click.option(
    "--model-onnx",
    "-m",
    "model_onnx",
    required=True,
    help="Finish the run behind this artifact path (must end in .onnx): move the model here, delete the "
    "folder of the same name.",
)
def clean_cmd(model_onnx):
    """Finish a run: move its fused model to `--model-onnx` and delete the working folder.

    `runs/foo.onnx` is built in `runs/foo/`; this moves `runs/foo/model.onnx` out and removes the rest.
    The artifact is self-describing — column names, metrics, featurizer, RDKit build, reference library
    and provenance all live inside it — so the manifest, targets, splits and per-column directories have
    no consumer once the fuse succeeds. On a real run they are almost all of the bytes.

    This **ends the run**: `learn-hard` and `export` read the manifest, so neither can run afterwards.
    `predict` is unaffected. `olinda fit` does this for you as its last stage; run it by hand when you
    drove the steps individually. Doing it twice is harmless.
    """
    from olinda import run as runlib
    from olinda.console import (
        STEP_COLORS,
        detail,
        echo,
        filesize,
        rule,
        set_active_color,
        success,
    )
    from olinda.console import (
        path as cpath,
    )

    set_active_color(STEP_COLORS["clean"])
    try:
        work = runlib.work_dir_for(model_onnx)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    rule("olinda · clean", style=STEP_COLORS["clean"], right=cpath(work))

    try:
        artifact, removed = runlib.finish_run(model_onnx)
    except FileNotFoundError as exc:
        echo(str(exc), "error")
        raise click.ClickException(
            f"nothing to finish — `olinda export -m {work}` first, or check the path"
        ) from exc

    if not removed:
        success(f"already clean · [dim]{cpath(artifact)}[/]")
        return
    detail([(name, filesize(nbytes)) for name, nbytes in removed])
    success(
        f"reclaimed [bold]{filesize(sum(n for _, n in removed))}[/] · "
        f"[dim]{cpath(artifact)} is all that remains[/]"
    )


@cli.command(
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


@cli.command(
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


def create_olinda_cli():
    """Configure the help rendering, register every command, and return the group.

    Commands are attached with ``add_command`` rather than the ``@cli.command`` decorator so that no
    command module has to import this one. That keeps the import graph a DAG which cannot be made
    cyclic by a plausible edit — which matters because `fit` imports five of its siblings to drive
    them directly.
    """
    configure_help()
    return cli
