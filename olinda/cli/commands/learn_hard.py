from __future__ import annotations

from pathlib import Path

import rich_click as click

from olinda.cli.guards import require_train


@click.command("learn-hard")
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
