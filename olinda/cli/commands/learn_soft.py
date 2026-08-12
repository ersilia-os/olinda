from __future__ import annotations

from pathlib import Path

import rich_click as click

from olinda.cli.guards import require_train


@click.command("learn-soft")
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
