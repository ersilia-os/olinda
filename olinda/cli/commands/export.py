from __future__ import annotations

from pathlib import Path

import rich_click as click

from olinda.cli.guards import require_train


@click.command("export")
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
