from __future__ import annotations

import rich_click as click


@click.command("clean")
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
