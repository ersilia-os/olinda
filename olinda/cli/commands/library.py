from __future__ import annotations

from pathlib import Path

import rich_click as click

from olinda.cli.guards import require_train

# The library is 1.36M rows. Streamed in chunks so the SMILES are never all resident at once —
# h5py's `asstr()` view decodes lazily, and writing per chunk keeps this at a few MB.
_CHUNK = 100_000


@click.command("library")
@click.option(
    "--output",
    "-o",
    "out_path",
    required=True,
    help="Output CSV for the SMILES (one column, header `smiles`).",
)
def library_cmd(out_path):
    """Write the reference library's SMILES to a CSV — one column, header `smiles`.

    The library that `olinda setup` downloads is an HDF5 file pairing each compound's Morgan
    fingerprint with its SMILES. This dumps just the SMILES, in library order, which is the order a
    teacher file has to be in: score a model over this CSV and the result is a valid `--soft-labels`
    input without any further alignment.
    """
    require_train()

    import csv

    import h5py

    from olinda.console import STEP_COLORS, echo, rule, success, sweep_progress
    from olinda.console import path as cpath
    from olinda.data.fetch import MORGAN_FINGERPRINTS_FILENAME, OLINDA_HOME

    source = Path(OLINDA_HOME) / MORGAN_FINGERPRINTS_FILENAME
    if not source.exists():
        raise click.ClickException(
            f"no reference library at {source} — run `olinda setup` first."
        )

    out = Path(out_path)
    # `-o results/nested/x.csv` should not need the folders made first. A bare filename gives a parent
    # of `.`, which mkdir(exist_ok=True) accepts, so this needs no special case.
    out.parent.mkdir(parents=True, exist_ok=True)

    rule("olinda · library", style=STEP_COLORS["library"])
    echo(f"reading [dim]{cpath(source)}[/]", "run")

    with h5py.File(str(source), "r") as handle:
        # `asstr()` is a decoded view of the variable-length UTF-8 dataset, so this never builds the
        # bytes objects that `[:]` would.
        smiles = handle["input"].asstr()
        total = int(handle["input"].shape[0])
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["smiles"])
            with sweep_progress("writing", total) as tick:
                for start in range(0, total, _CHUNK):
                    stop = min(start + _CHUNK, total)
                    writer.writerows((s,) for s in smiles[start:stop])
                    tick(stop)

    success(f"{total:,} SMILES → {cpath(out)}")
