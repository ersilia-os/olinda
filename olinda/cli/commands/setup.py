from __future__ import annotations

import rich_click as click


@click.command("setup")
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
