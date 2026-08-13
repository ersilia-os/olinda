"""Support ``python -m olinda.cli``, which is how the CLI runs without the console script."""

from __future__ import annotations

from olinda.cli import cli

if __name__ == "__main__":
    cli()
