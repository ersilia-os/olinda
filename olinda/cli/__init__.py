"""The olinda command-line interface.

``from olinda.cli import cli`` is the entry point — used by :mod:`olinda._entry`, by the tests through
click's ``CliRunner``, and by ``python -m olinda.cli``.
"""

from __future__ import annotations

from olinda.cli.create_cli import create_olinda_cli

cli = create_olinda_cli()
