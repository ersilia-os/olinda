"""One module per command, each defining a standalone ``click.Command``.

**A module in here may import, at module scope, only:** ``__future__``, the standard library,
``click``/``rich_click``, and other ``olinda.cli.*`` modules. Nothing from ``olinda.*`` outside
``olinda/cli/`` — not even :mod:`olinda.console`. Every such import goes inside the command body.

That is not stylistic. numpy, pandas, xgboost, rdkit and matplotlib cost about 12 seconds to import,
and ``olinda --help`` must not pay it; ``create_cli`` imports all ten of these modules to register
them, so one module-scope import here would land on every invocation. The rule has no exceptions
because a rule with exceptions is one nobody can apply — and it is enforced by
``tests/test_cli_surface.py``, which names the offending file, plus
``tests/test_inference_install.py``, which checks the same contract from a clean interpreter.

Commands are registered by :func:`olinda.cli.create_cli.create_olinda_cli` with ``add_command``, so no
module here imports the group. `fit` imports five of its siblings, because it drives the pipeline by
calling their callbacks directly.
"""
