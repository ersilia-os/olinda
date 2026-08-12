"""Structural guards on the CLI: what it may import, what its help shows, how `fit` drives the rest.

These assert on shape rather than behaviour, and they are cheap — no fitting, no fixtures. They exist
because the three things they check all fail *silently*: a stray module-scope import only shows up as
a slow start or a broken base install, a command missing from the help panels quietly grows a third
panel, and `fit` reaching a sibling's callback with the wrong keyword is only caught when someone runs
that path.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from unittest.mock import create_autospec

import pytest

ROOT = Path(__file__).resolve().parents[1]

# What a module under olinda/cli/ may import at module scope. Everything else — including
# olinda.console — belongs inside a command body. See olinda/cli/commands/__init__.py.
ALLOWED_ROOTS = {"click", "rich_click"}


def _cli_sources():
    """Every source file that makes up the CLI, flat module or package."""
    package = ROOT / "olinda" / "cli"
    if package.is_dir():
        return sorted(package.rglob("*.py"))
    return [ROOT / "olinda" / "cli.py"]


def _module_scope_imports(tree: ast.Module):
    """Dotted names imported at module scope, skipping ``if TYPE_CHECKING:`` blocks."""
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            # A relative import can only reach inside olinda.cli, which is allowed by construction.
            yield "olinda.cli" if node.level else (node.module or "")


@pytest.mark.parametrize("source", _cli_sources(), ids=lambda p: p.name)
def test_the_cli_imports_nothing_heavy_at_module_scope(source):
    """Heavy imports live in the command bodies, so `olinda --help` costs one rich_click import.

    ``tests/test_inference_install.py`` checks the same contract from the outside, but only in
    aggregate and only for modules that belong to an extra — it cannot see an ``olinda.console``
    import, because console is in the base install. This one names the file and catches that too.
    """
    offenders = [
        name
        for name in _module_scope_imports(ast.parse(source.read_text()))
        if name
        and name.split(".")[0] not in sys.stdlib_module_names
        and name.split(".")[0] not in ALLOWED_ROOTS
        and not name.startswith("olinda.cli")
    ]
    assert not offenders, (
        f"{source.relative_to(ROOT)} imports {offenders} at module scope; "
        "move it into the command body"
    )


def test_every_command_appears_in_exactly_one_help_panel():
    """A command absent from the panel config silently gets its own extra "Commands" panel."""
    import rich_click.rich_click as rc

    from olinda.cli import cli

    panels = rc.COMMAND_GROUPS["olinda"]
    listed = [name for panel in panels for name in panel["commands"]]

    assert sorted(cli.commands) == sorted(listed), (
        "help panels and registered commands disagree"
    )
    assert len(listed) == len(set(listed)), "a command is listed in two panels"


def test_fit_calls_every_pipeline_callback_with_arguments_they_accept():
    """`fit` drives the pipeline by calling its siblings' callbacks directly, not through click.

    That bypasses click's parameter handling, so a renamed or dropped option is not caught anywhere
    else — and nothing in the suite runs `olinda tune` or `fit --tune` at all. ``create_autospec``
    raises TypeError on any keyword the real callback would reject, so this validates all five call
    sites against their real signatures without doing any work.
    """
    from click.testing import CliRunner

    from olinda.cli import cli

    fit = cli.commands["fit"]
    siblings = ["prepare", "tune", "learn-soft", "learn-hard", "clean"]
    module = sys.modules[fit.callback.__module__]

    calls = {}
    originals = {}
    for name in siblings:
        command = cli.commands[name]
        spec = create_autospec(command.callback, spec_set=True)
        calls[name] = spec
        originals[name] = command.callback
        command.callback = spec
        # `fit` reaches its siblings as module globals, so the stand-in has to be visible there too.
        for attribute, value in vars(module).items():
            if value is command:
                setattr(module, attribute, command)

    try:
        result = CliRunner().invoke(
            cli, ["fit", "-s", "teacher.csv", "-m", "out/model.onnx", "--tune"]
        )
    finally:
        for name, callback in originals.items():
            cli.commands[name].callback = callback

    # Every callback `fit` reached must have been reached with keywords it actually accepts;
    # create_autospec raises TypeError otherwise, which surfaces as the command's exception.
    assert not isinstance(result.exception, TypeError), result.exception
    assert calls["prepare"].call_count == 1, (
        f"fit did not reach prepare (exit={result.exit_code}): {result.output}"
    )
    assert calls["tune"].call_count == 1, "fit --tune did not reach tune"
    assert calls["learn-soft"].call_count == 1, "fit did not reach learn-soft"
