"""Assemble the olinda CLI: one group, ten commands, each defined in its own module."""

from __future__ import annotations

import rich_click as click

from olinda.cli.commands.clean import clean_cmd
from olinda.cli.commands.export import export_cmd
from olinda.cli.commands.fit import fit_cmd
from olinda.cli.commands.learn_hard import learn_hard_cmd
from olinda.cli.commands.learn_soft import learn_soft_cmd
from olinda.cli.commands.predict import predict_cmd
from olinda.cli.commands.prepare import prepare_cmd
from olinda.cli.commands.setup import setup_cmd
from olinda.cli.commands.tune import tune_cmd
from olinda.cli.commands.validate import validate_cmd
from olinda.cli.rendering import COMMAND_PANELS, configure_help

_COMMANDS = {
    "setup": setup_cmd,
    "fit": fit_cmd,
    "prepare": prepare_cmd,
    "learn-soft": learn_soft_cmd,
    "tune": tune_cmd,
    "learn-hard": learn_hard_cmd,
    "export": export_cmd,
    "clean": clean_cmd,
    "predict": predict_cmd,
    "validate": validate_cmd,
}


@click.group()
def cli():
    pass


def create_olinda_cli():
    """Configure the help rendering, register every command, and return the group.

    Commands are attached with ``add_command`` rather than the ``@cli.command`` decorator so that no
    command module has to import this one. That keeps the import graph a DAG which cannot be made
    cyclic by a plausible edit — which matters because `fit` imports five of its siblings to drive
    them directly.

    Registration follows the help panels, so a command absent from
    :data:`~olinda.cli.rendering.COMMAND_PANELS` is a KeyError here rather than a silent extra panel
    later. ``tests/test_cli_surface.py`` checks the two agree.
    """
    configure_help()
    for panel in COMMAND_PANELS:
        for name in panel["commands"]:
            cli.add_command(_COMMANDS[name])
    return cli
