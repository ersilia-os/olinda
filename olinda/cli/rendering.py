"""How the help screens look: rich-click styling, the two command panels, aligned name columns.

All of this is global rich-click state, so :func:`configure_help` has to run before anything renders.
:func:`olinda.cli.create_cli.create_olinda_cli` calls it, and ``olinda/cli/__init__.py`` is the only
route to the group, so there is no path that renders help without it.
"""

from __future__ import annotations

import rich_click as click
import rich_click.rich_click as rc

# Two-level help grouping (a la zairachem): top-level user commands, then the lower-level pipeline steps.
COMMAND_PANELS = [
    {
        "name": "Main commands",
        "commands": ["setup", "library", "fit", "predict", "validate"],
    },
    {
        "name": "Fit pipeline commands",
        "commands": [
            "prepare",
            "tune",
            "learn-soft",
            "learn-hard",
            "export",
            "clean",
        ],
    },
]


def _align_command_columns(name_width: int = 11) -> None:
    """Pin the command-name column to a fixed width across every help panel.

    rich-click sizes each group's name column to *that group's* longest command, so the help text in the
    "Main commands" and "Fit pipeline commands" panels doesn't line up; its only knob is a *proportional*
    ratio, which either leaves a big gap or truncates names. A fixed width (just past the longest command,
    "learn-soft"/"learn-hard") keeps the help text tight to the names AND aligned across panels. rich-click exposes no
    config for this, so we defensively wrap the internal command-table builder — on any API drift it silently
    falls back to the default rendering rather than breaking the CLI.
    """
    try:
        from rich_click.rich_panel import RichCommandPanel
    except Exception:
        return

    _orig_get_table = RichCommandPanel.get_table

    def _get_table(self, *args, **kwargs):
        table = _orig_get_table(self, *args, **kwargs)
        try:
            # Fix the name column and make the help column absorb ALL slack, so the name column stays exactly
            # `name_width` even on a wide terminal (otherwise `expand` inflates both columns, per panel).
            name_col, help_col = table.columns[0], table.columns[1]
            name_col.width, name_col.ratio, name_col.no_wrap = name_width, None, True
            help_col.ratio, help_col.width = 1, None
        except Exception:
            pass
        return table

    # Marks the wrapper so configure_help can tell it has already run; without it a second call
    # would wrap the wrapper, and every extra layer re-does the same column fixing per render.
    _get_table._olinda_aligned = True
    RichCommandPanel.get_table = _get_table


def configure_help() -> None:
    """Apply the styling, the panels and the column alignment. Safe to call more than once."""
    click.rich_click.TEXT_MARKUP = "rich"
    click.rich_click.SHOW_ARGUMENTS = True

    rc.TEXT_MARKUP = "rich"
    rc.SHOW_ARGUMENTS = True
    rc.COLOR_SYSTEM = "truecolor"
    rc.STYLE_OPTION = "bold magenta"
    rc.STYLE_COMMAND = "bold green"
    rc.STYLE_METAVAR = "italic yellow"
    rc.STYLE_SWITCH = "underline cyan"
    rc.STYLE_USAGE = "bold blue"
    rc.STYLE_OPTION_DEFAULT = "dim italic"
    rc.COMMAND_GROUPS = {"olinda": COMMAND_PANELS}

    # _align_command_columns wraps RichCommandPanel.get_table, so a second call would wrap the wrapper.
    from rich_click.rich_panel import RichCommandPanel

    if not getattr(RichCommandPanel.get_table, "_olinda_aligned", False):
        _align_command_columns()
