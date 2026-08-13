"""Options declared identically by more than one command, and the label-column parser.

Only the flags whose declaration is *byte-identical* everywhere it appears live here. Each is a
factory returning a click decorator, and each is always called at the use site — ``@options.task()``,
not ``@options.task`` — so a reader never has to work out which need parentheses.

The flags that differ are deliberately left with their commands. ``--model-dir`` is declared by five
commands with five different descriptions, and ``--soft-labels`` means something different to `fit`,
`prepare` and `validate`; a command's help text is its interface, so five descriptions of ``-m`` are
five commands saying what they will do with the directory, not duplication to collapse. A flag earns a
place here when its declaration is the same in every command that has it, and not before.
"""

from __future__ import annotations

import rich_click as click


def task():
    """``--task``: the hard-label type, auto-detected by default. Declared by `fit` and `prepare`."""
    return click.option(
        "--task",
        type=click.Choice(["auto", "binary", "regression"]),
        default="auto",
        show_default=True,
        help="Hard-label type (auto-detected by default); only used with --hard-labels.",
    )


def val_frac():
    """``--val-frac``: the per-column validation holdout. Declared by `fit` and `prepare`."""
    return click.option(
        "--val-frac",
        default=0.1,
        type=float,
        show_default=True,
        help="Fraction of each teacher column held back for validation — early stopping and the reported "
        "metrics both read it. Split per column and stratified by value, so the held-back rows span the "
        "whole range rather than landing in one part of it.",
    )


def soft_smiles_column():
    """``--soft-smiles-column``. Declared by `fit`, `prepare` and `validate`."""
    return click.option(
        "--soft-smiles-column",
        default=None,
        help="Name of the SMILES column in --soft-labels (default: `smiles`/`input`, else the first column).",
    )


def hard_smiles_column():
    """``--hard-smiles-column``. Declared by `fit`, `prepare` and `validate`."""
    return click.option(
        "--hard-smiles-column",
        default=None,
        help="Name of the SMILES column in --hard-labels (default: `smiles`/`input`, else the first column).",
    )


def parse_label_columns(value: str | None):
    """Parse a comma-separated ``--*-label-columns`` value into a list of names, or ``None``.

    An unset — or empty, or all-whitespace — flag means "work it out from the file", which is the
    behaviour every one of these commands had before the flags existed.
    """
    if value is None:
        return None
    names = [part.strip() for part in value.split(",") if part.strip()]
    return names or None
