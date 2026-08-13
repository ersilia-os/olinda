"""Refusals a command makes before it does any work, when an install is missing what it needs."""

from __future__ import annotations

import rich_click as click


def require_train() -> None:
    """Refuse a distilling command with guidance when the training extra is absent.

    The CLI ships with the base install, so these commands are reachable on an inference-only one and
    would otherwise die on whichever heavy ``import`` came first — a traceback naming ``h5py``, which
    says nothing about what to install. Raised as a ClickException so it prints in the error panel like
    any other user-facing refusal. `validate` does the same through ``require_report_extra``.
    """
    from olinda.train import require_train_extra

    try:
        require_train_extra()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
