"""Console-script entry point.

The CLI ships with the base install, so this should always import. It is guarded anyway: a partial
or broken environment is worth naming rather than dumping an ImportError traceback on someone who
only wanted `--help`. Commands that need an extra refuse individually — see
:func:`olinda.train.require_train_extra` and :func:`olinda.report.require_report_extra` — because
that is a normal, expected state, while this is not.
"""

from __future__ import annotations


def main() -> None:
    try:
        from olinda.cli import cli
    except (
        ImportError
    ) as exc:  # pragma: no cover - a broken install, not a supported tier
        missing = getattr(exc, "name", None) or "a dependency"
        raise SystemExit(
            f"The olinda CLI could not start: {missing} is missing.\n"
            "The CLI is part of the base install, so this environment looks incomplete. Reinstall with:\n\n"
            "    pip install --force-reinstall olinda\n\n"
            "Distilling additionally needs `olinda[train]`; scoring a model needs `olinda[report]`."
        ) from exc
    cli()
