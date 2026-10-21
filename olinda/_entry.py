"""Console-script entry point.

The base install is the Python API alone; the ``olinda`` command needs the ``[cli]`` extra. pip
installs the ``olinda`` script either way, since entry points cannot be optional, so on a base install
this module is what answers: a plain message naming the extra, using nothing but the standard library,
instead of an ImportError traceback. Commands that need more than ``[cli]`` refuse individually — see
:func:`olinda.train.require_train_extra` and :func:`olinda.report.require_report_extra`.
"""

from __future__ import annotations

# What the [cli] extra adds. A missing one of these is the base install being used as intended.
_CLI_PACKAGES = {"click", "rich_click", "rich", "loguru", "tqdm"}


def main() -> None:
    try:
        from olinda.cli import cli
    except ImportError as exc:
        missing = (getattr(exc, "name", None) or "").split(".")[0]
        if missing in _CLI_PACKAGES:
            raise SystemExit(
                "The olinda command needs the CLI extra:\n\n"
                '    pip install "olinda[cli]"\n\n'
                "Without it, olinda runs models from Python: `from olinda import OlindaArtifact`."
            ) from exc
        raise SystemExit(  # pragma: no cover - a broken install, not a supported tier
            f"The olinda CLI could not start: {missing or 'a dependency'} is missing.\n"
            "This environment looks incomplete. Reinstall with:\n\n"
            '    pip install --force-reinstall "olinda[cli]"\n\n'
            "Distilling additionally needs `olinda[train]`; scoring a model needs `olinda[report]`."
        ) from exc
    cli()
