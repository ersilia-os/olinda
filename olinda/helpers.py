"""The package logger: a loguru singleton on Rich, for diagnostics rather than status.

User-facing output belongs to :mod:`olinda.console` — steps, panels, progress, summaries. This is the
other half: silent DEBUG diagnostics, and genuine warnings and errors that surface. Both write through
the one shared Console, so they interleave correctly with the live progress regions.

    from olinda.helpers import logger

    logger.debug("gathered %d rows", n)
    logger.warning("skipping invalid SMILES: %s", smi)
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from loguru import logger as _loguru
from rich.logging import RichHandler

# One console for the whole package. A second Console() here would own its own cursor state and
# interleave badly with the live progress bars in olinda.console (Live repaints assume it is the only
# writer), producing torn or duplicated lines.
from olinda.console import console as console

_loguru.remove()

# WARNING+ only on the console: user-facing status goes through olinda.console, so anything quieter
# than a warning would just be noise competing with it.
_loguru.add(
    RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        log_time_format="%H:%M:%S",
        show_path=False,
    ),
    format="{message}",
    colorize=True,
    level="WARNING",
)


class Logger:
    """The usual levels plus ``success()``, following the Ersilia logging convention."""

    def debug(self, msg: str) -> None:
        _loguru.debug(msg)

    def info(self, msg: str) -> None:
        _loguru.info(msg)

    def warning(self, msg: str) -> None:
        _loguru.warning(msg)

    def error(self, msg: str) -> None:
        _loguru.error(msg)

    def critical(self, msg: str) -> None:
        _loguru.critical(msg)

    def success(self, msg: str) -> None:
        _loguru.success(msg)

    @contextmanager
    def stage(self, name: str):
        """Bracket a long step with a rule and a timed completion line."""
        console.rule(f"[bold]{name}[/bold]", style="cyan")
        self.info(f"Stage started: {name}")
        started = time.perf_counter()
        try:
            yield
        finally:
            self.success(
                f"Stage finished: {name}  ({time.perf_counter() - started:.2f}s)"
            )
            console.rule(style="cyan")


logger = Logger()
