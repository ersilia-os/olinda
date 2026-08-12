"""The package logger: a loguru singleton on Rich, for diagnostics rather than status.

User-facing output belongs to :mod:`olinda.console` — steps, panels, progress, summaries. This is the
other half: silent DEBUG diagnostics, and genuine warnings and errors that surface. Both write through
the one shared Console, so they interleave correctly with the live progress regions.

    from olinda.utils.logging import logger

    logger.debug("gathered %d rows", n)
    logger.warning("skipping invalid SMILES: %s", smi)

Note the console handler is attached at WARNING, so ``debug``, ``info`` and ``success`` are silent
by design — they exist for their side effect on a ``-v`` future and for the record, not to print.
Anything a user should read goes through :mod:`olinda.console`.
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
        """Record *msg* for diagnosis. Below the console handler's level, so nothing is printed."""
        _loguru.debug(msg)

    def info(self, msg: str) -> None:
        """Record *msg*. Also below the handler's level — user-facing status is console's job."""
        _loguru.info(msg)

    def warning(self, msg: str) -> None:
        """Report something suspect that did not stop the run. The quietest level that prints."""
        _loguru.warning(msg)

    def error(self, msg: str) -> None:
        """Report a failure the caller is expected to handle or surface."""
        _loguru.error(msg)

    def critical(self, msg: str) -> None:
        """Report a failure that ends the run."""
        _loguru.critical(msg)

    def success(self, msg: str) -> None:
        """Record completion of a step. Silent for the same reason as :meth:`info`."""
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
