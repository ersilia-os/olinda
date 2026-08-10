from __future__ import annotations

import time
from contextlib import contextmanager, suppress
from pathlib import Path

from loguru import logger as _loguru
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# One console for the whole package. A second Console() here would own its own cursor state and
# interleave badly with the live progress bars in olinda.console (Live repaints assume it is the only
# writer), producing torn or duplicated lines.
from olinda.console import console as console

_loguru.remove()

_rich_handler = RichHandler(
  console=console,
  rich_tracebacks=True,
  markup=True,
  log_time_format="%H:%M:%S",
  show_path=False,
)
# WARNING+ only on the console: user-facing status goes through olinda.console; loguru is for genuine
# warnings/errors and (silent) DEBUG diagnostics. A file sink (add_file) can still capture everything.
_console_id = _loguru.add(_rich_handler, format="{message}", colorize=True, level="WARNING")


class Logger:
  def __init__(self) -> None:
    self._loguru = _loguru
    self._console = console
    self._file_id: int | None = None

  def debug(self, msg: str) -> None:
    self._loguru.debug(msg)

  def info(self, msg: str) -> None:
    self._loguru.info(msg)

  def warning(self, msg: str) -> None:
    self._loguru.warning(msg)

  def error(self, msg: str) -> None:
    self._loguru.error(msg)

  def critical(self, msg: str) -> None:
    self._loguru.critical(msg)

  def success(self, msg: str) -> None:
    self._loguru.success(msg)

  @property
  def rich(self):
    """Direct access to the shared Rich console for advanced output."""
    return self._console

  def panel(
    self, text: str, *, title: str = "", style: str = "bold cyan", border_style: str = "cyan"
  ) -> None:
    """Print a Rich Panel to the console."""
    self._console.print(Panel(Text(text), title=title, style=style, border_style=border_style))

  def table(self, rows: list[list[str]], *, headers: list[str] | None = None, title: str = "") -> None:
    """Print a Rich Table to the console."""
    t = Table(title=title, show_header=bool(headers), header_style="bold magenta")
    if headers:
      for h in headers:
        t.add_column(h, justify="right" if h != headers[0] else "left")
    for row in rows:
      t.add_row(*[str(c) for c in row])
    self._console.print(t)

  def metrics_table(self, metrics: dict, *, title: str = "Metrics") -> None:
    """Pretty-print a {name: value} dict as a two-column Rich table."""
    rows = []
    for k, v in metrics.items():
      if isinstance(v, float):
        rows.append([k, f"{v:.6f}"])
      else:
        rows.append([k, str(v)])
    self.table(rows, headers=["Metric", "Value"], title=title)

  def rule(self, text: str = "", style: str = "cyan") -> None:
    self._console.rule(text, style=style)

  @contextmanager
  def stage(self, name: str):
    """Context manager that logs stage start/end with elapsed time."""
    self.rule(f"[bold]{name}[/bold]")
    self.info(f"Stage started: {name}")
    t0 = time.perf_counter()
    try:
      yield
    finally:
      elapsed = time.perf_counter() - t0
      self.success(f"Stage finished: {name}  ({elapsed:.2f}s)")
      self.rule()

  def add_file(self, path: str | Path, level: str = "DEBUG") -> None:
    """Add a file sink to loguru."""
    if self._file_id is not None:
      with suppress(Exception):
        self._loguru.remove(self._file_id)
    self._file_id = self._loguru.add(
      str(path),
      level=level,
      format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
      rotation="10 MB",
    )

  def remove_file(self) -> None:
    if self._file_id is not None:
      with suppress(Exception):
        self._loguru.remove(self._file_id)
      self._file_id = None


logger = Logger()
