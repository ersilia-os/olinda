"""Curated terminal output for olinda — matches the zairachem-docker look and feel.

A single ``Console(highlight=False)``, Ersilia-style status glyphs, left-aligned themed rules,
borderless detail blocks, rounded summary panels, and a minimal live progress bar. Diagnostic
logging stays in :mod:`olinda.helpers` (loguru); this module is for user-facing output.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from rich import box
from rich.console import Console
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table

console = Console(highlight=False)

_active_color = "cyan"


def set_active_color(color: str) -> None:
  """Set the accent color used by rules/panels when none is given."""
  global _active_color
  _active_color = color


def active_color() -> str:
  return _active_color


_ICONS = {
  "success": ("✓", "green"),
  "warning": ("⚠", "yellow"),
  "error": ("✖", "red"),
  "run": ("▪", "cyan"),
  "info": ("·", "dim"),
}


def echo(text: str, kind: str = "info") -> None:
  """One-line status message with an Ersilia-style icon and color."""
  icon, style = _ICONS.get(kind, _ICONS["info"])
  console.print(f"  [{style}]{icon}[/] {text}")


def step(index: int, total: int, title: str) -> None:
  """A numbered step header line: ``▪ Step i/N · title`` (the workhorse of a command's step separation)."""
  echo(f"[bold]Step {index}/{total}[/] · {title}", "run")


def success(text: str) -> None:
  """A green ``✓`` completion line (use ``→`` in *text* for produced paths)."""
  echo(text, "success")


def rule(title: str, *, style: str | None = None, right: str | None = None) -> None:
  """Left-aligned themed section divider, with an optional dim right-side caption."""
  style = style or active_color()
  label = f"[bold {style}]{title}[/]"
  if right:
    label += f"   [dim]{right}[/]"
  console.rule(label, align="left", style=style)


def detail(rows, *, indent: int = 3) -> None:
  """Borderless right-dim-label / left-value key-value block."""
  table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2))
  table.add_column(justify="right", style="dim", no_wrap=True)
  table.add_column(justify="left", overflow="fold")
  for key, value in rows:
    table.add_row(str(key), str(value))
  console.print(Padding(table, (1, 0, 0, indent)))


def summary_panel(title: str, rows, *, border_style: str | None = None, icon: str | None = None) -> None:
  """Rounded, left-titled panel of key-value rows (run header / final summary)."""
  color = border_style or active_color()
  table = Table(show_header=False, box=None, pad_edge=False, expand=False, padding=(0, 1))
  table.add_column(justify="right", style=f"bold {color}", no_wrap=True)
  table.add_column(justify="left", overflow="fold")
  for key, value in rows:
    table.add_row(str(key), str(value))
  heading = f"{icon}  {title}" if icon else title
  console.print(
    Panel(
      table,
      title=f"[bold {color}]{heading}[/]",
      title_align="left",
      border_style=color,
      box=box.ROUNDED,
      padding=(1, 2),
      expand=False,
    )
  )


_STRATEGY_STYLE = {
  "none": ("○", "dim white", "NO REWEIGHTING", "natural distribution · faithful ERM"),
  "kde": ("◆", "black on cyan", "KDE INVERSE-DENSITY", "skewed / bimodal / tail-robust"),
  "bins": ("▦", "black on yellow", "INVERSE-DENSITY BINS", "discrete / small-sample target"),
}


def strategy_banner(strategy: str, reason: str, weight_range=None) -> None:
  """Highly visible, color-coded banner announcing the reweighting strategy in effect."""
  icon, badge_style, label, blurb = _STRATEGY_STYLE.get(strategy, ("·", "reverse", strategy.upper(), ""))
  badge = f"[{badge_style}] {label} [/]"
  console.print(f"\n  [bold]{icon} REWEIGHTING[/]  {badge}  [dim]{blurb}[/]")
  console.print(f"      [dim]why:[/] {reason}")
  if weight_range is not None:
    lo, hi = weight_range
    console.print(
      f"      [dim]weights ∈[/] [bold]{lo:.2f}–{hi:.2f}[/]  "
      f"[dim]· applied to train + val (coherent early stopping)[/]\n"
    )
  else:
    console.print("")


_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def engine_banner(backend: str, device: str, reason: str) -> None:
  """One-line banner announcing the auto-selected gradient-boosting engine and device."""
  style = "black on cyan" if backend == "xgboost" else "black on magenta"
  console.print(f"\n  [bold]⚙ ENGINE[/]  [{style}] {backend} · {device} [/]  [dim]{reason}[/]")


def spinner(i: int) -> str:
  """Return the spinner glyph for frame ``i`` (callers that lack a timer can pass an iteration count)."""
  return _SPINNER[i % len(_SPINNER)]


@contextmanager
def live_status(*, enabled: bool = True):
  """Yield an ``update(markup)`` that redraws a single transient status line in place.

  Falls back to a no-op updater when ``enabled`` is False or the console is not a TTY (e.g. piped/CI
  output), so callers should also emit plain milestone lines in that case.
  """
  if not enabled or not console.is_terminal:
    yield lambda markup: None
    return
  live = Live(console=console, transient=True, refresh_per_second=12)
  live.__enter__()
  try:
    yield live.update
  finally:
    live.__exit__(None, None, None)


def _bar(frac: float, width: int = 12) -> str:
  filled = round(frac * width)
  return f"{'█' * filled}[dim]{'─' * (width - filled)}[/]"


class LiveBar:
  """Minimal single-line progress bar (spinner + block bar) in the zairachem style."""

  def __init__(self, title: str, total: int, *, color: str | None = None) -> None:
    self.title = title
    self.total = max(int(total), 1)
    self.color = color or active_color()
    self.done = 0
    self._live = Live(console=console, transient=True, refresh_per_second=8)

  def __enter__(self) -> "LiveBar":
    self._live.__enter__()
    self._render()
    return self

  def update(self, done: int) -> None:
    self.done = done
    self._render()

  def _render(self) -> None:
    frac = min(self.done / self.total, 1.0)
    head = _SPINNER[int(time.time() * 10) % len(_SPINNER)]
    self._live.update(
      f"  [bold {self.color}]{head} {self.title}[/]  {_bar(frac)} [dim]{self.done:,}/{self.total:,}[/]"
    )

  def __exit__(self, *exc) -> None:
    self._live.__exit__(*exc)
