"""The shared console and the one-shot output helpers: rules, steps, panels, banners and formatters."""

from __future__ import annotations

import time
from contextlib import contextmanager

from rich import box
from rich.console import Console
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
    """The colour of the step currently being shown, so nested output matches its rule."""
    return _active_color


_ICONS = {
    "success": ("✓", "green"),
    "warning": ("⚠", "yellow"),
    "error": ("✖", "red"),
    "run": ("▪", "cyan"),
    "info": ("·", "dim"),
}


def echo(text: str, kind: str = "info", *, sub: bool = False) -> None:
    """One-line status message with an Ersilia-style icon and color.

    ``sub`` indents by one more level, for a line that elaborates on the one above it rather than
    standing beside it. This exists so nesting is stated rather than faked: call sites used to open
    the *text* with two literal spaces, which put the padding inside the markup where nothing could
    keep it consistent, and left no way to tell a deliberate sub-item from a stray space.
    """
    icon, style = _ICONS.get(kind, _ICONS["info"])
    console.print(f"{'    ' if sub else '  '}[{style}]{icon}[/] {text}")


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


# One accent colour per command. Everything printed while a command runs is themed in its colour,
# so the terminal shifts hue as the pipeline advances and a long run stays readable at a glance.
STEP_COLORS = {
    "setup": "cyan",
    "library": "cyan",
    "prepare": "cyan",
    "learn-soft": "green",
    "learn-hard": "magenta",
    "export": "bright_cyan",
    "clean": "bright_black",
    "predict": "blue",
    "validate": "bright_magenta",
    "fit": "bright_green",
}


def path(value, keep: int = 3) -> str:
    """Render a filesystem path compactly: ``~/runs/my_model`` or ``…/run/columns/c0``.

    Paths under ``$HOME`` collapse to ``~``, which is both shorter and more readable. Anything still
    too long to fit a panel is elided from the head — absolute paths in temp or scratch directories
    otherwise fold mid-component and read as broken output, and only the tail identifies the artifact.
    """
    from pathlib import Path as _Path

    text = str(value)
    home = str(_Path.home())
    if text.startswith(home):
        text = "~" + text[len(home) :]
    if len(text) <= 44:
        return text
    parts = _Path(text).parts
    if len(parts) <= keep:
        return text
    return "…/" + "/".join(parts[-keep:])


def resources() -> str | None:
    """``CPU 62%  ·  RAM 18.4/32.0 GB (58%)`` — or ``None`` when psutil is unavailable.

    Shown on step rules and completion lines so resource use stays visible across a long run, not just
    during the stages that happen to print progress.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        cpu = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
    except Exception:
        return None
    return f"CPU {cpu:.0f}%  ·  RAM {vm.used / 1e9:.1f}/{vm.total / 1e9:.1f} GB ({vm.percent:.0f}%)"


def elapsed(seconds: float) -> str:
    """Human-readable duration: ``42s``, ``3m 07s``, ``1h 12m``."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def filesize(nbytes: float) -> str:
    """Human-readable size: ``812 B``, ``4.3 MB``, ``11.7 GB`` (decimal units, as disk tools report)."""
    nbytes = float(max(0, nbytes))
    if nbytes < 1000:
        return f"{nbytes:.0f} B"
    for unit in ("KB", "MB"):
        nbytes /= 1000
        if nbytes < 1000:
            return f"{nbytes:.1f} {unit}"
    return f"{nbytes / 1000:.1f} GB"


@contextmanager
def stage(title: str, *, color: str | None = None, right: str | None = None):
    """A themed section that opens with a rule and closes with a timed completion line.

    Sets the accent colour for everything printed inside, so nested helpers theme themselves without
    being passed the colour. The closing glyph is drawn in that accent (red only on failure), matching
    how a completed step reads as part of its section rather than as a generic success.

        with stage("olinda · learn-soft", color="green") as st:
            ...
            st.summary = "3 columns · R² 0.86"
    """
    color = color or active_color()
    previous = active_color()
    set_active_color(color)
    rule(title, style=color, right=right if right is not None else resources())

    class _Stage:
        summary: str | None = None

    handle = _Stage()
    started = time.time()
    ok = True
    try:
        yield handle
    except BaseException:
        ok = False
        raise
    finally:
        took = elapsed(time.time() - started)
        tail = f"{handle.summary} · {took}" if handle.summary else took
        res = resources()
        if res:
            tail += f" · {res}"
        glyph, gcolor = ("✓", color) if ok else ("✕", "red")
        console.print(f"  [{gcolor}]{glyph}[/] [dim]{tail}[/]")
        set_active_color(previous)


def detail(rows, *, indent: int = 3) -> None:
    """Borderless right-dim-label / left-value key-value block."""
    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2))
    table.add_column(justify="right", style="dim", no_wrap=True)
    table.add_column(justify="left", overflow="fold")
    for key, value in rows:
        table.add_row(str(key), str(value))
    console.print(Padding(table, (1, 0, 0, indent)))


def summary_panel(
    title: str, rows, *, border_style: str | None = None, icon: str | None = None
) -> None:
    """Rounded, left-titled panel of key-value rows (run header / final summary)."""
    color = border_style or active_color()
    table = Table(
        show_header=False, box=None, pad_edge=False, expand=False, padding=(0, 1)
    )
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
    "kde": (
        "◆",
        "black on cyan",
        "KDE INVERSE-DENSITY",
        "skewed / bimodal / tail-robust",
    ),
    "bins": (
        "▦",
        "black on yellow",
        "INVERSE-DENSITY BINS",
        "discrete / small-sample target",
    ),
}


def strategy_banner(strategy: str, reason: str, weight_range=None) -> None:
    """Highly visible, color-coded banner announcing the reweighting strategy in effect."""
    icon, badge_style, label, blurb = _STRATEGY_STYLE.get(
        strategy, ("·", "reverse", strategy.upper(), "")
    )
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


def engine_banner(backend: str, device: str, reason: str) -> None:
    """One-line banner announcing the auto-selected gradient-boosting engine and device."""
    style = "black on cyan" if backend == "xgboost" else "black on magenta"
    console.print(
        f"\n  [bold]⚙ ENGINE[/]  [{style}] {backend} · {device} [/]  [dim]{reason}[/]"
    )
