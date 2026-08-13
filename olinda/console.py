"""Curated terminal output for olinda — matches the zairachem-docker look and feel.

A single ``Console(highlight=False)``, Ersilia-style status glyphs, left-aligned themed rules,
borderless detail blocks, rounded summary panels, and a minimal live progress bar. Diagnostic
logging stays in :mod:`olinda.utils.logging` (loguru); this module is for user-facing output.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from rich import box
from rich.console import Console, Group
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
    "tune": "bright_yellow",
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


_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def engine_banner(backend: str, device: str, reason: str) -> None:
    """One-line banner announcing the auto-selected gradient-boosting engine and device."""
    style = "black on cyan" if backend == "xgboost" else "black on magenta"
    console.print(
        f"\n  [bold]⚙ ENGINE[/]  [{style}] {backend} · {device} [/]  [dim]{reason}[/]"
    )


def spinner(i: int | None = None) -> str:
    """The spinner glyph for frame ``i``, or for *now* when ``i`` is omitted.

    Prefer the no-argument form. Passing an iteration count ties the animation to how often the caller
    happens to repaint, and a caller that updates every 25 rounds produces a spinner that jumps in
    fives — which reads as a stalled process rather than a slow one.
    """
    if i is None:
        i = int(time.time() * 8)
    return _SPINNER[i % len(_SPINNER)]


def bar(frac: float, width: int = 24) -> str:
    """A ``━`` progress bar filled to *frac*, dim for the remainder."""
    filled = int(round(max(0.0, min(1.0, frac)) * width))
    return "━" * filled + "[dim]━[/]" * (width - filled)


def status_line(
    verb: str,
    *,
    frac: float | None = None,
    pairs=(),
    tail: str | None = None,
    width: int = 24,
) -> str:
    """Build the one transient line a running stage repaints in place.

    Every progress display in olinda goes through here, because six of them grew up separately and
    drifted: three hardcoded ``bold cyan`` while their command was themed green, three drove the
    spinner off an iteration counter, and each invented its own spacing. The vocabulary is fixed —
    spinner and verb in the command's accent, an optional bar and percentage, ``·``-separated
    ``label value`` pairs, and a dim tail — so a new caller inherits the look instead of choosing one.

    Parameters
    ----------
    verb : str
        What is happening, in the present participle: ``"training"``, ``"scoring"``.
    frac : float, optional
        Progress in ``[0, 1]``. Omit when the total is unknown — the spinner still turns.
    pairs : sequence of (str, str)
        Label/value pairs. Labels render dim, values bold, so the numbers carry the line.
    tail : str, optional
        Dim trailing note — elapsed time, a best-so-far marker.
    """
    parts = [f"[{active_color()}]{spinner()} {verb}[/]"]
    if frac is not None:
        parts.append(f"{bar(frac, width)} [bold]{frac:>4.0%}[/]")
    for label, value in pairs:
        # An empty label is a bare value — a count that needs no naming next to the bar beside it.
        parts.append(
            f"[dim]{label}[/] [bold]{value}[/]" if label else f"[bold]{value}[/]"
        )
    line = "  [dim]·[/]  ".join(parts)
    return f"{line}  [dim]· {tail}[/]" if tail else line


_live_owner: object | None = None


def live_region_taken() -> bool:
    """True while something already owns the terminal's live region.

    Rich's ``Live`` assumes it is the only writer repainting the screen, so two of them fight and tear.
    A long-running inner stage checks this and stays quiet rather than competing with the outer
    progress display that is already showing its status.
    """
    return _live_owner is not None


@contextmanager
def live_status(*, enabled: bool = True):
    """Yield an ``update(markup, detail=None, **values)`` that redraws one transient line in place.

    When an outer live region already owns the screen, the updates are *handed to that owner* rather
    than dropped — a :class:`LiveTable` shows them beside itself and folds ``values`` into the running
    row. Silencing them instead is what makes a long stage look hung: the inner stage is the only
    thing that knows how far along it is.

    Two channels, because the two destinations want different shapes. ``markup`` is a finished line
    for a bare live region, which has the full terminal width to itself. ``detail`` is a small mapping
    of label → value for a :class:`LiveTable`, which has only the margin beside its table and already
    shows most of these numbers in the row — so it needs the few that are missing, not a sentence.

    Falls back to a no-op updater when ``enabled`` is False or the console is not a TTY (piped/CI
    output), so callers should still emit plain milestone lines for the log.
    """
    owner = _live_owner
    if enabled and owner is not None and hasattr(owner, "progress"):
        yield owner.progress
        return
    if not enabled or live_region_taken() or not console.is_terminal:
        yield lambda markup, detail=None, **values: None
        return
    live = Live(console=console, transient=True, refresh_per_second=12)
    live.__enter__()
    try:
        yield lambda markup, detail=None, **values: live.update(markup)
    finally:
        live.__exit__(None, None, None)


@contextmanager
def sweep_progress(verb: str, total: int, *, width: int = 24):
    """Yield a ``tick(done)`` reporting a chunked pass over the reference library on ONE line.

    A full-library sweep runs to ~27 chunks, and echoing each one buries the surrounding steps in a wall
    of near-identical lines. On a terminal this repaints in place with a bar, rate and ETA; when the
    output is piped it falls back to a handful of milestones (every ~25%) so a log stays short but a
    stalled run is still visible.
    """
    started = time.time()
    total = max(1, int(total))
    state = {"milestone": 0}

    with live_status() as update:

        def tick(done: int) -> None:
            done = min(int(done), total)
            frac = done / total
            took = time.time() - started
            rate = done / took if took > 0 else 0.0
            eta = (
                f" · eta {elapsed((total - done) / rate)}"
                if rate > 0 and done < total
                else ""
            )
            if console.is_terminal and not live_region_taken():
                update(
                    status_line(
                        verb,
                        frac=frac,
                        pairs=[("", f"{done:,}/{total:,}")],
                        tail=f"{rate / 1000:.0f}k/s{eta}",
                        width=width,
                    )
                )
            elif frac >= state["milestone"] + 0.25 or done == total:
                state["milestone"] = frac
                echo(f"{verb} {frac:>4.0%} · {done:,}/{total:,}{eta}", "info", sub=True)

        yield tick
    echo(
        f"{verb} complete · [bold]{total:,}[/] compounds [dim]· {elapsed(time.time() - started)}[/]",
        "info",
        sub=True,
    )


@contextmanager
def epoch_progress(verb: str, total: int, *, width: int = 24):
    """Yield a ``report(epoch, loss, rmse, improved, waited, patience)`` for an iterative fit, on ONE line.

    The same reasoning as :func:`sweep_progress`, applied to epochs instead of chunks: fifteen lines
    differing only in their fourth decimal bury the steps around them, and the reader's question is not
    "what was epoch 11" but "did it converge, and to what". So the per-epoch detail repaints in place
    on a terminal and collapses to a single closing line that names the best loss and whether early
    stopping fired. When the output is piped there is no live region, so nothing is printed until that
    closing line — an epoch is seconds, not minutes, and a stalled fit shows up in the step above.
    """
    started = time.time()
    total = max(1, int(total))
    state: dict = {"best": None, "epochs": 0}

    with live_status() as update:

        def report(epoch, loss, rmse, improved, waited=0, patience=None):
            state["epochs"] = int(epoch)
            if improved or state["best"] is None:
                state["best"] = (float(loss), float(rmse))
            if not console.is_terminal or live_region_taken():
                return
            note = (
                "best"
                if improved
                else f"no gain ({waited}/{patience})"
                if patience
                else "no gain"
            )
            update(
                status_line(
                    verb,
                    frac=epoch / total,
                    pairs=[("epoch", f"{epoch}/{total}"), ("val MSE", f"{loss:.6f}")],
                    tail=f"±{rmse:.3f} · {note}",
                    width=width,
                )
            )

        yield report

    loss, rmse = state["best"] or (float("nan"), float("nan"))
    ran = state["epochs"]
    # Early stopping is worth a word: a fit that used its whole budget may simply have run out of it.
    reach = (
        f"[bold]{ran}[/] epochs"
        if ran >= total
        else f"stopped early at [bold]{ran}[/]/{total}"
    )
    echo(
        f"{verb} complete · {reach} · best val MSE {loss:.6f} [dim]· ±{rmse:.3f} similarity "
        f"· {elapsed(time.time() - started)}[/]",
        "info",
        sub=True,
    )


class _Dynamic:
    """Renderable proxy that re-renders its owner every time Rich asks for a frame."""

    def __init__(self, owner) -> None:
        self.owner = owner

    def __rich__(self):
        return self.owner._renderable()


class LiveTable:
    """A fixed-height live table with one row per work item.

    Every row exists from the start as ``queued``, so the table never grows and the display cannot
    reflow mid-run. While it is open it owns the terminal's live region, and inner stages fall silent
    (see :func:`live_region_taken`) rather than fighting it for the screen.

    Falls back to plain one-line-per-item output when the console is not a TTY, so CI logs stay
    readable and ordered.

    Because it owns the live region, it is also responsible for showing what the silenced inner stage
    would have said: pass :meth:`progress` as that stage's status updater and its line renders under
    the table, and feed :meth:`update` the metrics it computes so the row itself ticks. Without that a
    long single-item run shows a spinner and nothing else, which is indistinguishable from a hang.

        with LiveTable(["a", "b"], title="Training", fields=["R²", "Time"]) as table:
            table.start("a")
            table.update("a", **{"R²": "0.71"})   # while running
            table.finish("a", **{"R²": "0.86", "Time": "42s"})
    """

    def __init__(
        self,
        items,
        *,
        title: str,
        fields,
        item_label: str = "Item",
        running_verb: str = "running",
        color: str | None = None,
    ) -> None:
        self.items = [str(i) for i in items]
        self.title = title
        self.fields = list(fields)
        self.item_label = item_label
        self.running_verb = running_verb
        self.color = color or active_color()
        self._state = {
            i: {"status": "queued", "started": None, "elapsed": None, "values": {}}
            for i in self.items
        }
        self._live = None
        self._note = ""  # the fallback line, shown under the table when the detail cannot fit beside
        self._detail: dict = {}

    # -- rendering ---------------------------------------------------------

    def _status_cell(self, s: dict) -> str:
        if s["status"] == "queued":
            return "[dim]queued[/]"
        if s["status"] == "running":
            # Driven off the clock, not a call counter, so it keeps moving between updates — a stalled
            # spinner reads as a hung process even when the work underneath is fine.
            return f"[{self.color}]{spinner()} {self.running_verb}[/]"
        if s["status"] == "failed":
            return "[red]failed[/]"
        return "[green]✓ done[/]"

    def _elapsed_cell(self, s: dict) -> str | None:
        """Live elapsed for a running row; ``None`` when the caller supplies its own value."""
        if s["status"] == "running" and s["started"] is not None:
            return f"[dim]{elapsed(time.time() - s['started'])}[/]"
        return None

    def _render(self) -> Table:
        done = sum(1 for s in self._state.values() if s["status"] in ("done", "failed"))
        table = Table(
            title=f"[bold {self.color}]{self.title} · {done}/{len(self.items)}[/]",
            title_justify="left",
            box=box.SIMPLE_HEAD,
            border_style=self.color,
            header_style=f"bold {self.color}",
            pad_edge=False,
            expand=False,
        )
        # Both widths are sized to their content rather than generously: the margin they free is what
        # the detail panel occupies, and a task name is identified by its tail as readily as its head.
        table.add_column(
            self.item_label, no_wrap=True, overflow="ellipsis", max_width=24
        )
        table.add_column("Status", no_wrap=True, width=13)
        for name in self.fields:
            table.add_column(name, justify="right", no_wrap=True)
        ticking = self._elapsed_cell
        for item in self.items:
            s = self._state[item]
            row = [item, self._status_cell(s)]
            for f in self.fields:
                value = s["values"].get(f)
                if value is None and f == "Time":
                    value = ticking(s)
                row.append(str(value) if value is not None else "[dim]—[/]")
            table.add_row(*row)
        return table

    def _detail_panel(self):
        """The running item's extra numbers, as a narrow panel — or ``None`` when there are none."""
        if not self._detail:
            return None
        grid = Table.grid(padding=(0, 1))
        grid.add_column(justify="left", style="dim", no_wrap=True)
        grid.add_column(justify="right", no_wrap=True)
        for label, value in self._detail.items():
            grid.add_row(label, str(value))
        return Panel(
            grid,
            border_style=self.color,
            box=box.ROUNDED,
            padding=(0, 1),
            expand=False,
        )

    def _renderable(self):
        """The table, with the running item's detail beside it if it fits, else beneath it.

        Beside is the point — the detail describes the row it sits next to, and putting it underneath
        separated the two by the whole width of the table. It only fits because the detail carries just
        what the row does not: the numbers the row already shows are not repeated here.

        Rich will happily squeeze a table into whatever space is left, silently truncating cells to do
        it, so the fit is measured rather than assumed. On a terminal too narrow for both, the line
        goes back underneath, which is cramped but never mangles the table.
        """
        table = self._render()
        panel = self._detail_panel()
        if panel is not None:
            grid = Table.grid(padding=(0, 3))
            grid.add_column()
            grid.add_column(
                vertical="bottom"
            )  # beside the rows, not floating up by the title
            grid.add_row(table, panel)
            # Measured against an unbounded width, not the console's. Rich clamps a measurement to the
            # space available, so asking it directly reports "fits" for every terminal — while the
            # render quietly ellipsises the cells to make that true.
            wide = console.options.update(width=100_000)
            if console.measure(grid, options=wide).maximum <= console.width:
                return grid
            # Too narrow for both. Fold the same detail into one line rather than the caller's full
            # markup, which carries a progress bar and every pair and would wrap on exactly the narrow
            # terminal that sent us here — a wrapped status line reflows the display on every repaint.
            folded = "  [dim]·[/]  ".join(
                f"[dim]{label}[/] [bold]{value}[/]"
                for label, value in self._detail.items()
            )
            return Group(table, "", f"  {folded}")
        if not self._note:
            return table
        return Group(table, "", self._note)

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.refresh()

    # -- lifecycle ---------------------------------------------------------

    def start(self, item) -> None:
        """Mark *item* as running and begin timing it."""
        item = str(item)
        self._state[item].update(status="running", started=time.time())
        if self._live is None:
            i = self.items.index(item) + 1
            echo(f"{self.running_verb} {item} [dim]({i}/{len(self.items)})[/]", "run")
        self._refresh()

    def update(self, item, **values) -> None:
        """Set named cells on *item*'s row without changing its status."""
        self._state[str(item)]["values"].update(values)
        self._refresh()

    def progress(self, markup: str, detail=None, **values) -> None:
        """Show the running item's live detail beside the table, and tick its row with *values*.

        This is what :func:`live_status` hands to an inner stage while this table owns the live region,
        so the stage's own progress reporting lands here instead of being dropped. *detail* is what
        renders beside the table; *markup* is kept only for the narrow-terminal fallback.
        """
        self._note = markup
        self._detail = dict(detail or {})
        if values:
            running = next(
                (i for i, s in self._state.items() if s["status"] == "running"), None
            )
            if running is not None:
                self._state[running]["values"].update(values)
        self._refresh()

    def finish(self, item, ok: bool = True, **values) -> None:
        """Close *item* out as done or failed, stamping its elapsed time and final *values*."""
        item = str(item)
        s = self._state[item]
        s["status"] = "done" if ok else "failed"
        if s["started"] is not None:
            s["elapsed"] = time.time() - s["started"]
        s["values"].update(values)
        self._note = ""  # the finished row carries the numbers now
        self._detail = {}
        if self._live is None:
            shown = "  ".join(f"{k} {v}" for k, v in s["values"].items())
            echo(f"{'✓' if ok else '✖'} {item}  [dim]{shown}[/]", "info")
        self._refresh()

    def __enter__(self) -> LiveTable:
        global _live_owner
        if console.is_terminal and not live_region_taken():
            # Live is handed a proxy, not a rendered table: it re-renders on every auto-refresh tick, so
            # the spinner turns and the clock advances between updates. Handing it a built Table instead
            # would redraw the same frozen frame 8x/second — which is what a hung run looks like.
            self._live = Live(
                _Dynamic(self), console=console, transient=True, refresh_per_second=8
            )
            self._live.__enter__()
            _live_owner = self
        return self

    def __exit__(self, *exc) -> None:
        global _live_owner
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None
            _live_owner = None
            console.print(
                self._render()
            )  # persist the finished table as the step's record
