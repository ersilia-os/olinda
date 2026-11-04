"""``LiveTable``: a fixed-height live table with one row per work item."""

from __future__ import annotations

import time

from rich import box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from olinda.console.base import active_color, console, echo, elapsed
from olinda.console.live import (
    claim_live_region,
    live_region_taken,
    release_live_region,
    spinner,
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
        if console.is_terminal and not live_region_taken():
            # Live is handed a proxy, not a rendered table: it re-renders on every auto-refresh tick, so
            # the spinner turns and the clock advances between updates. Handing it a built Table instead
            # would redraw the same frozen frame 8x/second — which is what a hung run looks like.
            self._live = Live(
                _Dynamic(self), console=console, transient=True, refresh_per_second=8
            )
            self._live.__enter__()
            claim_live_region(self)
        return self

    def __exit__(self, *exc) -> None:
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None
            release_live_region()
            console.print(
                self._render()
            )  # persist the finished table as the step's record
