"""Live terminal regions: the spinner, status line, and the sweep and epoch progress bars."""

from __future__ import annotations

import time
from contextlib import contextmanager

from rich.live import Live

from olinda.console.base import active_color, console, echo, elapsed

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


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


def claim_live_region(owner: object) -> None:
    """Record *owner* as the one object repainting the terminal's live region."""
    global _live_owner
    _live_owner = owner


def release_live_region() -> None:
    """Forget the live-region owner, so the next display may take the terminal."""
    global _live_owner
    _live_owner = None


def live_region_owner() -> object | None:
    """The object that currently owns the live region, or ``None``."""
    return _live_owner


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
    owner = live_region_owner()
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
            """Record *done* items finished and refresh the bar, its rate and its ETA."""
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
            """Record one epoch's loss and RMSE, and redraw the status line when a terminal is attached."""
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
