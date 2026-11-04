"""Curated terminal output for olinda — matches the zairachem-docker look and feel.

A single ``Console(highlight=False)``, Ersilia-style status glyphs, left-aligned themed rules,
borderless detail blocks, rounded summary panels, and a minimal live progress bar. Diagnostic
logging stays in :mod:`olinda.utils.logging` (loguru); this module is for user-facing output.

Split by what each part is for: :mod:`.base` holds the shared console and one-shot output,
:mod:`.live` the live regions and progress bars, :mod:`.table` the per-item live table. Everything
callers use is re-exported here, so ``from olinda.console import echo`` keeps working.
"""

from __future__ import annotations

from olinda.console.base import (
    STEP_COLORS,
    active_color,
    console,
    detail,
    echo,
    elapsed,
    engine_banner,
    filesize,
    path,
    resources,
    rule,
    set_active_color,
    stage,
    step,
    strategy_banner,
    success,
    summary_panel,
)
from olinda.console.live import (
    bar,
    epoch_progress,
    live_region_taken,
    live_status,
    spinner,
    status_line,
    sweep_progress,
)
from olinda.console.table import LiveTable

__all__ = [
    "STEP_COLORS",
    "LiveTable",
    "active_color",
    "bar",
    "console",
    "detail",
    "echo",
    "elapsed",
    "engine_banner",
    "epoch_progress",
    "filesize",
    "live_region_taken",
    "live_status",
    "path",
    "resources",
    "rule",
    "set_active_color",
    "spinner",
    "stage",
    "status_line",
    "step",
    "strategy_banner",
    "success",
    "summary_panel",
    "sweep_progress",
]
