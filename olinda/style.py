"""What an olinda figure looks like — the one place that decides.

The terminal look lives in :mod:`olinda.console`; this is its counterpart for figures. Both the
validation report (:mod:`olinda.report.plots`) and the training diagnostics
(:mod:`olinda.train.plots`) draw through here, so a colour means the same thing in every figure
olinda produces, and the report page can paint its swatches from the same table via
:func:`hexcol` instead of restating the hexes in CSS.

Figures are **publication-oriented**: stylia's ``article`` style (the non-branded NPG palette,
with a neutral foreground rather than Ersilia plum) and its ``print`` format (the 7.09-inch
Nature two-column width), so a panel can be dropped straight into a paper. The Ersilia identity
belongs to the page around them, not to the data marks.

stylia is an optional dependency (the ``report`` extra). Nothing here imports it at module
level; every helper takes the already-imported module from its caller, which keeps CLI startup
fast and lets each caller decide whether a missing stylia is an error or a skipped plot.
"""

from __future__ import annotations

import numpy as np

# Screen resolution rather than stylia's 600 dpi: the PNGs are viewed in a browser a few hundred
# pixels wide, where 600 dpi costs encoding time and disk for no visible gain. The PDF stays vector.
REPORT_DPI = 200

# The reference grid every figure is sized on. A figure declares a footprint in cells and its size
# follows, which is what keeps a report's panels commensurate instead of each plot inventing an
# aspect ratio. Six cells span stylia's print width (18 cm ≈ 7.09 in), so a cell is 3 cm.
CELLS_PER_WIDTH = 6

# Semantic roles → stylia ArticleColors names. Plots ask for a *meaning*, never a colour, so the
# same quantity keeps its hue across the whole report. Taken by name rather than by palette index
# on purpose: CategoricalPalette.get(n) runs farthest-point selection and does NOT return the
# first n entries, so indices would shift as soon as a plot asked for a different count.
ROLES = {
  "model": "periwinkle",  # what olinda predicts — the surrogate, the blended output, ROC/PR
  "teacher": "cobalt",  # the teacher values the student is judged against
  "hard": "orchid",  # the ground-truth head and its calibration
  "gate": "amber",  # applicability / blend weight
  "active": "crimson",  # the positive class
  "inactive": "cobalt",  # the negative class
  "neutral": "silver",  # diagonals, chance lines, no-skill baselines
}


def setup(stylia) -> None:
  """Put stylia into the report's style and format. Cheap, idempotent, safe to call per figure."""
  stylia.set_style("article")
  stylia.set_format("print")


def rgb(stylia, role: str):
  """The matplotlib colour for a semantic role (see :data:`ROLES`)."""
  return stylia.ArticleColors().get(ROLES[role])


def hexcol(role: str) -> str:
  """The hex string for a semantic role, for the report page's CSS.

  Resolved from stylia's own table when it is installed, so the figures and the page cannot
  drift apart, and from a copy of that table when it is not — the HTML is written on a machine
  that may not have the plotting extra.
  """
  try:
    from stylia.colors.colors import _PAPER
  except ImportError:
    return _PAPER_FALLBACK[ROLES[role]]
  return _PAPER[ROLES[role]]


# stylia's article palette, copied so `hexcol` works without the report extra installed. Kept in
# sync by test_report.py, which asserts the two agree whenever stylia IS importable.
_PAPER_FALLBACK = {
  "crimson": "#E63946",
  "tangerine": "#F4845F",
  "amber": "#FCBF49",
  "lime": "#6BBF59",
  "turquoise": "#2EC4B6",
  "cobalt": "#457B9D",
  "periwinkle": "#6C5CE7",
  "orchid": "#B05CC8",
  "fuchsia": "#E91E8C",
  "silver": "#A0A0A0",
}


def figure(stylia, cells: tuple[int, int] = (2, 2)):
  """Create a one-panel figure occupying ``cells`` = ``(rows, cols)`` of the 3 cm grid.

  Returns ``(fig, ax)`` with the axis already prepared — see :func:`prepare` for why that
  matters. Sizes are fractions of stylia's format width, so ``(2, 2)`` is a 2.36-inch square
  and ``(2, 4)`` is twice as wide as it is tall.
  """
  rows, cols = cells
  fig, axs = stylia.create_figure(1, 1, width=cols / CELLS_PER_WIDTH, height=rows / CELLS_PER_WIDTH)
  return fig, prepare(axs.next())


def prepare(ax):
  """Blank the placeholder axis labels stylia stamps on every axis it hands out.

  stylia 1.0.1's ``AxisManager`` re-applies ``"X-axis / Units"`` / ``"Y-axis / Units"`` on
  *every* access, so an axis must be taken once and cleared immediately; fetching the same
  panel a second time silently restores them. Plots that set their own labels overwrite these
  blanks, and plots that legitimately have none (a bare distribution) then render clean.
  """
  ax.set_xlabel("")
  ax.set_ylabel("")
  return ax


def reference(ax, stylia, kind: str, *, value: float = 0.0, lo: float = 0.0, hi: float = 1.0):
  """Draw a baseline — one recipe, so every "this is chance" line in the report looks alike.

  ``kind`` is ``"diagonal"`` (y=x between *lo* and *hi*), ``"horizontal"`` or ``"vertical"``
  (at *value*). Always silver, dashed, and behind the data.
  """
  style = {"linestyle": "--", "color": rgb(stylia, "neutral"), "linewidth": 1, "zorder": 0}
  if kind == "diagonal":
    ax.plot([lo, hi], [lo, hi], **style)
  elif kind == "horizontal":
    ax.axhline(value, **style)
  elif kind == "vertical":
    ax.axvline(value, **style)
  else:
    raise ValueError(f"unknown reference line {kind!r}")


def density(ax, stylia, x, y, *, role: str = "teacher", label: str = "Compounds"):
  """Hexbin of ``x`` against ``y``, shaded by count, with a colourbar.

  A scatter cannot show this data: a validation set is tens of thousands of compounds and the
  reference library is 1.36 million, so markers overplot into a solid blob long before the
  interesting structure appears. Binning shows where the mass actually is, and costs the same
  whether there are ten thousand points or a million.

  Counts are shaded on a **log** scale. Chemical data of this kind is extremely peaked — one bin
  routinely holds a thousand compounds while the structure worth seeing sits in bins of two or
  three — and on a linear ramp that single bin takes the whole colour range and washes every
  other one to white.
  """
  from matplotlib.colors import LinearSegmentedColormap, LogNorm

  x = np.asarray(x, dtype=np.float64).ravel()
  y = np.asarray(y, dtype=np.float64).ravel()
  cmap = LinearSegmentedColormap.from_list("olinda_density", ["#FFFFFF", hexcol(role)])
  hb = ax.hexbin(x, y, gridsize=45, cmap=cmap, mincnt=1, linewidths=0, norm=LogNorm())
  bar = ax.figure.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
  bar.set_label(label)
  bar.outline.set_visible(False)
  bar.ax.tick_params(length=2)
  return hb


def limits(ax, *values, pad: float = 0.03) -> tuple[float, float]:
  """Set both axes to one shared, slightly padded range, and return it.

  Square-data-space plots (anything with a y=x line) are misleading without this: matplotlib
  scales each axis independently, so the diagonal is drawn at whatever angle the aspect happens
  to give and "above the line" stops meaning what it looks like.
  """
  flat = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in values])
  lo, hi = float(flat.min()), float(flat.max())
  margin = (hi - lo) * pad or 0.01
  lo, hi = lo - margin, hi + margin
  ax.set_xlim(lo, hi)
  ax.set_ylim(lo, hi)
  return lo, hi
