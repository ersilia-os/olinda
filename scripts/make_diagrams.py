"""Generate the explanatory diagrams for olinda (presentation slides).

Four standalone PNGs, drawn in the Ersilia house style via stylia — the same footing as
``olinda/train/plots.py``. These are schematic box-and-arrow figures rather than data plots, so each
panel is an axis with the frame switched off and a fixed 0-100 layout grid.

    python scripts/make_diagrams.py [--out-dir docs/diagrams]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import stylia
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Format: slide | Style: ersilia — change with stylia.set_format() / stylia.set_style()
stylia.set_format("slide")
stylia.set_style("ersilia")

NC = stylia.NamedColors()

# One semantic palette across all four figures, so a colour means the same thing on every slide.
TEACHER = NC.purple  # the teacher / hard-label side
STUDENT = NC.blue  # the surrogate
BUNDLE = NC.mint  # the fused ONNX artifact
GATE = NC.pink  # the applicability gate
DATA = NC.gray  # data and files
OUT = NC.orange  # outputs


def _canvas(ax) -> None:
  """Turn an axis into a blank 0-100 layout grid."""
  ax.set_xlim(0, 100)
  ax.set_ylim(0, 100)
  ax.axis("off")


def _box(ax, x, y, w, h, text, color, *, dashed=False, fontsize=None, bold_first=False, fill=0.16):
  """Rounded box centred on ``(x, y)`` with centred, optionally bold-first-line text."""
  ax.add_patch(
    FancyBboxPatch(
      (x - w / 2, y - h / 2),
      w,
      h,
      boxstyle="round,pad=0,rounding_size=1.4",
      linewidth=1.3,
      edgecolor=color,
      facecolor=to_rgba(color, fill),
      linestyle="--" if dashed else "-",
      mutation_aspect=0.45,
      zorder=2,
    )
  )
  if bold_first and "\n" in text:
    head, rest = text.split("\n", 1)
    ax.text(
      x,
      y + h * 0.20,
      head,
      ha="center",
      va="center",
      fontsize=fontsize or stylia.FONTSIZE_SMALL,
      fontweight="bold",
      color=NC.black,
      zorder=3,
    )
    ax.text(
      x,
      y - h * 0.16,
      rest,
      ha="center",
      va="center",
      fontsize=stylia.FONTSIZE_SMALL * 0.86,
      color=NC.black,
      zorder=3,
    )
  else:
    ax.text(
      x,
      y,
      text,
      ha="center",
      va="center",
      fontsize=fontsize or stylia.FONTSIZE_SMALL,
      color=NC.black,
      zorder=3,
    )


def _arrow(ax, start, end, *, color=None, dashed=False, elbow=None, label=None, label_dy=2.6):
  """Arrow from ``start`` to ``end``; ``elbow`` picks a right-angled connection style."""
  style = {"angle_h": "angle,angleA=0,angleB=90", "angle_v": "angle,angleA=90,angleB=0"}
  ax.add_patch(
    FancyArrowPatch(
      start,
      end,
      arrowstyle="-|>",
      mutation_scale=11,
      linewidth=1.2,
      color=color or NC.gray,
      linestyle="--" if dashed else "-",
      connectionstyle=style.get(elbow, "arc3,rad=0"),
      shrinkA=1,
      shrinkB=1,
      zorder=1,
    )
  )
  if label:
    ax.text(
      (start[0] + end[0]) / 2,
      (start[1] + end[1]) / 2 + label_dy,
      label,
      ha="center",
      va="bottom",
      fontsize=stylia.FONTSIZE_SMALL * 0.82,
      color=NC.gray,
      style="italic",
      zorder=3,
    )


def _note(ax, x, y, text, *, color=None, ha="center", italic=True):
  """Small free-standing annotation."""
  ax.text(
    x,
    y,
    text,
    ha=ha,
    va="center",
    fontsize=stylia.FONTSIZE_SMALL * 0.82,
    color=color or NC.gray,
    style="italic" if italic else "normal",
    zorder=3,
  )


# ── 1 · the big idea ─────────────────────────────────────────────────────────


def draw_distillation(ax) -> None:
  """Why olinda exists: score a heavy teacher once, then learn a fast student from it."""
  _canvas(ax)

  _box(ax, 15, 80, 26, 20, "TEACHER\noriginal model\naccurate · slow · heavy deps", TEACHER, bold_first=True)
  _box(ax, 50, 80, 26, 20, "REFERENCE LIBRARY\n~1.4M compounds\n2048-d Morgan counts", DATA, bold_first=True)
  _box(ax, 85, 80, 24, 20, "SOFT LABELS\none teacher value\nper compound", DATA, bold_first=True)

  _arrow(ax, (28.5, 80), (36.5, 80), label="score once")
  _arrow(ax, (63.5, 80), (72.5, 80))

  # fingerprints and teacher values meet in the student
  _arrow(ax, (50, 69.5), (50, 55), color=STUDENT)
  _arrow(ax, (85, 69.5), (67, 46), elbow="angle_v", color=STUDENT)
  _note(ax, 53.5, 62, "fingerprints", color=STUDENT, ha="left")
  _note(ax, 82, 58, "targets", color=STUDENT, ha="right")

  _box(
    ax,
    50,
    44,
    46,
    17,
    "STUDENT\ngradient boosting  ·  XGBoost on GPU / LightGBM on CPU",
    STUDENT,
    bold_first=True,
  )
  _arrow(ax, (50, 35), (50, 26), color=BUNDLE)
  _box(
    ax,
    50,
    18,
    56,
    13,
    "model.onnx  —  milliseconds per compound, none of the teacher's dependencies",
    BUNDLE,
  )
  _note(ax, 50, 6, "the teacher is never needed again", color=DATA)

  stylia.label(ax, xlabel="", ylabel="", title="olinda distils a slow teacher into a fast student")


# ── 2 · commands and artifacts ───────────────────────────────────────────────


def draw_pipeline(ax) -> None:
  """The CLI steps and what each one writes into the run directory."""
  _canvas(ax)

  steps = [
    ("setup", "erl0_morgan.h5\nreference fingerprints", False),
    ("prepare", "train.h5 · val.h5\nsoft.h5 · hard.h5", False),
    ("tune", "best_params.json", True),
    ("learn-soft", "booster + model.onnx\nval_metrics.json", False),
    ("learn-hard", "_ground_truth/\nmodel.onnx re-fused", True),
    ("export", "model.onnx\nrebuilt on demand", False),
  ]
  xs = [9.5, 25.5, 41.5, 57.5, 73.5, 89.5]
  w = 14.0

  for (name, artifact, optional), x in zip(steps, xs):
    _box(ax, x, 68, w, 11, name, STUDENT if not optional else DATA, dashed=optional)
    _box(ax, x, 46, w, 15, artifact, DATA, dashed=optional, fill=0.09)
    _arrow(ax, (x, 62), (x, 54), color=DATA)

  for x0, x1 in zip(xs[:-1], xs[1:]):
    _arrow(ax, (x0 + w / 2, 68), (x1 - w / 2, 68))

  _note(ax, 41.5, 78.5, "optional", color=DATA)
  _note(ax, 73.5, 78.5, "only with hard labels", color=DATA)

  # `fit` chains prepare → learn-hard
  ax.plot([18.5, 80.5], [90, 90], color=OUT, linewidth=1.3, zorder=1)
  for x in (18.5, 80.5):
    ax.plot([x, x], [86, 90], color=OUT, linewidth=1.3, zorder=1)
  ax.text(
    49.5,
    93,
    "olinda fit  — chains these in one command",
    ha="center",
    va="bottom",
    fontsize=stylia.FONTSIZE_SMALL,
    color=OUT,
    fontweight="bold",
    zorder=3,
  )

  _arrow(ax, (89.5, 38.5), (81, 24), elbow="angle_v", color=BUNDLE)
  _box(
    ax,
    46,
    18,
    64,
    13,
    "olinda predict  -m <run dir>  -i compounds.csv  -o predictions.csv",
    BUNDLE,
  )
  _note(ax, 46, 7, "every step shares one --model-dir / -m", color=DATA)

  stylia.label(ax, xlabel="", ylabel="", title="One run directory, one command per step")


# ── 3 · inside the fused bundle ──────────────────────────────────────────────


def draw_model_onnx(ax) -> None:
  """What the single self-describing model.onnx actually contains."""
  _canvas(ax)

  # the ONNX boundary
  ax.add_patch(
    FancyBboxPatch(
      (24, 16),
      57,
      66,
      boxstyle="round,pad=0,rounding_size=1.4",
      linewidth=1.6,
      edgecolor=BUNDLE,
      facecolor=to_rgba(BUNDLE, 0.06),
      mutation_aspect=0.45,
      zorder=0,
    )
  )
  ax.text(
    26,
    78,
    "model.onnx",
    ha="left",
    va="center",
    fontsize=stylia.FONTSIZE_SMALL,
    fontweight="bold",
    color=NC.black,
    zorder=3,
  )

  # featurization is the one stage that cannot live in the graph
  _box(ax, 9, 66, 15, 11, "SMILES", DATA)
  _box(ax, 9, 46, 16, 14, "RDKit\nMorgan featurizer", DATA, dashed=True, bold_first=True)
  _arrow(ax, (9, 60.5), (9, 53), color=DATA)
  _note(ax, 9, 33, "stays in Python\nRDKit has no ONNX op", color=DATA)
  _arrow(ax, (17, 46), (21, 46), color=DATA)
  _note(ax, 19.5, 51, "2048 counts", color=DATA)

  # three heads, fed from the same fingerprint
  _box(ax, 36, 66, 17, 13, "soft model\nGBM surrogate", STUDENT, bold_first=True)
  _box(ax, 36, 44, 17, 13, "hard model\nG", TEACHER, bold_first=True)
  _box(ax, 36, 24, 17, 13, "gate\n2 × Bernoulli NB", GATE, bold_first=True)
  for y in (66, 44, 24):
    _arrow(ax, (21.5, 46), (27, y), color=DATA)

  _box(ax, 57, 66, 15, 12, "isotonic\ncorrection", STUDENT, fill=0.09)
  _box(ax, 57, 44, 15, 12, "isotonic\nto soft scale", TEACHER, fill=0.09)
  _arrow(ax, (44.5, 66), (49.5, 66), color=STUDENT)
  _arrow(ax, (44.5, 44), (49.5, 44), color=TEACHER, label="ground_truth", label_dy=1.8)

  # everything converges on the blend
  _box(ax, 73, 44, 12, 20, "blend", OUT, bold_first=False)
  _arrow(ax, (64.5, 66), (69, 50), color=STUDENT)
  _note(ax, 63, 57, "surrogate", color=STUDENT, ha="right")
  _arrow(ax, (64.5, 44), (67, 44), color=TEACHER)
  _note(ax, 57, 35, "ground_truth_soft", color=TEACHER)
  _arrow(ax, (44.5, 24), (69, 38), elbow="angle_h", color=GATE)
  _note(ax, 56, 20, "applicability  a", color=GATE)

  _arrow(ax, (79, 44), (86, 44), color=OUT)
  _box(ax, 93, 44, 13, 13, "prediction", OUT)

  _note(
    ax,
    52,
    92,
    "prediction  =  (1 − a) · surrogate  +  a · ground_truth_soft",
    color=NC.black,
    italic=False,
  )
  _note(
    ax,
    52,
    9,
    "written to the output CSV:  prediction · surrogate · ground_truth · ground_truth_soft · applicability",
    color=DATA,
  )
  _note(
    ax,
    52,
    4,
    "metadata_props carries the featurizer config, the RDKit version and the reference library — "
    "so predict refuses a mismatched RDKit",
    color=DATA,
  )

  stylia.label(ax, xlabel="", ylabel="", title="One fused, self-describing artifact")


# ── 4 · ground truth and applicability ───────────────────────────────────────


def draw_ground_truth(ax) -> None:
  """The four learn-hard steps, and how the applicability gate weights the blend."""
  _canvas(ax)

  steps = [
    "1 · train G\non your hard labels",
    "2 · score G\nacross the library",
    "3 · calibrate\nisotonic → soft scale",
    "4 · fit the gate\nsimilarity buckets",
  ]
  xs = [14, 38, 62, 86]
  for text, x in zip(steps, xs):
    _box(ax, x, 80, 21, 16, text, TEACHER, bold_first=True)
  for x0, x1 in zip(xs[:-1], xs[1:]):
    _arrow(ax, (x0 + 10.5, 80), (x1 - 10.5, 80))
  _note(ax, 50, 68, "the isotonic direction is learned — a low G may map to a high soft label", color=DATA)

  # the gate: one arrow into a similarity ladder, so nothing is drawn through a box
  _box(ax, 14, 48, 22, 17, "nearest-neighbour\nTanimoto to your\nlabelled compounds", GATE, bold_first=True)
  _arrow(ax, (25.5, 48), (33.5, 48), color=GATE)
  buckets = [("NOT SIMILAR", "a = 0"), ("LOW", "a = 0.33"), ("HIGH", "a = 0.66")]
  bx = [44, 64, 84]
  for (name, weight), x in zip(buckets, bx):
    _box(ax, x, 48, 19, 17, f"{name}\n{weight}", GATE, bold_first=True)

  ax.annotate(
    "",
    xy=(93.5, 35),
    xytext=(34.5, 35),
    arrowprops={"arrowstyle": "-|>", "color": DATA, "linewidth": 1.1},
    zorder=1,
  )
  _note(ax, 64, 31, "increasing similarity to your labelled chemistry", color=DATA)

  _note(
    ax,
    50,
    22,
    "at predict time two Bernoulli NB classifiers reproduce the bucket — no similarity search",
    color=NC.black,
    italic=False,
  )
  _box(
    ax,
    50,
    11,
    66,
    13,
    "prediction  =  (1 − a) · surrogate  +  a · calibrated ground truth",
    OUT,
  )
  _note(ax, 50, 2.5, "far from your data the blend falls back to the surrogate", color=DATA)

  stylia.label(ax, xlabel="", ylabel="", title="Hard labels, calibrated onto the teacher's scale")


FIGURES = [
  ("olinda_01_distillation.png", draw_distillation, 0.52),
  ("olinda_02_pipeline.png", draw_pipeline, 0.50),
  ("olinda_03_model_onnx.png", draw_model_onnx, 0.56),
  ("olinda_04_ground_truth.png", draw_ground_truth, 0.50),
]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", default="docs/diagrams", help="Directory for the PNGs.")
  args = parser.parse_args()

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  for name, draw, height in FIGURES:
    # `height` is set per figure: these are schematics whose content dictates the aspect ratio,
    # not data plots where stylia's default slide ratio applies.
    fig, axs = stylia.create_figure(1, 1, height=height)
    draw(axs.next())
    stylia.save_figure(str(out_dir / name))
    print(f"wrote {out_dir / name}")


if __name__ == "__main__":
  main()
