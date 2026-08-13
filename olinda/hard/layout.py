"""Where a hard-label head keeps its files, and how to tell it finished.

Deliberately dependency-free — pathlib and nothing else. :mod:`olinda.export` needs these names and
:func:`has_hard_head` but none of the training machinery, so keeping them here means importing the
export path does not drag numpy and the featurizer in behind it.
"""

from __future__ import annotations

from pathlib import Path

# Layout under <model_dir>/
HARD_H5_NAME = "hard.h5"  # featurized hard labels written by `prepare --hard-labels`, consumed by `learn-hard`
HARD_DIRNAME = "_hard"
HARD_MODEL_SUBDIR = "h"
H_REFERENCE_NAME = (
    "h_reference.h5"  # H's score for every reference-library compound (row-aligned)
)
H_TO_S_NAME = "h_to_s.json"  # the isotonic map carrying H onto S's scale, producing H_S
TANIMOTO_DIRNAME = "tanimoto"  # the similarity regressor gating the hard signal
HARD_META_NAME = "hard_meta.json"
HARD_EVAL_NAME = "hard_eval.json"


def has_hard_head(model_dir: str | Path) -> bool:
    """True iff *model_dir* has a **complete** hard-label head.

    ``learn-hard`` writes `H` first and its metadata last, with several minutes of reference scoring in
    between, so the presence of the model says only that the step *started*. Interrupt it and the
    column would claim a head with no calibrator and no gate — which then fails the fuse with a missing
    file rather than simply being treated as soft-only. The metadata is written last, so it is the
    completion marker.
    """
    return (Path(model_dir) / HARD_DIRNAME / HARD_META_NAME).exists()
