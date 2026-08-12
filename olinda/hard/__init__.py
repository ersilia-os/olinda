"""Learn from hard (experimental) labels and calibrate them onto the teacher's soft-label scale.

olinda's surrogate ``S(x)`` distills a teacher's *soft* labels. When real experimental (*hard*) labels of
the same endpoint are available, ``learn-hard`` runs four steps, each printed clearly:

1. **Train ``H``** — a binary hard-label classifier: a plain, portfolio-selected XGBoost booster via `lazy-qsar
   <https://github.com/ersilia-os/lazy-qsar>`_'s ``BaseXGBClassifier(calibrated=False)`` on olinda's Morgan
   count fingerprints (:class:`~olinda.featurizer.MorganCountFeaturizer`). Output is raw ``predict_proba`` —
   lazy-qsar's internal probability calibrator is off. (Continuous hard labels are a raises-for-now
   placeholder — see :func:`_new_hard_model`.)
2. **Score ``H`` across the reference library** (``erl0_morgan.h5``) → one hard score per reference
   compound, saved to ``h_reference.h5``.
3. **Calibrate** ``H`` onto the soft-label scale — a monotonic isotonic map fit on the reference library
   (where both ``H``'s score and the teacher's soft label exist), with the **direction learned from the
   data** (a low hard score may map to a high soft label). Saved as ``h_to_s.json``.
4. **Learn T** — label every reference compound with its exact 1-NN Tanimoto similarity
   to the labeled set, then fit a small MLP that predicts that number from the fingerprint alone (saved
   under ``tanimoto/`` as ``t.onnx`` + ``t_meta.json``). At predict time two matrix multiplies
   estimate the similarity and a linear ramp turns it into ``a``, so nothing searches the labeled set and
   the labeled fingerprints never leave the run — see :mod:`olinda.tanimoto`.

The end goal is to predict the soft-label distribution informed by the hard labels. Artifacts land under
``<model_dir>/_hard/``. The gate decides *where* to trust ``H``: the blend
``prediction = (1-a)·S + a·G_soft`` leans on the hard signal only near the labeled chemistry, and how far it
can ever lean is capped by ``a_max``, which the head earns from how well its calibrated output reproduces
the teacher's scale (:func:`_blend_ceiling`) — a head that loses to the surrogate earns zero and the model
ships soft-only. All stages are fused into a single ``model.onnx`` (see :mod:`olinda.export`) and served by
:class:`~olinda.artifact.OlindaArtifact`.
"""

from __future__ import annotations

from olinda.hard.gate import _blend_ceiling as _blend_ceiling
from olinda.hard.labels import (
    MIN_HARD_ROWS as MIN_HARD_ROWS,
)
from olinda.hard.labels import (
    MIN_PER_CLASS as MIN_PER_CLASS,
)
from olinda.hard.labels import (
    prepare_hard_labels_wide as prepare_hard_labels_wide,
)
from olinda.hard.layout import (
    H_REFERENCE_NAME as H_REFERENCE_NAME,
)
from olinda.hard.layout import (
    H_TO_S_NAME as H_TO_S_NAME,
)
from olinda.hard.layout import (
    HARD_DIRNAME as HARD_DIRNAME,
)
from olinda.hard.layout import (
    HARD_EVAL_NAME as HARD_EVAL_NAME,
)
from olinda.hard.layout import (
    HARD_H5_NAME as HARD_H5_NAME,
)
from olinda.hard.layout import (
    HARD_META_NAME as HARD_META_NAME,
)
from olinda.hard.layout import (
    HARD_MODEL_SUBDIR as HARD_MODEL_SUBDIR,
)
from olinda.hard.layout import (
    TANIMOTO_DIRNAME as TANIMOTO_DIRNAME,
)
from olinda.hard.layout import (
    has_hard_head as has_hard_head,
)
from olinda.hard.train import train_hard as train_hard
