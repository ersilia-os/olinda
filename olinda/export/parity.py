"""Does the fused graph still compute what the Python stages compute?

`build_bundle` refuses to ship a bundle whose ONNX output drifts from a Python recomposition of the
same stages, which is the check that makes the fuse trustworthy rather than merely plausible.
"""

from __future__ import annotations

import numpy as np

from olinda.hard.layout import HARD_H5_NAME

_PARITY_TOL = 1e-4  # build_bundle raises if the fused graph drifts from the Python reference beyond this
_SAMPLE_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "OCc1ccccc1"]
_PARITY_LABELLED = (
    8  # labelled compounds per hard column added to the probe, to exercise the blend
)


def _parity_probe(plan: list) -> np.ndarray:
    """Fingerprints to check the fused graph against the Python pipeline.

    A handful of fixed molecules is not enough on its own: ``T`` may score them all
    zero, in which case the blend collapses to the surrogate and the hard model, its calibrator and the
    gate are compared against nothing — a cross-wired column would pass. A hard column's own labelled
    compounds are at Tanimoto 1.0 from the labelled set, which is the top of the gate's ramp, so a few
    of them are appended to force the whole blend to be exercised.
    """
    import h5py

    from olinda.featurizer import MorganCountFeaturizer

    blocks = [MorganCountFeaturizer().transform(_SAMPLE_SMILES).astype(np.float32)]
    for entry in plan:
        hard = entry["dir"] / HARD_H5_NAME
        if not entry["has_hard"] or not hard.exists():
            continue
        with h5py.File(hard, "r") as f:
            blocks.append(np.asarray(f["x"][:_PARITY_LABELLED], dtype=np.float32))
    return np.concatenate(blocks, axis=0)


def _assert_model_belongs_to(sm, entry: dict) -> None:
    """Refuse to fuse a model that was trained for a different column.

    Column directories are named positionally (``c0``, ``c1``, …), so re-preparing a run directory with
    a different teacher file rebinds those names while leaving the previous run's artifacts in place.
    Without this check the stale booster is fused under the new column's name, and the parity check
    agrees because it validates against the same stale files.
    """
    trained_for = sm.metadata.get("column")
    if trained_for is not None and trained_for != entry["name"]:
        raise ValueError(
            f"{entry['dir']} holds a model trained for column {trained_for!r}, but this run calls "
            f"{entry['id']} {entry['name']!r}. The directory is stale — re-run `olinda learn-soft`, or "
            "prepare into a clean directory."
        )
