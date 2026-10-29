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


_PARITY_LIBRARY = (
    2048  # reference compounds drawn at random: deep trees have many rarely-taken paths
)
_PARITY_RANDOM = (
    1024  # random sparse count vectors, which exercise values the library rarely shows
)


_RAW_SLACK = 3e-5  # float32-vs-float64 slack on a raw score: tree sums agree to ~1e-6 (XGBoost) and ~5e-7 (LightGBM)


def value_scale(values: np.ndarray) -> float:
    """How much larger than one the column's values run, which is how far float32 rounding scales with them.

    A float32 holds about seven digits, so on a column of values near 5,000 it is off by ~5e-4 per value
    before any model is involved. The tolerances above are absolute and were set on probabilities.
    """
    values = np.asarray(values, dtype=np.float64)
    return max(1.0, float(np.abs(values).max())) if values.size else 1.0


def _calibrated(calibrator, raw: np.ndarray) -> np.ndarray:
    """A raw score through its calibrator, or unchanged when there is none."""
    if calibrator is None:
        return np.asarray(raw, dtype=np.float64)
    return np.asarray(calibrator.transform(raw, dtype=np.float64)).ravel()


def _step_envelope(calibrator, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The lowest and highest calibrated value a raw score could take, give or take float rounding.

    Calibrators are isotonic step functions, so an output can jump a whole step when float32 and float64
    disagree by 1e-7 about which side of a boundary a score sits on. The jump is bounded by the envelope
    here, which is what the parity check compares against instead of one flat number.
    """
    raw = np.asarray(raw, dtype=np.float64)
    slack = _RAW_SLACK * value_scale(raw)
    a = _calibrated(calibrator, raw - slack)
    b = _calibrated(calibrator, raw + slack)
    return np.minimum(a, b), np.maximum(a, b)


def _library_rows(model_dir, n: int) -> np.ndarray:
    """*n* random decoded rows of the run's reference library, or none if the library is not at hand.

    The handful of fixed molecules in :func:`_parity_probe` take the same few paths through a deep
    ensemble, so a converter fault on a rarely-taken branch would slip past them. Real library rows
    spread over the whole feature space. A library that has moved is not an error: the other probes
    still run, and the export must not depend on a 2.8 GB file.
    """
    import h5py

    from olinda import run as runlib
    from olinda.data.fetch import MORGAN_FINGERPRINTS_FILENAME, OLINDA_HOME

    path = OLINDA_HOME / MORGAN_FINGERPRINTS_FILENAME
    if not path.exists():
        return np.zeros((0, 0), dtype=np.uint8)
    limit = runlib.row_limit(runlib.read_manifest(model_dir))
    with h5py.File(path, "r") as f:
        total = int(f["data"].shape[0])
        total = min(total, limit) if limit else total
        idx = np.sort(
            np.random.default_rng(0).choice(total, size=min(n, total), replace=False)
        )
        return np.asarray(f["data"][idx], dtype=np.uint8)


def _random_rows(n: int, width: int) -> np.ndarray:
    """*n* sparse count vectors of *width* slots, counts 1-15, deterministic."""
    rng = np.random.default_rng(1)
    x = np.zeros((n, width), dtype=np.uint8)
    for i in range(n):
        k = int(rng.integers(5, 120))
        cols = rng.choice(width, size=k, replace=False)
        x[i, cols] = np.minimum(rng.geometric(0.45, size=k), 15)
    return x


def _parity_probe(
    plan: list, spec: dict, model_dir=None
) -> tuple[np.ndarray, np.ndarray]:
    """``(bytes, fingerprints)`` to check the fused graph against the Python pipeline.

    The graph gets the bytes, the Python reference gets the decoded fingerprints, so the check covers
    the graph's own decoder as well as every head.

    A handful of fixed molecules is not enough on its own: ``T`` may score them all
    zero, in which case the blend collapses to the surrogate and the hard model, its calibrator and the
    gate are compared against nothing — a cross-wired column would pass. A hard column's own labelled
    compounds are at Tanimoto 1.0 from the labelled set, which is the top of the gate's ramp, so a few
    of them are appended to force the whole blend to be exercised.
    """
    import h5py

    from olinda.featurizer import QmfpFeaturizer
    from olinda.featurizer.codec import decode, pack

    blocks = [QmfpFeaturizer(spec).featurize_bytes(_SAMPLE_SMILES)[0]]
    for entry in plan:
        hard = entry["dir"] / HARD_H5_NAME
        if not entry["has_hard"] or not hard.exists():
            continue
        with h5py.File(hard, "r") as f:
            # stored decoded, so packed back into the bytes a runtime would send
            blocks.append(pack(np.asarray(f["x"][:_PARITY_LABELLED], dtype=np.uint8)))
    if model_dir is not None:
        lib = _library_rows(model_dir, _PARITY_LIBRARY)
        if lib.size:
            blocks.append(pack(lib))
            blocks.append(pack(_random_rows(_PARITY_RANDOM, lib.shape[1])))
    packed = np.concatenate(blocks, axis=0)
    return packed, decode(packed).astype(np.float32)


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
