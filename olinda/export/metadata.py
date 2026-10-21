"""What the bundle says about itself, in ``metadata_props`` under ``olinda.*`` keys.

The artifact is self-describing, because a runtime in any language has no other way to know how to
build the input it expects. Every value is a string; structured ones are JSON:

``olinda.schema_version``
    ``"MAJOR.MINOR"``. Runtimes refuse a major version they do not know.
``olinda.fingerprint``
    The qmfp spec (:mod:`olinda.featurizer.spec`): the RDKit MinimalLib options that turn
    SMILES into the graph's input bytes. Runtimes read it rather than hard-coding featurization.
``olinda.rdkit_version``
    The RDKit release that featurized training and computed the golden vectors.
``olinda.golden``
    SMILES with their fingerprint digests and predictions. A runtime featurizes and scores them on
    load: agreement shows its RDKit reproduces the training fingerprints, which a version string
    cannot.
``olinda.model``
    Everything else: when and how the model was trained, the reference library, and one entry per
    column describing its task and the heads behind it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from olinda.hard.layout import HARD_DIRNAME, HARD_EVAL_NAME, HARD_META_NAME

MODEL_NAME = "model.onnx"
# The metadata layout, versioned. OlindaArtifact refuses a major version it does not know and says to
# rebuild, rather than reading a different layout field by field and getting it wrong.
SCHEMA_VERSION = "1.0"
PRODUCER_NAME = "olinda"


def _task(kind: str, value_range=None) -> dict:
    """One task descriptor: what a head or a column predicts, and over what range.

    Every predicting thing in the bundle carries one of these, so nothing has to be inferred from which
    metric keys happen to be present. ``kind`` is ``"regression"`` or ``"classification"``; ``range`` is
    the observed span of the values, or ``None`` when it was not measured. ``units`` is reserved and
    always ``None`` today — olinda never learns what a teacher's numbers mean.
    """
    return {
        "type": "classification"
        if kind in ("binary", "classification")
        else "regression",
        "range": list(value_range) if value_range else None,
        "units": None,
    }


def _soft_head(entry: dict) -> dict:
    """The surrogate's entry in a column's ``heads`` list."""
    training = entry.get("training") or {}
    return {
        "role": "soft",
        # The surrogate is fitted with squared error against the teacher's values (see
        # CANONICAL_DEFAULTS), so it regresses whatever scale the teacher emits — including a
        # probability. A future classification surrogate declares itself here and nowhere else.
        "task": _task("regression", training.get("value_range")),
        "source": {"kind": "teacher", "column": entry["name"]},
        "training": {
            "n": training.get("n_finite"),
            "n_train": training.get("n_train"),
            "n_val": training.get("n_val"),
        },
        "metrics": entry.get("metrics"),
    }


def _hard_head(entry: dict) -> dict:
    """The hard-label head's entry in a column's ``heads`` list, read back off the run.

    Its metrics are the alignment numbers, not classification scores: what decides whether this head is
    trusted is how well its *calibrated* output reproduces the teacher's scale, and that is what
    ``a_max`` is derived from.
    """
    meta_path = entry["dir"] / HARD_DIRNAME / HARD_META_NAME
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    eval_path = entry["dir"] / HARD_DIRNAME / HARD_EVAL_NAME
    evaluation = json.loads(eval_path.read_text()) if eval_path.exists() else {}
    prepared = entry.get("hard_meta") or {}
    gate = meta.get("tanimoto") or {}
    calibration = evaluation.get("calibration") or {}

    return {
        "role": "hard",
        # Taken from what the run resolved, not hardcoded: the labels decide, and a regression head is
        # the case this field exists for.
        "task": _task(prepared.get("task") or meta.get("task") or "binary"),
        "source": {"kind": "measured", "column": prepared.get("source_column")},
        "training": {"n": meta.get("n"), "n_positive": prepared.get("n_positive")},
        "metrics": {
            "r2_calibrated_soft": calibration.get("r2_calibrated_soft"),
            "pearson_calibrated_soft": calibration.get("pearson_calibrated_soft"),
            "spearman_h_soft": calibration.get("spearman_g_soft"),
        },
        "calibration": {"direction": calibration.get("direction")},
        # The gate is a model in the graph like any other, so record what it is and the knees it ramps
        # between. a_max of 0 means this head earned no weight and the column ships soft-only.
        "gate": {
            "signal": gate.get("signal"),
            "a_max": gate.get("a_max"),
            "t_lo": gate.get("sim_lo"),
            "t_hi": gate.get("sim_hi"),
        },
        "provenance": {"lazyqsar_version": meta.get("lazyqsar_version")},
    }


def _run_backend(plan: list, manifest: dict) -> str | None:
    """Which engine trained this run — recorded per column, never on the manifest itself."""
    if manifest.get("backend"):
        return manifest["backend"]
    meta_path = plan[0]["dir"] / "train_meta.json" if plan else None
    if meta_path and meta_path.exists():
        return json.loads(meta_path.read_text()).get("backend")
    return None


def _bundle_metadata(manifest: dict, plan: list) -> dict:
    """The ``olinda.model`` block: when the model was trained, from what, and what each column predicts.

    The fingerprint and the RDKit release have keys of their own, since every runtime needs them before
    it can run anything; this block is the rest.

    The shape is N-ary throughout: ``columns`` is a list, and a one-endpoint model is simply its
    one-element case. Within a column, every model that contributes is an entry in ``heads``, each
    carrying its own ``task`` — so a classification surrogate, or a regression hard head, needs a
    different ``type`` and no new keys. How the heads combine is *not* described here: the graph is the
    only statement of that, and its single output per column is the only thing a consumer reads.

    Nothing derivable is stored. ``n_columns``, the flat output list and "does this have a hard head"
    are all properties of ``columns``, and duplicating them here is how they come to disagree with it.
    """
    import importlib.metadata

    from olinda.data.fetch import MORGAN_FINGERPRINTS_FILENAME, MORGAN_FINGERPRINTS_URL

    try:
        olinda_version = importlib.metadata.version("olinda")
    except Exception:
        olinda_version = "unknown"

    columns = []
    for entry in plan:
        heads = [_soft_head(entry)]
        if entry["has_hard"]:
            heads.append(_hard_head(entry))
        columns.append(
            {
                "id": entry["id"],
                "name": entry["name"],
                "output": entry["output"],
                # What the column itself emits. The blend maps the hard head onto the teacher's scale before
                # mixing, so the column's task is the soft head's task whatever else contributes.
                "task": heads[0]["task"],
                "heads": heads,
            }
        )

    split = manifest.get("split") or {}
    return {
        "producer": "olinda",
        "olinda_version": olinda_version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_library": {
            "name": MORGAN_FINGERPRINTS_FILENAME,
            "url": MORGAN_FINGERPRINTS_URL,
            **{
                k: v
                for k, v in (manifest.get("reference_library") or {}).items()
                if k in ("n_rows", "dim")
            },
        },
        "run": {
            "backend": _run_backend(plan, manifest),
            "created": manifest.get("created"),
            **{k: split.get(k) for k in ("val_frac", "seed", "limit")},
        },
        "columns": columns,
    }
