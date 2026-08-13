"""``build_bundle``: fuse a trained run into one ``model.onnx``, then prove it still works."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from olinda.console import STEP_COLORS, echo, rule, success, summary_panel
from olinda.console import path as cpath
from olinda.export.fuse import _fuse
from olinda.export.graph import _save
from olinda.export.metadata import MODEL_NAME
from olinda.export.parity import _PARITY_TOL, _parity_probe
from olinda.hard.layout import (
    H_TO_S_NAME,
    HARD_DIRNAME,
    HARD_MODEL_SUBDIR,
    TANIMOTO_DIRNAME,
)


def build_bundle(model_dir: str | Path) -> dict:
    """Fuse a run directory into ``model.onnx``, gated on numeric parity against the Python pipeline.

    Every column is checked independently: the graph's output for that column must match a pure-Python
    recomposition of exactly the same artifacts, so a mis-wired subgraph cannot slip through.
    """
    import onnxruntime as ort

    from olinda.featurizer import featurizer_from_meta
    from olinda.models.bundle import StudentModel

    model_dir = Path(model_dir)
    rule("olinda · export", style=STEP_COLORS["export"], right=cpath(model_dir))
    model, plan, outputs = _fuse(model_dir)
    echo(f"fusing {len(plan)} column(s) → model.onnx", "run")
    _save(model, model_dir / MODEL_NAME)

    echo("checking parity: model.onnx vs Python reference", "run")
    ref: dict = {}
    fp = _parity_probe(plan)
    for entry in plan:
        sm = StudentModel.load(entry["dir"], featurizer_factory=featurizer_from_meta)
        raw = np.asarray(sm.predict(X=fp, calibrate=False)).ravel()
        surrogate = (
            np.asarray(sm.calibrator.transform(raw)).ravel()
            if sm.calibrator is not None
            else raw.astype(np.float64)
        )
        if not entry["has_hard"]:
            ref[entry["output"]] = surrogate
            continue

        from lazyqsar.base.xgboost import BaseXGBArtifact

        from olinda.calibrate import IsotonicCalibrator
        from olinda.tanimoto import TanimotoRegressor

        hard_root = entry["dir"] / HARD_DIRNAME
        g = np.asarray(
            BaseXGBArtifact.load(str(hard_root / HARD_MODEL_SUBDIR)).run(fp)
        )[:, 1].astype(np.float64)
        gsoft = np.asarray(
            IsotonicCalibrator.load(hard_root / H_TO_S_NAME).transform(g)
        ).ravel()
        gate = TanimotoRegressor.load(hard_root / TANIMOTO_DIRNAME)
        a = np.asarray(gate.weight(fp)).ravel()
        # a == 0 everywhere has two very different causes. If the ceiling itself is zero the hard head did
        # not earn any weight, the blend is off by design, and the fused output simply *is* the surrogate —
        # there is nothing left to cross-check. If the ceiling is positive and the gate still never fires,
        # the gate is broken and the hard branch would ship unverified.
        if gate.a_max > 0 and not (a > 0).any():
            raise RuntimeError(
                f"parity probe for column {entry['name']!r} scored zero blend weight on every molecule, so "
                "the blend collapses to the surrogate and the hard head would go unchecked. This should not "
                "happen — the probe includes that column's own labelled compounds."
            )
        ref[entry["output"]] = (1.0 - a) * surrogate + a * gsoft

    sess = ort.InferenceSession(
        (model_dir / MODEL_NAME).read_bytes(), providers=["CPUExecutionProvider"]
    )
    got = {
        o.name: np.asarray(v).ravel()
        for o, v in zip(sess.get_outputs(), sess.run(None, {"input": fp}))
    }
    parity = {k: float(np.max(np.abs(got[k] - ref[k]))) for k in outputs}
    worst = max(parity.values())
    if worst > _PARITY_TOL:
        offender = max(parity, key=parity.get)
        raise RuntimeError(
            f"model.onnx parity failed on column {offender!r}: max abs diff {worst:.2e} > {_PARITY_TOL:.0e}"
        )

    n_hard = sum(1 for e in plan if e["has_hard"])
    size_b = (model_dir / MODEL_NAME).stat().st_size
    size_txt = f"{size_b / 1e6:.1f} MB" if size_b >= 1e6 else f"{size_b / 1e3:.0f} KB"
    summary_panel(
        "olinda · export",
        [
            ("Columns", f"[bold]{len(plan)}[/] · {n_hard} with a hard head"),
            ("Outputs", " · ".join(outputs)),
            ("Parity (max)", f"[bold]{worst:.2e}[/] ≤ {_PARITY_TOL:.0e}"),
            ("Size", f"[bold]{size_txt}[/]"),
            ("Saved", f"[dim]{cpath(model_dir / MODEL_NAME)}[/]"),
        ],
        border_style=STEP_COLORS["export"],
        icon="✓",
    )
    success(
        f"fused model.onnx built and parity-checked → [dim]{cpath(model_dir / MODEL_NAME)}[/]"
    )
    return {"model": str(model_dir / MODEL_NAME), "columns": outputs, "parity": parity}
