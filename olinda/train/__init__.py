"""Training the student: splits, the boosting backends, the tuner and the per-column loop.

Nothing here is importable on a base install, which is the point — the boosting stack is an order of
magnitude larger than what running a model needs. :func:`require_train_extra` is the one exception,
and is deliberately dependency-free so a command can call it *before* touching anything heavy.
"""

from __future__ import annotations

# Probed in the order a run would hit them, so the first name in the message is the first thing that
# would have failed.
_TRAIN_MODULES = (
    ("h5py", "read the prepared run"),
    ("xgboost", "train on GPU"),
    ("lightgbm", "train on CPU"),
    ("onnx", "build the fused model"),
    ("onnxmltools", "convert boosters to ONNX"),
    ("optuna", "tune hyperparameters"),
    ("lazyqsar", "fit the hard-label head"),
)


def require_train_extra() -> None:
    """Raise a single actionable error if the training dependencies are absent.

    The mirror of :func:`olinda.report.require_report_extra`, and load-bearing for the same reason: the
    CLI ships with the base install, so a training command on an inference-only install reaches its own
    body and would otherwise die on whichever ``import`` came first — a traceback naming ``h5py``,
    which tells the reader nothing about what to install. Called at the top of every command that
    distils, before any work.
    """
    missing = []
    for module, why in _TRAIN_MODULES:
        try:
            __import__(module)
        except ImportError:
            missing.append(f"{module} (to {why})")
    if missing:
        raise RuntimeError(
            "this command needs the training extra — missing: "
            + ", ".join(missing)
            + '.\nInstall it with:  pip install "olinda[train]"'
            + "\nRunning a model you already have needs no extras — `olinda predict` works on the base "
            "install."
        )
