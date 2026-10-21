"""olinda — a model distillation library.

Whether a runtime reproduces the fingerprints a model was trained on is a property of the *model*, not
of the package: every ``model.onnx`` carries golden vectors, and :class:`~olinda.artifact.OlindaArtifact`
re-scores them at load time, warning (or raising, when strict) with :class:`GoldenMismatch` if they
differ. olinda itself never imports rdkit; it featurizes through RDKit compiled to WebAssembly.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # import-time cost avoided at runtime; see __getattr__ below
    from olinda.artifact import GoldenMismatch as GoldenMismatch
    from olinda.artifact import GoldenMismatchWarning as GoldenMismatchWarning
    from olinda.artifact import OlindaArtifact as OlindaArtifact

# Silence matplotlib's "Matplotlib is building the font cache; this may take a moment." notice (emitted
# by the font_manager logger on first import / stale cache). Set here — before matplotlib is ever lazily
# imported (via stylia in plotting paths) — so it never reaches the console.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def __getattr__(name):
    """Expose the inference API lazily, so `import olinda` stays cheap for CLI startup."""
    if name in ("OlindaArtifact", "GoldenMismatch", "GoldenMismatchWarning"):
        from olinda import artifact

        return getattr(artifact, name)
    raise AttributeError(f"module 'olinda' has no attribute {name!r}")


__all__ = ["GoldenMismatch", "GoldenMismatchWarning", "OlindaArtifact"]
