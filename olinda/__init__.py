"""olinda — a model distillation library.

Morgan fingerprints only reproduce bit-for-bit on the RDKit build a model was fused against, so
loading an artifact under a different one is refused. That check belongs to the *model*, not to the
package: every ``model.onnx`` records the build it needs, and :class:`~olinda.artifact.OlindaArtifact`
compares against it at load time. Importing olinda under any RDKit is fine — training is what pins a
version, and that constraint lives in ``pyproject.toml``.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # import-time cost avoided at runtime; see __getattr__ below
  from olinda.artifact import OlindaArtifact as OlindaArtifact
  from olinda.artifact import RDKitVersionMismatch as RDKitVersionMismatch

# Silence matplotlib's "Matplotlib is building the font cache; this may take a moment." notice (emitted
# by the font_manager logger on first import / stale cache). Set here — before matplotlib is ever lazily
# imported (via stylia in plotting paths) — so it never reaches the console.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def __getattr__(name):
  """Expose the inference API lazily, so `import olinda` stays cheap for CLI startup."""
  if name in ("OlindaArtifact", "RDKitVersionMismatch"):
    from olinda import artifact

    return getattr(artifact, name)
  raise AttributeError(f"module 'olinda' has no attribute {name!r}")


__all__ = ["OlindaArtifact", "RDKitVersionMismatch"]
