"""olinda — a model distillation library.

Molecular fingerprints, and therefore the embeddings produced by the bundled ONNX compound
encoders, are only reproducible with the exact RDKit build the encoders were trained against.
This package refuses to import under any other RDKit version.
"""

import logging
from typing import TYPE_CHECKING

import rdkit

if TYPE_CHECKING:  # import-time cost avoided at runtime; see __getattr__ below
  from olinda.artifact import OlindaArtifact as OlindaArtifact
  from olinda.artifact import RDKitVersionMismatch as RDKitVersionMismatch

# Silence matplotlib's "Matplotlib is building the font cache; this may take a moment." notice (emitted
# by the font_manager logger on first import / stale cache). Set here — before matplotlib is ever lazily
# imported (via stylia in plotting paths) — so it never reaches the console.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

REQUIRED_RDKIT_VERSION = "2025.09.4"

if rdkit.__version__ != REQUIRED_RDKIT_VERSION:
  raise RuntimeError(
    f"olinda requires RDKit {REQUIRED_RDKIT_VERSION}, but found {rdkit.__version__}. "
    "Fingerprint reproducibility for the ONNX compound encoders depends on this exact version. "
    "Install it with: pip install rdkit==2025.9.4"
  )


def __getattr__(name):
  """Expose the inference API lazily, so `import olinda` stays cheap for CLI startup."""
  if name in ("OlindaArtifact", "RDKitVersionMismatch"):
    from olinda import artifact

    return getattr(artifact, name)
  raise AttributeError(f"module 'olinda' has no attribute {name!r}")


__all__ = ["OlindaArtifact", "RDKitVersionMismatch", "REQUIRED_RDKIT_VERSION"]
