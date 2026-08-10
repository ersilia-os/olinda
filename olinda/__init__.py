"""olinda — a model distillation library.

Molecular fingerprints, and therefore the embeddings produced by the bundled ONNX compound
encoders, are only reproducible with the exact RDKit build the encoders were trained against.
This package refuses to import under any other RDKit version.
"""

import logging

import rdkit

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
