"""Fuse a trained model dir into a single, self-describing ``model.onnx``.

Every olinda-owned transform downstream of RDKit is ONNX-able, so a bundle collapses to ONE graph that
runs on onnxruntime alone. Its input is qmfp bytes, as RDKit MinimalLib or RDKit.js returns them, and a
decoder at the front turns them into the fingerprint every head reads. Two shapes:

- **soft-only**: ``bytes → decoder → soft_model → [soft_correction] → prediction`` (= ``S``).
- **hard present**: fuse ``S``, ``H → h_correction`` (``H_S``), ``T → ramp`` (the weight ``a``) and the
  blender, ``(1-a)·S + a·H_S``.

Either way a column declares **one output, its prediction**, named after the teacher column. ``S``,
``H_S`` and ``a`` stay internal tensors: they are how the answer is computed, not part of the answer,
and declaring them would let callers depend on a wiring that has already changed once.

The **fingerprint spec, golden vectors and provenance travel inside ``model.onnx`` metadata**
(``olinda.*`` keys, see :mod:`olinda.export.metadata`), so the file is self-describing: a runtime reads
the spec to featurize SMILES with RDKit (there is no ONNX op for that), checks itself against the golden
vectors, and runs the single graph.

The hard head is task-aware: a **classifier** exposes ``probabilities`` (we take column 1); a **regressor**
would expose ``variable`` directly (seam — only classifier is enabled today, see :func:`olinda.hard`).
"""

from __future__ import annotations

from olinda.export.bundle import build_bundle as build_bundle
from olinda.export.metadata import MODEL_NAME as MODEL_NAME
from olinda.export.metadata import PRODUCER_NAME as PRODUCER_NAME
from olinda.export.metadata import SCHEMA_VERSION as SCHEMA_VERSION
