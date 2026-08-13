"""Fuse a trained model dir into a single, self-describing ``model.onnx``.

Every olinda-owned transform downstream of the RDKit featurizer is ONNX-able, so a bundle collapses to ONE
graph that runs on onnxruntime alone. Two shapes:

- **soft-only**: ``fp → soft_model → [soft_correction] → prediction`` (= ``S``).
- **hard present**: fuse ``S``, ``H → h_correction`` (``H_S``), ``T → ramp`` (the weight ``a``) and the
  blender, ``(1-a)·S + a·H_S``.

Either way a column declares **one output, its prediction**, named after the teacher column. ``S``,
``H_S`` and ``a`` stay internal tensors: they are how the answer is computed, not part of the answer,
and declaring them would let callers depend on a wiring that has already changed once.

The **featurizer config + provenance travel inside ``model.onnx`` metadata** (``metadata_props["olinda"]``),
so the file is self-describing — a consumer reads the Morgan config (and RDKit version) to build the 2048-count
fingerprint in Python (no ONNX op for featurization) and runs the single graph.

The hard head is task-aware: a **classifier** exposes ``probabilities`` (we take column 1); a **regressor**
would expose ``variable`` directly (seam — only classifier is enabled today, see :func:`olinda.hard`).
"""

from __future__ import annotations

from olinda.export.bundle import build_bundle as build_bundle
from olinda.export.metadata import BUNDLE_SCHEMA as BUNDLE_SCHEMA
from olinda.export.metadata import MODEL_NAME as MODEL_NAME
from olinda.export.metadata import PRODUCER_NAME as PRODUCER_NAME
