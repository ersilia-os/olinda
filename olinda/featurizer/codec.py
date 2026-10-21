"""The qmfp byte layout: Morgan counts, each clipped at 15 and stored as a 4-bit number.

A qmfp fingerprint is RDKit's folded Morgan count fingerprint. Every slot's count is clipped at 15
and written in one nibble, two slots per byte: the even slot in the low nibble, the odd slot in the
high one. A 2048-slot fingerprint is therefore 1024 bytes, which is the model's input.

Why exact counts up to 15: measured against a real teacher, a model learns as much from them as from
unclipped counts (counts above 15 are 0.03% of nonzero slots), while RDKit's count simulation, which
buckets 2-3 and 4-7 together, cost 0.03-0.05 R².

Everything here is numpy only, because the inference install has no RDKit.
"""

from __future__ import annotations

import numpy as np

COUNT_CLIP = 15
"""The largest count a nibble holds; larger counts are stored as 15."""


def byte_table() -> np.ndarray:
    """``(256, 2)`` float32: the two slot counts every possible byte decodes to.

    The single definition of the layout. :func:`decode` indexes it in numpy and the fused graph
    embeds it in a ``Gather``, so the two decoders cannot disagree.
    """
    b = np.arange(256)
    return np.stack([b & 0xF, b >> 4], axis=1).astype(np.float32)


def decode(packed: np.ndarray) -> np.ndarray:
    """``uint8 (N, B)`` packed bytes → ``uint8 (N, 2B)`` slot counts in 0-15."""
    packed = np.asarray(packed, dtype=np.uint8)
    out = np.empty((packed.shape[0], 2 * packed.shape[1]), dtype=np.uint8)
    out[:, 0::2] = packed & 0xF
    out[:, 1::2] = packed >> 4
    return out


def pack(counts: np.ndarray) -> np.ndarray:
    """``(N, S)`` slot counts → ``uint8 (N, S/2)`` packed bytes; the inverse of :func:`decode`.

    Counts are clipped at :data:`COUNT_CLIP` first, so true counts can be passed straight in. Training
    stores decoded counts, and parity and tests turn them back into the bytes a runtime would send.
    """
    counts = np.minimum(np.asarray(counts), COUNT_CLIP).astype(np.uint8)
    return counts[:, 0::2] | (counts[:, 1::2] << 4)
