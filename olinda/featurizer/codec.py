"""The qmfp byte layout: quantized Morgan counts packed as RDKit count-simulation bits.

RDKit's *count simulation* gives every folded slot four bits, switched on by the count
thresholds 1, 2, 4 and 8. RDKit MinimalLib and RDKit.js expose fingerprints only as bit
vectors, so this is the one count-like fingerprint they can produce, and it is the one
olinda models consume: a slot's level is how many of its four bits are on (0-4), read as
the lower bound of a count bucket::

    true count : 0   1   2-3   4-7   >=8
    level      : 0   1    2     3     4
    value      : 0   1    2     4     8

Bits are little-endian within each byte, so one byte holds two slots: the even slot in the
low nibble, the odd slot in the high one, each as a thermometer code (level L sets its
lowest L bits). A 2048-slot fingerprint is therefore 8192 bits, or 1024 bytes.

Everything here is numpy only, because the inference install has no RDKit.
"""

from __future__ import annotations

import numpy as np

LEVEL_VALUES = np.array([0, 1, 2, 4, 8], dtype=np.uint8)
"""The count each level stands for: the lower bound of its bucket."""

COUNT_TO_LEVEL = np.array([0, 1, 2, 2, 3, 3, 3, 3, 4], dtype=np.uint8)
"""Level of a true count, for counts clipped at 8."""

_NIBBLE = np.array([0x0, 0x1, 0x3, 0x7, 0xF], dtype=np.uint8)
_POPCOUNT4 = np.array([bin(v).count("1") for v in range(16)], dtype=np.uint8)
_VALUE_TO_LEVEL = np.full(256, 255, dtype=np.uint8)
_VALUE_TO_LEVEL[LEVEL_VALUES] = np.arange(len(LEVEL_VALUES), dtype=np.uint8)


def byte_table(levels=LEVEL_VALUES) -> np.ndarray:
    """``(256, 2)`` float32: the two slot values every possible byte decodes to.

    The single definition of the layout. :func:`decode` indexes it in numpy and the fused
    graph embeds it in a ``Gather``, so the two decoders cannot disagree.
    """
    levels = np.asarray(levels, dtype=np.float32)
    b = np.arange(256)
    return np.stack(
        [levels[_POPCOUNT4[b & 0xF]], levels[_POPCOUNT4[b >> 4]]], axis=1
    ).astype(np.float32)


def decode(packed: np.ndarray) -> np.ndarray:
    """``uint8 (N, B)`` packed bytes → ``uint8 (N, 2B)`` slot values in {0, 1, 2, 4, 8}."""
    packed = np.asarray(packed, dtype=np.uint8)
    table = byte_table().astype(np.uint8)
    return table[packed].reshape(packed.shape[0], -1)


def pack_levels(levels: np.ndarray) -> np.ndarray:
    """``uint8 (N, S)`` levels 0-4 → ``uint8 (N, S/2)`` packed bytes."""
    levels = np.asarray(levels, dtype=np.uint8)
    return _NIBBLE[levels[:, 0::2]] | (_NIBBLE[levels[:, 1::2]] << 4)


def pack(values: np.ndarray) -> np.ndarray:
    """``uint8 (N, S)`` slot values in {0, 1, 2, 4, 8} → packed bytes; the inverse of :func:`decode`.

    Training stores decoded values, and parity and tests need to turn them back into the bytes a
    runtime would feed the graph. A value outside the five levels is a bug upstream, so it raises.
    """
    levels = _VALUE_TO_LEVEL[np.asarray(values, dtype=np.uint8)]
    if (levels == 255).any():
        raise ValueError("qmfp values must be 0, 1, 2, 4 or 8")
    return pack_levels(levels)
