"""Stable scientific RNG keys. Not Python's salted hash()."""

from __future__ import annotations

import hashlib
import struct

import numpy as np


def digest_u64(*parts: object) -> int:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            h.update(p)
        elif isinstance(p, str):
            h.update(p.encode("utf-8"))
        elif isinstance(p, (int, np.integer)):
            h.update(struct.pack("<q", int(p)))
        elif isinstance(p, np.datetime64):
            h.update(str(p).encode("utf-8"))
        else:
            h.update(repr(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest()[:8], "little", signed=False)


def noise_key(stream: str, split: str, window_id: object, draw: int, eval_seed: int) -> int:
    return digest_u64("prigm-noise", stream, split, window_id, int(draw), int(eval_seed))


def pcg64(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(int(seed) & ((1 << 64) - 1)))
