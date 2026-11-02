"""Serialisation shared by sync (Flower) and async paths so byte counts are comparable.

Flower serialises each ndarray with numpy.save; we use the same encoding so the bytes
counted for FedAsync's HTTP transport equal what Flower would put on the wire.
"""
from __future__ import annotations

import io
import struct

import numpy as np


def ndarray_to_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, a, allow_pickle=False)
    return buf.getvalue()


def bytes_to_ndarray(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b), allow_pickle=False)


def pack(arrays: list[np.ndarray]) -> bytes:
    """Length-prefixed concatenation of numpy.save payloads."""
    out = [struct.pack("<I", len(arrays))]
    for a in arrays:
        b = ndarray_to_bytes(a)
        out += [struct.pack("<Q", len(b)), b]
    return b"".join(out)


def unpack(blob: bytes) -> list[np.ndarray]:
    (n,) = struct.unpack_from("<I", blob, 0)
    off, arrays = 4, []
    for _ in range(n):
        (ln,) = struct.unpack_from("<Q", blob, off)
        off += 8
        arrays.append(bytes_to_ndarray(blob[off:off + ln]))
        off += ln
    return arrays


def payload_bytes(arrays: list[np.ndarray]) -> int:
    """Bytes of the tensor payload as Flower encodes it (sum of numpy.save buffers)."""
    return int(sum(len(ndarray_to_bytes(a)) for a in arrays))
