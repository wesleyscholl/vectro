"""Python shim that mimics the Mojo-compiled `vectro.src.quantizer` module.

This module implements the same per-vector int8 quantize/reconstruct API but
uses NumPy for fast vectorized operations. It exists so the repo can be used
without a real Mojo toolchain; when a real Mojo-compiled Python module is
available it can replace this file.
"""
from __future__ import annotations
import numpy as np


def quantize_int8(emb_flat, n: int, d: int):
    """Quantize flat embeddings array to int8 with per-vector scales.

    Parameters
    - emb_flat: array-like of length n*d (float32)
    - n: number of vectors
    - d: dims per vector

    Returns (q_flat_list, scales_list)
    """
    arr = np.asarray(emb_flat, dtype=np.float32)
    if arr.size != n * d:
        raise ValueError("emb_flat length does not match n*d")
    emb = arr.reshape((n, d))
    max_abs = np.max(np.abs(emb), axis=1)
    scales = np.where(max_abs == 0.0, 1.0, max_abs / 127.0).astype(np.float32)
    # avoid division by zero
    scales_safe = scales.copy()
    scales_safe[scales_safe == 0] = 1.0
    q = np.round(emb / scales_safe[:, None]).astype(np.int32)
    # clamp to [-127,127]
    q = np.clip(q, -127, 127).astype(np.int8)
    return q.ravel().tolist(), scales.tolist()


def reconstruct_int8(q_flat, scales, n: int, d: int):
    """Reconstruct float32 embeddings from int8 q_flat and per-vector scales.

    Returns a flat list of floats length n*d.
    """
    q = np.asarray(q_flat, dtype=np.int8)
    if q.size != n * d:
        raise ValueError("q_flat length does not match n*d")
    q2 = q.reshape((n, d)).astype(np.float32)
    s = np.asarray(scales, dtype=np.float32)
    if s.size != n:
        raise ValueError("scales length must match n")
    out = q2 * s[:, None]
    return out.ravel().tolist()
