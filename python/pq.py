"""Product Quantization (PQ) - simple NumPy implementation for MVP.

Provides:
 - train_pq(embeddings, m, ks, iters=20) -> codebooks (m, ks, d_sub)
 - encode_pq(embeddings, codebooks) -> codes (n, m) dtype=int32
 - decode_pq(codes, codebooks) -> reconstructed embeddings (n, d)
 - encode_pq_bytes / decode_pq_bytes helpers with optional zlib compression

This is intentionally small and dependency-free (NumPy only) for iteration.
"""
from __future__ import annotations
import numpy as np
import zlib
from typing import Tuple, Optional


def _kmeans_1d(X: np.ndarray, k: int, iters: int = 20) -> np.ndarray:
    # Simple k-means for rows in X (shape (n, dsub)), returns k centroids (k, dsub)
    n, d = X.shape
    if n == 0:
        return np.zeros((k, d), dtype=np.float32)
    # init: pick k samples or repeat last
    rng = np.random.default_rng(0)
    if n >= k:
        idx = rng.choice(n, size=k, replace=False)
        centroids = X[idx].astype(np.float32).copy()
    else:
        centroids = np.zeros((k, d), dtype=np.float32)
        centroids[:n] = X.astype(np.float32)

    for _ in range(iters):
        # assign
        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        assign = np.argmin(dists, axis=1)
        changed = False
        for j in range(k):
            sel = X[assign == j]
            if len(sel) > 0:
                newc = np.mean(sel, axis=0)
                if not np.allclose(newc, centroids[j]):
                    centroids[j] = newc
                    changed = True
        if not changed:
            break
    return centroids


def train_pq(emb: np.ndarray, m: int, ks: int, iters: int = 20) -> np.ndarray:
    """Train PQ codebooks.

    emb: (n, d)
    m: number of subquantizers
    ks: number of centroids per subquantizer
    returns: codebooks shape (m, ks, d_sub)
    """
    n, d = emb.shape
    if d % m != 0:
        raise ValueError('d must be divisible by m')
    dsub = d // m
    codebooks = np.zeros((m, ks, dsub), dtype=np.float32)
    for i in range(m):
        block = emb[:, i * dsub:(i + 1) * dsub]
        codebooks[i] = _kmeans_1d(block, ks, iters=iters)
    return codebooks


def encode_pq(emb: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    """Encode embeddings to PQ codes.

    returns codes shape (n, m) dtype=int32
    """
    n, d = emb.shape
    m, ks, dsub = codebooks.shape
    if d != m * dsub:
        raise ValueError('dimension mismatch')
    codes = np.empty((n, m), dtype=np.int32)
    for i in range(m):
        block = emb[:, i * dsub:(i + 1) * dsub]
        # compute squared distances to codebook
        cbs = codebooks[i]
        dists = np.sum((block[:, None, :] - cbs[None, :, :]) ** 2, axis=2)
        codes[:, i] = np.argmin(dists, axis=1)
    return codes


def decode_pq(codes: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    n, m = codes.shape
    m2, ks, dsub = codebooks.shape
    if m != m2:
        raise ValueError('codes/codebooks mismatch')
    out = np.empty((n, m * dsub), dtype=np.float32)
    for i in range(m):
        out[:, i * dsub:(i + 1) * dsub] = codebooks[i][codes[:, i]]
    return out


def encode_pq_bytes(codes: np.ndarray, codebooks: np.ndarray, compress: bool = True) -> Tuple[bytes, bytes]:
    """Return tuple (codes_bytes, codebooks_bytes). codebooks serialized as float32 array bytes.
    codes are stored as int32 flattened bytes; if compress=True, codes_bytes is zlib-compressed.
    """
    codes_bytes = codes.astype(np.int32).tobytes()
    if compress:
        codes_bytes = zlib.compress(codes_bytes)
    codebooks_bytes = codebooks.astype(np.float32).tobytes()
    return codes_bytes, codebooks_bytes


def decode_pq_bytes(codes_bytes: bytes, codebooks_bytes: bytes, m: int, ks: int, dsub: int, compressed: bool = True) -> np.ndarray:
    """Decode bytes back into embeddings. Requires shape metadata: m, ks, dsub."""
    if compressed:
        codes_raw = zlib.decompress(codes_bytes)
    else:
        codes_raw = codes_bytes
    codes = np.frombuffer(codes_raw, dtype=np.int32)
    if codes.size % m != 0:
        raise ValueError('codes bytes length not divisible by m')
    n = codes.size // m
    codes = codes.reshape((n, m))
    codebooks = np.frombuffer(codebooks_bytes, dtype=np.float32).reshape((m, ks, dsub))
    return decode_pq(codes, codebooks)
