"""
Python fallback quantizer for Vectro MVP.
This provides a simple per-vector int8 quantizer and reconstruction.
API:
 - quantize_embeddings(embeddings: np.ndarray) -> dict with keys:
    'q': np.ndarray (int8, flat), 'scales': np.ndarray (float32), 'dims': int, 'n': int
 - reconstruct_embeddings(q_flat, scales, dims) -> np.ndarray of shape (n, dims)
 - mean_cosine_similarity(orig, recon) -> float
"""
from __future__ import annotations
import importlib
import numpy as np


# Try to import the high-performance Mojo backend if available. The Mojo build
# should expose a Python module at `vectro.src.quantizer` or similar. If it
# isn't present, we'll fall back to the pure-NumPy implementation below.
_mojo_quant = None
try:
    _mojo_quant = importlib.import_module("vectro.src.quantizer")
except Exception:
    _mojo_quant = None


def quantize_embeddings(embeddings: np.ndarray) -> dict:
    """Quantize embeddings to int8 with per-vector scale.

    embeddings: shape (n, d), dtype float32
    returns: dict with 'q' flattened int8 array, 'scales' float32 array, 'dims' int, 'n' int
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D array of shape (n, d)")
    n, d = embeddings.shape

    # If Mojo backend available, use it for quantization (expects flat arrays)
    if _mojo_quant is not None:
        emb_flat = embeddings.astype(np.float32).ravel()
        # Mojo signature: quantize_int8(emb_flat: [f32], n: i32, d: i32) -> (q: [i8], scales: [f32])
        q_flat, scales = _mojo_quant.quantize_int8(emb_flat, int(n), int(d))
        # q_flat is expected to be a flat array of length n*d; ensure numpy types
        q_np = np.asarray(q_flat, dtype=np.int8)
        scales_np = np.asarray(scales, dtype=np.float32)
        return {"q": q_np, "scales": scales_np, "dims": d, "n": n}

    emb = embeddings.astype(np.float32)
    scales = np.empty((n,), dtype=np.float32)
    q = np.empty((n, d), dtype=np.int8)
    for i in range(n):
        v = emb[i]
        max_abs = np.max(np.abs(v))
        if max_abs == 0:
            scale = 1.0
        else:
            scale = max_abs / 127.0
        # avoid division by zero
        if scale == 0:
            scale = 1.0
        q[i] = np.round(v / scale).astype(np.int8)
        scales[i] = np.float32(scale)
    return {"q": q.ravel(), "scales": scales, "dims": d, "n": n}


def reconstruct_embeddings(q_flat: np.ndarray, scales: np.ndarray, dims: int) -> np.ndarray:
    """Reconstruct embeddings from flattened int8 q and per-vector scales."""
    q = np.asarray(q_flat, dtype=np.int8)
    n = int(len(q) // dims)
    # If Mojo backend exists, use its reconstruct function
    if _mojo_quant is not None:
        # Mojo signature: reconstruct_int8(q_flat: [i8], scales: [f32], n: i32, d: i32) -> [f32]
        out_flat = _mojo_quant.reconstruct_int8(q.tolist(), scales.tolist(), int(n), int(dims))
        return np.asarray(out_flat, dtype=np.float32).reshape((n, dims))

    q2 = q.reshape((n, dims)).astype(np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    if scales.shape[0] != n:
        raise ValueError("scales length must match number of vectors")
    recon = q2 * scales[:, None]
    return recon


def mean_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine similarity between rows of a and b."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if a.shape != b.shape:
        raise ValueError("shapes must match")
    # compute norms
    na = a
    nb = b
    norms_a = np.linalg.norm(na, axis=1)
    norms_b = np.linalg.norm(nb, axis=1)

    # prepare normalized arrays but avoid division by zero by temporarily replacing zeros with 1
    safe_a = na.copy()
    safe_b = nb.copy()
    norms_a_safe = norms_a.copy()
    norms_b_safe = norms_b.copy()
    norms_a_safe[norms_a_safe == 0] = 1.0
    norms_b_safe[norms_b_safe == 0] = 1.0
    safe_a = safe_a / norms_a_safe[:, None]
    safe_b = safe_b / norms_b_safe[:, None]

    dots = np.sum(safe_a * safe_b, axis=1)

    # handle zero-vector cases: if both are zero, define cosine = 1.0; if one is zero, define cosine = 0.0
    both_zero = (norms_a == 0) & (norms_b == 0)
    one_zero = ((norms_a == 0) ^ (norms_b == 0))
    dots[both_zero] = 1.0
    dots[one_zero] = 0.0

    return float(np.mean(dots))


if __name__ == "__main__":
    # simple demo
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((100, 768)).astype(np.float32)
    out = quantize_embeddings(emb)
    recon = reconstruct_embeddings(out['q'], out['scales'], out['dims'])
    print("Mean cosine:", mean_cosine_similarity(emb, recon))
