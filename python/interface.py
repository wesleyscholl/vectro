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


# Try to import the high-performance backends if available
_cython_quant = None
try:
    _cython_quant = importlib.import_module("vectro.quantizer_cython")
except Exception:
    _cython_quant = None

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

    # Priority: Mojo > Cython > NumPy
    if _mojo_quant is not None:
        # Convert to lists for Mojo
        emb_list = embeddings.astype(np.float32).ravel().tolist()
        q_list, scales_list = _mojo_quant.quantize_int8_py(emb_list, n, d)
        return {"q": np.asarray(q_list, dtype=np.int8), "scales": np.asarray(scales_list, dtype=np.float32), "dims": d, "n": n}
    elif _cython_quant is not None:
        quantized, scales = _cython_quant.quantize_embeddings_cython(embeddings.astype(np.float32))
        return {"q": quantized.ravel(), "scales": scales, "dims": d, "n": n}

    # Fallback to NumPy implementation
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
    q_reshaped = q.reshape((n, dims))

    # Priority: Mojo > Cython > NumPy
    if _mojo_quant is not None:
        # Convert to lists for Mojo
        q_list = q_flat.tolist() if hasattr(q_flat, 'tolist') else list(q_flat)
        scales_list = scales.tolist() if hasattr(scales, 'tolist') else list(scales)
        recon_list = _mojo_quant.reconstruct_int8_py(q_list, scales_list, n, d)
        return np.asarray(recon_list, dtype=np.float32).reshape((n, d))
    elif _cython_quant is not None:
        q_reshaped = q.reshape((n, dims))
        return _cython_quant.reconstruct_embeddings_cython(q_reshaped, scales.astype(np.float32))

    # Fallback to NumPy implementation
    q2 = q_reshaped.astype(np.float32)
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

    # If Cython backend available, use it for similarity calculation
    if _cython_quant is not None:
        return _cython_quant.mean_cosine_similarity_cython(a, b)

    # Fallback to NumPy implementation
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
