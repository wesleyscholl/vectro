"""
Python fallback quantizer for Vectro MVP.
This provides a simple per-vector int8 quantizer and reconstruction.
API:
 - quantize_embeddings(embeddings: np.ndarray) -> QuantizationResult
 - reconstruct_embeddings(result: QuantizationResult) -> np.ndarray
 - mean_cosine_similarity(orig, recon) -> float
"""
from __future__ import annotations
import importlib
import numpy as np
import os
import subprocess
import tempfile
from typing import NamedTuple


class QuantizationResult(NamedTuple):
    """Result of quantization operation."""
    quantized: np.ndarray  # int8 array, shape (n, d)
    scales: np.ndarray     # float32 array, shape (n,)
    dims: int             # dimension d
    n: int                # number of vectors


# Try to import the high-performance backends if available
_cython_quant = None
try:
    _cython_quant = importlib.import_module("python.quantizer_cython")
except Exception:
    try:
        # Fallback for development/local install
        _cython_quant = importlib.import_module("quantizer_cython")
    except Exception:
        _cython_quant = None

_mojo_available = False
_mojo_binary = None
try:
    # Check if Mojo binary exists
    import pathlib
    possible_paths = [
        pathlib.Path(__file__).parent.parent / "vectro_quantizer",
        pathlib.Path("vectro_quantizer"),
    ]
    for path in possible_paths:
        if path.exists():
            _mojo_binary = str(path.absolute())
            _mojo_available = True
            break
except Exception:
    pass


def _quantize_with_mojo(embeddings: np.ndarray) -> QuantizationResult:
    """Quantize using Mojo binary via simple algorithm (since subprocess overhead is high).
    
    For now, we use NumPy with Mojo-like algorithm until we have proper Python bindings.
    """
    n, d = embeddings.shape
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
        # Mojo-style quantization
        inv_scale = 1.0 / scale
        raw = v * inv_scale
        # Clamp to [-127, 127]
        raw = np.clip(raw, -127.0, 127.0)
        q[i] = np.round(raw).astype(np.int8)
        scales[i] = np.float32(scale)
    
    return QuantizationResult(quantized=q, scales=scales, dims=d, n=n)


def _reconstruct_with_mojo(result: QuantizationResult) -> np.ndarray:
    """Reconstruct using Mojo-style algorithm."""
    q = result.quantized
    scales = result.scales
    dims = result.dims
    n = result.n
    
    q2 = q.astype(np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    recon = q2 * scales[:, None]
    return recon


def get_backend_info():
    """Get information about available backends."""
    info = {
        "mojo": _mojo_available,
        "cython": _cython_quant is not None,
        "numpy": True,  # Always available
    }
    if _mojo_available:
        info["mojo_binary"] = _mojo_binary
    return info


def quantize_embeddings(embeddings: np.ndarray, backend: str = "auto") -> QuantizationResult:
    """Quantize embeddings to int8 with per-vector scale.

    embeddings: shape (n, d), dtype float32
    backend: 'auto', 'mojo', 'cython', or 'numpy'
    returns: QuantizationResult with quantized int8 array, scales, dims, n
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D array of shape (n, d)")
    n, d = embeddings.shape

    # Backend selection
    if backend == "auto":
        # Priority: Mojo > Cython > NumPy
        if _mojo_available:
            backend = "mojo"
        elif _cython_quant is not None:
            backend = "cython"
        else:
            backend = "numpy"
    
    # Use selected backend
    if backend == "mojo" and _mojo_available:
        return _quantize_with_mojo(embeddings)
    elif backend == "cython" and _cython_quant is not None:
        quantized, scales = _cython_quant.quantize_embeddings_cython(embeddings.astype(np.float32))
        return QuantizationResult(quantized=quantized, scales=scales, dims=d, n=n)
    elif backend == "numpy" or backend == "auto":
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
        return QuantizationResult(quantized=q, scales=scales, dims=d, n=n)
    else:
        raise ValueError(f"Unknown or unavailable backend: {backend}")


def reconstruct_embeddings(result: QuantizationResult, backend: str = "auto") -> np.ndarray:
    """Reconstruct embeddings from QuantizationResult.
    
    backend: 'auto', 'mojo', 'cython', or 'numpy'
    """
    # Backend selection
    if backend == "auto":
        if _mojo_available:
            backend = "mojo"
        elif _cython_quant is not None:
            backend = "cython"
        else:
            backend = "numpy"
    
    # Use selected backend
    if backend == "mojo" and _mojo_available:
        return _reconstruct_with_mojo(result)
    elif backend == "cython" and _cython_quant is not None:
        return _cython_quant.reconstruct_embeddings_cython(result.quantized, result.scales.astype(np.float32))
    elif backend == "numpy" or backend == "auto":
        # Fallback to NumPy implementation
        q2 = result.quantized.astype(np.float32)
        scales = np.asarray(result.scales, dtype=np.float32)
        if scales.shape[0] != result.n:
            raise ValueError("scales length must match number of vectors")
        recon = q2 * scales[:, None]
        return recon
    else:
        raise ValueError(f"Unknown or unavailable backend: {backend}")


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
    # Show available backends
    print("Available backends:")
    backend_info = get_backend_info()
    for name, available in backend_info.items():
        if name.endswith("_binary"):
            continue
        status = "✓" if available else "✗"
        print(f"  {status} {name}")
    print()
    
    # simple demo
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((100, 768)).astype(np.float32)
    result = quantize_embeddings(emb)
    recon = reconstruct_embeddings(result)
    print("Mean cosine:", mean_cosine_similarity(emb, recon))
