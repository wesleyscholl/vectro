"""
Cython implementation of embedding quantization for native performance.
This provides a compiled alternative to the Python NumPy implementation.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, fabs

# Define types for better performance
ctypedef cnp.float32_t DTYPE_FLOAT32
ctypedef cnp.int8_t DTYPE_INT8
ctypedef cnp.int32_t DTYPE_INT32

def quantize_embeddings_cython(cnp.ndarray[DTYPE_FLOAT32, ndim=2] embeddings):
    """
    Quantize embeddings using int8 quantization with Cython optimization.

    Args:
        embeddings: Input embeddings array (n_vectors, dim)

    Returns:
        Tuple of (quantized_embeddings, scale_factors)
    """
    cdef int n_vectors = embeddings.shape[0]
    cdef int dim = embeddings.shape[1]

    # Allocate output arrays
    cdef cnp.ndarray[DTYPE_INT8, ndim=2] quantized = np.empty((n_vectors, dim), dtype=np.int8)
    cdef cnp.ndarray[DTYPE_FLOAT32, ndim=1] scales = np.empty(n_vectors, dtype=np.float32)

    cdef int i, j
    cdef float max_abs, scale
    cdef float val

    # Process each embedding vector
    for i in range(n_vectors):
        # Find maximum absolute value for scaling
        max_abs = 0.0
        for j in range(dim):
            val = fabs(embeddings[i, j])
            if val > max_abs:
                max_abs = val

        # Calculate scale factor (avoid division by zero)
        if max_abs == 0.0:
            scale = 1.0
        else:
            scale = 127.0 / max_abs

        scales[i] = scale

        # Quantize the vector
        for j in range(dim):
            quantized[i, j] = <DTYPE_INT8>(embeddings[i, j] * scale)

    return quantized, scales

def reconstruct_embeddings_cython(cnp.ndarray[DTYPE_INT8, ndim=2] quantized,
                                  cnp.ndarray[DTYPE_FLOAT32, ndim=1] scales):
    """
    Reconstruct embeddings from quantized representation.

    Args:
        quantized: Quantized embeddings array (n_vectors, dim)
        scales: Scale factors for each vector

    Returns:
        Reconstructed embeddings array
    """
    cdef int n_vectors = quantized.shape[0]
    cdef int dim = quantized.shape[1]

    cdef cnp.ndarray[DTYPE_FLOAT32, ndim=2] reconstructed = np.empty((n_vectors, dim), dtype=np.float32)

    cdef int i, j

    # Reconstruct each vector
    for i in range(n_vectors):
        for j in range(dim):
            reconstructed[i, j] = <float>quantized[i, j] / scales[i]

    return reconstructed

def mean_cosine_similarity_cython(cnp.ndarray[DTYPE_FLOAT32, ndim=2] original,
                                  cnp.ndarray[DTYPE_FLOAT32, ndim=2] reconstructed):
    """
    Calculate mean cosine similarity between original and reconstructed embeddings.

    Args:
        original: Original embeddings
        reconstructed: Reconstructed embeddings

    Returns:
        Mean cosine similarity score
    """
    cdef int n_vectors = original.shape[0]
    cdef int dim = original.shape[1]

    cdef float total_similarity = 0.0
    cdef float dot_product, norm_orig, norm_recon
    cdef int i, j

    for i in range(n_vectors):
        dot_product = 0.0
        norm_orig = 0.0
        norm_recon = 0.0

        for j in range(dim):
            dot_product += original[i, j] * reconstructed[i, j]
            norm_orig += original[i, j] * original[i, j]
            norm_recon += reconstructed[i, j] * reconstructed[i, j]

        norm_orig = sqrt(norm_orig)
        norm_recon = sqrt(norm_recon)

        if norm_orig > 0 and norm_recon > 0:
            total_similarity += dot_product / (norm_orig * norm_recon)

    return total_similarity / n_vectors