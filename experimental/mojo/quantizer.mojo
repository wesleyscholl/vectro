"""
Mojo implementation of high-performance embedding quantization.

Per-vector INT8 quantization algorithm (v3):
- scale = max_abs(v) / 127  (1.0 when the vector is all-zeros)
- q = round(v / scale) clamped to [-127, 127]
- reconstruction: v' = q * scale

The functions operate on flat row-major arrays for easy Python / Mojo interop.
"""

from algorithm import vectorize, parallelize
from math import sqrt
from sys.info import simdwidthof


fn quantize_int8(emb_flat: List[Float32], n: Int, d: Int) -> (List[Int8], List[Float32]):
    """Quantize a flat embeddings array (row-major) to INT8 with per-vector scales.

    Args:
        emb_flat: Flat array of length n*d (row-major).
        n: Number of vectors.
        d: Dimensions per vector.

    Returns:
        Tuple of (quantized_int8_flat, scales_per_vector).
    """
    var q = List[Int8](capacity=n * d)
    var scales = List[Float32](capacity=n)

    # Pre-allocate storage so indexed writes work
    for _ in range(n * d):
        q.append(0)
    for _ in range(n):
        scales.append(0.0)

    for i in range(n):
        var base = i * d
        var max_abs: Float32 = 0.0

        # Pass 1: find max absolute value for this vector
        for j in range(d):
            var v = emb_flat[base + j]
            var a = v if v >= 0.0 else -v
            if a > max_abs:
                max_abs = a

        var scale: Float32 = 1.0
        if max_abs > 0.0:
            scale = max_abs / 127.0
        scales[i] = scale

        # Pass 2: quantize
        var inv_scale = 1.0 / scale
        for j in range(d):
            var raw = emb_flat[base + j] * inv_scale
            if raw > 127.0:
                raw = 127.0
            elif raw < -127.0:
                raw = -127.0
            var qv = Int(raw + 0.5) if raw >= 0.0 else Int(raw - 0.5)
            q[base + j] = Int8(qv)

    return (q^, scales^)


fn reconstruct_int8(q_flat: List[Int8], scales: List[Float32], n: Int, d: Int) -> List[Float32]:
    """Reconstruct float32 embeddings from INT8 quantized data.

    Args:
        q_flat: Flat int8 array of length n*d.
        scales: Per-vector scale factors (length n).
        n: Number of vectors.
        d: Dimensions per vector.

    Returns:
        Reconstructed float32 array of length n*d.
    """
    var out = List[Float32](capacity=n * d)
    for _ in range(n * d):
        out.append(0.0)

    for i in range(n):
        var base = i * d
        var s = scales[i]
        for j in range(d):
            out[base + j] = Float32(q_flat[base + j]) * s

    return out^


# Python-compatible interface functions
fn quantize_int8_py(emb_flat: List[Float32], n: Int, d: Int) -> (List[Int8], List[Float32]):
    """Python-compatible quantize function."""
    return quantize_int8(emb_flat, n, d)

fn reconstruct_int8_py(q_flat: List[Int8], scales: List[Float32], n: Int, d: Int) -> List[Float32]:
    """Python-compatible reconstruct function."""
    return reconstruct_int8(q_flat, scales, n, d)
