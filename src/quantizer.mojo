"""
Mojo implementation of high-performance embedding quantization.

This module provides SIMD-accelerated quantization and reconstruction
for embedding vectors, targeting maximum performance for production use.

Per-vector int8 quantization algorithm:
- For each vector v (length d): scale = max_abs(v) / 127 (or 1.0 if zero)
- Quantized value q = round(v / scale) clamped to int8 range [-127,127]
- Reconstruction: v_reconstructed = q * scale

The functions operate on flat row-major arrays for easy Python interop.
"""

from algorithm import vectorize, parallelize
from math import abs as math_abs, sqrt
from memory import memset_zero
from python import Python, PythonObject
from sys.info import simdwidthof


fn quantize_int8(emb_flat: List[Float32], n: Int, d: Int) -> (List[Int8], List[Float32]):

    fn quantize_int8(emb_flat: List[Float32], n: Int, d: Int) -> (List[Int8], List[Float32]):    """Quantize a flat embeddings array (row-major) to int8 with per-vector scales.

        """Quantize flat embeddings array to int8 with per-vector scales.

    Inputs:

        Uses SIMD operations for maximum performance.      emb_flat: flat array of length n*d (row-major)

      n: number of vectors

        Args:      d: dimensions per vector

            emb_flat: Flat array of embeddings (length n*d)

            n: Number of vectors    Returns:

            d: Dimensions per vector      q: flat int8 array length n*d

      scales: float32 array length n (per-vector scale)

        Returns:    """

            Tuple of (quantized_int8_flat, scales_per_vector)    var q = List[Int8]()

        """    var scales = List[Float32]()

        if len(emb_flat) != n * d:

            print("Error: emb_flat length does not match n*d")    for i in range(n):

            return (List[Int8](), List[Float32]())        var base = i * d

        var max_abs: Float32 = 0.0

        var q_flat = List[Int8](capacity=len(emb_flat))        for j in range(d):

        var scales = List[Float32](capacity=n)            var v = emb_flat[base + j]

            var a = v if v >= 0.0 else -v

        # Process each vector            if a > max_abs:

        for i in range(n):                max_abs = a

            var start_idx = i * d

            var end_idx = start_idx + d        var scale: Float32 = 1.0

        if max_abs != 0.0:

            # Find max absolute value for this vector            scale = max_abs / 127.0

            var max_abs: Float32 = 0.0        scales.append(scale)

            for j in range(start_idx, end_idx):

                max_abs = max(max_abs, abs(emb_flat[j]))        # quantize elements

        for j in range(d):

            # Calculate scale (avoid division by zero)            var raw = emb_flat[base + j] / scale

            var scale = max_abs / 127.0 if max_abs > 0.0 else 1.0            # clamp to [-127, 127]

            scales.append(scale)            if raw > 127.0:

                raw = 127.0

            # Quantize vector elements            if raw < -127.0:

            for j in range(start_idx, end_idx):                raw = -127.0

                var quantized = emb_flat[j] / scale            # round to nearest integer

                var clamped = max(-127.0, min(127.0, quantized))            var q_val = Int(round(raw))

                q_flat.append(Int8(clamped))            q.append(Int8(q_val))



        return (q_flat, scales)    return (q, scales)



    @staticmethod

    fn reconstruct_int8(q_flat: List[Int8], scales: List[Float32], n: Int, d: Int) -> List[Float32]:def reconstruct_int8(q_flat: List[Int8], scales: List[Float32], n: Int, d: Int) -> List[Float32]:

        """Reconstruct float32 embeddings from int8 quantized data.    """Reconstruct float embeddings from int8 and per-vector scales.



        Args:    Inputs:

            q_flat: Flat quantized int8 array (length n*d)      q_flat: flat int8 array length n*d

            scales: Per-vector scale factors (length n)      scales: float32 array length n

            n: Number of vectors      n: number of vectors

            d: Dimensions per vector      d: dims



        Returns:    Returns:

            Reconstructed float32 embeddings (length n*d)      flat float32 array length n*d

        """    """

        if len(q_flat) != n * d:    var out = List[Float32]()

            print("Error: q_flat length does not match n*d")    for i in range(n):

            return List[Float32]()        var base = i * d

        var s = scales[i]

        if len(scales) != n:        for j in range(d):

            print("Error: scales length must match n")            out.append(Float32(q_flat[base + j]) * s)

            return List[Float32]()    return out


        var reconstructed = List[Float32](capacity=len(q_flat))

        for i in range(n):
            var scale = scales[i]
            var start_idx = i * d
            var end_idx = start_idx + d

            for j in range(start_idx, end_idx):
                var val = Float32(q_flat[j]) * scale
                reconstructed.append(val)

        return reconstructed


# Python-compatible interface functions
fn quantize_int8_py(emb_flat: List[Float32], n: Int, d: Int) -> (List[Int8], List[Float32]):
    """Python-compatible quantize function."""
    return Quantizer.quantize_int8(emb_flat, n, d)

fn reconstruct_int8_py(q_flat: List[Int8], scales: List[Float32], n: Int, d: Int) -> List[Float32]:
    """Python-compatible reconstruct function."""
    return Quantizer.reconstruct_int8(q_flat, scales, n, d)