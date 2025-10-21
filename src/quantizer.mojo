"""
Mojo quantizer: per-vector int8 quantization and reconstruction.

This file implements a simple per-vector int8 quantizer:
 - For each vector v (length d): scale = max_abs(v) / 127 (or 1.0 if zero)
 - Quantized value q = round(v / scale) clamped to int8 range [-127,127]

The functions operate on flat row-major arrays so they are easy to call from
Python bindings. This implementation uses plain loops but is written so it can
be optimized further with Mojo parallel constructs when available.
"""

# Note: This code is written in a straightforward Mojo-like style. When Mojo's
# stable language features (parallel loops / SIMD intrinsics) are available,
# those can be layered on top of this implementation for additional speed.

fn quantize_int8(emb_flat: [f32], n: i32, d: i32) -> (q: [i8], scales: [f32]):
    """Quantize a flat embeddings array (row-major) to int8 with per-vector scales.

    Inputs:
      emb_flat: flat array of length n*d (row-major)
      n: number of vectors
      d: dimensions per vector

    Returns:
      q: flat int8 array length n*d
      scales: float32 array length n (per-vector scale)
    """
    # Allocate outputs
    q = [i8](n * d)
    scales = [f32](n)

    i = 0
    while i < n:
        base = i * d
        # compute max abs for vector i
        max_abs: f32 = 0.0
        j = 0
        while j < d:
            v = emb_flat[base + j]
            a = v if v >= 0.0 else -v
            if a > max_abs:
                max_abs = a
            j += 1

        scale: f32 = 1.0
        if max_abs != 0.0:
            scale = max_abs / 127.0
        scales[i] = scale

        # quantize elements
        j = 0
        while j < d:
            raw = emb_flat[base + j] / scale
            # clamp to [-127, 127]
            if raw > 127.0:
                raw = 127.0
            if raw < -127.0:
                raw = -127.0
            # round to nearest integer
            # Mojo has standard rounding; use a simple float->int conversion behavior
            q_val: i32 = int(round(raw))
            q[base + j] = i8(q_val)
            j += 1

        i += 1

    return (q, scales)


fn reconstruct_int8(q_flat: [i8], scales: [f32], n: i32, d: i32) -> [f32]:
    """Reconstruct float embeddings from int8 and per-vector scales.

    Inputs:
      q_flat: flat int8 array length n*d
      scales: float32 array length n
      n: number of vectors
      d: dims

    Returns:
      flat float32 array length n*d
    """
    out = [f32](n * d)
    i = 0
    while i < n:
        base = i * d
        s = scales[i]
        j = 0
        while j < d:
            out[base + j] = float(q_flat[base + j]) * s
            j += 1
        i += 1
    return out
