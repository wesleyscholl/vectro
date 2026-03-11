"""
SIMD-accelerated INT8 vector quantizer for Vectro v3, Phase 1.

Key improvements over the scalar quantizer.mojo:
  - vectorize[SIMD_WIDTH]() for abs-max reduction and quantize inner loops
  - abs()-based max via SIMD reduce_max(), no branching per element
  - Symmetric abs-max scaling (zero-centred, correct for LLM embeddings)
  - perf_counter_ns benchmark included

Target: >= 5 M vec/s for INT8 on d=768 (vs ~70K in v2 scalar path).
"""

from algorithm import vectorize
from math import sqrt
from sys.info import simdwidthof
from time import perf_counter_ns

alias SIMD_W = simdwidthof[DType.float32]()


fn quantize_int8_simd(
    emb_flat: List[Float32], n: Int, d: Int
) -> (List[Int8], List[Float32]):
    """Quantize n*d float32 values to INT8 using SIMD inner loops.

    Args:
        emb_flat: Flat row-major array of length n*d.
        n: Number of vectors.
        d: Dimensions per vector.
    Returns:
        (q_flat, scales) — INT8 array of length n*d and float32 scales of length n.
    """
    var q = List[Int8](capacity=n * d)
    var scales = List[Float32](capacity=n)

    for _ in range(n * d):
        q.append(0)
    for _ in range(n):
        scales.append(0.0)

    for i in range(n):
        var base = i * d
        var ptr = emb_flat.unsafe_ptr() + base

        # ── Pass 1: SIMD abs-max reduction ──────────────────────────────────
        var acc_max: Float32 = 0.0

        @parameter
        fn _max_kernel[w: Int](j: Int):
            var v = SIMD[DType.float32, w].load(ptr + j)
            var a = v.abs()
            acc_max = max(acc_max, a.reduce_max())

        vectorize[_max_kernel, SIMD_W](d)

        var scale: Float32 = 1.0
        if acc_max > 0.0:
            scale = acc_max / 127.0
        scales[i] = scale

        # ── Pass 2: SIMD quantize ────────────────────────────────────────────
        var inv_scale = 1.0 / scale
        var qptr = q.unsafe_ptr() + base

        @parameter
        fn _quant_kernel[w: Int](j: Int):
            var raw = SIMD[DType.float32, w].load(ptr + j) * inv_scale
            # Clamp to [-127, 127]
            raw = raw.max(SIMD[DType.float32, w](-127.0))
            raw = raw.min(SIMD[DType.float32, w](127.0))
            # Round: add 0.5 for positive, sub 0.5 for negative, then truncate
            var sign = (raw >= SIMD[DType.float32, w](0.0)).select(
                SIMD[DType.float32, w](0.5),
                SIMD[DType.float32, w](-0.5),
            )
            var rounded = (raw + sign).__int__()
            # Store as Int8 — must be scalar tail for safety
            for k in range(w):
                qptr[j + k] = Int8(rounded[k])

        vectorize[_quant_kernel, SIMD_W](d)

    return (q^, scales^)


fn reconstruct_int8_simd(
    q_flat: List[Int8], scales: List[Float32], n: Int, d: Int
) -> List[Float32]:
    """Reconstruct float32 embeddings from INT8 + per-vector scales.

    Args:
        q_flat: Flat int8 array (length n*d).
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
        var q_ptr = q_flat.unsafe_ptr() + base
        var o_ptr = out.unsafe_ptr() + base

        @parameter
        fn _recon_kernel[w: Int](j: Int):
            # Widen Int8 → Float32 element-by-element (no SIMD cast yet in Mojo List)
            for k in range(w):
                o_ptr[j + k] = Float32(q_ptr[j + k]) * s

        vectorize[_recon_kernel, SIMD_W](d)

    return out^


fn benchmark_quantize_simd(n: Int, d: Int, warmup: Int = 3, iters: Int = 20):
    """Measure and print quantization + reconstruction throughput.

    Args:
        n: Number of vectors per batch.
        d: Vector dimension.
        warmup: Number of warm-up iterations (not timed).
        iters: Number of timed iterations.
    """
    from random import random_float64

    print("Benchmark: n=", n, " d=", d, " iters=", iters)

    var data = List[Float32](capacity=n * d)
    for _ in range(n * d):
        data.append(Float32(random_float64() * 2.0 - 1.0))

    # Warm up
    for _ in range(warmup):
        var r = quantize_int8_simd(data, n, d)
        _ = reconstruct_int8_simd(r.0, r.1, n, d)

    # Timed quantization
    var t0 = perf_counter_ns()
    for _ in range(iters):
        _ = quantize_int8_simd(data, n, d)
    var quant_ns = perf_counter_ns() - t0

    var quant_throughput = Float64(n * iters) / (Float64(quant_ns) / 1e9)
    print("  Quantize  throughput:", Int(quant_throughput), "vec/s")

    # Timed reconstruction
    var r2 = quantize_int8_simd(data, n, d)
    t0 = perf_counter_ns()
    for _ in range(iters):
        _ = reconstruct_int8_simd(r2.0, r2.1, n, d)
    var recon_ns = perf_counter_ns() - t0

    var recon_throughput = Float64(n * iters) / (Float64(recon_ns) / 1e9)
    print("  Reconstruct throughput:", Int(recon_throughput), "vec/s")


fn main():
    """Smoke-test and benchmark SIMD quantizer."""
    print("=" * 70)
    print("Vectro SIMD Quantizer (Phase 1)")
    print("SIMD width:", SIMD_W, "x Float32")
    print("=" * 70)

    # Correctness check
    var data = List[Float32]()
    for i in range(12):
        data.append(Float32(i + 1) * 0.5)

    var r = quantize_int8_simd(data, 3, 4)
    var recon = reconstruct_int8_simd(r.0, r.1, 3, 4)

    print("\nCorrectness check (3 vectors, d=4):")
    print("  Original  :", data[0], data[1], data[2], data[3], "...")
    print("  Scales     :", r.1[0], r.1[1], r.1[2])
    print("  Reconstructed:", recon[0], recon[1], recon[2], recon[3], "...")

    var max_err: Float32 = 0.0
    for i in range(12):
        var err = data[i] - recon[i]
        var a = err if err >= 0.0 else -err
        if a > max_err:
            max_err = a
    print("  Max abs error:", max_err)

    print("\nPerformance benchmarks:")
    benchmark_quantize_simd(n=1000, d=768)
    benchmark_quantize_simd(n=10000, d=384)

    print("\n" + "=" * 70)
    print("SIMD quantizer ready.")
    print("=" * 70)
