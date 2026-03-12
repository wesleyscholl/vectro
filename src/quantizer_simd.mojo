"""
SIMD-accelerated INT8 vector quantizer for Vectro v3, Phase 1.

Key improvements over the scalar quantizer.mojo:
  - vectorize[SIMD_WIDTH]() for abs-max reduction and quantize inner loops
  - abs()-based max via SIMD reduce_max(), no branching per element
  - Symmetric abs-max scaling (zero-centred, correct for LLM embeddings)
  - parallelize[_process_row](n) for multi-core throughput
  - perf_counter_ns benchmark included

Target: >= 5 M vec/s for INT8 on d=768 (vs ~70K in v2 scalar path).
"""

from algorithm import vectorize, parallelize
from io import FileDescriptor
from math import copysign
from time import perf_counter_ns

# Tile 4 NEON lanes in software (LLVM pipelines the 4 loads better than scalar).
alias SIMD_W: Int = 16


fn quantize_int8_simd(
    emb_flat: List[Float32], n: Int, d: Int
) -> (List[Int8], List[Float32]):
    """Quantize n*d float32 values to INT8 using SIMD inner loops + parallel rows.

    Args:
        emb_flat: Flat row-major array of length n*d.
        n: Number of vectors.
        d: Dimensions per vector.
    Returns:
        (q_flat, scales) — INT8 array of length n*d and float32 scales of length n.
    """
    var q = List[Int8](capacity=n * d)
    var scales = List[Float32](capacity=n)

    q.resize(n * d, Int8(0))
    scales.resize(n, Float32(0.0))

    var emb_ptr    = emb_flat.unsafe_ptr()
    var q_ptr_out  = q.unsafe_ptr()
    var scales_ptr = scales.unsafe_ptr()

    @parameter
    fn _process_row(i: Int):
        var ptr  = emb_ptr   + i * d
        var qptr = q_ptr_out + i * d

        # ── Pass 1: SIMD abs-max reduction ──────────────────────────────────
        var acc_max: Float32 = 0.0

        @parameter
        fn _max_kernel[w: Int](j: Int):
            acc_max = max(acc_max, abs(ptr.load[width=w](j)).reduce_max())

        vectorize[_max_kernel, SIMD_W](d)

        var scale: Float32 = acc_max / 127.0 if acc_max > 0.0 else Float32(1.0)
        scales_ptr[i] = scale

        # ── Pass 2: SIMD quantize + SIMD store ───────────────────────────────
        var inv_scale = Float32(1.0) / scale

        @parameter
        fn _quant_kernel[w: Int](j: Int):
            var raw = ptr.load[width=w](j) * inv_scale
            raw = max(raw, SIMD[DType.float32, w](-127.0))
            raw = min(raw, SIMD[DType.float32, w](127.0))
            var half = copysign(SIMD[DType.float32, w](0.5), raw)
            qptr.store(j, (raw + half).cast[DType.int32]().cast[DType.int8]())

        vectorize[_quant_kernel, SIMD_W](d)

    parallelize[_process_row](n)

    return (q^, scales^)


fn reconstruct_int8_simd(
    q_flat: List[Int8], scales: List[Float32], n: Int, d: Int
) -> List[Float32]:
    """Reconstruct float32 embeddings from INT8 + per-vector scales.

    Uses SIMD int8→float32 cast + multiply and parallelises over rows.

    Args:
        q_flat: Flat int8 array (length n*d).
        scales: Per-vector scale factors (length n).
        n: Number of vectors.
        d: Dimensions per vector.
    Returns:
        Reconstructed float32 array of length n*d.
    """
    var out = List[Float32](capacity=n * d)
    out.resize(n * d, Float32(0.0))

    var q_ptr_in   = q_flat.unsafe_ptr()
    var o_ptr_out  = out.unsafe_ptr()
    var scales_ptr = scales.unsafe_ptr()

    @parameter
    fn _recon_row(i: Int):
        var qp = q_ptr_in  + i * d
        var op = o_ptr_out + i * d
        var s  = scales_ptr[i]

        @parameter
        fn _recon_kernel[w: Int](j: Int):
            # SIMD int8 load → widen to float32 → multiply by scale → store
            var qi = qp.load[width=w](j)
            op.store(j, qi.cast[DType.float32]() * SIMD[DType.float32, w](s))

        vectorize[_recon_kernel, SIMD_W](d)

    parallelize[_recon_row](n)

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
