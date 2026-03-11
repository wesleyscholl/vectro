"""
NF4 (Normal Float 4-bit) quantizer for Vectro v3, Phase 2.

NF4 levels are placed at the 16 quantiles of N(0,1) as introduced by
Dettmers et al., QLoRA (2023).  Compared with linear INT4 this reduces
reconstruction error by ~20% for normally-distributed embedding data.

Key design decisions:
  - Per-vector abs-max normalisation to [-1, 1] before encoding.
  - Nearest-level lookup via a simple linear scan of the 16 constants
    (fits in one SIMD[float32, 16] register on AVX-512/NEON).
  - Two NF4 indices packed per byte (low nibble = even dim, high = odd).
  - Reconstruct: unpack nibbles → table lookup → multiply by scale.

Target: cosine_sim >= 0.985 at d=768; throughput >= 2 M vec/s.
"""

from algorithm import vectorize
from sys.info import simdwidthof
from time import perf_counter_ns

# NF4 codebook — quantiles of N(0,1) from Dettmers et al. 2023
alias NF4 = SIMD[DType.float32, 16](
    -1.0,        -0.6961928,  -0.5250730,  -0.3949003,
    -0.2844677,  -0.1848745,  -0.09105004,  0.0,
     0.07958031,  0.16093908,  0.24611496,  0.33791524,
     0.44070983,  0.56266755,  0.72295761,  1.0,
)

alias SIMD_W = simdwidthof[DType.float32]()


# ────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────

fn _nearest_nf4(v: Float32) -> Int:
    """Return the NF4 index (0-15) nearest to v in [-1, 1].

    Args:
        v: A float32 value, already normalised to [-1, 1].
    Returns:
        Index 0-15 of the closest NF4 level.
    """
    var best_idx = 0
    var best_dist = (v - NF4[0]) * (v - NF4[0])
    for i in range(1, 16):
        var d = (v - NF4[i]) * (v - NF4[i])
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


fn _abs_max_simd(ptr: UnsafePointer[Float32], d: Int) -> Float32:
    """SIMD abs-max reduction over d float32 values at ptr.

    Args:
        ptr: Pointer to the first element.
        d: Number of elements.
    Returns:
        Maximum absolute value found.
    """
    var acc: Float32 = 0.0

    @parameter
    fn _k[w: Int](i: Int):
        var v = SIMD[DType.float32, w].load(ptr + i)
        acc = max(acc, v.abs().reduce_max())

    vectorize[_k, SIMD_W](d)
    return acc


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

fn quantize_nf4(
    emb_flat: List[Float32], n: Int, d: Int
) -> (List[UInt8], List[Float32]):
    """Quantize n*d float32 values to NF4 (4-bit) packed representation.

    Each pair of consecutive dimensions is packed into one byte:
        byte = (idx[dim+1] << 4) | idx[dim]

    Args:
        emb_flat: Flat row-major array of length n*d.
        n: Number of vectors.
        d: Dimensions per vector (must be even).
    Returns:
        (packed, scales) where
          packed[i*bytes_per_vec .. (i+1)*bytes_per_vec] holds NF4 nibbles
          and scales[i] is the per-vector abs-max normalisation factor.
    """
    var bytes_per_vec = (d + 1) // 2
    var packed = List[UInt8](capacity=n * bytes_per_vec)
    var scales = List[Float32](capacity=n)

    for _ in range(n * bytes_per_vec):
        packed.append(0)
    for _ in range(n):
        scales.append(0.0)

    for i in range(n):
        var base = i * d
        var pbase = i * bytes_per_vec
        var ptr = emb_flat.unsafe_ptr() + base

        var abs_max = _abs_max_simd(ptr, d)
        scales[i] = abs_max

        var inv_scale: Float32 = 1.0
        if abs_max > 0.0:
            inv_scale = 1.0 / abs_max

        var j = 0
        while j + 1 < d:
            var lo = _nearest_nf4(emb_flat[base + j] * inv_scale)
            var hi = _nearest_nf4(emb_flat[base + j + 1] * inv_scale)
            packed[pbase + j // 2] = UInt8((hi << 4) | lo)
            j += 2

        # Handle odd trailing dimension
        if j < d:
            var lo = _nearest_nf4(emb_flat[base + j] * inv_scale)
            packed[pbase + j // 2] = UInt8(lo)

    return (packed^, scales^)


fn dequantize_nf4(
    packed: List[UInt8], scales: List[Float32], n: Int, d: Int
) -> List[Float32]:
    """Reconstruct float32 embeddings from NF4 packed bytes + per-vector scales.

    Args:
        packed: NF4 packed bytes, length n * ((d+1)//2).
        scales: Per-vector abs-max scale factors, length n.
        n: Number of vectors.
        d: Dimensions per vector.
    Returns:
        Reconstructed float32 array of length n*d.
    """
    var bytes_per_vec = (d + 1) // 2
    var out = List[Float32](capacity=n * d)
    for _ in range(n * d):
        out.append(0.0)

    for i in range(n):
        var base = i * d
        var pbase = i * bytes_per_vec
        var s = scales[i]

        var j = 0
        while j + 1 < d:
            var byte_val = Int(packed[pbase + j // 2])
            var lo = byte_val & 0xF
            var hi = (byte_val >> 4) & 0xF
            out[base + j] = NF4[lo] * s
            out[base + j + 1] = NF4[hi] * s
            j += 2

        if j < d:
            var lo = Int(packed[pbase + j // 2]) & 0xF
            out[base + j] = NF4[lo] * s

    return out^


fn benchmark_nf4(n: Int, d: Int, iters: Int = 20):
    """Print NF4 quantize + dequantize throughput for a synthetic batch.

    Args:
        n: Vectors per batch.
        d: Vector dimension.
        iters: Number of timed iterations.
    """
    from random import random_float64

    print("NF4 Benchmark: n=", n, " d=", d, " iters=", iters)

    var data = List[Float32](capacity=n * d)
    for _ in range(n * d):
        data.append(Float32(random_float64() * 2.0 - 1.0))

    # Warmup
    for _ in range(3):
        var r = quantize_nf4(data, n, d)
        _ = dequantize_nf4(r.0, r.1, n, d)

    var t0 = perf_counter_ns()
    for _ in range(iters):
        _ = quantize_nf4(data, n, d)
    var q_ns = perf_counter_ns() - t0
    print("  Quantize    :", Int(Float64(n * iters) / (Float64(q_ns) / 1e9)), "vec/s")

    var r2 = quantize_nf4(data, n, d)
    t0 = perf_counter_ns()
    for _ in range(iters):
        _ = dequantize_nf4(r2.0, r2.1, n, d)
    var dq_ns = perf_counter_ns() - t0
    print("  Dequantize  :", Int(Float64(n * iters) / (Float64(dq_ns) / 1e9)), "vec/s")

    # Cosine similarity check
    var recon = dequantize_nf4(r2.0, r2.1, n, d)
    var total_cos_: Float32 = 0.0
    for vi in range(n):
        var dot: Float32 = 0.0
        var n1: Float32 = 0.0
        var n2: Float32 = 0.0
        for k in range(d):
            var a = data[vi * d + k]
            var b = recon[vi * d + k]
            dot += a * b
            n1 += a * a
            n2 += b * b
        if n1 > 0.0 and n2 > 0.0:
            total_cos_ += dot / (n1.sqrt() * n2.sqrt())
    print("  Avg cos_sim :", total_cos_ / Float32(n))
    print("  Compression : 8x vs FP32")


fn main():
    """Smoke-test and benchmark NF4 quantizer."""
    print("=" * 70)
    print("Vectro NF4 Quantizer (Phase 2)")
    print("=" * 70)

    var data = List[Float32]()
    for i in range(8):
        data.append(Float32(i - 3) * 0.3)

    var r = quantize_nf4(data, 2, 4)
    var recon = dequantize_nf4(r.0, r.1, 2, 4)

    print("\nCorrectness check (2 vectors, d=4):")
    print("  Original     :", data[0], data[1], data[2], data[3])
    print("  Scales       :", r.1[0], r.1[1])
    print("  Packed bytes :", Int(r.0[0]), Int(r.0[1]))
    print("  Reconstructed:", recon[0], recon[1], recon[2], recon[3])

    print("\nPerformance benchmarks:")
    benchmark_nf4(n=1000, d=768)
    benchmark_nf4(n=10000, d=384)

    print("\n" + "=" * 70)
    print("NF4 quantizer ready.")
    print("=" * 70)
