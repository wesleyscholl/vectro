"""
Product Quantization (PQ) engine for Vectro v3, Phase 3 — Mojo core.

This file contains the Mojo-side PQ encode/decode and the Asymmetric
Distance Computation (ADC) table loop.  Codebook training (K-means) is
done in the Python layer (pq_api.py) and the resulting float32 centroid
table is passed in at encode/decode time.

Compression target: d=768, M=96 sub-spaces → 32× vs FP32.
Encode throughput target: >= 500 K vec/s.

Coordinate convention:
    sub-vector s has dims [s * sub_dim .. (s+1) * sub_dim).
    Code for sub-vector s is the nearest centroid index (0..K-1).
    Packed representation: one UInt8 per sub-space (K <= 256).
"""

from algorithm import vectorize, parallelize
from sys.info import simdwidthof
from time import perf_counter_ns

alias SIMD_W = simdwidthof[DType.float32]()


fn pq_encode_row(
    vec:       UnsafePointer[Float32],
    centroids: UnsafePointer[Float32],   # [M * K * sub_dim] row-major
    M:         Int,
    K:         Int,
    sub_dim:   Int,
    out:       UnsafePointer[UInt8],
):
    """Encode one vector into M PQ codes (nearest centroid per sub-space).

    Args:
        vec:       Pointer to the source vector (length M*sub_dim).
        centroids: Flat centroid table, layout centroids[m][k][j].
        M:         Number of sub-spaces.
        K:         Centroids per sub-space (must be <= 256).
        sub_dim:   Dimension of each sub-space (d // M).
        out:       Output buffer of M bytes.
    """
    var K_sub_dim = K * sub_dim

    for m in range(M):
        var v_ptr = vec + m * sub_dim
        var c_ptr = centroids + m * K_sub_dim
        var best_dist: Float32 = 1e38
        var best_k: Int = 0

        for k in range(K):
            var c = c_ptr + k * sub_dim
            var dist: Float32 = 0.0

            @parameter
            fn _l2[w: Int](j: Int):
                var diff = SIMD[DType.float32, w].load(v_ptr + j) - \
                           SIMD[DType.float32, w].load(c + j)
                dist += (diff * diff).reduce_add()

            vectorize[_l2, SIMD_W](sub_dim)

            if dist < best_dist:
                best_dist = dist
                best_k = k

        out[m] = UInt8(best_k)


fn pq_encode_batch(
    emb_flat:  List[Float32],          # [n, d] flat
    centroids: List[Float32],          # [M * K * sub_dim]
    n:         Int,
    M:         Int,
    K:         Int,
    sub_dim:   Int,
) -> List[UInt8]:
    """Encode n vectors into M-byte PQ codes.

    Args:
        emb_flat:  Flat input array (length n * M * sub_dim).
        centroids: Centroid table (length M * K * sub_dim).
        n:         Number of vectors.
        M:         Sub-spaces.
        K:         Centroids per sub-space.
        sub_dim:   d // M.
    Returns:
        Flat UInt8 array of length n*M.
    """
    var d = M * sub_dim
    var codes = List[UInt8](capacity=n * M)
    for _ in range(n * M):
        codes.append(0)

    var vec_ptr = emb_flat.unsafe_ptr()
    var cen_ptr = centroids.unsafe_ptr()
    var out_ptr = codes.unsafe_ptr()

    for i in range(n):
        pq_encode_row(
            vec_ptr + i * d,
            cen_ptr,
            M, K, sub_dim,
            out_ptr + i * M,
        )

    return codes^


fn pq_decode_batch(
    codes:     List[UInt8],    # [n * M]
    centroids: List[Float32],  # [M * K * sub_dim]
    n:         Int,
    M:         Int,
    K:         Int,
    sub_dim:   Int,
) -> List[Float32]:
    """Decode PQ codes back to approximate float32 vectors.

    Args:
        codes:     Flat UInt8 array (length n*M).
        centroids: Centroid table (length M*K*sub_dim).
        n:         Number of vectors.
        M:         Sub-spaces.
        K:         Centroids per sub-space.
        sub_dim:   d // M.
    Returns:
        Reconstructed float32 array of length n*M*sub_dim.
    """
    var d = M * sub_dim
    var out = List[Float32](capacity=n * d)
    for _ in range(n * d):
        out.append(0.0)

    var K_sub_dim = K * sub_dim

    for i in range(n):
        for m in range(M):
            var k = Int(codes[i * M + m])
            var c_ptr = centroids.unsafe_ptr() + m * K_sub_dim + k * sub_dim
            var o_ptr = out.unsafe_ptr() + i * d + m * sub_dim
            for j in range(sub_dim):
                o_ptr[j] = c_ptr[j]

    return out^


fn pq_distance_table(
    query:     UnsafePointer[Float32],   # [d]
    centroids: UnsafePointer[Float32],   # [M * K * sub_dim]
    M:         Int,
    K:         Int,
    sub_dim:   Int,
    table_out: UnsafePointer[Float32],   # [M * K]
):
    """Pre-compute per-query ADC table for fast batch distance.

    For each sub-space m and centroid k, stores the squared L2 distance
    between the query sub-vector and centroid k.

    Args:
        query:     Pointer to query vector (length M*sub_dim).
        centroids: Centroid table.
        M:         Sub-spaces.
        K:         Centroids per sub-space.
        sub_dim:   d // M.
        table_out: Output buffer of M*K floats.
    """
    var K_sub_dim = K * sub_dim

    for m in range(M):
        var q_ptr = query + m * sub_dim
        var c_ptr = centroids + m * K_sub_dim

        for k in range(K):
            var ck = c_ptr + k * sub_dim
            var dist: Float32 = 0.0

            @parameter
            fn _l2[w: Int](j: Int):
                var diff = SIMD[DType.float32, w].load(q_ptr + j) - \
                           SIMD[DType.float32, w].load(ck + j)
                dist += (diff * diff).reduce_add()

            vectorize[_l2, SIMD_W](sub_dim)
            table_out[m * K + k] = dist


fn pq_distance_batch_adc(
    codes:     UnsafePointer[UInt8],    # [n * M]
    table:     UnsafePointer[Float32],  # [M * K]
    n:         Int,
    M:         Int,
    K:         Int,
    dist_out:  UnsafePointer[Float32],  # [n]
):
    """Accumulate ADC distances for n PQ-coded vectors.

    Args:
        codes:    PQ codes of n database vectors.
        table:    Pre-computed query distance table [M * K].
        n:        Number of database vectors.
        M:        Sub-spaces.
        K:        Centroids per sub-space.
        dist_out: Output distances (length n).
    """
    for i in range(n):
        var acc: Float32 = 0.0
        for m in range(M):
            var k = Int(codes[i * M + m])
            acc += table[m * K + k]
        dist_out[i] = acc


fn benchmark_pq(n: Int, d: Int, M: Int, K: Int, iters: Int = 10):
    """Benchmark PQ encode throughput with random data and centroids.

    Args:
        n:     Vectors per batch.
        d:     Must be divisible by M.
        M:     Sub-spaces.
        K:     Centroids per sub-space (usually 256).
        iters: Timed iterations.
    """
    from random import random_float64

    var sub_dim = d // M
    print("PQ Benchmark: n=", n, " d=", d, " M=", M, " K=", K, " sub_dim=", sub_dim)

    var data = List[Float32](capacity=n * d)
    for _ in range(n * d):
        data.append(Float32(random_float64() * 2.0 - 1.0))

    var centroids = List[Float32](capacity=M * K * sub_dim)
    for _ in range(M * K * sub_dim):
        centroids.append(Float32(random_float64() * 2.0 - 1.0))

    # Warmup
    for _ in range(2):
        _ = pq_encode_batch(data, centroids, n, M, K, sub_dim)

    var t0 = perf_counter_ns()
    for _ in range(iters):
        _ = pq_encode_batch(data, centroids, n, M, K, sub_dim)
    var ns = perf_counter_ns() - t0
    print("  Encode throughput:", Int(Float64(n * iters) / (Float64(ns) / 1e9)), "vec/s")


fn main():
    """Smoke-test PQ encode/decode and benchmark."""
    print("=" * 70)
    print("Vectro PQ Engine (Phase 3)")
    print("=" * 70)

    var d = 8
    var M = 4
    var K = 4
    var sub_dim = d // M
    var n = 2

    var data = List[Float32]()
    for i in range(n * d):
        data.append(Float32(i + 1) * 0.1)

    var centroids = List[Float32]()
    for _ in range(M * K * sub_dim):
        centroids.append(0.0)
    # Fill c0 with identity-ish values
    for m in range(M):
        for j in range(sub_dim):
            centroids[m * K * sub_dim + j] = Float32(m + 1) * 0.1

    var codes = pq_encode_batch(data, centroids, n, M, K, sub_dim)
    print("\nPQ codes (2 vectors):", Int(codes[0]), Int(codes[1]), "...")

    var recon = pq_decode_batch(codes, centroids, n, M, K, sub_dim)
    print("Reconstructed[0]:", recon[0], recon[1], "...")

    print("\nBenchmark:")
    benchmark_pq(n=100, d=16, M=4, K=16)

    print("\n" + "=" * 70)
    print("PQ engine ready.")
    print("=" * 70)
