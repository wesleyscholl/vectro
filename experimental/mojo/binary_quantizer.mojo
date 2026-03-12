"""
Binary (1-bit) quantization for Vectro v3, Phase 4.

sign(v) → 1-bit per dimension, 8 bits packed per byte.
Hamming distance via XOR + POPCOUNT for ultra-fast ANN search.

Compression target: 32× vs FP32 (same ratio as PQ, zero training cost).
Scan throughput target: >= 50 M vec/s at d=768.
"""

from algorithm import vectorize
from sys.info import simdwidthof
from time import perf_counter_ns

alias SIMD_W8 = simdwidthof[DType.uint8]()
alias SIMD_WF = simdwidthof[DType.float32]()


fn quantize_binary(
    emb_flat: List[Float32], n: Int, d: Int
) -> List[UInt8]:
    """Quantize n*d float32 values to binary (sign bit), packed 8 per byte.

    Vectors SHOULD be L2-normalised before calling this function for best
    recall quality.

    Args:
        emb_flat: Flat row-major float32 array, length n*d.
        n: Number of vectors.
        d: Dimensions per vector.
    Returns:
        Packed binary array; each vector occupies ceil(d/8) bytes.
    """
    var bytes_per_vec = (d + 7) // 8
    var out = List[UInt8](capacity=n * bytes_per_vec)
    for _ in range(n * bytes_per_vec):
        out.append(0)

    for i in range(n):
        var base_f = i * d
        var base_b = i * bytes_per_vec

        for j in range(d):
            if emb_flat[base_f + j] > 0.0:
                var byte_idx = j >> 3       # j // 8
                var bit_pos  = j & 7        # j %  8
                out[base_b + byte_idx] |= UInt8(1 << bit_pos)

    return out^


fn dequantize_binary(
    packed: List[UInt8], n: Int, d: Int
) -> List[Float32]:
    """Reconstruct {-1, +1} float32 vectors from binary packed bytes.

    Args:
        packed: Packed binary array (length n * ceil(d/8)).
        n: Number of vectors.
        d: Original dimension.
    Returns:
        Float32 array of n*d values; each element is +1.0 or -1.0.
    """
    var bytes_per_vec = (d + 7) // 8
    var out = List[Float32](capacity=n * d)
    for _ in range(n * d):
        out.append(-1.0)

    for i in range(n):
        var base_f = i * d
        var base_b = i * bytes_per_vec

        for j in range(d):
            var byte_idx = j >> 3
            var bit_pos  = j & 7
            if (Int(packed[base_b + byte_idx]) >> bit_pos) & 1 == 1:
                out[base_f + j] = 1.0

    return out^


fn hamming_distance(
    a_ptr: UnsafePointer[UInt8],
    b_ptr: UnsafePointer[UInt8],
    bytes: Int,
) -> Int:
    """Compute Hamming distance between two binary vectors (byte arrays).

    Uses XOR + popcount (Mojo's Int.popcount() where available, else manual).

    Args:
        a_ptr: Pointer to first vector.
        b_ptr: Pointer to second vector.
        bytes: Byte length (= ceil(d/8)).
    Returns:
        Hamming distance (number of differing bits).
    """
    var dist = 0
    for i in range(bytes):
        var xor_byte = Int(a_ptr[i] ^ b_ptr[i])
        # Brian Kernighan popcount
        var v = xor_byte
        while v != 0:
            v &= v - 1
            dist += 1
    return dist


fn hamming_batch(
    query_ptr:  UnsafePointer[UInt8],   # [bytes_per_vec]
    db_ptr:     UnsafePointer[UInt8],   # [n * bytes_per_vec]
    n:          Int,
    bytes_per_vec: Int,
    dist_out:   UnsafePointer[Int32],   # [n]
):
    """Compute Hamming distance from one query to n database binary vectors.

    Args:
        query_ptr:     Pointer to query binary vector.
        db_ptr:        Pointer to flat database binary array.
        n:             Number of database vectors.
        bytes_per_vec: Bytes per vector = ceil(d/8).
        dist_out:      Output buffer of n Int32 distances.
    """
    for i in range(n):
        var db = db_ptr + i * bytes_per_vec
        dist_out[i] = Int32(hamming_distance(query_ptr, db, bytes_per_vec))


fn top_k_hamming(
    query_packed:  List[UInt8],            # single query, ceil(d/8) bytes
    db_packed:     List[UInt8],            # n database vectors
    n:             Int,
    bytes_per_vec: Int,
    k:             Int,
) -> List[Int]:
    """Return indices of the k nearest binary vectors by Hamming distance.

    Args:
        query_packed:  Packed query binary vector.
        db_packed:     Flat packed database.
        n:             Number of database vectors.
        bytes_per_vec: Bytes per vector.
        k:             Number of nearest neighbours to return.
    Returns:
        List of k indices (ascending Hamming distance order).
    """
    var q_ptr = query_packed.unsafe_ptr()
    var db_ptr = db_packed.unsafe_ptr()

    # Compute all distances
    var dists = List[Int32](capacity=n)
    for i in range(n):
        dists.append(Int32(hamming_distance(q_ptr, db_ptr + i * bytes_per_vec, bytes_per_vec)))

    # Partial selection sort for top-k
    var result = List[Int](capacity=k)
    var used = List[Bool](capacity=n)
    for _ in range(n):
        used.append(False)

    for _ in range(k):
        var best_d = Int32(2147483647)
        var best_i = 0
        for j in range(n):
            if not used[j] and dists[j] < best_d:
                best_d = dists[j]
                best_i = j
        result.append(best_i)
        used[best_i] = True

    return result^


fn benchmark_binary(n: Int, d: Int, iters: Int = 20):
    """Benchmark binary quantize + Hamming scan throughput.

    Args:
        n:     Vectors in database.
        d:     Vector dimension.
        iters: Timed iterations.
    """
    from random import random_float64

    var bytes_per_vec = (d + 7) // 8
    print("Binary Benchmark: n=", n, " d=", d, " bytes_per_vec=", bytes_per_vec)

    var data = List[Float32](capacity=n * d)
    for _ in range(n * d):
        data.append(Float32(random_float64() * 2.0 - 1.0))

    # Warmup
    for _ in range(3):
        _ = quantize_binary(data, n, d)

    var t0 = perf_counter_ns()
    for _ in range(iters):
        _ = quantize_binary(data, n, d)
    var q_ns = perf_counter_ns() - t0
    print("  Quantize  :", Int(Float64(n * iters) / (Float64(q_ns) / 1e9)), "vec/s")

    var db = quantize_binary(data, n, d)
    var query_data = List[Float32](capacity=d)
    for _ in range(d):
        query_data.append(Float32(random_float64() * 2.0 - 1.0))
    var qb = quantize_binary(query_data, 1, d)

    t0 = perf_counter_ns()
    for _ in range(iters):
        var q_ptr = qb.unsafe_ptr()
        var db_ptr = db.unsafe_ptr()
        var dist_acc = 0
        for i in range(n):
            dist_acc += hamming_distance(q_ptr, db_ptr + i * bytes_per_vec, bytes_per_vec)
        _ = dist_acc
    var h_ns = perf_counter_ns() - t0
    print("  Hamming scan:", Int(Float64(n * iters) / (Float64(h_ns) / 1e9)), "vec/s")
    print("  Compression :", Float64(d * 4) / Float64(bytes_per_vec), "x vs FP32")


fn main():
    """Smoke-test and benchmark binary quantizer."""
    print("=" * 70)
    print("Vectro Binary Quantizer (Phase 4)")
    print("=" * 70)

    var d = 8
    var data = List[Float32]()
    var vals = List[Float32]()
    vals.append( 0.5); vals.append(-0.3); vals.append( 0.1); vals.append(-0.9)
    vals.append(-0.2); vals.append( 0.8); vals.append(-0.4); vals.append( 0.6)
    for v in vals:
        data.append(v[])

    var packed = quantize_binary(data, 1, d)
    var recon  = dequantize_binary(packed, 1, d)

    print("\nInput  :", data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7])
    print("Packed :", Int(packed[0]))
    print("Recon  :", recon[0], recon[1], recon[2], recon[3], recon[4], recon[5], recon[6], recon[7])

    # Hamming distance test
    var a = quantize_binary(data, 1, d)
    var d2 = List[Float32]()
    for v in vals:
        d2.append(-v[])
    var b = quantize_binary(d2, 1, d)
    var h = hamming_distance(a.unsafe_ptr(), b.unsafe_ptr(), 1)
    print("\nHamming(v, -v) =", h, "  (expected 8)")

    print("\nPerformance:")
    benchmark_binary(n=10000, d=768)

    print("\n" + "=" * 70)
    print("Binary quantizer ready.")
    print("=" * 70)
