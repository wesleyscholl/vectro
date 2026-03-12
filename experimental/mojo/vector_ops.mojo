"""
High-performance vector operations for similarity and distance computations.

Phase 1 (v3): scalar loops replaced with SIMD vectorize() calls.
All hot functions operate on List[Float32] and use the native SIMD width
so the compiler can emit AVX2/NEON instructions automatically.
"""

from algorithm import vectorize
from math import sqrt
from sys.info import simdwidthof

alias SIMD_WIDTH = simdwidthof[DType.float32]()


fn cosine_similarity(vec1: List[Float32], vec2: List[Float32]) -> Float32:
    """Compute cosine similarity between two float32 vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.
    Returns:
        Cosine similarity in [-1, 1].  Returns 0.0 for zero vectors.
    """
    var dim = len(vec1)
    if dim != len(vec2):
        print("Error: Vector dimensions don't match")
        return 0.0

    var dot: Float32 = 0.0
    var n1: Float32 = 0.0
    var n2: Float32 = 0.0

    @parameter
    fn _kernel[w: Int](i: Int):
        var v1 = SIMD[DType.float32, w].load(vec1.unsafe_ptr() + i)
        var v2 = SIMD[DType.float32, w].load(vec2.unsafe_ptr() + i)
        dot += (v1 * v2).reduce_add()
        n1  += (v1 * v1).reduce_add()
        n2  += (v2 * v2).reduce_add()

    vectorize[_kernel, SIMD_WIDTH](dim)

    var denom = sqrt(n1) * sqrt(n2)
    if denom < 1e-10:
        return 0.0
    return dot / denom


fn euclidean_distance(vec1: List[Float32], vec2: List[Float32]) -> Float32:
    """Compute Euclidean (L2) distance between two float32 vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.
    Returns:
        L2 distance.
    """
    var dim = len(vec1)
    if dim != len(vec2):
        print("Error: Vector dimensions don't match")
        return 0.0

    var sum_sq: Float32 = 0.0

    @parameter
    fn _kernel[w: Int](i: Int):
        var d = SIMD[DType.float32, w].load(vec1.unsafe_ptr() + i) \
              - SIMD[DType.float32, w].load(vec2.unsafe_ptr() + i)
        sum_sq += (d * d).reduce_add()

    vectorize[_kernel, SIMD_WIDTH](dim)
    return sqrt(sum_sq)


fn manhattan_distance(vec1: List[Float32], vec2: List[Float32]) -> Float32:
    """Compute Manhattan (L1) distance between two float32 vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.
    Returns:
        L1 distance.
    """
    var dim = len(vec1)
    if dim != len(vec2):
        print("Error: Vector dimensions don't match")
        return 0.0

    var sum_abs: Float32 = 0.0

    @parameter
    fn _kernel[w: Int](i: Int):
        var d = SIMD[DType.float32, w].load(vec1.unsafe_ptr() + i) \
              - SIMD[DType.float32, w].load(vec2.unsafe_ptr() + i)
        sum_abs += d.abs().reduce_add()

    vectorize[_kernel, SIMD_WIDTH](dim)
    return sum_abs


fn dot_product(vec1: List[Float32], vec2: List[Float32]) -> Float32:
    """Compute dot product of two float32 vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.
    Returns:
        Dot product.
    """
    var dim = len(vec1)
    if dim != len(vec2):
        print("Error: Vector dimensions don't match")
        return 0.0

    var result: Float32 = 0.0

    @parameter
    fn _kernel[w: Int](i: Int):
        var v1 = SIMD[DType.float32, w].load(vec1.unsafe_ptr() + i)
        var v2 = SIMD[DType.float32, w].load(vec2.unsafe_ptr() + i)
        result += (v1 * v2).reduce_add()

    vectorize[_kernel, SIMD_WIDTH](dim)
    return result


fn vector_norm(vec: List[Float32]) -> Float32:
    """Compute L2 norm of a float32 vector.

    Args:
        vec: Input vector.
    Returns:
        L2 norm.
    """
    var dim = len(vec)
    var sum_sq: Float32 = 0.0

    @parameter
    fn _kernel[w: Int](i: Int):
        var v = SIMD[DType.float32, w].load(vec.unsafe_ptr() + i)
        sum_sq += (v * v).reduce_add()

    vectorize[_kernel, SIMD_WIDTH](dim)
    return sqrt(sum_sq)


fn normalize_vector(var vec: List[Float32]) -> List[Float32]:
    """Normalize a float32 vector to unit L2 length in-place and return it.

    Args:
        vec: Input vector (consumed).
    Returns:
        Normalised vector (same buffer, unit length).
    """
    var norm = vector_norm(vec)
    if norm < 1e-10:
        return vec^

    var inv_norm = 1.0 / norm
    var dim = len(vec)

    @parameter
    fn _kernel[w: Int](i: Int):
        var v = SIMD[DType.float32, w].load(vec.unsafe_ptr() + i)
        (v * inv_norm).store(vec.unsafe_ptr() + i)

    vectorize[_kernel, SIMD_WIDTH](dim)
    return vec^


struct VectorOps:
    """Collection of SIMD-accelerated vector operations."""

    @staticmethod
    fn batch_cosine_similarity(
        vectors1: List[List[Float32]],
        vectors2: List[List[Float32]]
    ) -> List[Float32]:
        """Compute cosine similarities for batches of vectors.
        Args:
            vectors1: First batch of vectors.
            vectors2: Second batch of vectors.
        Returns:
            List of similarity scores.
        """
        var batch_size = len(vectors1)
        var similarities = List[Float32]()
        for i in range(batch_size):
            similarities.append(cosine_similarity(vectors1[i], vectors2[i]))
        return similarities^

    @staticmethod
    fn batch_euclidean_distance(
        vectors1: List[List[Float32]],
        vectors2: List[List[Float32]]
    ) -> List[Float32]:
        """Compute Euclidean distances for batches of vectors.
        Args:
            vectors1: First batch of vectors.
            vectors2: Second batch of vectors.
        Returns:
            List of distance values.
        """
        var batch_size = len(vectors1)
        var distances = List[Float32]()
        for i in range(batch_size):
            distances.append(euclidean_distance(vectors1[i], vectors2[i]))
        return distances^


fn main():
    """Smoke-test and throughput check for SIMD vector ops."""
    print("=" * 70)
    print("Vectro Vector Operations Module (SIMD, Phase 1)")
    print("=" * 70)

    var vec1 = List[Float32]()
    vec1.append(1.0); vec1.append(2.0); vec1.append(3.0)

    var vec2 = List[Float32]()
    vec2.append(4.0); vec2.append(5.0); vec2.append(6.0)

    print("\nvec1 = [1.0, 2.0, 3.0]")
    print("vec2 = [4.0, 5.0, 6.0]")
    print("  Cosine similarity :", cosine_similarity(vec1, vec2))
    print("  Euclidean distance:", euclidean_distance(vec1, vec2))
    print("  Manhattan distance:", manhattan_distance(vec1, vec2))
    print("  Dot product       :", dot_product(vec1, vec2))
    print("  vec1 norm         :", vector_norm(vec1))

    # Throughput benchmark
    from time import perf_counter_ns

    var dim = 768
    var n_iters = 10_000
    var a = List[Float32](capacity=dim)
    var b = List[Float32](capacity=dim)
    for i in range(dim):
        a.append(Float32(i + 1) * 0.001)
        b.append(Float32(dim - i) * 0.001)

    var start = perf_counter_ns()
    for _ in range(n_iters):
        _ = cosine_similarity(a, b)
    var elapsed_ns = perf_counter_ns() - start
    var ops_per_sec = Float64(n_iters) / (Float64(elapsed_ns) / 1e9)
    print("\nCosine similarity throughput (dim=768):", Int(ops_per_sec), "ops/sec")
    print("=" * 70)
