"""
High-performance vector operations for similarity and distance computations.
"""

fn cosine_similarity(vec1: List[Float32], vec2: List[Float32]) -> Float32:
    """Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector.
        vec2: Second vector.
        
    Returns:
        Cosine similarity value between -1 and 1.
    """
    var dim = len(vec1)
    if dim != len(vec2):
        print("Error: Vector dimensions don't match")
        return 0.0
    
    var dot_product: Float32 = 0.0
    var norm1: Float32 = 0.0
    var norm2: Float32 = 0.0
    
    for i in range(dim):
        var v1 = vec1[i]
        var v2 = vec2[i]
        dot_product += v1 * v2
        norm1 += v1 * v1
        norm2 += v2 * v2
    
    from math import sqrt
    var denom = sqrt(norm1) * sqrt(norm2)
    if denom < 1e-10:
        return 0.0
    return dot_product / denom


fn euclidean_distance(vec1: List[Float32], vec2: List[Float32]) -> Float32:
    """Compute Euclidean (L2) distance between two vectors.
    
    Args:
        vec1: First vector.
        vec2: Second vector.
        
    Returns:
        L2 distance value.
    """
    var dim = len(vec1)
    if dim != len(vec2):
        print("Error: Vector dimensions don't match")
        return 0.0
    
    var sum_squared: Float32 = 0.0
    
    for i in range(dim):
        var diff = vec1[i] - vec2[i]
        sum_squared += diff * diff
    
    from math import sqrt
    return sqrt(sum_squared)


fn manhattan_distance(vec1: List[Float32], vec2: List[Float32]) -> Float32:
    """Compute Manhattan (L1) distance between two vectors.
    
    Args:
        vec1: First vector.
        vec2: Second vector.
        
    Returns:
        L1 distance value.
    """
    var dim = len(vec1)
    if dim != len(vec2):
        print("Error: Vector dimensions don't match")
        return 0.0
    
    var sum_abs: Float32 = 0.0
    
    for i in range(dim):
        var diff = vec1[i] - vec2[i]
        var abs_diff = diff if diff >= 0 else -diff
        sum_abs += abs_diff
    
    return sum_abs


fn dot_product(vec1: List[Float32], vec2: List[Float32]) -> Float32:
    """Compute dot product of two vectors.
    
    Args:
        vec1: First vector.
        vec2: Second vector.
        
    Returns:
        Dot product value.
    """
    var dim = len(vec1)
    if dim != len(vec2):
        print("Error: Vector dimensions don't match")
        return 0.0
    
    var result: Float32 = 0.0
    
    for i in range(dim):
        result += vec1[i] * vec2[i]
    
    return result


fn vector_norm(vec: List[Float32]) -> Float32:
    """Compute L2 norm of a vector.
    
    Args:
        vec: Input vector.
        
    Returns:
        L2 norm value.
    """
    var sum_squared: Float32 = 0.0
    
    for i in range(len(vec)):
        var v = vec[i]
        sum_squared += v * v
    
    from math import sqrt
    return sqrt(sum_squared)


fn normalize_vector(var vec: List[Float32]) -> List[Float32]:
    """Normalize a vector to unit length.
    
    Args:
        vec: Input vector.
        
    Returns:
        Normalized vector.
    """
    var norm = vector_norm(vec)
    
    if norm < 1e-10:
        return vec
    
    var inv_norm = 1.0 / norm
    var result = List[Float32]()
    
    for i in range(len(vec)):
        result.append(vec[i] * inv_norm)
    
    return result^


struct VectorOps:
    """Collection of vector operations for embedding processing."""
    
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
    """Test vector operations."""
    print("=" * 70)
    print("Vectro Vector Operations Module")
    print("=" * 70)
    
    # Create test vectors
    var vec1 = List[Float32]()
    vec1.append(1.0)
    vec1.append(2.0)
    vec1.append(3.0)
    
    var vec2 = List[Float32]()
    vec2.append(4.0)
    vec2.append(5.0)
    vec2.append(6.0)
    
    print("\nTesting vector operations:")
    print("  vec1 = [1.0, 2.0, 3.0]")
    print("  vec2 = [4.0, 5.0, 6.0]")
    
    print("\nResults:")
    print("  Cosine similarity:", cosine_similarity(vec1, vec2))
    print("  Euclidean distance:", euclidean_distance(vec1, vec2))
    print("  Manhattan distance:", manhattan_distance(vec1, vec2))
    print("  Dot product:", dot_product(vec1, vec2))
    print("  vec1 norm:", vector_norm(vec1))
    print("  vec2 norm:", vector_norm(vec2))
    
    print("\n" + "=" * 70)
    print("Available functions:")
    print("  - cosine_similarity()")
    print("  - euclidean_distance()")
    print("  - manhattan_distance()")
    print("  - dot_product()")
    print("  - vector_norm()")
    print("  - normalize_vector()")
    print("=" * 70)

