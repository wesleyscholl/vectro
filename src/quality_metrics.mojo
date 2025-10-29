"""
Advanced quality metrics and validation for quantized embeddings.
Mojo implementation for error analysis and quality assessment.
"""
from math import sqrt


struct QualityMetrics:
    """Comprehensive quality metrics for quantization."""
    var mean_absolute_error: Float32
    var max_absolute_error: Float32
    var mean_squared_error: Float32
    var root_mean_squared_error: Float32
    var mean_cosine_similarity: Float32
    var min_cosine_similarity: Float32
    var num_vectors: Int
    var vector_dim: Int
    var error_percentiles: List[Float32]  # 25th, 50th, 75th, 95th, 99th
    
    fn __init__(
        out self,
        mae: Float32,
        max_err: Float32,
        mse: Float32,
        rmse: Float32,
        mean_cos: Float32,
        min_cos: Float32,
        n: Int,
        d: Int,
        var percentiles: List[Float32]
    ):
        """Initialize quality metrics.
        Args:
            mae: Mean absolute error.
            max_err: Maximum absolute error.
            mse: Mean squared error.
            rmse: Root mean squared error.
            mean_cos: Mean cosine similarity.
            min_cos: Minimum cosine similarity.
            n: Number of vectors analyzed.
            d: Vector dimension.
            percentiles: Error percentile values.
        """
        self.mean_absolute_error = mae
        self.max_absolute_error = max_err
        self.mean_squared_error = mse
        self.root_mean_squared_error = rmse
        self.mean_cosine_similarity = mean_cos
        self.min_cosine_similarity = min_cos
        self.num_vectors = n
        self.vector_dim = d
        self.error_percentiles = percentiles^
    
    fn print_metrics(self):
        """Print formatted quality metrics."""
        print("=" * 70)
        print("Quality Metrics Report")
        print("=" * 70)
        print("Dataset:")
        print("  Vectors:", self.num_vectors)
        print("  Dimensions:", self.vector_dim)
        print()
        print("Error Metrics:")
        print("  Mean Absolute Error:", self.mean_absolute_error)
        print("  Max Absolute Error:", self.max_absolute_error)
        print("  Mean Squared Error:", self.mean_squared_error)
        print("  Root Mean Squared Error:", self.root_mean_squared_error)
        print()
        print("Similarity Metrics:")
        print("  Mean Cosine Similarity:", self.mean_cosine_similarity)
        print("  Min Cosine Similarity:", self.min_cosine_similarity)
        print("  Similarity %:", self.mean_cosine_similarity * 100.0, "%")
        print()
        print("Error Distribution (Percentiles):")
        if len(self.error_percentiles) >= 5:
            print("  25th:", self.error_percentiles[0])
            print("  50th (Median):", self.error_percentiles[1])
            print("  75th:", self.error_percentiles[2])
            print("  95th:", self.error_percentiles[3])
            print("  99th:", self.error_percentiles[4])
        print("=" * 70)
    
    fn is_acceptable(self, mae_threshold: Float32 = 0.01, cos_threshold: Float32 = 0.99) -> Bool:
        """Check if quality metrics meet acceptable thresholds.
        Args:
            mae_threshold: Maximum acceptable MAE.
            cos_threshold: Minimum acceptable cosine similarity.
        Returns:
            True if metrics are acceptable.
        """
        return (self.mean_absolute_error <= mae_threshold and 
                self.mean_cosine_similarity >= cos_threshold)


fn compute_vector_error(original: List[Float32], reconstructed: List[Float32]) -> Float32:
    """Compute mean absolute error between two vectors.
    Args:
        original: Original vector.
        reconstructed: Reconstructed vector.
    Returns:
        Mean absolute error.
    """
    var dim = len(original)
    var total_error: Float32 = 0.0
    
    for i in range(dim):
        var diff = original[i] - reconstructed[i]
        var abs_diff = diff if diff >= 0 else -diff
        total_error += abs_diff
    
    return total_error / Float32(dim)


fn compute_cosine_similarity_quality(original: List[Float32], reconstructed: List[Float32]) -> Float32:
    """Compute cosine similarity between original and reconstructed vector.
    Args:
        original: Original vector.
        reconstructed: Reconstructed vector.
    Returns:
        Cosine similarity (0 to 1).
    """
    var dim = len(original)
    var dot_product: Float32 = 0.0
    var norm_orig: Float32 = 0.0
    var norm_recon: Float32 = 0.0
    
    for i in range(dim):
        dot_product += original[i] * reconstructed[i]
        norm_orig += original[i] * original[i]
        norm_recon += reconstructed[i] * reconstructed[i]
    
    var denom = sqrt(norm_orig) * sqrt(norm_recon)
    if denom < 1e-10:
        return 0.0
    
    return dot_product / denom


fn sort_list(var lst: List[Float32]) -> List[Float32]:
    """Simple bubble sort for small lists (for percentile calculation).
    Args:
        lst: List to sort.
    Returns:
        Sorted list.
    """
    var n = len(lst)
    for i in range(n):
        for j in range(0, n - i - 1):
            if lst[j] > lst[j + 1]:
                var temp = lst[j]
                lst[j] = lst[j + 1]
                lst[j + 1] = temp
    return lst^


fn calculate_percentiles(var errors: List[Float32]) -> List[Float32]:
    """Calculate error percentiles.
    Args:
        errors: List of error values.
    Returns:
        List containing 25th, 50th, 75th, 95th, 99th percentiles.
    """
    var sorted_errors = sort_list(errors^)
    var n = len(sorted_errors)
    
    var percentiles = List[Float32]()
    var indices = List[Int]()
    indices.append(Int(Float32(n) * 0.25))  # 25th
    indices.append(Int(Float32(n) * 0.50))  # 50th
    indices.append(Int(Float32(n) * 0.75))  # 75th
    indices.append(Int(Float32(n) * 0.95))  # 95th
    indices.append(Int(Float32(n) * 0.99))  # 99th
    
    for i in range(5):
        var idx = indices[i]
        if idx >= n:
            idx = n - 1
        percentiles.append(sorted_errors[idx])
    
    return percentiles^


fn evaluate_quality(
    originals: List[List[Float32]],
    reconstructed: List[List[Float32]]
) -> QualityMetrics:
    """Evaluate comprehensive quality metrics.
    Args:
        originals: List of original vectors.
        reconstructed: List of reconstructed vectors.
    Returns:
        QualityMetrics with comprehensive analysis.
    """
    var num_vectors = len(originals)
    if num_vectors == 0:
        var empty_percentiles = List[Float32]()
        return QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, empty_percentiles^)
    
    var vector_dim = len(originals[0])
    
    # Accumulate errors
    var total_mae: Float32 = 0.0
    var max_error: Float32 = 0.0
    var total_mse: Float32 = 0.0
    var total_cos_sim: Float32 = 0.0
    var min_cos_sim: Float32 = 1.0
    var all_errors = List[Float32]()
    
    print("Evaluating quality for", num_vectors, "vectors...")
    
    for i in range(num_vectors):
        var orig = originals[i].copy()
        var recon = reconstructed[i].copy()
        
        # Calculate errors for this vector
        var vec_mae = compute_vector_error(orig, recon)
        total_mae += vec_mae
        all_errors.append(vec_mae)
        
        if vec_mae > max_error:
            max_error = vec_mae
        
        # Calculate MSE
        var vec_mse: Float32 = 0.0
        for j in range(vector_dim):
            var diff = orig[j] - recon[j]
            vec_mse += diff * diff
        vec_mse /= Float32(vector_dim)
        total_mse += vec_mse
        
        # Calculate cosine similarity
        var cos_sim = compute_cosine_similarity_quality(orig, recon)
        total_cos_sim += cos_sim
        if cos_sim < min_cos_sim:
            min_cos_sim = cos_sim
    
    # Calculate averages
    var mean_mae = total_mae / Float32(num_vectors)
    var mean_mse = total_mse / Float32(num_vectors)
    var rmse = sqrt(mean_mse)
    var mean_cos = total_cos_sim / Float32(num_vectors)
    
    # Calculate percentiles
    var percentiles = calculate_percentiles(all_errors^)
    
    return QualityMetrics(
        mean_mae,
        max_error,
        mean_mse,
        rmse,
        mean_cos,
        min_cos_sim,
        num_vectors,
        vector_dim,
        percentiles^
    )


struct ValidationResult:
    """Result of quality validation."""
    var passed: Bool
    var message: String
    var mean_mae: Float32
    var mean_cos: Float32
    
    fn __init__(out self, p: Bool, msg: String, mae: Float32, cos: Float32):
        """Initialize validation result.
        Args:
            p: Whether validation passed.
            msg: Validation message.
            mae: Mean absolute error.
            cos: Mean cosine similarity.
        """
        self.passed = p
        self.message = msg
        self.mean_mae = mae
        self.mean_cos = cos
    
    fn print_result(self):
        """Print validation result."""
        print("\n" + "=" * 70)
        if self.passed:
            print("✓ VALIDATION PASSED")
        else:
            print("✗ VALIDATION FAILED")
        print("=" * 70)
        print("Message:", self.message)
        print("  Mean MAE:", self.mean_mae)
        print("  Mean Cosine:", self.mean_cos)


fn validate_quantization_quality(
    originals: List[List[Float32]],
    reconstructed: List[List[Float32]],
    max_mae: Float32 = 0.01,
    min_cosine: Float32 = 0.99
) -> ValidationResult:
    """Validate quantization quality against thresholds.
    Args:
        originals: Original vectors.
        reconstructed: Reconstructed vectors.
        max_mae: Maximum acceptable MAE.
        min_cosine: Minimum acceptable cosine similarity.
    Returns:
        ValidationResult with pass/fail and metrics.
    """
    print("\nValidating quantization quality...")
    print("  Thresholds:")
    print("    Max MAE:", max_mae)
    print("    Min Cosine:", min_cosine)
    
    var metrics = evaluate_quality(originals, reconstructed)
    
    var passed = metrics.is_acceptable(max_mae, min_cosine)
    var message: String
    
    if passed:
        message = "Quality metrics meet acceptable thresholds"
    else:
        message = "Quality metrics do not meet acceptable thresholds"
    
    return ValidationResult(passed, message, metrics.mean_absolute_error, metrics.mean_cosine_similarity)


fn main():
    """Test quality metrics."""
    print("=" * 70)
    print("Vectro Quality Metrics Module (Mojo)")
    print("=" * 70)
    
    # Create sample data
    var originals = List[List[Float32]]()
    var reconstructed = List[List[Float32]]()
    
    print("\nGenerating test data (10 vectors, 8 dimensions)...")
    for i in range(10):
        var orig = List[Float32]()
        var recon = List[Float32]()
        for j in range(8):
            var val = Float32(i * 8 + j) * 0.1
            orig.append(val)
            # Add small reconstruction error
            recon.append(val * 0.998 + 0.001)
        originals.append(orig^)
        reconstructed.append(recon^)
    
    # Evaluate quality
    print("\nEvaluating quality metrics...")
    var metrics = evaluate_quality(originals, reconstructed)
    metrics.print_metrics()
    
    # Validate
    print("\nRunning validation...")
    var validation = validate_quantization_quality(originals, reconstructed, 0.05, 0.95)
    validation.print_result()
    
    print("\n" + "=" * 70)
    print("Quality metrics module ready!")
    print("=" * 70)
