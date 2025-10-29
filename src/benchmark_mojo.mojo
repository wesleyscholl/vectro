"""
Comprehensive benchmarking suite for quantization performance.
Mojo implementation for high-precision timing and throughput measurement.
"""
from time import now


struct BenchmarkResult:
    """Results from a benchmark run."""
    var name: String
    var iterations: Int
    var total_vectors: Int
    var elapsed_seconds: Float64
    var throughput_vec_per_sec: Float64
    var avg_time_per_vector_us: Float64
    
    fn __init__(
        out self,
        name: String,
        iters: Int,
        vecs: Int,
        elapsed: Float64
    ):
        """Initialize benchmark result.
        Args:
            name: Benchmark name.
            iters: Number of iterations.
            vecs: Total vectors processed.
            elapsed: Elapsed time in seconds.
        """
        self.name = name
        self.iterations = iters
        self.total_vectors = vecs
        self.elapsed_seconds = elapsed
        self.throughput_vec_per_sec = Float64(vecs) / elapsed
        self.avg_time_per_vector_us = (elapsed / Float64(vecs)) * 1_000_000.0
    
    fn print_result(self):
        """Print formatted benchmark result."""
        print("\n" + "=" * 70)
        print("Benchmark:", self.name)
        print("=" * 70)
        print("Iterations:", self.iterations)
        print("Total vectors:", self.total_vectors)
        print("Elapsed time:", self.elapsed_seconds, "seconds")
        print("Throughput:", Int(self.throughput_vec_per_sec), "vectors/sec")
        print("Avg time per vector:", self.avg_time_per_vector_us, "microseconds")
        print("=" * 70)


struct BenchmarkSuite:
    """Suite of benchmarks for performance testing."""
    var results: List[BenchmarkResult]
    var suite_name: String
    
    fn __init__(out self, name: String):
        """Initialize benchmark suite.
        Args:
            name: Suite name.
        """
        self.results = List[BenchmarkResult]()
        self.suite_name = name
    
    fn add_result(inout self, var result: BenchmarkResult):
        """Add a benchmark result to the suite.
        Args:
            result: BenchmarkResult to add.
        """
        self.results.append(result^)
    
    fn print_summary(self):
        """Print summary of all benchmark results."""
        print("\n" + "=" * 70)
        print("Benchmark Suite:", self.suite_name)
        print("=" * 70)
        print("Total benchmarks:", len(self.results))
        print()
        
        if len(self.results) == 0:
            print("No results yet")
            return
        
        # Print each result
        for i in range(len(self.results)):
            var r = self.results[i]
            print(str(i+1) + ".", r.name)
            print("   Throughput:", Int(r.throughput_vec_per_sec), "vec/s")
            print("   Time:", r.elapsed_seconds, "sec")
        
        # Find best performance
        var best_idx = 0
        var best_throughput = self.results[0].throughput_vec_per_sec
        for i in range(1, len(self.results)):
            if self.results[i].throughput_vec_per_sec > best_throughput:
                best_throughput = self.results[i].throughput_vec_per_sec
                best_idx = i
        
        print("\nBest Performance:", self.results[best_idx].name)
        print("  Throughput:", Int(best_throughput), "vectors/sec")
        print("=" * 70)


fn benchmark_quantization_simple(
    vector_dim: Int,
    num_vectors: Int,
    iterations: Int
) -> BenchmarkResult:
    """Benchmark simple quantization performance.
    Args:
        vector_dim: Dimension of vectors.
        num_vectors: Number of vectors per iteration.
        iterations: Number of iterations to run.
    Returns:
        BenchmarkResult with timing data.
    """
    print("\nRunning quantization benchmark...")
    print("  Vector dim:", vector_dim)
    print("  Num vectors:", num_vectors)
    print("  Iterations:", iterations)
    
    # Create test data
    var test_data = List[List[Float32]]()
    for i in range(num_vectors):
        var vec = List[Float32]()
        for j in range(vector_dim):
            vec.append(Float32(i * vector_dim + j) * 0.001)
        test_data.append(vec^)
    
    # Warm-up
    for i in range(10):
        for v in range(num_vectors):
            var vec = test_data[v]
            var max_val: Float32 = 0.0
            for j in range(vector_dim):
                if vec[j] > max_val:
                    max_val = vec[j]
    
    # Actual benchmark
    var start = now()
    
    for iter in range(iterations):
        for v in range(num_vectors):
            var vec = test_data[v]
            
            # Find max
            var max_val: Float32 = 0.0
            for j in range(vector_dim):
                if vec[j] > max_val:
                    max_val = vec[j]
            
            # Quantize
            var scale = max_val / 127.0
            var inv_scale = 1.0 / scale
            
            var quantized = List[Int8]()
            for j in range(vector_dim):
                var val = vec[j] * inv_scale
                var quant = Int8(val + 0.5)
                quantized.append(quant)
    
    var end = now()
    var elapsed = Float64(end - start) / 1_000_000_000.0
    var total_vecs = num_vectors * iterations
    
    return BenchmarkResult("Simple Quantization", iterations, total_vecs, elapsed)


fn benchmark_reconstruction_simple(
    vector_dim: Int,
    num_vectors: Int,
    iterations: Int
) -> BenchmarkResult:
    """Benchmark simple reconstruction performance.
    Args:
        vector_dim: Dimension of vectors.
        num_vectors: Number of vectors per iteration.
        iterations: Number of iterations to run.
    Returns:
        BenchmarkResult with timing data.
    """
    print("\nRunning reconstruction benchmark...")
    print("  Vector dim:", vector_dim)
    print("  Num vectors:", num_vectors)
    print("  Iterations:", iterations)
    
    # Create test quantized data
    var test_quantized = List[List[Int8]]()
    var test_scales = List[Float32]()
    
    for i in range(num_vectors):
        var vec = List[Int8]()
        for j in range(vector_dim):
            vec.append(Int8((i * vector_dim + j) % 127))
        test_quantized.append(vec^)
        test_scales.append(Float32(i) * 0.01 + 0.1)
    
    # Warm-up
    for i in range(10):
        for v in range(num_vectors):
            var qvec = test_quantized[v]
            var scale = test_scales[v]
            for j in range(vector_dim):
                var val = Float32(qvec[j]) * scale
    
    # Actual benchmark
    var start = now()
    
    for iter in range(iterations):
        for v in range(num_vectors):
            var qvec = test_quantized[v]
            var scale = test_scales[v]
            
            var reconstructed = List[Float32]()
            for j in range(vector_dim):
                reconstructed.append(Float32(qvec[j]) * scale)
    
    var end = now()
    var elapsed = Float64(end - start) / 1_000_000_000.0
    var total_vecs = num_vectors * iterations
    
    return BenchmarkResult("Simple Reconstruction", iterations, total_vecs, elapsed)


fn benchmark_end_to_end(
    vector_dim: Int,
    num_vectors: Int,
    iterations: Int
) -> BenchmarkResult:
    """Benchmark complete quantization + reconstruction cycle.
    Args:
        vector_dim: Dimension of vectors.
        num_vectors: Number of vectors per iteration.
        iterations: Number of iterations to run.
    Returns:
        BenchmarkResult with timing data.
    """
    print("\nRunning end-to-end benchmark...")
    print("  Vector dim:", vector_dim)
    print("  Num vectors:", num_vectors)
    print("  Iterations:", iterations)
    
    # Create test data
    var test_data = List[List[Float32]]()
    for i in range(num_vectors):
        var vec = List[Float32]()
        for j in range(vector_dim):
            vec.append(Float32(i * vector_dim + j) * 0.001)
        test_data.append(vec^)
    
    var start = now()
    
    for iter in range(iterations):
        for v in range(num_vectors):
            var vec = test_data[v]
            
            # Quantize
            var max_val: Float32 = 0.0
            for j in range(vector_dim):
                if vec[j] > max_val:
                    max_val = vec[j]
            
            var scale = max_val / 127.0
            var inv_scale = 1.0 / scale
            
            var quantized = List[Int8]()
            for j in range(vector_dim):
                var val = vec[j] * inv_scale
                quantized.append(Int8(val + 0.5))
            
            # Reconstruct
            var reconstructed = List[Float32]()
            for j in range(vector_dim):
                reconstructed.append(Float32(quantized[j]) * scale)
    
    var end = now()
    var elapsed = Float64(end - start) / 1_000_000_000.0
    var total_vecs = num_vectors * iterations
    
    return BenchmarkResult("End-to-End (Quant+Recon)", iterations, total_vecs, elapsed)


fn run_comprehensive_benchmarks() -> BenchmarkSuite:
    """Run comprehensive benchmark suite.
    Returns:
        BenchmarkSuite with all results.
    """
    var suite = BenchmarkSuite("Vectro Comprehensive Benchmarks")
    
    print("=" * 70)
    print("Starting Comprehensive Benchmark Suite")
    print("=" * 70)
    
    # Small vectors (128D) - typical embeddings
    print("\n[1/6] Small vectors (128D)...")
    var result1 = benchmark_quantization_simple(128, 100, 1000)
    suite.add_result(result1^)
    
    print("\n[2/6] Small vector reconstruction (128D)...")
    var result2 = benchmark_reconstruction_simple(128, 100, 1000)
    suite.add_result(result2^)
    
    # Medium vectors (768D) - BERT embeddings
    print("\n[3/6] Medium vectors (768D)...")
    var result3 = benchmark_quantization_simple(768, 100, 500)
    suite.add_result(result3^)
    
    print("\n[4/6] Medium vector reconstruction (768D)...")
    var result4 = benchmark_reconstruction_simple(768, 100, 500)
    suite.add_result(result4^)
    
    # Large vectors (1536D) - Large model embeddings
    print("\n[5/6] Large vectors (1536D)...")
    var result5 = benchmark_quantization_simple(1536, 50, 200)
    suite.add_result(result5^)
    
    # End-to-end benchmark
    print("\n[6/6] End-to-end benchmark (768D)...")
    var result6 = benchmark_end_to_end(768, 100, 300)
    suite.add_result(result6^)
    
    return suite^


fn main():
    """Run benchmark suite."""
    print("=" * 70)
    print("Vectro Benchmark Module (Mojo)")
    print("=" * 70)
    
    var suite = run_comprehensive_benchmarks()
    
    print("\n\n")
    suite.print_summary()
    
    print("\n" + "=" * 70)
    print("Benchmarking complete!")
    print("=" * 70)
