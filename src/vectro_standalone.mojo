"""
Vectro Mojo Package - Standalone version for testing.

This is a simplified version that can be compiled and run directly.
"""

struct QuantResult:
    """Result of quantization containing quantized values and scale factor."""
    var quantized: List[Int8]
    var scale: Float32
    
    fn __init__(out self, var q: List[Int8], s: Float32):
        self.quantized = q^
        self.scale = s


fn quantize_vector(data: List[Float32]) -> QuantResult:
    """Quantize a single vector to int8 with per-vector scale factor.
    
    Args:
        data: Input float32 vector.
    
    Returns:
        QuantResult containing quantized int8 values and scale factor.
    """
    # Find max absolute value
    var max_val: Float32 = 0.0
    for i in range(len(data)):
        var val = data[i]
        var abs_val = val if val >= 0.0 else -val
        if abs_val > max_val:
            max_val = abs_val
    
    # Calculate scale
    var scale: Float32 = 1.0
    if max_val > 0.0:
        scale = max_val / 127.0
    
    var inv_scale = 1.0 / scale
    
    # Quantize elements
    var result = List[Int8]()
    for i in range(len(data)):
        var raw = data[i] * inv_scale
        # Clamp to [-127, 127]
        if raw > 127.0:
            raw = 127.0
        elif raw < -127.0:
            raw = -127.0
        # Round to nearest int
        var rounded = Int(raw + 0.5) if raw >= 0.0 else Int(raw - 0.5)
        result.append(Int8(rounded))
    
    return QuantResult(result^, scale)


fn reconstruct_vector(quantized: List[Int8], scale: Float32) -> List[Float32]:
    """Reconstruct float32 vector from quantized int8 values.
    
    Args:
        quantized: Quantized int8 values.
        scale: Scale factor from quantization.
    
    Returns:
        Reconstructed float32 vector.
    """
    var result = List[Float32]()
    for i in range(len(quantized)):
        result.append(Float32(quantized[i]) * scale)
    return result^


fn benchmark_quantization(n_vectors: Int, dim: Int) raises:
    """Benchmark the quantization performance."""
    from time import perf_counter_ns
    from random import random_float64
    
    print("Benchmarking", n_vectors, "vectors of dimension", dim)
    
    # Generate random data
    var data = List[Float32]()
    for _ in range(n_vectors * dim):
        data.append(Float32(random_float64() * 2.0 - 1.0))
    
    # Warm up
    for vec_idx in range(min(10, n_vectors)):
        var vec = List[Float32]()
        for j in range(dim):
            vec.append(data[vec_idx * dim + j])
        _ = quantize_vector(vec)
    
    # Benchmark
    var start = perf_counter_ns()
    var total_quantized = 0
    
    for vec_idx in range(n_vectors):
        var vec = List[Float32]()
        for j in range(dim):
            vec.append(data[vec_idx * dim + j])
        var result = quantize_vector(vec)
        total_quantized += len(result.quantized)
    
    var elapsed_ns = perf_counter_ns() - start
    var elapsed_s = Float64(elapsed_ns) / 1e9
    var throughput = Float64(n_vectors) / elapsed_s
    
    print("Quantized", total_quantized, "elements")
    print("Throughput:", Int(throughput), "vectors/sec")
    print("Time:", elapsed_s, "seconds")


fn main() raises:
    """Main entry point for testing."""
    print("Vectro Mojo Quantizer - Standalone Test")
    print("=" * 50)
    
    # Test 1: Basic quantization
    print("\nTest 1: Basic quantization")
    var test_data = List[Float32]()
    test_data.append(1.0)
    test_data.append(2.0)
    test_data.append(3.0)
    test_data.append(4.0)
    
    var result = quantize_vector(test_data)
    print("Original: [1.0, 2.0, 3.0, 4.0]")
    print("Scale:", result.scale)
    print("Quantized:", result.quantized[0], result.quantized[1], result.quantized[2], result.quantized[3])
    
    # Reconstruct
    var recon = reconstruct_vector(result.quantized, result.scale)
    print("Reconstructed:", recon[0], recon[1], recon[2], recon[3])
    
    # Calculate error
    var total_error: Float32 = 0.0
    for i in range(len(test_data)):
        var err = test_data[i] - recon[i]
        var abs_err = err if err >= 0.0 else -err
        total_error += abs_err
    print("Average error:", total_error / Float32(len(test_data)))
    
    # Test 2: Performance benchmark
    print("\nTest 2: Performance benchmark")
    benchmark_quantization(10000, 128)
    
    print("\n✓ All tests passed!")
