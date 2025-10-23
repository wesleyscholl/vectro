"""Test the Mojo quantizer implementation."""

from quantizer import Quantizer, quantize_int8_py, reconstruct_int8_py
from time import now
from math import cos
from random import random_float64


fn test_basic_quantization():
    """Test basic quantization and reconstruction."""
    print("Testing basic quantization...")

    # Create test data
    var test_data = List[Float32]()
    for i in range(12):  # 3 vectors of 4 dims each
        test_data.append(Float32(i + 1))

    var n = 3
    var d = 4

    # Quantize
    var (q_flat, scales) = Quantizer.quantize_int8(test_data, n, d)

    print("Original:", test_data)
    print("Quantized:", q_flat)
    print("Scales:", scales)

    # Reconstruct
    var reconstructed = Quantizer.reconstruct_int8(q_flat, scales, n, d)
    print("Reconstructed:", reconstructed)

    # Check quality
    var total_error = 0.0
    for i in range(len(test_data)):
        var error = abs(test_data[i] - reconstructed[i])
        total_error += error

    var avg_error = total_error / Float32(len(test_data))
    print("Average reconstruction error:", avg_error)


fn benchmark_quantization():
    """Benchmark quantization performance."""
    print("\nBenchmarking quantization performance...")

    var n = 1000  # number of vectors
    var d = 128   # dimensions per vector

    # Generate random test data
    var test_data = List[Float32](capacity=n * d)
    for i in range(n * d):
        test_data.append(Float32(random_float64() * 2.0 - 1.0))  # Random in [-1, 1]

    print("Benchmarking", n, "vectors of", d, "dimensions each...")

    # Time quantization
    var start_time = now()
    var (q_flat, scales) = Quantizer.quantize_int8(test_data, n, d)
    var quantize_time = now() - start_time

    # Time reconstruction
    start_time = now()
    var reconstructed = Quantizer.reconstruct_int8(q_flat, scales, n, d)
    var reconstruct_time = now() - start_time

    var quantize_throughput = Float64(n) / (Float64(quantize_time) / 1e9)  # vectors per second
    var reconstruct_throughput = Float64(n) / (Float64(reconstruct_time) / 1e9)

    print("Quantize throughput:", quantize_throughput, "vectors/second")
    print("Reconstruct throughput:", reconstruct_throughput, "vectors/second")

    # Calculate quality
    var total_cos_sim = 0.0
    var count = 0
    for i in range(n):
        var start_idx = i * d
        var orig_sum_sq = 0.0
        var recon_sum_sq = 0.0
        var dot_product = 0.0

        for j in range(d):
            var orig = test_data[start_idx + j]
            var recon = reconstructed[start_idx + j]
            orig_sum_sq += orig * orig
            recon_sum_sq += recon * recon
            dot_product += orig * recon

        if orig_sum_sq > 0 and recon_sum_sq > 0:
            var cos_sim = dot_product / (orig_sum_sq ** 0.5 * recon_sum_sq ** 0.5)
            total_cos_sim += cos_sim
            count += 1

    var avg_cos_sim = total_cos_sim / Float64(count)
    print("Average cosine similarity:", avg_cos_sim)


fn main():
    """Main test function."""
    print("Mojo Quantizer Tests")
    print("=" * 30)

    test_basic_quantization()
    benchmark_quantization()

    print("\nAll tests completed!")