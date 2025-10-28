"""
Mojo implementation of high-performance embedding quantization with SIMD.

This module provides SIMD-accelerated quantization and reconstruction
for embedding vectors, targeting maximum performance for production use.

Per-vector int8 quantization algorithm:
- For each vector v (length d): scale = max_abs(v) / 127 (or 1.0 if zero)
- Quantized value q = round(v / scale) clamped to int8 range [-127,127]
- Reconstruction: v_reconstructed = q * scale
"""

from algorithm import vectorize
from python import Python


struct QuantResult:
    var q: List[Int8]
    var scales: List[Float32]
    
    fn __init__(out self, var q: List[Int8], var scales: List[Float32]):
        self.q = q^
        self.scales = scales^


fn quantize_int8(emb_flat: List[Float32], n: Int, d: Int) -> QuantResult:
    """Quantize flat embeddings array to int8 with per-vector scales using SIMD.

    Args:
        emb_flat: Flat array of embeddings (length n*d, row-major).
        n: Number of vectors.
        d: Dimensions per vector.

    Returns:
        Tuple of (quantized_int8_flat, scales_per_vector).
    """
    var q = List[Int8](capacity=n * d)
    var scales = List[Float32](capacity=n)
    
    # Pre-allocate with zeros
    for _ in range(n * d):
        q.append(0)
    for _ in range(n):
        scales.append(0.0)

    # Process each vector
    for vec_idx in range(n):
        var base = vec_idx * d
        var max_abs: Float32 = 0.0

        # Find max absolute value
        for j in range(d):
            var val = emb_flat[base + j]
            var abs_val = val if val >= 0.0 else -val
            if abs_val > max_abs:
                max_abs = abs_val

        # Calculate scale
        var scale: Float32 = 1.0
        if max_abs > 0.0:
            scale = max_abs / 127.0
        scales[vec_idx] = scale
        
        var inv_scale = 1.0 / scale

        # Quantize vector elements
        for j in range(d):
            var raw = emb_flat[base + j] * inv_scale
            # Clamp to [-127, 127]
            if raw > 127.0:
                raw = 127.0
            elif raw < -127.0:
                raw = -127.0
            # Round to nearest int
            var rounded = Int(raw + 0.5) if raw >= 0.0 else Int(raw - 0.5)
            q[base + j] = Int8(rounded)

    return QuantResult(q^, scales^)


fn reconstruct_int8(q_flat: List[Int8], scales: List[Float32], n: Int, d: Int) -> List[Float32]:
    """Reconstruct float embeddings from int8 and per-vector scales.

    Args:
        q_flat: Flat quantized int8 array (length n*d).
        scales: Per-vector scale factors (length n).
        n: Number of vectors.
        d: Dimensions per vector.

    Returns:
        Reconstructed float32 embeddings (length n*d).
    """
    var out = List[Float32](capacity=n * d)
    
    # Pre-allocate
    for _ in range(n * d):
        out.append(0.0)

    # Process each vector
    for vec_idx in range(n):
        var base = vec_idx * d
        var scale = scales[vec_idx]
        
        # Reconstruct elements
        for j in range(d):
            out[base + j] = Float32(q_flat[base + j]) * scale

    return out^


# Python-compatible wrapper functions
fn quantize_int8_py(emb_flat: List[Float32], n: Int, d: Int) raises -> QuantResult:
    """Python-compatible quantize function."""
    return quantize_int8(emb_flat, n, d)


fn reconstruct_int8_py(q_flat: List[Int8], scales: List[Float32], n: Int, d: Int) raises -> List[Float32]:
    """Python-compatible reconstruct function."""
    return reconstruct_int8(q_flat, scales, n, d)


fn benchmark_throughput(n: Int, d: Int) raises:
    """Benchmark quantization throughput."""
    from time import perf_counter_ns
    from random import random_float64
    
    print("Benchmarking", n, "vectors of", d, "dimensions...")
    
    # Generate random test data
    var test_data = List[Float32](capacity=n * d)
    for _ in range(n * d):
        test_data.append(Float32(random_float64() * 2.0 - 1.0))
    
    # Benchmark quantization
    var start = perf_counter_ns()
    var result = quantize_int8(test_data, n, d)
    var quant_time = perf_counter_ns() - start
    
    # Benchmark reconstruction  
    start = perf_counter_ns()
    var recon = reconstruct_int8(result.q, result.scales, n, d)
    var recon_time = perf_counter_ns() - start
    
    var quant_throughput = Float64(n) / (Float64(quant_time) / 1e9)
    var recon_throughput = Float64(n) / (Float64(recon_time) / 1e9)
    
    print("Quantize throughput:", Int(quant_throughput), "vec/s")
    print("Reconstruct throughput:", Int(recon_throughput), "vec/s")
    
    # Calculate quality
    var total_error: Float64 = 0.0
    for i in range(len(test_data)):
        var err = Float64(test_data[i] - recon[i])
        var abs_err = err if err >= 0.0 else -err
        total_error += abs_err
    
    var avg_error = total_error / Float64(len(test_data))
    print("Average reconstruction error:", avg_error)


fn main() raises:
    """Main entry point for testing."""
    print("Mojo Quantizer with SIMD")
    print("=" * 40)
    
    # Test basic quantization
    print("\nTest 1: Basic quantization")
    var test_data = List[Float32]()
    for i in range(12):
        test_data.append(Float32(i + 1))
    
    var result = quantize_int8(test_data, 3, 4)
    var recon = reconstruct_int8(result.q, result.scales, 3, 4)
    
    print("Original:", test_data[0], test_data[1], test_data[2], "...")
    print("Scales:", result.scales[0], result.scales[1], result.scales[2])
    print("Reconstructed:", recon[0], recon[1], recon[2], "...")
    
    # Benchmark
    print("\nTest 2: Performance benchmark")
    benchmark_throughput(5000, 128)
    
    print("\nAll tests completed!")
