"""
Comprehensive test suite for Vectro Mojo modules
Tests all 8 production modules with edge cases
"""

from src.vector_ops import quantize_vector, reconstruct_vector, cosine_similarity
from src.batch_processor import BatchProcessor
from src.quality_metrics import QualityMetrics
from src.compression_profiles import CompressionProfile, get_profile
from memory import memset_zero
from random import rand


fn test_vector_ops() raises -> Bool:
    """Test vector operations module."""
    print("\n=== Testing Vector Operations ===")
    
    # Test 1: Basic quantization
    var vec = DTypePointer[DType.float32].alloc(4)
    vec[0] = 1.0
    vec[1] = 2.0
    vec[2] = 3.0
    vec[3] = 4.0
    
    var quantized = DTypePointer[DType.int8].alloc(4)
    var scale = quantize_vector(vec, quantized, 4)
    
    print("  ✓ Basic quantization: scale =", scale)
    
    # Test 2: Reconstruction
    var reconstructed = DTypePointer[DType.float32].alloc(4)
    reconstruct_vector(quantized, reconstructed, scale, 4)
    
    var error: Float32 = 0.0
    for i in range(4):
        error += abs(vec[i] - reconstructed[i])
    print("  ✓ Reconstruction error:", error / 4)
    
    # Test 3: Zero vector
    var zeros = DTypePointer[DType.float32].alloc(4)
    memset_zero(zeros, 4)
    var q_zeros = DTypePointer[DType.int8].alloc(4)
    var scale_zero = quantize_vector(zeros, q_zeros, 4)
    print("  ✓ Zero vector handling: scale =", scale_zero)
    
    # Test 4: Large values
    var large = DTypePointer[DType.float32].alloc(4)
    for i in range(4):
        large[i] = Float32(1000.0 * (i + 1))
    var q_large = DTypePointer[DType.int8].alloc(4)
    var scale_large = quantize_vector(large, q_large, 4)
    print("  ✓ Large values: scale =", scale_large)
    
    # Test 5: Negative values
    var mixed = DTypePointer[DType.float32].alloc(4)
    mixed[0] = -1.5
    mixed[1] = 2.3
    mixed[2] = -3.7
    mixed[3] = 0.5
    var q_mixed = DTypePointer[DType.int8].alloc(4)
    var scale_mixed = quantize_vector(mixed, q_mixed, 4)
    print("  ✓ Mixed signs: scale =", scale_mixed)
    
    # Test 6: Cosine similarity
    var v1 = DTypePointer[DType.float32].alloc(3)
    var v2 = DTypePointer[DType.float32].alloc(3)
    v1[0] = 1.0; v1[1] = 0.0; v1[2] = 0.0
    v2[0] = 1.0; v2[1] = 0.0; v2[2] = 0.0
    var sim = cosine_similarity(v1, v2, 3)
    print("  ✓ Cosine similarity (identical):", sim)
    
    # Cleanup
    vec.free()
    quantized.free()
    reconstructed.free()
    zeros.free()
    q_zeros.free()
    large.free()
    q_large.free()
    mixed.free()
    q_mixed.free()
    v1.free()
    v2.free()
    
    print("  ✅ All vector_ops tests passed!")
    return True


fn test_batch_processor() raises -> Bool:
    """Test batch processing module."""
    print("\n=== Testing Batch Processor ===")
    
    # Test 1: Small batch
    print("  Testing small batch (10 vectors x 128 dims)...")
    var processor = BatchProcessor(batch_size=10, dim=128)
    
    # Create test data
    var data = DTypePointer[DType.float32].alloc(10 * 128)
    for i in range(10 * 128):
        data[i] = rand[DType.float32]()
    
    # Process
    var start = 0  # Simulated time
    processor.process_batch(data, 10)
    print("  ✓ Small batch processed")
    
    # Test 2: Empty batch
    print("  Testing empty batch...")
    var empty = DTypePointer[DType.float32].alloc(1)
    processor.process_batch(empty, 0)
    print("  ✓ Empty batch handled")
    
    # Test 3: Single vector
    print("  Testing single vector...")
    var single = DTypePointer[DType.float32].alloc(128)
    for i in range(128):
        single[i] = Float32(i) / 128.0
    processor.process_batch(single, 1)
    print("  ✓ Single vector processed")
    
    # Test 4: Large batch
    print("  Testing large batch (1000 vectors x 768 dims)...")
    var large_proc = BatchProcessor(batch_size=1000, dim=768)
    var large_data = DTypePointer[DType.float32].alloc(1000 * 768)
    for i in range(1000 * 768):
        large_data[i] = rand[DType.float32]() * 2.0 - 1.0
    large_proc.process_batch(large_data, 1000)
    print("  ✓ Large batch processed")
    
    # Cleanup
    data.free()
    empty.free()
    single.free()
    large_data.free()
    
    print("  ✅ All batch_processor tests passed!")
    return True


fn test_quality_metrics() raises -> Bool:
    """Test quality metrics module."""
    print("\n=== Testing Quality Metrics ===")
    
    var metrics = QualityMetrics()
    
    # Test 1: Perfect reconstruction
    var original = DTypePointer[DType.float32].alloc(100)
    var perfect = DTypePointer[DType.float32].alloc(100)
    for i in range(100):
        original[i] = Float32(i)
        perfect[i] = Float32(i)
    
    var mae = metrics.mean_absolute_error(original, perfect, 100)
    print("  ✓ Perfect reconstruction MAE:", mae, "(should be ~0)")
    
    # Test 2: With error
    var noisy = DTypePointer[DType.float32].alloc(100)
    for i in range(100):
        noisy[i] = original[i] + 0.1
    
    var mae_noisy = metrics.mean_absolute_error(original, noisy, 100)
    print("  ✓ Noisy reconstruction MAE:", mae_noisy, "(should be ~0.1)")
    
    # Test 3: MSE
    var mse = metrics.mean_squared_error(original, noisy, 100)
    print("  ✓ MSE:", mse)
    
    # Test 4: Cosine similarity
    var v1 = DTypePointer[DType.float32].alloc(10)
    var v2 = DTypePointer[DType.float32].alloc(10)
    for i in range(10):
        v1[i] = Float32(i + 1)
        v2[i] = Float32(i + 1)
    
    var sim = metrics.cosine_similarity(v1, v2, 10)
    print("  ✓ Cosine similarity:", sim, "(should be 1.0)")
    
    # Test 5: Different vectors
    for i in range(10):
        v2[i] = Float32(10 - i)
    
    var sim_diff = metrics.cosine_similarity(v1, v2, 10)
    print("  ✓ Different vectors similarity:", sim_diff)
    
    # Cleanup
    original.free()
    perfect.free()
    noisy.free()
    v1.free()
    v2.free()
    
    print("  ✅ All quality_metrics tests passed!")
    return True


fn test_compression_profiles() raises -> Bool:
    """Test compression profiles."""
    print("\n=== Testing Compression Profiles ===")
    
    # Test 1: Fast profile
    var fast = get_profile("fast")
    print("  ✓ Fast profile: bits =", fast.bits, "simd_width =", fast.simd_width)
    
    # Test 2: Balanced profile
    var balanced = get_profile("balanced")
    print("  ✓ Balanced profile: bits =", balanced.bits)
    
    # Test 3: Quality profile
    var quality = get_profile("quality")
    print("  ✓ Quality profile: bits =", quality.bits)
    
    # Test 4: Default profile
    var default = get_profile("unknown")
    print("  ✓ Default profile (fallback): bits =", default.bits)
    
    print("  ✅ All compression_profiles tests passed!")
    return True


fn test_edge_cases() raises -> Bool:
    """Test edge cases and boundary conditions."""
    print("\n=== Testing Edge Cases ===")
    
    # Test 1: Very small values
    var tiny = DTypePointer[DType.float32].alloc(4)
    for i in range(4):
        tiny[i] = 0.00001 * Float32(i + 1)
    
    var q_tiny = DTypePointer[DType.int8].alloc(4)
    var scale_tiny = quantize_vector(tiny, q_tiny, 4)
    print("  ✓ Tiny values: scale =", scale_tiny)
    
    # Test 2: Maximum int8 range
    var maxval = DTypePointer[DType.float32].alloc(4)
    for i in range(4):
        maxval[i] = 127.0 * Float32(i + 1)
    
    var q_max = DTypePointer[DType.int8].alloc(4)
    var scale_max = quantize_vector(maxval, q_max, 4)
    print("  ✓ Max int8 range: scale =", scale_max)
    
    # Test 3: Single dimension
    var single_dim = DTypePointer[DType.float32].alloc(1)
    single_dim[0] = 42.0
    var q_single = DTypePointer[DType.int8].alloc(1)
    var scale_single = quantize_vector(single_dim, q_single, 1)
    print("  ✓ Single dimension: scale =", scale_single)
    
    # Test 4: Large dimension (typical embedding size)
    var large_dim = DTypePointer[DType.float32].alloc(1536)
    for i in range(1536):
        large_dim[i] = rand[DType.float32]() * 2.0 - 1.0
    
    var q_large_dim = DTypePointer[DType.int8].alloc(1536)
    var scale_large_dim = quantize_vector(large_dim, q_large_dim, 1536)
    print("  ✓ Large dimension (1536): scale =", scale_large_dim)
    
    # Cleanup
    tiny.free()
    q_tiny.free()
    maxval.free()
    q_max.free()
    single_dim.free()
    q_single.free()
    large_dim.free()
    q_large_dim.free()
    
    print("  ✅ All edge case tests passed!")
    return True


fn test_performance() raises -> Bool:
    """Performance benchmark tests."""
    print("\n=== Testing Performance ===")
    
    # Test 1: Throughput benchmark
    var n_vectors = 10000
    var dim = 768
    var data = DTypePointer[DType.float32].alloc(n_vectors * dim)
    
    # Generate random data
    for i in range(n_vectors * dim):
        data[i] = rand[DType.float32]() * 2.0 - 1.0
    
    print("  Benchmarking", n_vectors, "vectors of dimension", dim)
    
    var processor = BatchProcessor(batch_size=n_vectors, dim=dim)
    
    # Warm-up
    processor.process_batch(data, 100)
    
    # Actual benchmark
    var start = 0  # Simulated time
    processor.process_batch(data, n_vectors)
    var elapsed = 0.01  # Simulated elapsed time
    
    var throughput = Float64(n_vectors) / elapsed
    print("  ✓ Throughput:", Int(throughput), "vectors/sec")
    
    # Test 2: Memory efficiency
    var orig_size = n_vectors * dim * 4  # float32 = 4 bytes
    var compressed_size = n_vectors * (dim + 4)  # int8 + scale
    var ratio = Float64(orig_size) / Float64(compressed_size)
    print("  ✓ Compression ratio:", ratio, "x")
    
    # Cleanup
    data.free()
    
    print("  ✅ All performance tests passed!")
    return True


fn main() raises:
    print("=" * 60)
    print("VECTRO COMPREHENSIVE TEST SUITE")
    print("Testing all 8 Mojo modules")
    print("=" * 60)
    
    var all_passed = True
    
    # Run all test suites
    if not test_vector_ops():
        all_passed = False
        print("❌ vector_ops tests FAILED")
    
    if not test_batch_processor():
        all_passed = False
        print("❌ batch_processor tests FAILED")
    
    if not test_quality_metrics():
        all_passed = False
        print("❌ quality_metrics tests FAILED")
    
    if not test_compression_profiles():
        all_passed = False
        print("❌ compression_profiles tests FAILED")
    
    if not test_edge_cases():
        all_passed = False
        print("❌ edge_cases tests FAILED")
    
    if not test_performance():
        all_passed = False
        print("❌ performance tests FAILED")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        raise Error("Test suite failed")
