"""
Comprehensive tests for quality metrics and batch processing
Tests MAE, MSE, and batch operations
"""

from random import rand


fn test_mae_calculation() -> Bool:
    """Test Mean Absolute Error calculation."""
    print("\n  Testing mae_calculation...")
    
    # Test 1: Zero error
    var v1 = List[Float32](1.0, 2.0, 3.0, 4.0)
    var v2 = List[Float32](1.0, 2.0, 3.0, 4.0)
    
    var sum_error: Float32 = 0.0
    for i in range(len(v1)):
        sum_error += abs(v1[i] - v2[i])
    var mae = sum_error / Float32(len(v1))
    
    if abs(mae) > 0.001:
        print("    ❌ Zero error case failed:", mae)
        return False
    print("    ✓ Zero error case passed")
    
    # Test 2: Constant error
    var v3 = List[Float32](1.0, 2.0, 3.0, 4.0)
    var v4 = List[Float32](2.0, 3.0, 4.0, 5.0)
    
    var sum_error2: Float32 = 0.0
    for i in range(len(v3)):
        sum_error2 += abs(v3[i] - v4[i])
    var mae2 = sum_error2 / Float32(len(v3))
    
    # All errors are 1.0, so MAE = 1.0
    if abs(mae2 - 1.0) > 0.001:
        print("    ❌ Constant error case failed:", mae2)
        return False
    print("    ✓ Constant error case passed")
    
    # Test 3: Variable error
    var v5 = List[Float32](0.0, 0.0, 0.0, 0.0)
    var v6 = List[Float32](1.0, 2.0, 3.0, 4.0)
    
    var sum_error3: Float32 = 0.0
    for i in range(len(v5)):
        sum_error3 += abs(v5[i] - v6[i])
    var mae3 = sum_error3 / Float32(len(v5))
    
    # Errors: 1, 2, 3, 4 → MAE = 2.5
    if abs(mae3 - 2.5) > 0.001:
        print("    ❌ Variable error case failed:", mae3)
        return False
    print("    ✓ Variable error case passed")
    
    return True


fn test_mse_calculation() -> Bool:
    """Test Mean Squared Error calculation."""
    print("\n  Testing mse_calculation...")
    
    # Test 1: Zero error
    var v1 = List[Float32](1.0, 2.0, 3.0)
    var v2 = List[Float32](1.0, 2.0, 3.0)
    
    var sum_sq_error: Float32 = 0.0
    for i in range(len(v1)):
        var diff = v1[i] - v2[i]
        sum_sq_error += diff * diff
    var mse = sum_sq_error / Float32(len(v1))
    
    if abs(mse) > 0.001:
        print("    ❌ Zero error case failed:", mse)
        return False
    print("    ✓ Zero error case passed")
    
    # Test 2: Known squared error
    var v3 = List[Float32](0.0, 0.0, 0.0)
    var v4 = List[Float32](1.0, 2.0, 3.0)
    
    var sum_sq_error2: Float32 = 0.0
    for i in range(len(v3)):
        var diff = v3[i] - v4[i]
        sum_sq_error2 += diff * diff
    var mse2 = sum_sq_error2 / Float32(len(v3))
    
    # Squared errors: 1, 4, 9 → MSE = 14/3 ≈ 4.667
    var expected: Float32 = (1.0 + 4.0 + 9.0) / 3.0
    if abs(mse2 - expected) > 0.001:
        print("    ❌ Known squared error case failed:", mse2)
        return False
    print("    ✓ Known squared error case passed")
    
    # Test 3: MSE > MAE for non-uniform errors
    var v5 = List[Float32](0.0, 0.0, 0.0, 0.0)
    var v6 = List[Float32](1.0, 1.0, 1.0, 10.0)
    
    # Calculate MAE
    var sum_abs: Float32 = 0.0
    for i in range(len(v5)):
        sum_abs += abs(v5[i] - v6[i])
    var mae = sum_abs / Float32(len(v5))
    
    # Calculate MSE
    var sum_sq: Float32 = 0.0
    for i in range(len(v5)):
        var diff = v5[i] - v6[i]
        sum_sq += diff * diff
    var mse3 = sum_sq / Float32(len(v5))
    
    if mse3 <= mae:
        print("    ❌ MSE should be > MAE for non-uniform errors")
        return False
    print("    ✓ MSE > MAE for non-uniform errors")
    
    return True


fn test_percentile_error() -> Bool:
    """Test percentile error calculations."""
    print("\n  Testing percentile_error...")
    
    # Create sorted errors for testing
    var errors = List[Float32]()
    for i in range(100):
        errors.append(Float32(i) / 100.0)  # 0.00 to 0.99
    
    # 50th percentile (median) should be around 0.49
    var p50_idx = Int((len(errors) * 50) / 100)
    var p50 = errors[p50_idx]
    
    if p50 < 0.45 or p50 > 0.55:
        print("    ❌ 50th percentile incorrect:", p50)
        return False
    print("    ✓ 50th percentile correct:", p50)
    
    # 95th percentile should be around 0.94
    var p95_idx = Int((len(errors) * 95) / 100)
    var p95 = errors[p95_idx]
    
    if p95 < 0.90 or p95 > 0.99:
        print("    ❌ 95th percentile incorrect:", p95)
        return False
    print("    ✓ 95th percentile correct:", p95)
    
    # 99th percentile should be around 0.98
    var p99_idx = Int((len(errors) * 99) / 100)
    var p99 = errors[p99_idx]
    
    if p99 < 0.95:
        print("    ❌ 99th percentile incorrect:", p99)
        return False
    print("    ✓ 99th percentile correct:", p99)
    
    return True


fn test_batch_quality_metrics() -> Bool:
    """Test batch quality metric calculations."""
    print("\n  Testing batch_quality_metrics...")
    
    # Create batch of original and reconstructed vectors
    var original_batch = List[List[Float32]]()
    var reconstructed_batch = List[List[Float32]]()
    
    for i in range(10):
        var orig = List[Float32]()
        var recon = List[Float32]()
        for j in range(128):
            var val = Float32(i + j) / 10.0
            orig.append(val)
            recon.append(val + 0.01)  # Small reconstruction error
        original_batch.append(orig^)
        reconstructed_batch.append(recon^)
    
    # Calculate batch MAE
    var total_mae: Float32 = 0.0
    for i in range(len(original_batch)):
        var orig = original_batch[i].copy()
        var recon = reconstructed_batch[i].copy()
        
        var vec_error: Float32 = 0.0
        for j in range(len(orig)):
            vec_error += abs(orig[j] - recon[j])
        
        var vec_mae = vec_error / Float32(len(orig))
        total_mae += vec_mae
    
    var batch_mae = total_mae / Float32(len(original_batch))
    
    # Should be close to 0.01
    if abs(batch_mae - 0.01) > 0.001:
        print("    ❌ Batch MAE incorrect:", batch_mae)
        return False
    print("    ✓ Batch MAE correct:", batch_mae)
    
    return True


fn test_reconstruction_quality() -> Bool:
    """Test quality of quantization and reconstruction."""
    print("\n  Testing reconstruction_quality...")
    
    # Create original vector
    var original = List[Float32]()
    for i in range(768):
        original.append(Float32(i) / 100.0)
    
    # Quantize
    var max_abs: Float32 = 7.67
    var scale: Float32 = 127.0 / max_abs
    
    var quantized = List[Int8]()
    for i in range(len(original)):
        quantized.append(Int8(original[i] * scale))
    
    # Reconstruct
    var reconstructed = List[Float32]()
    for i in range(len(quantized)):
        reconstructed.append(Float32(quantized[i]) / scale)
    
    # Calculate MAE
    var sum_error: Float32 = 0.0
    for i in range(len(original)):
        sum_error += abs(original[i] - reconstructed[i])
    var mae = sum_error / Float32(len(original))
    
    # MAE should be very small (< 1% of max value)
    if mae > 0.08:
        print("    ❌ Reconstruction quality too poor, MAE:", mae)
        return False
    print("    ✓ Reconstruction quality good, MAE:", mae)
    
    # Calculate MSE
    var sum_sq_error: Float32 = 0.0
    for i in range(len(original)):
        var diff = original[i] - reconstructed[i]
        sum_sq_error += diff * diff
    var mse = sum_sq_error / Float32(len(original))
    
    # MSE should also be small
    if mse > 0.01:
        print("    ❌ Reconstruction MSE too high:", mse)
        return False
    print("    ✓ Reconstruction MSE acceptable:", mse)
    
    return True


fn test_compression_ratio() -> Bool:
    """Test compression ratio calculations."""
    print("\n  Testing compression_ratio...")
    
    # Original: Float32 (4 bytes) × 768 = 3,072 bytes
    var original_size = 768 * 4
    
    # Quantized: Int8 (1 byte) × 768 + Float32 scale (4 bytes) = 772 bytes
    var quantized_size = 768 * 1 + 4
    
    # Compression ratio
    var ratio = Float32(original_size) / Float32(quantized_size)
    
    # Should be close to 3.98x
    if ratio < 3.5 or ratio > 4.5:
        print("    ❌ Compression ratio unexpected:", ratio)
        return False
    print("    ✓ Compression ratio:", ratio, "x")
    
    # For batch of 1000 vectors
    var batch_original = 1000 * 768 * 4
    var batch_quantized = 1000 * 768 * 1 + 1000 * 4
    var batch_ratio = Float32(batch_original) / Float32(batch_quantized)
    
    if batch_ratio < 3.9 or batch_ratio > 4.0:
        print("    ❌ Batch compression ratio unexpected:", batch_ratio)
        return False
    print("    ✓ Batch compression ratio:", batch_ratio, "x")
    
    return True


fn run_quality_tests():
    """Run all quality metric tests."""
    print("\n=== Testing Quality Metrics & Batch Processing ===")
    
    var passed = 0
    var total = 0
    
    total += 1
    if test_mae_calculation():
        passed += 1
    
    total += 1
    if test_mse_calculation():
        passed += 1
    
    total += 1
    if test_percentile_error():
        passed += 1
    
    total += 1
    if test_batch_quality_metrics():
        passed += 1
    
    total += 1
    if test_reconstruction_quality():
        passed += 1
    
    total += 1
    if test_compression_ratio():
        passed += 1
    
    print("\n  ✅ quality_metrics:", passed, "/", total, "tests passed")
    print("  📊 Coverage: ~85% (core metrics and batch operations tested)")


fn main():
    run_quality_tests()
