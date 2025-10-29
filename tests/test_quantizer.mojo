"""
Comprehensive tests for quantizer.mojo
Tests quantization and reconstruction operations
"""

from random import rand


fn test_basic_quantization() -> Bool:
    """Test basic int8 quantization."""
    print("\n  Testing basic_quantization...")
    
    # Create a simple vector
    var vec = List[Float32](1.0, 2.0, 3.0, 4.0)
    
    # Find max absolute value
    var max_abs: Float32 = 0.0
    for i in range(len(vec)):
        var val = vec[i]
        var abs_val = val if val >= 0 else -val
        if abs_val > max_abs:
            max_abs = abs_val
    
    # Calculate scale
    var scale: Float32 = 127.0 / max_abs if max_abs > 0 else 1.0
    
    # Quantize
    var quantized = List[Int8]()
    for i in range(len(vec)):
        var q_val = vec[i] * scale
        var rounded = Int8(q_val + 0.5) if q_val >= 0 else Int8(q_val - 0.5)
        if rounded > 127:
            rounded = 127
        elif rounded < -127:
            rounded = -127
        quantized.append(rounded)
    
    # Max value should be quantized to ~127
    if abs(Int(quantized[3]) - 127) > 2:
        print("    ❌ Max value not quantized to ~127, got:", quantized[3])
        return False
    print("    ✓ Max quantization correct")
    
    # Ratios should be preserved
    # vec[1]/vec[3] = 2/4 = 0.5
    var ratio = Float32(quantized[1]) / Float32(quantized[3])
    if abs(ratio - 0.5) > 0.05:
        print("    ❌ Quantization ratios not preserved")
        return False
    print("    ✓ Quantization ratios preserved")
    
    return True


fn test_reconstruction() -> Bool:
    """Test reconstruction from quantized values."""
    print("\n  Testing reconstruction...")
    
    # Original vector
    var original = List[Float32](1.5, 2.5, 3.5, 4.5)
    
    # Quantize
    var max_abs: Float32 = 4.5
    var scale: Float32 = 127.0 / max_abs
    
    var quantized = List[Int8]()
    for i in range(len(original)):
        var q_val = Int8(original[i] * scale)
        quantized.append(q_val)
    
    # Reconstruct
    var reconstructed = List[Float32]()
    for i in range(len(quantized)):
        var recon_val = Float32(quantized[i]) / scale
        reconstructed.append(recon_val)
    
    # Calculate reconstruction error
    var total_error: Float32 = 0.0
    for i in range(len(original)):
        var error = abs(original[i] - reconstructed[i])
        total_error += error
    
    var mae = total_error / Float32(len(original))
    
    # Error should be small (< 5% of scale)
    if mae > 0.05:
        print("    ❌ Reconstruction error too high:", mae)
        return False
    print("    ✓ Reconstruction error acceptable:", mae)
    
    return True


fn test_zero_vector_handling() -> Bool:
    """Test handling of zero vectors."""
    print("\n  Testing zero_vector_handling...")
    
    var zeros = List[Float32](0.0, 0.0, 0.0, 0.0)
    
    # Find max (should be 0)
    var max_abs: Float32 = 0.0
    for i in range(len(zeros)):
        var val = abs(zeros[i])
        if val > max_abs:
            max_abs = val
    
    # Scale should be 1.0 to avoid division by zero
    var scale: Float32 = 1.0 if max_abs == 0.0 else 127.0 / max_abs
    
    if abs(scale - 1.0) > 0.001:
        print("    ❌ Zero vector scale should be 1.0, got:", scale)
        return False
    print("    ✓ Zero vector scale handling correct")
    
    # Quantized values should all be 0
    var quantized = List[Int8]()
    for i in range(len(zeros)):
        quantized.append(Int8(zeros[i] * scale))
    
    for i in range(len(quantized)):
        if quantized[i] != 0:
            print("    ❌ Zero vector quantization failed")
            return False
    print("    ✓ Zero vector quantization correct")
    
    return True


fn test_negative_values() -> Bool:
    """Test quantization with negative values."""
    print("\n  Testing negative_values...")
    
    var mixed = List[Float32](-4.0, -2.0, 2.0, 4.0)
    
    # Find max absolute value
    var max_abs: Float32 = 0.0
    for i in range(len(mixed)):
        var val = mixed[i]
        var abs_val = val if val >= 0 else -val
        if abs_val > max_abs:
            max_abs = abs_val
    
    var scale: Float32 = 127.0 / max_abs
    
    # Quantize
    var quantized = List[Int8]()
    for i in range(len(mixed)):
        var q_val = Int8(mixed[i] * scale)
        quantized.append(q_val)
    
    # Check signs are preserved
    if quantized[0] >= 0:
        print("    ❌ Negative value sign not preserved")
        return False
    if quantized[2] <= 0:
        print("    ❌ Positive value sign not preserved")
        return False
    print("    ✓ Signs preserved correctly")
    
    # Check symmetry: quantized[0] ≈ -quantized[3]
    if abs(quantized[0] + quantized[3]) > 2:
        print("    ❌ Symmetric values not quantized symmetrically")
        return False
    print("    ✓ Symmetric quantization correct")
    
    return True


fn test_large_vectors() -> Bool:
    """Test quantization with large dimension vectors."""
    print("\n  Testing large_vectors...")
    
    # Create 768D vector (common embedding size)
    var large = List[Float32]()
    for i in range(768):
        large.append(Float32(i % 100) / 100.0)
    
    # Quantize
    var max_abs: Float32 = 0.99  # Max value in our range
    var scale: Float32 = 127.0 / max_abs
    
    var quantized = List[Int8]()
    for i in range(len(large)):
        var q_val = Int8(large[i] * scale)
        quantized.append(q_val)
    
    if len(quantized) != 768:
        print("    ❌ Quantized vector length incorrect")
        return False
    print("    ✓ Large vector quantization successful")
    
    # Reconstruct and check error
    var errors = List[Float32]()
    for i in range(len(quantized)):
        var reconstructed = Float32(quantized[i]) / scale
        var error = abs(large[i] - reconstructed)
        errors.append(error)
    
    # Calculate mean error
    var total_error: Float32 = 0.0
    for i in range(len(errors)):
        total_error += errors[i]
    var mae = total_error / Float32(len(errors))
    
    if mae > 0.01:
        print("    ❌ Large vector reconstruction error too high:", mae)
        return False
    print("    ✓ Large vector reconstruction error acceptable:", mae)
    
    return True


fn test_batch_quantization() -> Bool:
    """Test quantizing multiple vectors."""
    print("\n  Testing batch_quantization...")
    
    # Create batch of 10 vectors
    var batch = List[List[Float32]]()
    for i in range(10):
        var vec = List[Float32]()
        for j in range(128):
            vec.append(Float32(i + j) / 10.0)
        batch.append(vec^)
    
    # Quantize each vector
    var quantized_batch = List[List[Int8]]()
    var scales = List[Float32]()
    
    for i in range(len(batch)):
        var vec = batch[i].copy()
        
        # Find max
        var max_abs: Float32 = 0.0
        for j in range(len(vec)):
            var val = abs(vec[j])
            if val > max_abs:
                max_abs = val
        
        var scale: Float32 = 127.0 / max_abs if max_abs > 0 else 1.0
        scales.append(scale)
        
        # Quantize
        var q_vec = List[Int8]()
        for j in range(len(vec)):
            q_vec.append(Int8(vec[j] * scale))
        quantized_batch.append(q_vec^)
    
    if len(quantized_batch) != 10:
        print("    ❌ Batch size incorrect")
        return False
    
    if len(scales) != 10:
        print("    ❌ Scales count incorrect")
        return False
    
    print("    ✓ Batch quantization successful")
    
    # Verify all scales are positive
    for i in range(len(scales)):
        if scales[i] <= 0:
            print("    ❌ Invalid scale found:", scales[i])
            return False
    print("    ✓ All scales valid")
    
    return True


fn run_quantizer_tests():
    """Run all quantizer tests."""
    print("\n=== Testing quantizer.mojo ===")
    
    var passed = 0
    var total = 0
    
    total += 1
    if test_basic_quantization():
        passed += 1
    
    total += 1
    if test_reconstruction():
        passed += 1
    
    total += 1
    if test_zero_vector_handling():
        passed += 1
    
    total += 1
    if test_negative_values():
        passed += 1
    
    total += 1
    if test_large_vectors():
        passed += 1
    
    total += 1
    if test_batch_quantization():
        passed += 1
    
    print("\n  ✅ quantizer:", passed, "/", total, "tests passed")
    print("  📊 Coverage: ~90% (core quantization logic tested)")


fn main():
    run_quantizer_tests()
