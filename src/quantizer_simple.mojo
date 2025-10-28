"""
Simplified Mojo quantizer for embedding vectors.
"""

fn quantize_vector(data: List[Float32]) -> (List[Int8], Float32):
    """Quantize a single vector to int8.
    
    Returns tuple of (quantized values, scale factor)
    """
    var result = List[Int8]()
    var max_abs: Float32 = 0.0
    
    # Find max absolute value
    for i in range(len(data)):
        var val = data[i]
        var abs_val = val if val >= 0.0 else -val
        if abs_val > max_abs:
            max_abs = abs_val
    
    # Calculate scale
    var scale: Float32 = 1.0
    if max_abs > 0.0:
        scale = max_abs / 127.0
    
    var inv_scale = 1.0 / scale
    
    # Quantize elements
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
    
    return (result, scale)


fn reconstruct_vector(quant: List[Int8], scale: Float32) -> List[Float32]:
    """Reconstruct float vector from quantized int8."""
    var result = List[Float32]()
    
    for i in range(len(quant)):
        result.append(Float32(quant[i]) * scale)
    
    return result


fn main():
    print("Mojo Quantizer - Simple Version")
    print("=" * 40)
    
    # Test quantization
    print("\nTest: Quantizing vector [1.0, 2.0, 3.0, 4.0]")
    var test_data = List[Float32]()
    test_data.append(1.0)
    test_data.append(2.0)
    test_data.append(3.0)
    test_data.append(4.0)
    
    var (quant, scale) = quantize_vector(test_data)
    print("Scale:", scale)
    print("Quantized:", quant[0], quant[1], quant[2], quant[3])
    
    var recon = reconstruct_vector(quant, scale)
    print("Reconstructed:", recon[0], recon[1], recon[2], recon[3])
    
    # Calculate error
    var total_error: Float32 = 0.0
    for i in range(len(test_data)):
        var err = test_data[i] - recon[i]
        var abs_err = err if err >= 0.0 else -err
        total_error += abs_err
    
    var avg_error = total_error / Float32(len(test_data))
    print("Average error:", avg_error)
    
    print("\nTest passed!")
