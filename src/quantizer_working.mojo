"""
Working Mojo quantizer using struct return type.
"""

struct QuantResult:
    var quantized: List[Int8]
    var scale: Float32
    
    fn __init__(out self, var q: List[Int8], s: Float32):
        self.quantized = q^
        self.scale = s


fn quantize_vector(data: List[Float32]) -> QuantResult:
    """Quantize a vector to int8 with scale factor."""
    # Find max
    var max_val: Float32 = 0.0
    for i in range(len(data)):
        var val = data[i]
        if val > max_val:
            max_val = val
    
    # Calculate scale
    var scale = max_val / 127.0
    var inv_scale = 1.0 / scale
    
    # Quantize
    var result = List[Int8]()
    for i in range(len(data)):
        var quant_val = Int(data[i] * inv_scale + 0.5)
        result.append(Int8(quant_val))
    
    return QuantResult(result^, scale)


fn main():
    print("Mojo Quantizer - Struct Version")
    print("=" * 40)
    
    # Test data
    var data = List[Float32]()
    data.append(1.0)
    data.append(2.0) 
    data.append(3.0)
    data.append(4.0)
    
    # Quantize
    var result = quantize_vector(data)
    
    print("\nOriginal: [1.0, 2.0, 3.0, 4.0]")
    print("Scale:", result.scale)
    print("Quantized:", result.quantized[0], result.quantized[1], result.quantized[2], result.quantized[3])
    
    # Reconstruct
    print("\nReconstruction:")
    for i in range(len(result.quantized)):
        var recon = Float32(result.quantized[i]) * result.scale
        print("  Position", i, ":", recon)
    
    print("\n✓ Mojo quantization working!")
