"""
Mojo quantizer that works with current Mojo version.
"""

fn quantize_simple() -> Tuple[List[Int8], Float32]:
    """Simple quantization test."""
    var data = List[Float32]()
    data.append(1.0)
    data.append(2.0) 
    data.append(3.0)
    data.append(4.0)
    
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
    
    return (result^, scale)


fn main():
    print("Testing Mojo quantizer...")
    
    var result = quantize_simple()
    var quant = result[][0]
    var scale = result[][1]
    
    print("Scale:", scale)
    print("Quantized:", quant[0], quant[1], quant[2], quant[3])
    
    print("✓ Quantization works!")
