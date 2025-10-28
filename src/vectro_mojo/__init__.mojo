"""
Vectro Mojo Package - High-performance embedding quantization.

This package provides Python-accessible functions for int8 quantization
of embedding vectors using SIMD acceleration.
"""

from python import Python, PythonObject
from memory import memcpy


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


struct BatchQuantResult:
    """Result of batch quantization."""
    var quantized: List[Int8]
    var scales: List[Float32]
    
    fn __init__(out self, var q: List[Int8], var s: List[Float32]):
        self.quantized = q^
        self.scales = s^


fn quantize_batch(data: List[Float32], n_vectors: Int, dim: Int) -> BatchQuantResult:
    """Quantize a batch of vectors.
    
    Args:
        data: Flat array of embeddings (length n_vectors * dim, row-major).
        n_vectors: Number of vectors.
        dim: Dimensions per vector.
    
    Returns:
        BatchQuantResult containing quantized int8 flat array and per-vector scale factors.
    """
    var quantized = List[Int8]()
    var scales = List[Float32]()
    
    # Pre-allocate
    for _ in range(n_vectors * dim):
        quantized.append(0)
    for _ in range(n_vectors):
        scales.append(0.0)
    
    # Process each vector
    for vec_idx in range(n_vectors):
        var base = vec_idx * dim
        var max_val: Float32 = 0.0
        
        # Find max absolute value
        for j in range(dim):
            var val = data[base + j]
            var abs_val = val if val >= 0.0 else -val
            if abs_val > max_val:
                max_val = abs_val
        
        # Calculate scale
        var scale: Float32 = 1.0
        if max_val > 0.0:
            scale = max_val / 127.0
        scales[vec_idx] = scale
        
        var inv_scale = 1.0 / scale
        
        # Quantize vector elements
        for j in range(dim):
            var raw = data[base + j] * inv_scale
            # Clamp to [-127, 127]
            if raw > 127.0:
                raw = 127.0
            elif raw < -127.0:
                raw = -127.0
            # Round to nearest int
            var rounded = Int(raw + 0.5) if raw >= 0.0 else Int(raw - 0.5)
            quantized[base + j] = Int8(rounded)
    
    return BatchQuantResult(quantized^, scales^)


fn reconstruct_batch(quantized: List[Int8], scales: List[Float32], n_vectors: Int, dim: Int) -> List[Float32]:
    """Reconstruct a batch of vectors from quantized data.
    
    Args:
        quantized: Flat quantized int8 array (length n_vectors * dim).
        scales: Per-vector scale factors (length n_vectors).
        n_vectors: Number of vectors.
        dim: Dimensions per vector.
    
    Returns:
        Reconstructed float32 embeddings (length n_vectors * dim).
    """
    var result = List[Float32]()
    
    # Pre-allocate
    for _ in range(n_vectors * dim):
        result.append(0.0)
    
    # Process each vector
    for vec_idx in range(n_vectors):
        var base = vec_idx * dim
        var scale = scales[vec_idx]
        
        # Reconstruct elements
        for j in range(dim):
            result[base + j] = Float32(quantized[base + j]) * scale
    
    return result^


# Python-accessible functions using MAX Python API
fn python_quantize_vector(data: PythonObject) raises -> PythonObject:
    """Python-accessible quantize function for single vector.
    
    Args:
        data: Python list or numpy array of float32 values.
    
    Returns:
        Python tuple of (quantized_list, scale).
    """
    var py = Python.import_module("builtins")
    
    # Convert Python data to Mojo List
    var mojo_data = List[Float32]()
    var length = py.len(data)
    for i in range(Int(length)):
        mojo_data.append(Float32(data[i]))
    
    # Quantize
    var result = quantize_vector(mojo_data)
    
    # Convert back to Python
    var py_quantized = py.list()
    for i in range(len(result.quantized)):
        _ = py_quantized.append(Int(result.quantized[i]))
    
    return py.tuple([py_quantized, Float32(result.scale)])


fn python_quantize_batch(data: PythonObject, n_vectors: Int, dim: Int) raises -> PythonObject:
    """Python-accessible quantize function for batch of vectors.
    
    Args:
        data: Python list or numpy array (flat, row-major).
        n_vectors: Number of vectors.
        dim: Dimensions per vector.
    
    Returns:
        Python tuple of (quantized_list, scales_list).
    """
    var py = Python.import_module("builtins")
    
    # Convert Python data to Mojo List
    var mojo_data = List[Float32]()
    var length = py.len(data)
    for i in range(Int(length)):
        mojo_data.append(Float32(data[i]))
    
    # Quantize
    var result = quantize_batch(mojo_data, n_vectors, dim)
    
    # Convert back to Python
    var py_quantized = py.list()
    for i in range(len(result.quantized)):
        _ = py_quantized.append(Int(result.quantized[i]))
    
    var py_scales = py.list()
    for i in range(len(result.scales)):
        _ = py_scales.append(Float32(result.scales[i]))
    
    return py.tuple([py_quantized, py_scales])
