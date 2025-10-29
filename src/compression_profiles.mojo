"""
Compression quality profiles for different use cases.
Balances speed, memory, and accuracy based on requirements.
"""

struct CompressionProfile:
    """Defines parameters for different compression quality levels."""
    
    var name: String
    var bits_per_value: Int
    var min_value: Float32
    var max_value: Float32
    
    fn __init__(
        out self,
        name: String,
        bits_per_value: Int,
        min_value: Float32,
        max_value: Float32
    ):
        """Initialize compression profile.
        
        Args:
            name: Profile name.
            bits_per_value: Number of bits per quantized value.
            min_value: Minimum quantized value.
            max_value: Maximum quantized value.
        """
        self.name = name
        self.bits_per_value = bits_per_value
        self.min_value = min_value
        self.max_value = max_value
    
    fn get_range(self) -> Float32:
        """Get the quantization range."""
        return self.max_value - self.min_value


fn create_fast_profile() -> CompressionProfile:
    """Create profile optimized for speed.
    
    Returns:
        Fast compression profile (int8, full range).
    """
    return CompressionProfile(
        name="fast",
        bits_per_value=8,
        min_value=-127.0,
        max_value=127.0
    )


fn create_balanced_profile() -> CompressionProfile:
    """Create profile with balanced speed and quality.
    
    Returns:
        Balanced compression profile (int8, standard range).
    """
    return CompressionProfile(
        name="balanced",
        bits_per_value=8,
        min_value=-127.0,
        max_value=127.0
    )


fn create_quality_profile() -> CompressionProfile:
    """Create profile optimized for quality.
    
    Returns:
        Quality compression profile (int8, conservative range).
    """
    return CompressionProfile(
        name="quality",
        bits_per_value=8,
        min_value=-100.0,
        max_value=100.0
    )


fn quantize_with_profile(
    data: List[Float32],
    profile: CompressionProfile
) -> (List[Int8], Float32):
    """Quantize a vector using a specific compression profile.
    
    Args:
        data: Input float32 vector.
        profile: Compression profile to use.
        
    Returns:
        Tuple of (quantized values, scale factor).
    """
    var dim = len(data)
    
    # Find max absolute value
    var max_abs: Float32 = 0.0
    for i in range(dim):
        var val = data[i]
        var abs_val = val if val >= 0 else -val
        if abs_val > max_abs:
            max_abs = abs_val
    
    # Compute scale
    var scale: Float32
    if max_abs < 1e-10:
        scale = 1.0
    else:
        scale = max_abs / profile.max_value
    
    var inv_scale = 1.0 / scale
    
    # Quantize
    var quantized = List[Int8]()
    for i in range(dim):
        var val = data[i] * inv_scale
        
        # Clamp to profile range
        if val > profile.max_value:
            val = profile.max_value
        elif val < profile.min_value:
            val = profile.min_value
        
        # Round to nearest int8
        var quant_val: Int
        if val >= 0:
            quant_val = Int(val + 0.5)
        else:
            quant_val = Int(val - 0.5)
        
        quantized.append(Int8(quant_val))
    
    return (quantized^, scale)


fn reconstruct_with_profile(
    quantized: List[Int8],
    scale: Float32
) -> List[Float32]:
    """Reconstruct a vector from quantized form.
    
    Args:
        quantized: Input int8 quantized values.
        scale: Scale factor used during quantization.
        
    Returns:
        Reconstructed float32 vector.
    """
    var reconstructed = List[Float32]()
    
    for i in range(len(quantized)):
        reconstructed.append(Float32(quantized[i]) * scale)
    
    return reconstructed^


struct ProfileManager:
    """Manages multiple compression profiles."""
    
    @staticmethod
    fn get_profile(name: String) -> CompressionProfile:
        """Get a compression profile by name.
        
        Args:
            name: Profile name ("fast", "balanced", or "quality").
            
        Returns:
            Compression profile.
        """
        if name == "fast":
            return create_fast_profile()
        elif name == "balanced":
            return create_balanced_profile()
        elif name == "quality":
            return create_quality_profile()
        else:
            return create_balanced_profile()
    
    @staticmethod
    fn list_profiles():
        """Print available compression profiles."""
        print("Available Compression Profiles:")
        print("  - fast: Maximum speed (full int8 range)")
        print("  - balanced: Good speed/quality tradeoff (standard)")
        print("  - quality: Maximum quality (conservative range)")


fn main():
    """Test compression profiles."""
    print("=" * 70)
    print("Vectro Compression Profiles Module")
    print("=" * 70)
    print()
    
    ProfileManager.list_profiles()
    
    print("\nTesting profiles with sample vector:")
    var test_data = List[Float32]()
    test_data.append(1.0)
    test_data.append(2.0)
    test_data.append(3.0)
    test_data.append(4.0)
    
    print("  Input: [1.0, 2.0, 3.0, 4.0]")
    print()
    
    # Test each profile
    var profiles = List[String]()
    profiles.append("fast")
    profiles.append("balanced")
    profiles.append("quality")
    
    for i in range(len(profiles)):
        var profile_name = profiles[i]
        var profile = ProfileManager.get_profile(profile_name)
        var result = quantize_with_profile(test_data, profile)
        var quantized = result.0
        var scale = result.1
        
        print("Profile:", profile_name)
        print("  Scale factor:", scale)
        print("  Range: [", profile.min_value, ",", profile.max_value, "]")
        print("  Quantized: [", quantized[0], ",", quantized[1], ",", 
              quantized[2], ",", quantized[3], "]")
        
        var recon = reconstruct_with_profile(quantized, scale)
        print("  Reconstructed: [", recon[0], ",", recon[1], ",", 
              recon[2], ",", recon[3], "]")
        print()
    
    print("=" * 70)

