"""
Vectro Mojo - High-performance int8 quantization for embeddings.

Simple Python wrapper for testing the Mojo quantizer.
"""

from python import Python


def test_mojo_quantizer():
    """Test the Mojo quantizer from Python."""
    print("Testing Mojo Quantizer from Python...")
    print("=" * 50)
    
    # For now, we'll use Python directly until MAX Python API is available
    # This is a placeholder for the integration
    
    # Test data
    test_data = [1.0, 2.0, 3.0, 4.0]
    print(f"\nTest data: {test_data}")
    
    # Simple Python quantization (placeholder)
    max_val = max(abs(x) for x in test_data)
    scale = max_val / 127.0
    quantized = [int(x / scale + 0.5) for x in test_data]
    
    print(f"Scale: {scale}")
    print(f"Quantized: {quantized}")
    
    # Reconstruct
    reconstructed = [q * scale for q in quantized]
    print(f"Reconstructed: {reconstructed}")
    
    # Error
    errors = [abs(orig - recon) for orig, recon in zip(test_data, reconstructed)]
    avg_error = sum(errors) / len(errors)
    print(f"Average error: {avg_error}")
    
    print("\n✓ Python wrapper test complete")
    print("Note: Full MAX Python API integration coming next")


if __name__ == "__main__":
    test_mojo_quantizer()
