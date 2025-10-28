"""
Python integration test for Vectro Mojo quantizer.

This demonstrates how to integrate Mojo performance into the Python vectro project.
"""

import numpy as np
import time
import subprocess
import json


def test_mojo_performance():
    """Test and compare Mojo vs Python/Cython performance."""
    print("Vectro Performance Comparison")
    print("=" * 60)
    
    # Test configuration
    n_vectors = 10000
    dimensions = 128
    
    print(f"\nTest setup: {n_vectors} vectors × {dimensions} dimensions")
    print("-" * 60)
    
    # Generate test data
    embeddings = np.random.randn(n_vectors, dimensions).astype(np.float32)
    
    # Test 1: Python/NumPy baseline
    print("\n1. NumPy Baseline:")
    start = time.time()
    for vec in embeddings:
        max_val = np.abs(vec).max()
        scale = max_val / 127.0
        quantized = np.round(vec / scale).astype(np.int8)
    elapsed_numpy = time.time() - start
    throughput_numpy = n_vectors / elapsed_numpy
    print(f"   Time: {elapsed_numpy:.4f}s")
    print(f"   Throughput: {throughput_numpy:,.0f} vectors/sec")
    
    # Test 2: Cython (if available)
    try:
        from interface import QuantizerInterface
        print("\n2. Cython Backend:")
        qi = QuantizerInterface(backend="cython")
        start = time.time()
        result = qi.quantize(embeddings)
        elapsed_cython = time.time() - start
        throughput_cython = n_vectors / elapsed_cython
        print(f"   Time: {elapsed_cython:.4f}s")
        print(f"   Throughput: {throughput_cython:,.0f} vectors/sec")
        print(f"   Speedup vs NumPy: {elapsed_numpy/elapsed_cython:.2f}x")
    except Exception as e:
        print(f"\n2. Cython Backend: Not available ({e})")
        elapsed_cython = None
        throughput_cython = None
    
    # Test 3: Mojo (via compiled binary)
    print("\n3. Mojo Backend:")
    try:
        result = subprocess.run(
            ['./vectro_quantizer'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Parse the output
        for line in result.stdout.split('\n'):
            if 'Throughput:' in line:
                throughput_str = line.split(':')[1].strip().split()[0]
                throughput_mojo = float(throughput_str.replace(',', ''))
                print(f"   Throughput: {throughput_mojo:,.0f} vectors/sec")
                
                if throughput_cython:
                    print(f"   Speedup vs Cython: {throughput_mojo/throughput_cython:.2f}x")
                print(f"   Speedup vs NumPy: {throughput_mojo/throughput_numpy:.2f}x")
                break
        else:
            print("   Could not parse throughput from output")
            
    except FileNotFoundError:
        print("   Mojo binary not found. Run: mojo build src/vectro_standalone.mojo -o vectro_quantizer")
    except Exception as e:
        print(f"   Error running Mojo: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Performance Summary:")
    print("-" * 60)
    print(f"NumPy:   {throughput_numpy:>12,.0f} vec/s")
    if throughput_cython:
        print(f"Cython:  {throughput_cython:>12,.0f} vec/s ({throughput_cython/throughput_numpy:>5.2f}x)")
    print(f"Mojo:    {throughput_mojo:>12,.0f} vec/s (estimated)")
    print("=" * 60)
    
    # Expected performance targets
    print("\nExpected Performance Targets:")
    print("  • NumPy (baseline):     50,000 - 100,000 vec/s")
    print("  • Cython (current):    300,000 - 400,000 vec/s")
    print("  • Mojo (SIMD):         500,000 - 1,000,000 vec/s")
    print("\n✓ Performance test complete")


def test_accuracy():
    """Test quantization accuracy."""
    print("\n" + "=" * 60)
    print("Accuracy Test")
    print("=" * 60)
    
    # Test data
    test_vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    
    # Quantize
    max_val = np.abs(test_vec).max()
    scale = max_val / 127.0
    quantized = np.round(test_vec / scale).astype(np.int8)
    
    # Reconstruct
    reconstructed = quantized.astype(np.float32) * scale
    
    # Error
    error = np.abs(test_vec - reconstructed)
    avg_error = error.mean()
    max_error = error.max()
    
    print(f"\nOriginal:      {test_vec}")
    print(f"Scale:         {scale:.6f}")
    print(f"Quantized:     {quantized}")
    print(f"Reconstructed: {reconstructed}")
    print(f"Average error: {avg_error:.6f} ({avg_error/test_vec.mean()*100:.2f}%)")
    print(f"Max error:     {max_error:.6f}")
    
    print("\n✓ Accuracy test complete")


if __name__ == "__main__":
    test_mojo_performance()
    test_accuracy()
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("  1. Integrate Mojo into interface.py as 'mojo' backend")
    print("  2. Update setup.py to include Mojo package")
    print("  3. Add automatic backend selection (Mojo > Cython > NumPy)")
    print("  4. Update documentation and README")
    print("=" * 60)
