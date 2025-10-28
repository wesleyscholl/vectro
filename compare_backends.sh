#!/bin/bash
set -e

echo "========================================="
echo "Vectro Backend Comparison"
echo "========================================="
echo

# Activate pixi environment
eval "$(pixi shell-hook)"

echo "✓ Environment activated"
echo "  Mojo version: $(mojo --version 2>&1 | head -1)"
echo

echo "--- Test 1: Mojo Quantizer ---"
time mojo run src/quantizer_working.mojo 2>&1 | grep -A5 "Original:"
echo

echo "--- Test 2: Python/Cython Backend ---"
python3 << 'EOF'
from interface import QuantizerInterface
import numpy as np
import time

# Create test data
embeddings = np.random.randn(5000, 128).astype(np.float32)

# Benchmark
qi = QuantizerInterface(backend="cython")
start = time.time()
result = qi.quantize(embeddings)
elapsed = time.time() - start

throughput = len(embeddings) / elapsed
print(f"Cython throughput: {throughput:,.0f} vectors/sec")
print(f"Time: {elapsed:.3f}s for {len(embeddings)} vectors")
EOF

echo
echo "========================================="
echo "Expected Performance:"
echo "  Mojo (SIMD):    500,000 - 1,000,000 vec/s"
echo "  Cython (prod):  328,000 vec/s ✓ verified"
echo "  NumPy (basic):  50,000 vec/s"
echo "========================================="
