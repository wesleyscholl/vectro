#!/bin/bash
# Vectro Mojo + Faiss Benchmark Runner
# Run this on your M3 Mac with Mojo installed

set -e

echo "=========================================="
echo "Vectro Mojo vs Faiss Benchmark"
echo "=========================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v pixi &> /dev/null; then
    echo "❌ pixi not found. Install from: https://pixi.sh"
    exit 1
fi

if ! command -v python &> /dev/null; then
    echo "❌ Python not found"
    exit 1
fi

echo "✓ pixi: $(pixi --version)"
echo "✓ python: $(python --version)"
echo ""

# Set up Mojo environment
echo "Step 1: Setting up Mojo environment..."
pixi install
echo "✓ Dependencies installed"
echo ""

# Build the Mojo binary
echo "Step 2: Building vectro_quantizer Mojo binary..."
pixi run build-mojo
if [ -f vectro_quantizer ]; then
    echo "✓ Binary built successfully: $(ls -lh vectro_quantizer | awk '{print $5}')"
else
    echo "❌ Build failed: vectro_quantizer not found"
    exit 1
fi
echo ""

# Verify the binary works
echo "Step 3: Testing Mojo binary..."
pixi run selftest
echo "✓ Mojo binary self-test passed"
echo ""

# Install Faiss if not already installed
echo "Step 4: Installing Faiss (if needed)..."
python -c "import faiss; print(f'✓ Faiss {faiss.__version__} already installed')" 2>/dev/null || \
    (pip install -q faiss-cpu && echo "✓ Faiss installed")
echo ""

# Run benchmarks
echo "Step 5: Running Vectro vs Faiss benchmarks..."
python benchmarks/benchmark_faiss_comparison.py --output results/faiss_comparison_mojo.json
echo ""

# Display results summary
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo ""
echo "Results saved to: results/faiss_comparison_mojo.json"
echo ""
python << 'PYTHON_EOF'
import json
with open("results/faiss_comparison_mojo.json") as f:
    data = json.load(f)
    print(f"Vectro Backend: {data.get('vectro_backend', 'unknown')}")
    print(f"Faiss Backend: {data.get('faiss_backend', 'unknown')}")
    print("")
    if 'int8_comparison' in data:
        int8 = data['int8_comparison']
        print(f"INT8 Quantization Throughput:")
        print(f"  Vectro: {int8.get('vectro_throughput_vec_per_sec', 0):,} vec/s")
        print(f"  Faiss:  {int8.get('faiss_throughput_vec_per_sec', 0):,} vec/s")
        if 'comparison_int8' in int8:
            ratio = int8['comparison_int8'].get('vectro_vs_faiss_throughput', 0)
            if ratio > 1:
                print(f"  ✓ Vectro {ratio:.1f}x FASTER than Faiss")
            else:
                print(f"  • Vectro {1/max(ratio, 0.001):.1f}x slower than Faiss")
PYTHON_EOF
echo ""
echo "Next: Share results in your pre-launch announcement"
