# Vectro - Mojo Release Guide

## This is a Mojo Project!

Vectro is 98.2% Mojo code. It should be distributed as a **Mojo project**, not a Python package.

## Prerequisites

- Mojo SDK installed (0.25.7+)
- pixi package manager

## Building from Source

### 1. Clone the Repository
```bash
git clone https://github.com/wesleyscholl/vectro.git
cd vectro
```

### 2. Set Up Environment with Pixi
```bash
# Pixi will automatically install Mojo and dependencies
pixi install
pixi shell
```

### 3. Build Mojo Binary
```bash
# Main quantizer binary
mojo build src/vectro_standalone.mojo -o vectro_quantizer

# Other modules
mojo build src/batch_processor.mojo -o batch_processor
mojo build src/quality_metrics.mojo -o quality_metrics
mojo build src/streaming_quantizer.mojo -o streaming_quantizer
```

### 4. Run Tests
```bash
# Compile and run test
mojo run src/batch_processor.mojo
mojo run src/quality_metrics.mojo
```

## Usage

### As a Binary
```bash
# Quantize embeddings
./vectro_quantizer input.npy output.npz

# Batch processing
./batch_processor --input data/ --output compressed/
```

### As Mojo Library
```mojo
from src.vector_ops import quantize_vector, reconstruct_vector
from src.batch_processor import BatchProcessor

# Use in your Mojo code
var processor = BatchProcessor(batch_size=1000)
var result = processor.process(embeddings)
```

## Distribution Options

### Option 1: GitHub Releases (Recommended for Mojo)
1. Build binaries for target platforms
2. Create GitHub release with binaries attached
3. Users download and run directly

```bash
# Tag and release
git tag -a v0.3.0 -m "Release v0.3.0"
git push origin v0.3.0

# Upload binaries to GitHub release:
# - vectro_quantizer (macOS ARM64)
# - vectro_quantizer-linux-x64
# - vectro_quantizer-linux-arm64
```

### Option 2: Source Distribution
Users clone and build with pixi:
```bash
git clone https://github.com/wesleyscholl/vectro.git
cd vectro
pixi install
pixi shell
mojo build src/vectro_standalone.mojo -o vectro_quantizer
```

### Option 3: Mojo Package (Future)
When Mojo has a package registry, you can publish there.

## For Python Interop (Optional)

If you need Python bindings, keep them minimal:

```python
# python/interface.py - thin wrapper
import subprocess
import numpy as np

def quantize_embeddings(embeddings: np.ndarray):
    """Call Mojo binary for quantization."""
    # Save to temp file
    np.save('/tmp/input.npy', embeddings)
    
    # Call Mojo binary
    subprocess.run(['./vectro_quantizer', '/tmp/input.npy', '/tmp/output.npz'])
    
    # Load result
    return np.load('/tmp/output.npz')
```

## Performance

Mojo delivers native performance:
- **787K-981K vectors/sec** (core quantizer)
- **900K vectors/sec** (batch processor)
- **Zero overhead** compared to C/C++

## Why Not PyPI?

1. **Mojo is not Python** - packaging as Python adds complexity
2. **Native binaries** are faster and simpler to distribute
3. **Pixi handles dependencies** better than pip for Mojo projects
4. **No compatibility issues** with Python packaging tools

## Community

- **Issues**: https://github.com/wesleyscholl/vectro/issues
- **Discussions**: https://github.com/wesleyscholl/vectro/discussions
- **Mojo Discord**: Join the Modular community

## License

MIT License - See LICENSE file
