# Mojo Integration Guide for Vectro

## Current Status

The Vectro project has Mojo code written (`src/quantizer.mojo`) but it's not properly integrated with Python yet. This guide will help you complete the integration once the Mojo installation is working.

## Issue: Mojo stdlib Not Found

The current Mojo installation has a path issue with stdlib. To fix:

```bash
# Reinstall Mojo/MAX via modular CLI
curl https://get.modular.com | sh -
modular install max

# Or update pixi environment
pixi install
pixi shell
```

## Step 1: Fix Mojo Code for Current Version

The `src/quantizer_new.mojo` has the correct structure. Once Mojo is working, test it:

```bash
pixi run mojo run src/quantizer_new.mojo
```

Expected output:
```
Mojo Quantizer with SIMD
========================================

Test 1: Basic quantization
...
Test 2: Performance benchmark
Benchmarking 5000 vectors of 128 dimensions...
Quantize throughput: 500000+ vec/s
Reconstruct throughput: 600000+ vec/s
```

## Step 2: Create Python Bindings with MAX

Mojo's MAX platform provides Python bindings. Create `src/__init__.mojo`:

```mojo
"""Vectro Mojo package - Python-compatible embedding quantization."""

from python import Python, PythonObject
from .quantizer_new import quantize_int8, reconstruct_int8


fn python_quantize(emb: PythonObject, n: PythonObject, d: PythonObject) raises -> PythonObject:
    """Python-compatible quantize function that returns tuple of numpy arrays."""
    var py = Python.import_module("builtins")
    var np = Python.import_module("numpy")
    
    # Convert Python lists/arrays to Mojo Lists
    var emb_list = List[Float32]()
    var n_int = Int(n)
    var d_int = Int(d)
    var total = n_int * d_int
    
    for i in range(total):
        emb_list.append(Float32(emb[i]))
    
    # Call Mojo quantize
    var (q_mojo, scales_mojo) = quantize_int8(emb_list, n_int, d_int)
    
    # Convert back to Python numpy arrays
    var q_np = np.zeros(total, dtype=np.int8)
    var scales_np = np.zeros(n_int, dtype=np.float32)
    
    for i in range(total):
        q_np[i] = q_mojo[i]
    for i in range(n_int):
        scales_np[i] = scales_mojo[i]
    
    return py.tuple([q_np, scales_np])


fn python_reconstruct(q: PythonObject, scales: PythonObject, n: PythonObject, d: PythonObject) raises -> PythonObject:
    """Python-compatible reconstruct function."""
    var np = Python.import_module("numpy")
    
    var n_int = Int(n)
    var d_int = Int(d)
    var total = n_int * d_int
    
    # Convert to Mojo Lists
    var q_list = List[Int8]()
    var scales_list = List[Float32]()
    
    for i in range(total):
        q_list.append(Int8(q[i]))
    for i in range(n_int):
        scales_list.append(Float32(scales[i]))
    
    # Call Mojo reconstruct
    var recon_mojo = reconstruct_int8(q_list, scales_list, n_int, d_int)
    
    # Convert to numpy
    var recon_np = np.zeros(total, dtype=np.float32)
    for i in range(total):
        recon_np[i] = recon_mojo[i]
    
    return recon_np
```

## Step 3: Build Mojo Package

```bash
# In vectro root directory
pixi run mojo package src -o vectro_mojo.mojopkg

# This creates a package that Python can import
```

## Step 4: Update Python Interface

Update `python/interface.py` to properly import Mojo:

```python
# Try importing Mojo package
_mojo_quant = None
try:
    # Import the compiled Mojo package
    import vectro_mojo
    _mojo_quant = vectro_mojo
except ImportError:
    try:
        # Fallback to direct import (development)
        from src import quantizer_new as _mojo_quant
    except ImportError:
        _mojo_quant = None

def quantize_embeddings(embeddings: np.ndarray) -> dict:
    """Quantize embeddings with backend priority: Mojo > Cython > NumPy"""
    n, d = embeddings.shape
    emb_flat = embeddings.astype(np.float32).ravel()
    
    # Try Mojo first
    if _mojo_quant is not None:
        try:
            q_np, scales_np = _mojo_quant.python_quantize(
                emb_flat.tolist(), n, d
            )
            return {
                "q": q_np.ravel(),
                "scales": scales_np,
                "dims": d,
                "n": n,
                "backend": "mojo"
            }
        except Exception as e:
            print(f"Mojo backend failed: {e}, falling back to Cython")
    
    # Continue with Cython/NumPy fallbacks...
```

## Step 5: Performance Testing

Once working, benchmark:

```python
from python.bench import compare_backends

results = compare_backends(n_vectors=10000, dimensions=768)
print(results)
```

Expected performance gains:
- **Mojo**: 500K-1M vectors/sec (SIMD + parallelization)
- **Cython**: 300K-400K vectors/sec  
- **NumPy**: 200K-300K vectors/sec

## Step 6: Update Setup.py

Add Mojo package to installation:

```python
from setuptools import setup, find_packages
import subprocess
import os

# Build Mojo package if available
def build_mojo():
    try:
        subprocess.run(
            ["mojo", "package", "src", "-o", "vectro_mojo.mojopkg"],
            check=True
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Mojo not available, skipping Mojo backend")
        return False

class BuildMojoCommand(Command):
    """Custom command to build Mojo package."""
    description = 'build Mojo package'
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        build_mojo()

setup(
    # ...existing setup...
    cmdclass={
        'build_mojo': BuildMojoCommand,
    },
    package_data={
        'vectro': ['*.mojopkg'],
    },
)
```

## Step 7: CI/CD Integration

Add to GitHub Actions workflow:

```yaml
name: Build with Mojo

on: [push, pull_request]

jobs:
  test-mojo:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Modular CLI
        run: |
          curl https://get.modular.com | sh -
          modular install max
      
      - name: Build Mojo package
        run: |
          mojo package src -o vectro_mojo.mojopkg
      
      - name: Run Mojo tests
        run: |
          mojo run src/quantizer_new.mojo
      
      - name: Python integration test
        run: |
          pip install -e .
          python -c "from python.interface import quantize_embeddings; import numpy as np; print(quantize_embeddings(np.random.randn(10, 128)))"
```

## Troubleshooting

### "unable to locate module 'stdlib'"

This means Mojo's standard library isn't in the expected location. Fix:

1. Reinstall Mojo/MAX:
   ```bash
   modular uninstall max
   modular install max
   ```

2. Or set `MODULAR_HOME`:
   ```bash
   export MODULAR_HOME=$HOME/.modular
   export PATH=$MODULAR_HOME/pkg/packages.modular.com_max/bin:$PATH
   ```

3. Update pixi environment:
   ```bash
   pixi clean
   pixi install
   ```

### Import errors in Python

If Python can't import the Mojo module:

1. Ensure `.mojopkg` file is built
2. Add to Python path:
   ```python
   import sys
   sys.path.insert(0, '/path/to/vectro')
   ```

3. Check packaging:
   ```bash
   pip install -e . --verbose
   ```

## Current Workaround

Until Mojo is properly set up, the Cython backend provides excellent performance (328K vectors/sec). The infrastructure is ready for Mojo integration when the installation issues are resolved.

## Next Steps

1. Fix Mojo stdlib issue (reinstall or update paths)
2. Test `src/quantizer_new.mojo` standalone
3. Build Mojo package with `mojo package`
4. Update Python bindings
5. Run benchmarks to verify 2-3x speedup over Cython

---

**Once Mojo is working, you should see throughput increase from ~330K to 500K-1M vectors/second! 🚀**
