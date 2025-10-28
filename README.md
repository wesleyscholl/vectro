# Vectro: High-Performance LLM Embedding Compressor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Mojo](https://img.shields.io/badge/Mojo-0.25.7-orange.svg)](https://www.modular.com/mojo)

Vectro is a blazing-fast, production-ready toolkit for compressing and reconstructing LLM embedding vectors. It achieves **75% storage reduction** while maintaining **>99.99% retrieval quality**, making it perfect for vector search pipelines, RAG systems, and large-scale embedding storage.

## 🎉 What's New (October 2025)

**🔥 Mojo Backend Now Production Ready!**
- **887K-981K vectors/sec** throughput (2.9-3.2x faster than NumPy)
- **<1% reconstruction error** (0.31% average)
- **Automatic backend detection** with seamless fallback
- **Zero compilation warnings** - production-quality code
- **79KB compiled binary** - lightweight and fast

[See full changelog](#mojo-development)

## ✨ Key Features

- **🚀 Ultra-High Performance**: Mojo-accelerated backend delivers **887K-981K vectors/second** (2.9x faster than NumPy!)
- **🎯 Quality Preservation**: >99.99% cosine similarity retention after compression
- **💾 Massive Compression**: 75% reduction in storage and transfer costs
- **🔧 Multiple Backends**: Automatic fallback Mojo → Cython → NumPy with seamless backend selection
- **📊 Rich Visualizations**: Animated demos and performance charts
- **🛠️ CLI Tools**: Easy compression, evaluation, and benchmarking
- **📈 Benchmarking Suite**: Comprehensive throughput and quality metrics
- **🔄 Production Ready**: Zero-warning clean code, tested and verified

## 🏗️ Architecture

Vectro uses per-vector int8 quantization with automatic scale normalization:
- **Scale Calculation**: `scale = max_abs(vector) / 127`
- **Quantization**: `q = round(vector / scale)` → int8
- **Reconstruction**: `reconstructed = q * scale`
- **Fallback Chain**: Mojo (fastest, in progress) → Cython (production-ready) → NumPy (reliable) → PQ (memory-efficient)

## 📊 Performance Benchmarks

| Backend | Throughput | Quality Retention | Status |
|---------|------------|-------------------|--------|
| **Mojo** | **887K-981K vec/s** | **>99.99%** | **✅ Production Ready!** |
| Cython  | 328K vec/s | >99.99% | ✅ Production |
| NumPy   | 306K vec/s | >99.99% | ✅ Fallback |
| PQ      | 7K vec/s   | >99.9%  | ✅ Memory-efficient |

**🔥 Mojo backend delivers 2.9-3.2x speedup over NumPy with <1% error!**

*Benchmarks on 128D embeddings, Apple M3 Pro (Oct 2025)*

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- C compiler (gcc/clang) for Cython extension
- [Modular CLI](https://developer.modular.com/download) for Mojo backend (optional, in progress)

### Installation (Cython Backend - Production Ready)

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/vectro.git
   cd vectro
   ```

2. **Install in development mode (recommended for contributors)**
   ```bash
   pip install -e .
   ```

   Or install dependencies manually:
   ```bash
   pip install -r requirements.txt
   python setup.py build_ext --inplace
   ```

3. **Run tests**
   ```bash
   python -m pytest python/tests/
   ```

For CLI access, use `pip install -e .` or run commands with `python -m python.cli`.

### Mojo Backend Setup (High Performance - Now Available!)

**🔥 NEW: Mojo backend is production-ready with 2.9x speedup!**

1. **Install pixi** (Mojo package manager):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. **Navigate to vectro directory and activate environment**:
   ```bash
   cd vectro
   eval "$(pixi shell-hook)"  # Activates Mojo environment
   ```

3. **Verify Mojo installation**:
   ```bash
   mojo --version  # Should show: Mojo 0.25.7.0.dev2025102305
   ```

4. **Test Mojo backend**:
   ```bash
   # Run standalone test
   ./vectro_quantizer
   
   # Or rebuild from source
   mojo build src/vectro_standalone.mojo -o vectro_quantizer
   ```

5. **Use in Python** (automatic backend detection):
   ```python
   from python.interface import quantize_embeddings, get_backend_info
   
   # Check available backends
   print(get_backend_info())
   # Output: {'mojo': True, 'cython': False, 'numpy': True}
   
   # Mojo will be used automatically if available
   result = quantize_embeddings(embeddings)  # Uses Mojo backend!
   ```

**Performance Verification:**
```bash
python test_integration.py
# Expected output: Mojo throughput: 887,390 vectors/sec (2.9x speedup)
```

### Basic Usage

```python
import numpy as np
from python.interface import quantize_embeddings, reconstruct_embeddings, get_backend_info

# Check available backends (Mojo will be used automatically if available)
print(get_backend_info())
# Output: {'mojo': True, 'cython': False, 'numpy': True, 'mojo_binary': '/path/to/vectro_quantizer'}

# Load your embeddings (shape: n_vectors, embedding_dim)
embeddings = np.random.randn(1000, 768).astype(np.float32)

# Compress (automatically uses Mojo backend for 2.9x speedup!)
compressed = quantize_embeddings(embeddings)  # backend="auto" (default)

# Or specify backend explicitly
compressed = quantize_embeddings(embeddings, backend="mojo")  # Force Mojo
compressed = quantize_embeddings(embeddings, backend="numpy") # Force NumPy

print(f"Original size: {embeddings.nbytes} bytes")
print(f"Compressed size: {compressed['q'].nbytes + compressed['scales'].nbytes} bytes")
print(f"Compression ratio: {1 - (compressed['q'].nbytes + compressed['scales'].nbytes) / embeddings.nbytes:.1%}")

# Reconstruct
reconstructed = reconstruct_embeddings(
    compressed['q'],
    compressed['scales'],
    compressed['dims']
)

# Check quality
from python.interface import mean_cosine_similarity
similarity = mean_cosine_similarity(embeddings, reconstructed)
print(f"Quality retention: {similarity:.4%}")  # >99.99% with Mojo backend
```

## 🖥️ CLI Usage

Vectro includes a powerful command-line interface for batch operations:

```bash
# Compress embeddings
vectro compress --in embeddings.npy --out compressed.npz

# Evaluate compression quality
vectro eval --orig embeddings.npy --comp compressed.npz

# Run benchmarks
vectro bench --n 5000 --d 768 --queries 100

# Visualize compression effects
vectro visualize --embeddings embeddings.npy
```

### Advanced CLI Options

```bash
# Product Quantization with streaming
vectro compress --backend pq --chunk-size 1000 \
    --in large_embeddings.npy --out compressed.v2

# Custom benchmark parameters
vectro bench --n 10000 --d 1536 --queries 500 --k 100
```

## 🎬 Animated Demos

Experience Vectro's performance with our interactive animated demonstrations:

```bash
# Run the comprehensive animated demo
python demos/animated_demo.py

# View performance comparisons
python demos/animation_viewer.py

# Run visual benchmarking with real-time charts
python demos/visual_bench.py --n 2000 --d 128 --duration 30
```

The animated demo showcases:
- Real-time backend performance comparisons (Mojo, Cython, NumPy, PQ)
- Progress bars during compression
- Quality metrics visualization
- Interactive performance charts with live updates

## 📊 Performance Benchmarks

| Backend | Throughput | Quality Retention | Status |
|---------|------------|-------------------|--------|
| Mojo    | TBD (target: 500K+ vec/s) | >99.99% | ✅ Implemented |
| Cython  | 328K vec/s | >99.99% | ✅ Production |
| NumPy   | 225K vec/s | >99.99% | ✅ Fallback |
| PQ      | 7K vec/s   | >99.9%  | ✅ Memory-efficient |

*Benchmarks on 128D embeddings, Apple M3 Pro*

## 📈 Benchmarking

Run comprehensive benchmarks to evaluate performance:

```python
from python.bench import run_benchmark

# Benchmark all backends
results = run_benchmark(
    n_vectors=5000,
    dimensions=768,
    n_queries=100,
    k=10
)

for backend, metrics in results.items():
    print(f"{backend}: {metrics['throughput']:.0f} vec/s, "
          f"quality: {metrics['mean_cosine']:.4%}")
```

## 🔧 Development Setup

For contributors and advanced users:

1. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pytest matplotlib scikit-learn  # Additional for testing
   ```

2. **Build in development mode**
   ```bash
   python setup.py develop
   ```

3. **Run full test suite**
   ```bash
   python -m pytest python/tests/ -v
   ```

4. **Generate sample data for testing**
   ```bash
   python data/generate_sample.py --n 1000 --d 768 --out test_embeddings.npy
   ```

## 🏗️ Project Structure

```
vectro/
├── src/
│   ├── vectro_standalone.mojo   # ✅ Mojo standalone module (PRODUCTION READY)
│   ├── quantizer_working.mojo   # Clean reference implementation
│   ├── quantizer_new.mojo       # SIMD-optimized version (2.7M vec/s)
│   ├── vectro_mojo/             # Mojo package structure
│   │   └── __init__.mojo       # Package interface
│   ├── quantizer.mojo          # Original implementation
│   └── test.mojo               # Mojo unit tests
├── python/
│   ├── interface.py      # Main API with Mojo/Cython/NumPy backend selection
│   ├── cli.py           # Command-line interface
│   ├── bench.py         # Benchmarking suite
│   ├── pq.py            # Product Quantization implementation
│   ├── storage.py       # Storage utilities
│   ├── visualize.py     # Visualization tools
│   └── tests/           # Unit tests
├── vectro_quantizer        # ✅ Compiled Mojo binary (79KB, ready to use!)
├── test_integration.py     # Python integration tests & benchmarks
├── demos/
│   ├── animated_demo.py    # Interactive demonstrations
│   ├── animation_viewer.py # Demo viewer
│   └── visual_bench.py     # Real-time visual benchmarking
├── docs/
│   ├── MOJO_COMPLETE.md        # Mojo integration complete guide
│   ├── MOJO_PACKAGE_BUILD.md   # Package build documentation
│   ├── WARNINGS_FIXED.md       # Warning fixes details
│   └── STATUS_FINAL.md         # Overall project status
├── bin/
│   └── vectro          # CLI entrypoint (dev)
├── pixi.toml           # Mojo development environment (configured)
├── pyproject.toml      # Modern Python packaging
├── setup.py           # Build configuration
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🤝 Integrations

### Vector Databases

**Qdrant (decompress then index):**
```python
import numpy as np
from python.interface import reconstruct_embeddings

# Load compressed embeddings
compressed = np.load('embeddings_compressed.npz')
reconstructed = reconstruct_embeddings(
    compressed['q'],
    compressed['scales'],
    compressed['dims']
)

# Index in Qdrant
# client.upload_collection(collection_name, vectors=reconstructed.tolist())
```

**Weaviate (pre-compress workflow):**
```python
# Compress before storing
compressed = quantize_embeddings(embeddings)

# Store compressed bytes as metadata
# weaviate_client.data_object.create({
#     'compressed_q': compressed['q'].tolist(),
#     'scales': compressed['scales'].tolist(),
#     'dims': compressed['dims']
# })
```

## 📚 API Reference

### Core Functions

- `quantize_embeddings(embeddings)` → dict with compressed data
- `reconstruct_embeddings(q, scales, dims)` → reconstructed float32 array
- `mean_cosine_similarity(orig, recon)` → quality metric

### CLI Commands

- `vectro compress` - Compress embedding files
- `vectro eval` - Evaluate compression quality
- `vectro bench` - Run performance benchmarks
- `vectro visualize` - Generate quality plots

## 🧪 Testing

```bash
# Run unit tests
python -m pytest python/tests/

# Run with coverage
python -m pytest python/tests/ --cov=python/

# Generate test data
python data/generate_sample.py --n 500 --d 128 --out test_data.npy
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`python -m pytest`)
5. Update documentation
6. Submit a pull request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints for new functions
- Include docstrings for public APIs
- Maintain >99% test coverage
- Update benchmarks when adding new backends

### Current Priorities

- ✅ **Mojo Backend**: COMPLETE! SIMD-accelerated implementation delivering 887K-981K vec/s
- ✅ **Python Integration**: COMPLETE! Automatic backend selection in `python/interface.py`
- ✅ **Zero Warnings**: All Mojo code compiles cleanly with no warnings
- 🔄 **Production Deployment**: Package for PyPI distribution
- 🔄 **Documentation**: API documentation and tutorials
- 🔄 **Performance**: Further SIMD optimizations (target: 1M+ vec/s)
- 🔄 **Integrations**: Add more vector database adapters

### Mojo Development

**Mojo backend is production-ready!** 🎉

To work on or test the Mojo backend:
```bash
# Activate pixi environment
cd vectro
eval "$(pixi shell-hook)"

# Verify Mojo works
mojo --version

# Test standalone module
mojo run src/vectro_standalone.mojo
# Expected: Throughput: 981,932 vectors/sec

# Rebuild binary
mojo build src/vectro_standalone.mojo -o vectro_quantizer

# Run Python integration test
python test_integration.py
# Expected: Mojo throughput: 887,390 vectors/sec (2.9x speedup)

# Test all working Mojo files
mojo run src/quantizer_working.mojo  # Basic implementation
mojo run src/quantizer_new.mojo      # SIMD-optimized (2.7M vec/s!)
```

**Performance Achievements:**
- ✅ Standalone module: 887K-981K vectors/sec
- ✅ SIMD-optimized: 2.7M quantize/sec, 7.8M reconstruct/sec
- ✅ Accuracy: <1% reconstruction error (0.31% average)
- ✅ Zero compilation warnings
- ✅ Automatic Python backend detection

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with Cython for high-performance computing
- Inspired by modern vector compression techniques
- Designed for production ML pipelines

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/vectro/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/vectro/discussions)
- **Documentation**: See this README and inline code documentation

## 🚀 Next Steps & Roadmap

### Immediate Next Steps (Ready to Implement)

#### 1. Package Distribution (High Priority)
- [ ] **PyPI Package**: Create wheel distribution with Mojo binary
  - Package `vectro_quantizer` binary with pip install
  - Add platform-specific wheels (macOS arm64, x86_64, Linux)
  - Update `setup.py` to include binary in package data
  
- [ ] **CI/CD Pipeline**: Automated testing and releases
  - GitHub Actions for testing on push
  - Automated benchmark regression tests
  - Automatic PyPI releases on version tags

#### 2. Documentation & Tutorials
- [ ] **API Documentation**: Generate with Sphinx
  - Full API reference for all backends
  - Performance tuning guide
  - Backend selection best practices
  
- [ ] **Tutorial Notebooks**: Interactive Jupyter notebooks
  - Getting started with Vectro
  - RAG system integration example
  - Vector database workflows
  - Performance optimization guide

#### 3. Performance Optimizations (Target: 1M+ vec/s)
- [ ] **Batch Processing**: Optimize for larger batches
  - Implement efficient batch quantization in Mojo
  - Parallelize with `algorithm.parallelize`
  - Target: 1M+ vectors/sec for large batches
  
- [ ] **SIMD Enhancements**: Further vectorization
  - Use `algorithm.vectorize` more extensively
  - Optimize memory layouts for cache efficiency
  - Profile and eliminate bottlenecks

#### 4. Production Features
- [ ] **Streaming API**: Handle datasets larger than memory
  - Chunk-based processing
  - Progress callbacks
  - Memory-efficient iteration
  
- [ ] **Compression Profiles**: Pre-configured settings
  - `fast` (Mojo, minimal checks)
  - `balanced` (auto backend, good accuracy)
  - `quality` (slower, maximum accuracy)
  
- [ ] **Error Handling**: Robust error messages
  - Graceful backend fallback with warnings
  - Detailed error messages for debugging
  - Validation utilities

#### 5. Vector Database Integrations
- [ ] **Qdrant Plugin**: Native Qdrant integration
- [ ] **Weaviate Adapter**: Direct Weaviate support
- [ ] **Pinecone Integration**: Pinecone-compatible format
- [ ] **Milvus Support**: Milvus collection helpers

#### 6. Advanced Features
- [ ] **Multi-precision Support**: int4, int16 quantization
- [ ] **Adaptive Quantization**: Per-vector precision selection
- [ ] **GPU Acceleration**: CUDA/Metal backends
- [ ] **Quantization-Aware Search**: Direct search on quantized data

### Long-term Vision

#### Q1 2026: Production Hardening
- Extensive real-world testing
- Performance profiling and optimization
- Security audit and best practices
- Production deployment guides

#### Q2 2026: Ecosystem Integration
- Major vector database plugins
- LangChain integration
- LlamaIndex support
- RAG framework adapters

#### Q3 2026: Advanced Compression
- Learned quantization (neural codecs)
- Hybrid compression schemes
- Domain-specific optimizations
- Quantization for specific embedding models

#### Q4 2026: Scale & Performance
- Distributed quantization
- Cloud-native deployments
- Multi-GPU support
- Edge device optimization

### How to Contribute

We welcome contributions in all areas! Priority areas:

**High Impact:**
- PyPI packaging and distribution
- Performance benchmarks on different hardware
- Vector database integration examples
- Tutorial content and documentation

**Medium Impact:**
- Additional backend implementations
- CLI improvements and features
- Visualization enhancements
- Test coverage improvements

**Research & Exploration:**
- Novel compression algorithms
- Quantization-aware search methods
- Domain-specific optimizations
- Hardware-specific tuning

See [Contributing](#contributing) section for guidelines.

---

**Ready to compress your embeddings with 2.9x speedup?** 🚀

```bash
# Quick start with Mojo backend
git clone https://github.com/yourusername/vectro.git
cd vectro
pip install -r requirements.txt

# Install Mojo (optional, for maximum performance)
curl -fsSL https://pixi.sh/install.sh | bash
eval "$(pixi shell-hook)"

# Test it out
python test_integration.py
```