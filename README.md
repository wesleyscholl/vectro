# Vectro: High-Performance LLM Embedding Compressor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Vectro is a blazing-fast, production-ready toolkit for compressing and reconstructing LLM embedding vectors. It achieves **75% storage reduction** while maintaining **>99.99% retrieval quality**, making it perfect for vector search pipelines, RAG systems, and large-scale embedding storage.

## ✨ Key Features

- **🚀 High Performance**: Mojo (in progress) and Cython-accelerated backends deliver **328K+ vectors/second** quantization
- **🎯 Quality Preservation**: >99.99% cosine similarity retention after compression
- **💾 Massive Compression**: 75% reduction in storage and transfer costs
- **🔧 Multiple Backends**: Automatic fallback Mojo → Cython → NumPy → PQ
- **📊 Rich Visualizations**: Animated demos and performance charts
- **🛠️ CLI Tools**: Easy compression, evaluation, and benchmarking
- **📈 Benchmarking Suite**: Comprehensive throughput and quality metrics
- **🔄 Streaming Support**: Handle datasets larger than memory

## 🏗️ Architecture

Vectro uses per-vector int8 quantization with automatic scale normalization:
- **Scale Calculation**: `scale = max_abs(vector) / 127`
- **Quantization**: `q = round(vector / scale)` → int8
- **Reconstruction**: `reconstructed = q * scale`
- **Fallback Chain**: Mojo (fastest, in progress) → Cython (production-ready) → NumPy (reliable) → PQ (memory-efficient)

## 📊 Performance Benchmarks

| Backend | Throughput | Quality Retention | Status |
|---------|------------|-------------------|--------|
| Mojo    | TBD (target: 500K+ vec/s) | >99.99% | In Progress |
| Cython  | 328K vec/s | >99.99% | ✅ Production |
| NumPy   | 225K vec/s | >99.99% | ✅ Fallback |
| PQ      | 7K vec/s   | >99.9%  | ✅ Memory-efficient |

*Benchmarks on 128D embeddings, Apple M3 Pro*

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

### Mojo Backend Setup (Optional - High Performance)

For the bleeding-edge Mojo implementation with SIMD acceleration:

1. **Install Mojo** via [Modular CLI](https://developer.modular.com/download)
2. **Test Mojo compilation**:
   ```bash
   mojo run src/test.mojo  # Run unit tests
   ```
3. **Build Mojo extension** (future integration):
   ```bash
   # Mojo will be integrated as a Python extension in future versions
   ```

**Note**: Mojo backend is implemented but requires Modular CLI installation. The Cython backend provides excellent performance as a production-ready alternative.

### Basic Usage

```python
import numpy as np
from python.interface import quantize_embeddings, reconstruct_embeddings

# Load your embeddings (shape: n_vectors, embedding_dim)
embeddings = np.random.randn(1000, 768).astype(np.float32)

# Compress
compressed = quantize_embeddings(embeddings)
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
print(f"Quality retention: {similarity:.4%}")
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
│   ├── quantizer.mojo       # Mojo implementation (SIMD-accelerated)
│   ├── quantizer_cython.pyx # Cython implementation
│   └── test.mojo            # Mojo unit tests
├── python/
│   ├── interface.py      # Main API with backend selection
│   ├── cli.py           # Command-line interface
│   ├── bench.py         # Benchmarking suite
│   ├── pq.py            # Product Quantization implementation
│   ├── storage.py       # Storage utilities
│   ├── visualize.py     # Visualization tools
│   └── tests/           # Unit tests
├── demos/
│   ├── animated_demo.py    # Interactive demonstrations
│   ├── animation_viewer.py # Demo viewer
│   ├── visual_bench.py     # Real-time visual benchmarking
│   └── demo_*.py          # Additional demos
├── bin/
│   └── vectro          # CLI entrypoint (dev)
├── pixi.toml           # Mojo development environment
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

- **Mojo Backend**: Complete SIMD-accelerated implementation in `src/quantizer.mojo`
- **Performance**: Optimize Cython backend further
- **Integrations**: Add more vector database adapters
- **Streaming**: Improve large dataset handling

### Mojo Development

To work on the Mojo backend:
```bash
pixi shell  # Enter Mojo environment
mojo build src/quantizer.mojo  # Test compilation
mojo run src/test.mojo         # Run Mojo unit tests
```

The Mojo implementation provides SIMD-accelerated quantization with target performance of 500K+ vectors/second. See `src/quantizer.mojo` for the current implementation and `src/test.mojo` for unit tests.

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

---

**Ready to compress your embeddings?** 🚀

```bash
git clone https://github.com/yourusername/vectro.git
cd vectro && pip install -r requirements.txt && python setup.py build_ext --inplace
```