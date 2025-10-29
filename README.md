# 🚀 Vectro - Ultra-High-Performance LLM Embedding Compressor

**98.2% Pure Mojo** | **787K-1.04M vectors/sec** | **3.98x Compression** | **99.97% Accuracy**

A Mojo-first vector quantization library for compressing LLM embeddings.

## ⚡ Quick Start

```bash
# Clone and build
git clone https://github.com/wesleyscholl/vectro.git
cd vectro
pixi install
pixi shell
mojo build src/vectro_standalone.mojo -o vectro_quantizer

# Run
./vectro_quantizer

# Test
mojo run tests/test_all_modules.mojo
```

## 📦 What's Included

- **8 Mojo Modules**: Complete quantization pipeline
- **Comprehensive Tests**: All modules tested
- **SIMD Optimized**: Native performance
- **Multiple Profiles**: Fast/Balanced/Quality modes

## 📖 Documentation

- [SIMPLE_MOJO_RELEASE.md](SIMPLE_MOJO_RELEASE.md) - How to release
- [MOJO_MODULES.md](MOJO_MODULES.md) - API reference
- [CHANGELOG.md](CHANGELOG.md) - Version history

## 🧪 Testing

```bash
mojo run tests/test_all_modules.mojo
```

Tests cover: vector ops, batch processing, quality metrics, compression profiles, edge cases, and performance.

## 📝 License

MIT - See LICENSE file
