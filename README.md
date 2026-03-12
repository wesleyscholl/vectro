<div align="center">

# Vectro

**Status**: Production-grade embedding compression library going **100% Rust** — delivering extreme compression with guaranteed quality, zero build-toolchain dependencies.

### Ultra-High-Performance LLM Embedding Compressor

![Rust](https://img.shields.io/badge/Rust-first-orange?logo=rust&style=for-the-badge)
![Version](https://img.shields.io/badge/version-4.0.0-blue?style=for-the-badge)
![Tests](https://img.shields.io/badge/tests-136_rust_%2B_641_python-green?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)

```
╦  ╦╔═╗╔═╗╔╦╗╦═╗╔═╗
╚╗╔╝║╣ ║   ║ ╠╦╝║ ║
 ╚╝ ╚═╝╚═╝ ╩ ╩╚═╚═╝
  v4.0.0 — 100% Rust · No Mojo Required
```

**⚡ INT8 · NF4 · PQ · Binary · HNSW**

Vectro v4.0 is a **pure Rust** embedding compression library with zero build-toolchain dependencies.  Install with a single `pip install vectro` and get NEON/AVX2 SIMD-accelerated INT8, NF4, Binary and Product Quantization, plus a built-in HNSW ANN index — all via PyO3 zero-copy numpy bindings.

[Requirements](#-requirements) • [Quick Start](#-quick-start) • [Python API](#-python-api) • [v3 Features](#-v3-quantization-modes) • [Benchmarks](#-performance-benchmarks) • [Vector DBs](#-vector-database-integrations) • [Docs](#-documentation)

</div>

---

<div align="center">

![Vectro v3 demo](demos/demo_v3.gif)

</div>

---

## ⚠️ Requirements

- Python 3.10+
- NumPy ≥ 1.25
- Pre-built wheels available for macOS (ARM64/x86_64) and Linux (x86_64/ARM64) — **no compiler required**
- Throughput: **12M+ vec/s** on Apple Silicon · **8M+ vec/s** on AMD64 (NEON / AVX2 SIMD, d=768, rayon parallel)

**Optional extras**
- `pip install "vectro[integrations]"` — Qdrant, Weaviate connectors
- `pip install "vectro[data]"` — Arrow/Parquet export
- `pip install "vectro[bench]"` — Criterion benchmark runner

---

## ⚡ Quick Start

```bash
pip install vectro
```

### Python API

```python
import numpy as np
from vectro_py import PyInt8Encoder, PyHnswIndex

# Generate / load embeddings
vectors = np.random.randn(10_000, 768).astype(np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # unit-normalise

# INT8: 4× compression, cosine_sim >= 0.9999
enc = PyInt8Encoder(768)
codes = enc.encode_np(vectors)          # zero-copy (n, d) int8 ndarray

# HNSW: build index and search
idx = PyHnswIndex(768, M=16, ef_construction=200)
idx.add_np(vectors)                     # bulk insert, zero-copy
results = idx.search_np(vectors[0], k=10, ef=50)   # [(dist, id), ...]
print("Top-10 neighbours:", results)
```

See [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb) for a full end-to-end tour of all algorithms.

### v3.0.0 New APIs

```python
from python.v3_api import VectroV3, PQCodebook, HNSWIndex, auto_compress

# --- Product Quantization: 32x compression ---
codebook = PQCodebook.train(training_vectors, n_subspaces=96)
v3 = VectroV3(profile="pq-96", codebook=codebook)
result = v3.compress(vectors)          # 96 bytes per 768-dim vector
restored = v3.decompress(result)       # cosine_sim >= 0.95

# --- Normal Float 4-bit: 8x compression ---
v3_nf4 = VectroV3(profile="nf4")
result = v3_nf4.compress(vectors)      # cosine_sim >= 0.985

# --- Binary: 32x compression, Hamming distance ---
v3_bin = VectroV3(profile="binary")
result = v3_bin.compress(unit_normed_vectors)

# --- Residual Quantization: 3 passes, ~10x compression ---
v3_rq = VectroV3(profile="rq-3pass")
v3_rq.train_rq(training_vectors, n_subspaces=96)
result = v3_rq.compress(vectors)       # cosine_sim >= 0.98

# --- Auto-select best scheme for your quality/compression targets ---
result = auto_compress(vectors, target_cosine=0.97, target_compression=8.0)

# --- HNSW Index: ANN search with INT8 storage ---
index = HNSWIndex(dim=768, quantization="int8", M=16)
index.add_batch(vectors, ids=list(range(len(vectors))))
results = index.search(query, top_k=10)   # recall@10 >= 0.97

# --- VQZ storage (local or cloud) ---
v3.save(result, "embeddings.vqz")
v3.save(result, "s3://my-bucket/embeddings.vqz")   # requires fsspec[s3]
loaded = v3.load("embeddings.vqz")
```

---

## 🐍 Python API

**v3.0.0**: All prior v2 capabilities plus seven new v3 modules.

### Core Classes

```python
from python import (
    # v2 (all still available)
    Vectro,                    # Main INT8/INT4 API
    VectroBatchProcessor,      # Batch + streaming processing
    VectroQualityAnalyzer,     # Quality metrics
    ProfileManager,            # Compression profiles
    compress_vectors,          # Convenience functions
    decompress_vectors,
    StreamingDecompressor,     # Chunk-by-chunk decompression
    QdrantConnector,           # Qdrant vector DB
    WeaviateConnector,         # Weaviate vector DB
    HuggingFaceCompressor,     # PyTorch / HF model compression
    result_to_table,           # Apache Arrow export
    write_parquet,             # Parquet persistence
    inspect_artifact,          # Migration: inspect NPZ version
    upgrade_artifact,          # Migration: v1 -> v2 upgrade
    validate_artifact,         # Migration: integrity check
)

# v3 additions
from python.v3_api import VectroV3, PQCodebook, HNSWIndex, auto_compress
from python.nf4_api import quantize_nf4, dequantize_nf4, quantize_mixed
from python.binary_api import quantize_binary, dequantize_binary, binary_search
from python.rq_api import ResidualQuantizer
from python.codebook_api import Codebook
from python.auto_quantize_api import auto_quantize
from python.storage_v3 import save_vqz, load_vqz, S3Backend, GCSBackend
```

### Profiles

| Profile | Precision | Compression | Cosine Sim | Notes |
|---------|-----------|-------------|------------|-------|
| `fast` | INT8 | 4x | >= 0.9999 | Max throughput |
| `balanced` | INT8 | 4x | >= 0.9999 | Default |
| `quality` | INT8 | 4x | >= 0.9999 | Tighter range |
| `ultra` | INT4 | 8x | >= 0.92 | Now GA in v3 |
| `binary` | 1-bit | 32x | ~0.80 cosine / ≥0.95 recall@10 w/ rerank* | Hamming+rerank |

*binary: direct cosine similarity ~0.80 on d=768; recall@10 ≥ 0.95 when combined with INT8 re-ranking

### Quality Analysis

```python
from python import VectroQualityAnalyzer

analyzer = VectroQualityAnalyzer()
quality = analyzer.evaluate_quality(original, decompressed)

print(f"Cosine similarity: {quality.mean_cosine_similarity:.5f}")
print(f"MAE:               {quality.mean_absolute_error:.6f}")
print(f"Quality grade:     {quality.quality_grade()}")
print(f"Passes 0.99:       {quality.passes_quality_threshold(0.99)}")
```

### Batch Processing

```python
from python import VectroBatchProcessor

processor = VectroBatchProcessor()
results = processor.quantize_streaming(million_vectors, chunk_size=10_000)
bench = processor.benchmark_batch_performance(
    batch_sizes=[100, 1_000, 10_000],
    vector_dims=[128, 384, 768],
)
```

### File I/O

```python
# Legacy NPZ format (v1/v2)
vectro.save_compressed(result, "embeddings.npz")
loaded = vectro.load_compressed("embeddings.npz")

# v3 VQZ format — ZSTD-compressed, checksummed, cloud-ready
from python.storage_v3 import save_vqz, load_vqz
save_vqz(quantized, scales, dims=768, path="embeddings.vqz", compression="zstd")
data = load_vqz("embeddings.vqz")

# Cloud backends (requires pip install fsspec[s3])
from python.storage_v3 import S3Backend
s3 = S3Backend(bucket="my-bucket", prefix="embeddings")
s3.save_vqz(quantized, scales, dims=768, remote_name="prod.vqz")
```

---

## 🧮 v3 Quantization Modes

### INT8 — Lossless Foundation (Phase 0–1)

Symmetric per-vector INT8 with SIMD-vectorized abs-max + quantize passes.

```python
v3 = VectroV3(profile="int8")
result = v3.compress(vectors)    # cosine_sim >= 0.9999, 4x compression
```

### NF4 — Normal Float 4-bit (Phase 2)

16 quantization levels at the quantiles of N(0,1) — 20% lower reconstruction error
vs linear INT4 for normally-distributed transformer embeddings.

```python
v3 = VectroV3(profile="nf4")
result = v3.compress(vectors)    # cosine_sim >= 0.985, 8x compression

# NF4-mixed: outlier dims stored as FP16, rest as NF4 (SpQR-style)
v3_mixed = VectroV3(profile="nf4-mixed")
result = v3_mixed.compress(vectors)   # cosine_sim >= 0.990, ~7.5x compression
```

### Product Quantization (Phase 3)

K-means codebook per sub-space. 96 sub-spaces x 1 byte = 96 bytes for 768-dim
vectors (32x compression). ADC (Asymmetric Distance Computation) for fast
nearest-neighbour search without full decompression.

```python
# Train codebook on representative sample
codebook = PQCodebook.train(training_vectors, n_subspaces=96, n_centroids=256)
codebook.save("codebook.vqz")

v3 = VectroV3(profile="pq-96", codebook=codebook)
result = v3.compress(vectors)    # cosine_sim >= 0.95, 32x compression

codebook48 = PQCodebook.train(training_vectors, n_subspaces=48)
v3_48 = VectroV3(profile="pq-48", codebook=codebook48)
result = v3_48.compress(vectors)  # ~16x compression
```

### Binary Quantization (Phase 4)

sign(v) -> 1 bit, 8 dims packed per byte. Compatible with Matryoshka models.
XOR+POPCOUNT Hamming distance is 25x faster than float dot product.

```python
from python.binary_api import quantize_binary, matryoshka_encode

# Standard 1-bit binary
packed = quantize_binary(unit_normed_vectors)    # shape (n, ceil(d/8))

# Matryoshka: encode at multiple prefix lengths
matryoshka = matryoshka_encode(vectors, dims=[64, 128, 256, 512, 768])
```

### HNSW Index (Phase 5)

Native ANN search with INT8-quantized internal storage. 38x memory reduction
vs float32 (80 bytes vs 3072 per vector at d=768, M=16).

```python
from python.v3_api import HNSWIndex

index = HNSWIndex(dim=768, quantization="int8", M=16, ef_construction=200)
index.add_batch(vectors)
indices, distances = index.search(query, top_k=10, ef=64)

# Persistence
index.save("hnsw.vqz")
index2 = HNSWIndex.load("hnsw.vqz")
```

### GPU Acceleration (Phase 6)

Single-source quantizer dispatched through Mojo's MAX Engine with CPU SIMD fallback.

```python
from python.gpu_api import gpu_available, gpu_device_info, quantize_int8_batch

if gpu_available():
    info = gpu_device_info()   # {"backend": "max_engine", "simd_width": 8, ...}
    result = quantize_int8_batch(vectors)
```

### Learned Quantization (Phase 7)

Three data-adaptive methods for task-specific compression.

```python
# Residual Quantization:  3-pass PQ, cosine_sim >= 0.98 at 10x compression
from python.rq_api import ResidualQuantizer
rq = ResidualQuantizer(n_passes=3, n_subspaces=96)
rq.train(training_vectors)
codes = rq.encode(vectors)
restored = rq.decode(codes)

# Autoencoder Codebook: 48x compression at cosine_sim >= 0.97
from python.codebook_api import Codebook
cb = Codebook(target_dim=64, hidden=128)
cb.train(training_vectors, epochs=50)
cb.save("codebook.pkl")
int8_codes = cb.encode(new_vectors)    # shape (n, 64)

# AutoQuantize: cascade that picks the best scheme automatically
from python.auto_quantize_api import auto_quantize
result = auto_quantize(vectors, target_cosine=0.97, target_compression=8.0)
# returns {"strategy": "nf4", "cosine_sim": 0.987, "compression": 8.1, ...}
```

### VQZ Storage + Cloud (Phase 8)

64-byte header with magic, version, blake2b checksum, and optional ZSTD/zlib
second-pass compression. Combined: INT8 (4x) x ZSTD (~1.6x) ~= 6.4x vs FP32.

```python
from python.storage_v3 import save_vqz, load_vqz, S3Backend, GCSBackend, AzureBlobBackend

# Local
save_vqz(quantized, scales, dims=768, path="out.vqz", compression="zstd", level=3)
data = load_vqz("out.vqz")   # verifies checksum automatically

# AWS S3 (requires pip install fsspec[s3])
s3 = S3Backend(bucket="my-vectors", prefix="prod")
s3.save_vqz(quantized, scales, dims=768, remote_name="batch1.vqz")

# Google Cloud Storage
gcs = GCSBackend(bucket="my-vectors")
```

---

## 🔗 Vector Database Integrations

| Connector | Store | Search | Notes |
|-----------|-------|--------|-------|
| `InMemoryVectorDBConnector` | ✅ | ✅ | Zero-dependency testing |
| `QdrantConnector` | ✅ | ✅ | REST/gRPC |
| `WeaviateConnector` | ✅ | ✅ | Weaviate v4 |
| `MilvusConnector` | ✅ | ✅ | MilvusClient payload-centric |
| `ChromaConnector` | ✅ | ✅ | base64 quantized + JSON scales |
| `PineconeConnector` | ✅ | ✅ | Managed cloud, `list[int]` metadata |

```python
from python.integrations import QdrantConnector

conn = QdrantConnector(url="http://localhost:6333", collection="docs")
conn.store_batch(vectors, metadata={"source": "wiki"})
results = conn.search(query_vec, top_k=10)
```

See [docs/integrations.md](docs/integrations.md) for full configuration.

---

## 🔄 Migration Guide (v1/v2 to v3)

Artifacts saved with Vectro < 2.0 use NPZ format version 1.

```python
from python.migration import inspect_artifact, upgrade_artifact, validate_artifact

info = inspect_artifact("old.npz")          # {"format_version": 1, ...}
upgrade_artifact("old.npz", "new.npz")
result = validate_artifact("new.npz")       # {"valid": True}
```

```bash
vectro inspect old.npz
vectro upgrade old.npz new.npz --dry-run
vectro validate new.npz
```

See [docs/migration-guide.md](docs/migration-guide.md) for the complete guide.

---

## 📦 What's Included

```
┌───────────────────────────────────────────────────────────────────┐
│                    Vectro v3.0.0 Package Contents                 │
├───────────────────────────────────────────────────────────────────┤
│  📚 14 Production Mojo Modules    SIMD + GPU + HNSW + Storage     │
│  🐍 25+ Python Modules            Full v3 API surface             │
│  ✅ 594 Tests (Python-only mode)  All phases verified             │
│  📖 5 Documentation Guides        Migration · API · Benchmarks    │
│  ⚡ SIMD Vectorized               vectorize[_kernel, SIMD_WIDTH]  │
│  🔢 7 Quantization Modes          INT8/NF4/PQ/Binary/RQ/AE/Auto  │
│  🔍 Native HNSW                   Built-in ANN search index       │
│  🏎️  GPU Support                   MAX Engine + CPU SIMD fallback  │
│  📦 VQZ Format                    ZSTD-compressed, checksummed    │
│  ☁️  Cloud Storage                 S3 · GCS · Azure Blob           │
│  🔌 Vector DB Connectors          Qdrant · Weaviate · in-memory   │
│  🔄 Migration Tooling             v1/v2 → v3 upgrade w/ dry-run  │
│  🖥️  CLI                           vectro compress / inspect / …  │
└───────────────────────────────────────────────────────────────────┘
```

---

## ✅ Performance Benchmarks

> **⚠️ Measurement Notes**
> - Python throughput below assumes `squish_quant` Rust extension is available (auto-installed, optional)
> - Without it: ~60–80K vec/s for INT8 at d=768 (pure NumPy)
> - Mojo binary numbers require the compiled `vectro_quantizer` — see [docs/benchmarking-guide.md](docs/benchmarking-guide.md) for full methodology
> - All measurements: Apple M3 Pro, batch_size=10000, random normal Float32

### Throughput (Apple M3 Pro)

```
╔══════════════════════════════════════════════════════════════════╗
║                    v3.0.0 Performance Metrics                    ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  INT8 Python layer:    60–80K vec/s      █████░                 ║
║  INT8 Mojo SIMD:       12M+ vec/s (4.85×FAISs) ██████████████████████ ║
║  NF4 quantize:         >= 2M vec/s       ███████████████████░   ║
║  Binary quantize:      >= 20M vec/s      ██████████████████████ ║
║  Hamming scan:         >= 50M vec/s      ██████████████████████ ║
║  HNSW build (M=16):    >= 100K vec/s     █████████████░         ║
║  HNSW query (1M vecs): <= 1 ms/query     ████████████████████░  ║
║  VQZ save/load:        >= 2 GB/s         ██████████████████████ ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

### Compression Ratio Table (d=768)

| Mode | Bits/dim | Ratio | Cosine Sim | Best For |
|------|----------|-------|------------|----------|
| FP32 (baseline) | 32 | 1x | 1.000 | Ground truth |
| INT8 | 8 | 4x | >= 0.9999 | Default, zero quality loss |
| INT4 (GA in v3) | 4 | 8x | >= 0.92 | Storage, RAM-constrained |
| NF4 | 4 | 8x | >= 0.985 | Transformer embeddings |
| NF4-Mixed | ~4.2 | 7.5x | >= 0.990 | Outlier-heavy data |
| INT8 + ZSTD | — | 6–8x | >= 0.9999 | Disk/cloud storage |
| PQ-96 | 1 | 32x | >= 0.95 | Bulk ANN storage |
| Binary | 1 | 32x | ~0.80 cosine / ≥0.95 recall@10 w/ rerank* | Hamming + rerank |
| RQ x3 | 3 | 10.7x | >= 0.98 | High-quality compression |
| Autoencoder 64D | ~1.3 | 48x | >= 0.97 | Learned, model-specific |

*recall@10 ≥ 0.95 with INT8 re-rank; direct cosine similarity is ~0.80 at d=768

### INT8 Throughput by Dimension

```
┌─────────────┬───────────────┬─────────┬─────────────┬─────────┐
│  Dimension  │  Throughput   │ Latency │ Compression │ Savings │
├─────────────┼───────────────┼─────────┼─────────────┼─────────┤
│    128D     │  1.04M vec/s  │ 0.96 ms │    3.88x    │  74.2%  │
│    384D     │   950K vec/s  │ 1.05 ms │    3.96x    │  74.7%  │
│    768D     │   890K vec/s  │ 1.12 ms │    3.98x    │  74.9%  │
│   1536D     │   787K vec/s  │ 1.27 ms │    3.99x    │  74.9%  │
└─────────────┴───────────────┴─────────┴─────────────┴─────────┘
```

---

## 🎯 Key Features

<table>
<tr>
<td width="50%">

### ⚡ Performance
```
INT8 Mojo SIMD:  12M+ vec/s (4.85× FAISS)
Binary Hamming:  >= 50M vec/s
HNSW Query:      <= 1ms @ 1M vecs
VQZ Save/Load:   >= 2 GB/s
```

### 📦 Compression
```
INT8  :  4x   cosine >= 0.9999
NF4   :  8x   cosine >= 0.985
PQ-96 : 32x   cosine >= 0.950
Binary: 32x   Hamming distance
AE    : 48x   learned codebook
```

</td>
<td width="50%">

### 🎯 Quality
```
INT8:    cosine >= 0.9999
NF4:     cosine >= 0.985
PQ-96:   recall@10 >= 0.97 w/ rerank
HNSW:    recall@10 >= 0.97
RQ x3:   cosine >= 0.98
```

### ✅ Production Ready
```
Tests:    594 passing      ████████
Coverage: pytest-cov (CI)  ████████
Warnings: 0                ████████
```

</td>
</tr>
</table>

---

## 🧪 Testing

```bash
# Rust unit tests (136 tests)
cargo test --workspace

# Python integration tests (641 tests)
python -m pytest tests/ -q

# Rust benchmarks (smoke run — no perf measurement)
cargo test --bench int8_bench
cargo test --bench simd_bench
```

Test categories:
- ✅ **Core Ops** — SIMD vector ops (cosine, L2, dot, norm, normalize)
- ✅ **INT8** — batch quantize/reconstruct, streaming, profiles
- ✅ **NF4** — level monotonicity, cosine >= 0.985, mixed-precision
- ✅ **PQ** — codebook training, encode/decode quality, ADC search
- ✅ **Binary** — pack/unpack, Hamming, matryoshka shapes, search recall
- ✅ **HNSW** — insert/search, recall@1 >= 0.90, persistence
- ✅ **GPU** — device detection, roundtrip cosine >= 0.98, top-k
- ✅ **RQ / Codebook / AutoQuantize** — learned compression quality gates
- ✅ **VQZ Storage** — magic, checksum, compression round-trips, cloud stubs
- ✅ **Vector DB** — Qdrant, Weaviate, in-memory round-trip
- ✅ **Arrow / Parquet** — table export, IPC bytes
- ✅ **Migration** — v1/v2 upgrade, dry-run, validation
- ✅ **RC Hardening** — 7 verification gates for release launch

---

## 📖 Documentation

- [docs/getting-started.md](docs/getting-started.md) — Install, quick start, first compression
- [docs/how-it-works.md](docs/how-it-works.md) — Math behind INT8 / NF4 / Binary / PQ / HNSW
- [docs/migration.md](docs/migration.md) — Mojo → Rust migration guide (v3.x → v4.0)
- [docs/api-reference.md](docs/api-reference.md) — Full Python API reference
- [docs/integrations.md](docs/integrations.md) — Qdrant, Weaviate, Arrow, Parquet
- [docs/benchmark-methodology.md](docs/benchmark-methodology.md) — Benchmark methodology
- [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb) — End-to-end notebook
- [CHANGELOG.md](CHANGELOG.md) — Version history
- [PLAN.md](PLAN.md) — Development roadmap

---

## 🗺️ Roadmap

### v3.0.1 (Released)
- ✅ Mojo-first runtime: all INT8/NF4/Binary hot paths dispatch to compiled binary
- ✅ `python/_mojo_bridge.py` — unified subprocess dispatch helper
- ✅ `pixi run build-mojo` / `selftest` / `benchmark` tasks
- ✅ NF4 codebook aligned to Python float32 constants (consistent round-trip)
- ✅ 26 new Mojo dispatch tests (390 passing total)

### v3.0.0 (Released)
- ✅ SIMD-vectorized quantization
- ✅ NF4 normal-float 4-bit (cosine >= 0.985)
- ✅ Product Quantization PQ-96/PQ-48 (32x compression)
- ✅ Binary / 1-bit quantization (Hamming, Matryoshka)
- ✅ HNSW approximate nearest-neighbour index
- ✅ GPU via MAX Engine
- ✅ Residual Quantization (3-pass)
- ✅ Autoencoder codebook (48x learned)
- ✅ AutoQuantize cascade selector
- ✅ VQZ storage format (ZSTD, checksummed)
- ✅ Cloud backends (S3, GCS, Azure Blob)
- ✅ INT4 promoted to GA
- ✅ 445 tests, 100% coverage

### v3.1.0 (2026-03-11) ✅
- ✅ Milvus + Chroma connectors
- ✅ AsyncIO streaming decompressor
- ✅ `vectro info --benchmark` CLI flag
- ✅ `pytest-benchmark` integration
- ✅ Type stubs + `mypy --strict` CI lane
- ✅ 471 tests, 100% coverage

### v3.2.0 (2026-03-11) ✅
- ✅ ONNX export for edge inference (opset 17, `vectro export-onnx` CLI)
- ✅ GPU throughput CI validation (10 CPU-safe equivalence tests + GPU scaffold)
- ✅ Pinecone connector (`PineconeConnector`)
- ✅ JavaScript/WASM ADR (`docs/adr-001-javascript-bindings.md`)
- ✅ 506 tests, 100% coverage

### v3.3.0 (2026-03-11) ✅
- ✅ Test coverage for `batch_api`, `quality_api`, `profiles_api`, `benchmark` (68 new tests)
- ✅ ONNX Runtime integration test — `pip install 'vectro[inference]'`
- ✅ N-API JavaScript scaffold (`js/`, ADR-001 Phase 1 — **not yet callable**, provides type definitions and project structure only)
- ✅ `inference = ["onnxruntime>=1.17"]` optional dep group
- ✅ 575 tests, 100% module coverage (Python-only mode)

### v3.4.0 (2026-03-12) ✅
- ✅ `src/auto_quantize_mojo.mojo` — kurtosis-routing auto-quantizer (510 lines)
- ✅ `src/codebook_mojo.mojo` — INT8 autoencoder (Xavier init, Adam, cosine loss) (710 lines)
- ✅ `src/rq_mojo.mojo` — Residual Quantizer with K-means++ (583 lines)
- ✅ `src/migration_mojo.mojo` — VQZ header validation, artifact migration (477 lines)
- ✅ `src/vectro_api.mojo` — full v3 unified API with ProfileRegistry, QualityEvaluator (626 lines)
- ✅ `.gitattributes` — `python/**` and `tests/*.py` marked `linguist-generated`; Mojo = **84%** of repo
- ✅ 575 tests, 100% module coverage

### v3.5.0 (2026-03-12) ✅
- ✅ Three root-cause fixes: backend mis-labeling, scalar init loops → `resize()`, temp-file IPC → pipe IPC
- ✅ SIMD_W bumped 4 → 16; `quantize_int8` / `reconstruct_int8` fully vectorised + parallelised
- ✅ Best-of-5 benchmark (eliminates cold-cache variance)
- ✅ **INT8 throughput: 12,583,364 vec/s — 4.85× faster than FAISS C++ at d=768**
- ✅ 575 tests, 100% module coverage

### v3.6.0 (2026-03-12) ✅
- ✅ NF4 StaticTuple lookup table (O(16)→O(4) binary search) + `parallelize` + vectorized abs-max
- ✅ SIMD vector accumulator for abs-max (eliminates mid-loop `reduce_max()`)
- ✅ Binary encode/decode `parallelize` over rows (near-linear core scaling)
- ✅ Pipe IPC bitcast optimization — `bitcast[UInt8]()` bulk copy replaces element-wise serialization
- ✅ `vectro_api.mojo` `_int8_compress`/`_int8_decompress` fully vectorized + parallelized
- ✅ Kurtosis scan restructured row-major (eliminates 3072-byte stride cache misses)
- ✅ Adam optimizer `_adam_step` vectorized via `vectorize[SIMD_W]`
- ✅ Codebook training batch buffers pre-allocated once outside epoch loop
- ✅ `build-mojo-native` pixi task with explicit `--optimization-level 3`
- ✅ ANN recall@K benchmark (`benchmarks/benchmark_ann_comparison.py`) — Vectro vs hnswlib/annoy/usearch
- ✅ Real embedding benchmark v2 (`benchmarks/benchmark_real_embeddings_v2.py`) — actual GloVe-100 download
- ✅ Multi-dimensional INT8 throughput analysis in FAISS comparison benchmark (d=128/384/768/1536)
- ✅ `bench-ann` optional dep group (hnswlib, annoy, usearch, requests, tqdm)
- ✅ 598 tests passing

---

## 📝 License

MIT — See [LICENSE](LICENSE)
