# Getting Started with Vectro

Vectro is an ultra-high-performance LLM embedding compressor. It compresses floating-point
embedding vectors to INT8 or INT4 using scalar quantization, achieving 4–8× storage savings
with minimal semantic quality loss.

---

## Installation

### From PyPI

```bash
pip install vectro
```

### With optional extras

```bash
# Arrow/Parquet I/O
pip install "vectro[data]"

# Vector DB and ML framework integrations
pip install "vectro[integrations]"

# Everything
pip install "vectro[data,integrations]"
```

### From source (requires [Pixi](https://prefix.dev))

```bash
git clone https://github.com/wesleyscholl/vectro
cd vectro
pixi install
pixi run python -c "import python; print(python.__version__)"
```

---

## Quickstart: Compress Your First Batch of Embeddings

```python
import numpy as np
from python import Vectro

# Simulate 1 000 embeddings of dimension 768 (BERT/MPNet size)
rng = np.random.default_rng(42)
embeddings = rng.standard_normal((1_000, 768)).astype(np.float32)

vectro = Vectro()

# Compress the whole batch at once
result = vectro.compress_batch(embeddings)
print(f"Compression ratio: {result.compression_ratio:.2f}×")
print(f"Original size : {result.total_original_bytes:,} bytes")
print(f"Compressed size: {result.total_compressed_bytes:,} bytes")
```

---

## Save and Load Artifacts

```python
# Save to disk
vectro.save_compressed(result, "embeddings.npz")

# Load back
result2 = vectro.load_compressed("embeddings.npz")

# Reconstruct float32 vectors
from python import decompress_vectors
restored = decompress_vectors(result2)
print(restored.shape)   # (1000, 768)
```

---

## Choosing a Compression Profile

Vectro ships with pre-tuned profiles that trade quality for speed and size:

| Profile | Precision | Group size | Compression | Use case |
|---------|-----------|------------|-------------|----------|
| `speed` | INT8 | 0 | ~4× | Real-time inference |
| `balanced` | INT8 | 32 | ~4× | General purpose |
| `quality` | INT8 | 16 | ~4× | Maximum recall |
| `extreme` | INT4 | 64 | ~8× | Storage-constrained |
| `adaptive` | INT8+INT4 | auto | 4–8× | Mixed workloads |

```python
from python import get_compression_profile

profile = get_compression_profile("quality")
result = vectro.compress_batch(embeddings, profile=profile)
```

---

## Streaming Decompression

For large datasets that do not fit in memory, use the `StreamingDecompressor`:

```python
from python import StreamingDecompressor

result = vectro.load_compressed("big_dataset.npz")

for chunk in StreamingDecompressor(result, chunk_size=256):
    # chunk is a float32 numpy array of shape (chunk_size, dim)
    process(chunk)
```

---

## Backend Selection

Vectro tries backends in this order: `mojo`, `rust`, `python`.

```python
from python import get_backend_info

print(get_backend_info())
# {'backend': 'mojo', 'available': ['mojo', 'python']}
```

Force a specific backend:

```python
vectro = Vectro(backend="python")    # pure-Python fallback
```

---

## Next Steps

* [Migration Guide](migration-guide.md) — upgrading from v1
* [Integrations](integrations.md) — Qdrant, Weaviate, PyTorch, Arrow/Parquet
* [Benchmark Methodology](benchmark-methodology.md) — measuring compression quality
* [API Reference](api-reference.md) — complete public API
