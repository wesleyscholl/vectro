# Getting Started with Vectro

Vectro is an ultra-high-performance LLM embedding compressor. It compresses floating-point
embedding vectors using nine quantization strategies, achieving 4–32× storage savings
with configurable quality trade-offs.

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

| Profile | Precision | Compression | Cosine sim | Use case |
|---------|-----------|-------------|------------|----------|
| `speed` | INT8 | ~4× | 99.97% | Real-time inference |
| `balanced` | INT8 | ~4× | 99.97% | General purpose |
| `quality` | INT8 | ~4× | 99.99% | Maximum recall |
| `extreme` | INT4 | ~8× | 97%+ | Storage-constrained |
| `adaptive` | INT8+INT4 | 4–8× | 97%+ | Mixed workloads |

```python
from python import get_compression_profile

profile = get_compression_profile("quality")
result = vectro.compress_batch(embeddings, profile=profile)
```

---

## v3 Quickstart: Advanced Compression

### Product Quantization (32× compression)

```python
from python.v3_api import VectroV3

v3 = VectroV3(profile="pq-96")

# train_data should be at least 256 * n_subspaces vectors
result = v3.compress(embeddings)   # trains internally on first call
restored = v3.decompress(result)

print(f"Ratio : {result.compression_ratio:.0f}×")
print(f"Cosine: {result.mean_cosine:.4f}")
```

### HNSW Approximate Nearest-Neighbour Index

```python
from python.v3_api import HNSWIndex

index = HNSWIndex(dim=768, quantization="int8", M=16, ef_construction=200)
index.add(database_vectors)            # INT8 internal storage — 4× memory savings

# k-NN query
indices, distances = index.search(query_vector, k=10, ef=50)

# Persist the index
index.save("index.hnsw")
index2 = HNSWIndex.load("index.hnsw")
```

### AutoQuantize (choose the best strategy automatically)

```python
from python.auto_quantize_api import auto_quantize

result = auto_quantize(
    embeddings,
    target_cosine=0.97,        # minimum acceptable cosine similarity
    target_compression=8.0,    # minimum acceptable compression ratio
)
print(f"Chosen : {result['strategy']}")
print(f"Ratio  : {result['compression_ratio']:.1f}×")
print(f"Cosine : {result['mean_cosine']:.4f}")
```

---

## VQZ Storage Format

Vectro v3 introduces the `.vqz` binary container — a 64-byte header followed by a
ZSTD/zlib-compressed body. It replaces `.npz` for v3 data and supports S3/GCS/Azure
via `fsspec`.

```python
from python.storage_v3 import save_vqz, load_vqz

# Save with ZSTD compression (default)
save_vqz(result.quantized, result.scales, dims=768, path="embeddings.vqz")

# Load back
data = load_vqz("embeddings.vqz")
print(data["n_vectors"], data["dims"])   # e.g. 1000 768
```

Cloud storage (requires `pip install fsspec s3fs`):

```python
from python.storage_v3 import S3Backend

s3 = S3Backend(bucket="my-bucket", prefix="vectro/")
s3.save_vqz(result.quantized, result.scales, dims=768, remote_name="embeddings.vqz")
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

* [API Reference](api-reference.md) — full Python API including v3 classes
* [Migration Guide](migration-guide.md) — upgrading from v1/v2
* [Integrations](integrations.md) — Qdrant, Weaviate, PyTorch, Arrow/Parquet
* [Benchmark Methodology](benchmark-methodology.md) — measuring compression quality
