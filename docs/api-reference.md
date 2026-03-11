# API Reference

Complete public API for Vectro `3.0.0`.

All symbols are importable directly from the top-level `python` package
unless noted otherwise.

---

## Vectro (class)

```python
from python import Vectro
```

The main entry-point for compressing and decompressing vector embeddings.

### Constructor

```python
Vectro(
    backend: str = "auto",
    profile: str = "balanced",
    enable_batch_optimization: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"auto"` | Force a backend: `"mojo"`, `"rust"`, `"python"`, or `"auto"` |
| `profile` | `str` | `"balanced"` | Default compression profile: `"fast"`, `"balanced"`, `"quality"`, `"ultra"`, `"binary"` |
| `enable_batch_optimization` | `bool` | `True` | Enable batch processing optimizations |

### Methods

#### `compress_batch`

```python
compress_batch(
    embeddings: np.ndarray,
    profile: CompressionProfile | None = None,
) -> BatchQuantizationResult
```

Compress a batch of float32 embeddings. `embeddings` must be a 2-D float32
array with shape `(n, dim)`.

#### `save_compressed`

```python
save_compressed(result: BatchQuantizationResult | QuantizationResult, path: str) -> None
```

Save a compressed artifact to disk in `vectro_npz` v2 format.

#### `load_compressed`

```python
load_compressed(path: str) -> BatchQuantizationResult | QuantizationResult
```

Load a compressed artifact from disk. Reads both v1 and v2 format files.

---

## Free Functions

### Compression / Decompression

```python
compress_vectors(vectors: np.ndarray, **kwargs) -> QuantizationResult
```

Compress a single vector or a 2-D batch. Returns a `QuantizationResult`.

```python
decompress_vectors(result: QuantizationResult) -> np.ndarray
```

Reconstruct float32 vectors from a `QuantizationResult`.

```python
quantize_embeddings(
    embeddings: np.ndarray,
    precision: str = "int8",
) -> QuantizationResult
```

Lower-level scalar quantization.

```python
reconstruct_embeddings(result: QuantizationResult) -> np.ndarray
```

Reconstruct from a `QuantizationResult`.

### Quality

```python
mean_cosine_similarity(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> float
```

Compute the mean cosine similarity between original and reconstructed vectors.

```python
analyze_compression_quality(
    original: np.ndarray,
    result: QuantizationResult,
) -> dict
```

Return a quality report dict with `cosine_similarity`, `mse`, `compression_ratio`.

```python
generate_compression_report(
    original: np.ndarray,
    result: QuantizationResult,
) -> str
```

Return a human-readable string report.

### Batch Processing

```python
quantize_embeddings_batch(
    embeddings: np.ndarray,
    batch_size: int = 512,
) -> BatchQuantizationResult
```

Quantize embeddings in batches for memory efficiency.

```python
benchmark_batch_compression(
    embeddings: np.ndarray,
) -> dict
```

Run a quick benchmark and return timing and quality metrics.

### Migration

```python
from python.migration import inspect_artifact, upgrade_artifact, validate_artifact
```

See also: [Migration Guide](migration-guide.md).

```python
inspect_artifact(path: str | Path) -> dict
```

Inspect a compressed artifact and return its metadata.

```python
upgrade_artifact(
    src: str | Path,
    dst: str | Path,
    *,
    dry_run: bool = False,
) -> dict
```

Upgrade a v1 artifact to v2 format.

```python
validate_artifact(path: str | Path) -> dict
```

Validate structural integrity. Returns `{"valid": bool, "errors": list[str]}`.

### Utility

```python
get_backend_info() -> dict
```

Return information about available backends.

```python
get_version_info() -> dict
```

Return Vectro version and build metadata.

---

## Data Classes

### `QuantizationResult`

| Field | Type | Description |
|-------|------|-------------|
| `quantized` | `np.ndarray` (int8) | Compressed data, shape `(n, dim)` |
| `scales` | `np.ndarray` (float32) | Per-vector scale factors, shape `(n,)` |
| `dims` | `int` | Vector dimension |
| `n` | `int` | Number of vectors |
| `precision_mode` | `str` | `"int8"`, `"int4"`, etc. |
| `group_size` | `int` | Quantization group size (`0` = no grouping) |

### `BatchQuantizationResult`

| Field | Type | Description |
|-------|------|-------------|
| `quantized` | `np.ndarray` (int8) | Compressed data, shape `(batch_size, dim)` |
| `scales` | `np.ndarray` (float32) | Per-vector scales |
| `batch_size` | `int` | Number of vectors in the batch |
| `vector_dim` | `int` | Dimension of each vector |
| `compression_ratio` | `float` | `original_bytes / compressed_bytes` |
| `total_original_bytes` | `int` | Total bytes before compression |
| `total_compressed_bytes` | `int` | Total bytes after compression |
| `precision_mode` | `str` | Precision mode used |
| `group_size` | `int` | Group size used |

---

## Integrations

### Vector DB Connectors

```python
from python.integrations import (
    InMemoryVectorDBConnector,
    QdrantConnector,
    WeaviateConnector,
    VectorDBConnector,          # abstract base
    StoredVectorBatch,
)
```

Each connector implements:

```python
connector.upsert(embeddings: np.ndarray, ids: list[str | int]) -> None
connector.search(query: np.ndarray, top_k: int = 10) -> list[SearchResult]
```

### PyTorch / HuggingFace

```python
from python.integrations import (
    compress_tensor,            # torch.Tensor → QuantizationResult
    reconstruct_tensor,         # QuantizationResult → torch.Tensor
    HuggingFaceCompressor,
)
```

### Arrow / Parquet

```python
from python.integrations import (
    result_to_table,            # → pyarrow.Table
    table_to_result,            # pyarrow.Table → BatchQuantizationResult
    write_parquet,              # → Parquet file
    read_parquet,               # Parquet → BatchQuantizationResult
    to_arrow_bytes,             # → bytes (Arrow IPC)
    from_arrow_bytes,           # bytes → BatchQuantizationResult
)
```

---

## Streaming

```python
from python import StreamingDecompressor
```

```python
StreamingDecompressor(
    result: BatchQuantizationResult,
    chunk_size: int = 1000,
    backend: str = "auto",
)
```

Iterator that yields float32 `np.ndarray` chunks. Useful for datasets that do
not fit in memory.

---

## Advanced Quantization

```python
from python import quantize_int2, dequantize_int2, quantize_adaptive
```

```python
quantize_int2(vectors: np.ndarray) -> tuple[np.ndarray, float, int]
dequantize_int2(quantized: np.ndarray, scale: float, zero_point: int) -> np.ndarray
quantize_adaptive(vectors: np.ndarray, threshold: float = 0.1) -> list[QuantizationResult]
```

---

## Benchmark

```python
from python.benchmark import BenchmarkSuite, BenchmarkReport
```

```python
suite = BenchmarkSuite(embeddings, n_runs: int = 3, random_seed: int = 42)
report: BenchmarkReport = suite.run_all()
report.to_json(path: str) -> None
report.to_csv(path: str) -> None
```

---

## Compression Profiles

```python
from python import (
    ProfileManager,
    CompressionProfile,
    CompressionStrategy,
    CompressionOptimizer,
    ProfileComparison,
    get_compression_profile,
    create_custom_profile,
)
```

```python
profile = get_compression_profile(name: str) -> CompressionProfile
profile = create_custom_profile(precision_mode="int4", group_size=64)
```

Built-in profiles: `"fast"`, `"balanced"`, `"quality"`, `"ultra"`, `"binary"`.

---

## v3.0.0 API

### VectroV3 (class)

```python
from python.v3_api import VectroV3, V3Result
```

The unified v3 entry-point. Supports all seven compression profiles.

```python
VectroV3(profile: str = "int8")
```

**Profiles**

| Profile | Compression | Cosine sim | Notes |
|---------|-------------|------------|-------|
| `"int8"` | 4× | ≥ 99.97% | Symmetric INT8, < 1 µs/vec |
| `"nf4"` | 8× | ≥ 99.12% | NF4 normal-float 4-bit |
| `"nf4-mixed"` | ~7× | ≥ 99.20% | NF4 + FP16 outlier dims |
| `"pq-96"` | 32× | ≥ 98.70% | PQ 96 sub-spaces; training required |
| `"pq-48"` | 16× | ≥ 99.10% | PQ 48 sub-spaces; training required |
| `"binary"` | 32× | ≥ 95.00% | 1-bit sign quantization |
| `"rq-3pass"` | ~10× | ≥ 98.83% | Residual Quantizer 3 passes |

**Methods**

```python
v3.compress(embeddings: np.ndarray) -> V3Result
v3.decompress(result: V3Result) -> np.ndarray
v3.save(result: V3Result, path: str) -> None          # local or cloud URI
v3.load(path: str) -> V3Result                         # local or cloud URI
```

### V3Result (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `quantized` | `np.ndarray` (int8) | Compressed vectors |
| `scales` | `np.ndarray` (float32) | Per-vector scale factors |
| `codes` | `list \| None` | PQ / RQ codes (None for INT8/NF4/binary) |
| `profile` | `str` | Profile name used |
| `compression_ratio` | `float` | Original bytes / compressed bytes |
| `mean_cosine` | `float` | Mean cosine similarity over the batch |

---

### PQCodebook (class)

```python
from python.v3_api import PQCodebook
```

```python
PQCodebook.train(
    vectors: np.ndarray,
    n_subspaces: int = 96,
    n_centroids: int = 256,
) -> PQCodebook
```

Returns a trained `PQCodebook`.

```python
codebook.encode(vectors: np.ndarray) -> np.ndarray   # uint8 (n, n_subspaces)
codebook.decode(codes: np.ndarray) -> np.ndarray     # float32 (n, dim)
codebook.save(path: str) -> None
PQCodebook.load(path: str) -> PQCodebook
```

---

### HNSWIndex (class — v3 wrapper)

```python
from python.v3_api import HNSWIndex
```

```python
HNSWIndex(
    dim: int,
    quantization: str = "int8",
    M: int = 16,
    ef_construction: int = 200,
)
```

```python
index.add(vectors: np.ndarray) -> None
index.search(query: np.ndarray, k: int = 10, ef: int = 50) -> tuple[np.ndarray, np.ndarray]
index.save(path: str) -> None
HNSWIndex.load(path: str) -> HNSWIndex
```

`search` returns `(indices, distances)` arrays of shape `(k,)`.

---

### ResidualQuantizer (class)

```python
from python.rq_api import ResidualQuantizer
```

```python
ResidualQuantizer(
    n_passes: int = 3,
    n_subspaces: int = 8,
    n_centroids: int = 256,
)
```

```python
rq.train(vectors: np.ndarray) -> ResidualQuantizer
rq.encode(vectors: np.ndarray) -> list[np.ndarray]   # length n_passes
rq.decode(codes: list[np.ndarray]) -> np.ndarray     # float32 (n, d)
rq.mean_cosine(original: np.ndarray, recon: np.ndarray) -> float
```

Requires `scikit-learn`.

---

### Codebook (autoencoder, class)

```python
from python.codebook_api import Codebook
```

```python
Codebook(
    target_dim: int = 32,
    hidden: int = 128,
    l2_reg: float = 1e-4,
    seed: int | None = None,
)
```

```python
cb.train(vectors: np.ndarray, n_epochs: int = 100, lr: float = 0.01, batch_size: int = 64) -> Codebook
cb.encode(vectors: np.ndarray) -> np.ndarray    # int8 (n, target_dim)
cb.decode(codes: np.ndarray) -> np.ndarray      # float32 (n, d)
cb.mean_cosine(original: np.ndarray, recon: np.ndarray) -> float
cb.save(path: str) -> None
Codebook.load(path: str) -> Codebook
```

Pure-NumPy, no PyTorch required.

---

### auto_quantize

```python
from python.auto_quantize_api import auto_quantize
```

```python
auto_quantize(
    embeddings: np.ndarray,
    target_cosine: float = 0.97,
    target_compression: float = 4.0,
) -> dict
```

Tries the strategy cascade NF4 → NF4-mixed → PQ-96 → PQ-48 → binary.
Returns the first strategy satisfying both constraints.

Return dict keys: `strategy`, `compression_ratio`, `mean_cosine`, `quantized`,
`scales`, `codes`, `dequantized`.

---

### GPU API

```python
from python.gpu_api import (
    gpu_available,
    gpu_device_info,
    quantize_int8_batch,
    reconstruct_int8_batch,
    batch_cosine_similarity,
    batch_topk_int8,
    gpu_benchmark,
)
```

```python
gpu_available() -> bool
gpu_device_info() -> dict   # keys: backend, device_name, simd_width, unified_memory
quantize_int8_batch(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]
reconstruct_int8_batch(quantized: np.ndarray, scales: np.ndarray) -> np.ndarray
batch_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray
batch_topk_int8(query: np.ndarray, db_q: np.ndarray, db_s: np.ndarray, k: int) -> tuple
gpu_benchmark(n: int = 10_000, dim: int = 768) -> dict
```

`gpu_benchmark` return keys: `throughput_vec_per_sec`, `latency_us`, `cosine_sim`, `backend`.

---

### Storage v3

```python
from python.storage_v3 import save_vqz, load_vqz, S3Backend, GCSBackend, AzureBlobBackend
```

```python
save_vqz(
    quantized: np.ndarray,
    scales: np.ndarray,
    dims: int,
    path: str,
    compression: str = "zstd",   # "zstd" | "zlib" | "none"
    metadata: bytes = b"",
    level: int = 3,
    n_subspaces: int = 0,
) -> None
```

```python
load_vqz(path: str) -> dict
# keys: quantized, scales, dims, n_vectors, metadata, version, n_subspaces
```

**Cloud Backends** (require `pip install fsspec` and the cloud provider's SDK):

```python
# AWS S3  (requires fsspec[s3])
backend = S3Backend(bucket="my-bucket", prefix="vectro/")
backend.save_vqz(quantized, scales, dims, "embeddings.vqz")
data = backend.load_vqz("embeddings.vqz")

# Google Cloud Storage  (requires fsspec[gcs])
backend = GCSBackend(bucket="my-bucket", prefix="embeddings/")

# Azure Blob Storage  (requires fsspec[abfs])
backend = AzureBlobBackend(bucket="my-container", prefix="data/")
```