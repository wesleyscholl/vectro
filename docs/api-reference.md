# API Reference

Complete public API for Vectro `1.2.0`.

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
Vectro(backend: str = "auto")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"auto"` | Force a backend: `"mojo"`, `"rust"`, `"python"`, or `"auto"` |

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

Built-in profiles: `"speed"`, `"balanced"`, `"quality"`, `"extreme"`, `"adaptive"`.
