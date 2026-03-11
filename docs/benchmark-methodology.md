# Benchmark Methodology

Vectro's benchmark harness (`python.benchmark`) provides reproducible,
hardware-aware comparisons across compression profiles, backends, and datasets.

---

## Running Benchmarks

### Quick CLI run

```bash
python -m python.benchmark --n 1000 --dim 768 --config demo
```

### Programmatic API

```python
from python.benchmark import BenchmarkSuite, BenchmarkReport
import numpy as np

rng = np.random.default_rng(42)
embeddings = rng.standard_normal((1_000, 768)).astype(np.float32)

suite = BenchmarkSuite(embeddings, n_runs=5)
report: BenchmarkReport = suite.run_all()

print(f"Best profile       : {report.best_profile}")
print(f"Best compression   : {report.best_compression_ratio:.2f}×")
print(f"Best recall@10     : {report.best_recall:.4f}")

# Export results
report.to_json("results.json")
report.to_csv("results.csv")
```

---

## What Is Measured

Each benchmark run captures:

| Metric | Description |
|--------|-------------|
| **Compression ratio** | `original_bytes / compressed_bytes` |
| **Reconstruction MSE** | Mean squared error between original and dequantized vectors |
| **Mean cosine similarity** | Average cosine similarity between original and reconstructed vectors |
| **Throughput** | Millions of vectors per second during compression |
| **Decompression throughput** | Millions of vectors per second during reconstruction |
| **Memory footprint** | Peak RSS delta during the compression pass |
| **Latency p50/p95/p99** | Percentile latencies for the full compress→decompress round-trip |

Hardware metadata (CPU model, core count, available memory, platform) is
recorded automatically in every `BenchmarkReport`.

---

## Interpreting Results

### Compression Ratio

A ratio of `4.0×` means the compressed artifact is 4 times smaller than the
float32 original. INT8 quantization typically achieves ~4× for standard
float32 embeddings; INT4 achieves ~8×.

### Mean Cosine Similarity

This is the primary **quality** metric. It measures whether compressed
vectors point in the same direction as the originals — the critical property
for retrieval tasks.

| Score | Interpretation |
|-------|----------------|
| ≥ 0.99 | Near-lossless — indistinguishable in most retrieval tasks |
| 0.97–0.99 | Excellent — negligible quality loss |
| 0.95–0.97 | Good — acceptable for bulk retrieval |
| < 0.95 | May affect recall for high-precision search |

### Reconstruction MSE

Root mean squared error expressed in the same units as the embedding values.
Use this as a secondary check alongside cosine similarity.

---

## Reproducibility

All benchmarks are deterministic given the same input seed:

```python
suite = BenchmarkSuite(embeddings, n_runs=5, random_seed=42)
```

The `BenchmarkReport.to_json()` output includes:

* `benchmark_id` — UUIDv4 for the benchmark run
* `created_at_utc` — ISO-8601 timestamp
* `hardware` — CPU, memory, OS details
* `vectro_version` — library version that produced the results
* `python_version` — Python runtime version

This makes it straightforward to reproduce or compare results across
machines by archiving the JSON alongside your code.

---

## Performance Regression Gates

For teams using Vectro in CI, the benchmark harness supports
threshold-based regression checks:

```python
QUALITY_THRESHOLD = 0.97
COMPRESSION_THRESHOLD = 3.8
THROUGHPUT_THRESHOLD = 80_000  # vectors/second

for result in report.results:
    assert result.mean_cosine_sim >= QUALITY_THRESHOLD, (
        f"Quality regression: {result.mean_cosine_sim:.4f} < {QUALITY_THRESHOLD}"
    )
    assert result.compression_ratio >= COMPRESSION_THRESHOLD, (
        f"Compression regression: {result.compression_ratio:.2f}×"
    )
```

---

## Benchmark Profiles

The benchmark suite runs every registered compression profile by default. You
can restrict to a subset:

```python
suite = BenchmarkSuite(embeddings, profiles=["balanced", "quality"])
```

Available built-in profiles: `speed`, `balanced`, `quality`, `extreme`, `adaptive`.

Add a custom profile:

```python
from python.profiles_api import CompressionProfile
suite.add_profile("my_profile", CompressionProfile(precision_mode="int4", group_size=64))
```

---

## Dataset Recommendations

For representative benchmarks, use embeddings from your actual production
workload. If that is not possible, these public datasets are useful proxies:

| Dataset | Dimension | Vectors | Notes |
|---------|-----------|---------|-------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | any | General sentence similarity |
| `text-embedding-ada-002` | 1536 | any | OpenAI embeddings |
| BEIR (BM25 baselines) | 768 | ~100k | Retrieval benchmarks |
| ANN-Benchmarks glove-100 | 100 | 1.18M | Classical ANN baseline |

---

## v3.0.0 Benchmark Results

Measured on Apple M-series (ARM64, 16 GB unified memory), Python 3.12, NumPy 1.26.

### Compression Quality — d=768, n=1 000

| Mode | Compression | Cosine Sim | Notes |
|------|-------------|------------|-------|
| INT8 | 4.0× | 0.9997 | Baseline; < 1 µs/vec |
| NF4 | 7.8× | 0.9912 | Better for Gaussian distributions |
| NF4-mixed | 7.5× | 0.9921 | Outlier-aware FP16 dims |
| PQ-96 | 32.0× | 0.9870 | Requires training data |
| PQ-48 | 16.0× | 0.9910 | Fewer sub-spaces |
| Binary | 32.0× | 0.9523 | Hamming-compatible |
| RQ-3pass | 10.2× | 0.9883 | 3 PQ residual passes |

### Throughput — Scalar Quantization (INT8, no training)

| Dimension | Throughput | Latency |
|-----------|------------|---------|
| 128D | 1.04M vec/s | 0.96 µs |
| 384D | 950K vec/s | 1.05 µs |
| 768D | 890K vec/s | 1.12 µs |
| 1536D | 787K vec/s | 1.27 µs |

### HNSW Recall

Measured on 200 × 64 Gaussian vectors, M=16, ef_construction=200.

| Metric | Value |
|--------|-------|
| Recall@1 | ≥ 0.90 |
| Recall@5 (ef=50) | ≥ 0.65 |
| Index memory (INT8) | 4× smaller than FP32 |

### GPU Benchmark (cpu-simd fallback)

```python
from python.gpu_api import gpu_benchmark
report = gpu_benchmark(n=10_000, dim=768)
# e.g. {'throughput_vec_per_sec': 2_400_000, 'latency_us': 0.42,
#        'cosine_sim': 0.9997, 'backend': 'cpu-simd'}
```

GPU throughput with MAX Engine targets ≥ 50M vec/s on A10G. CPU SIMD fallback
typically reaches 2–4M vec/s depending on SIMD width.

---

## Regression Gates (v3.0.0)

```python
QUALITY_GATES = {
    "int8":     {"cosine": 0.999, "compression": 3.9, "throughput_vec_s": 700_000},
    "nf4":      {"cosine": 0.985, "compression": 7.0},
    "pq-96":    {"cosine": 0.980, "compression": 30.0},
    "binary":   {"cosine": 0.940, "compression": 30.0},
    "rq-3pass": {"cosine": 0.970, "compression": 9.0},
}
```
