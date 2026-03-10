# Vectro v3 — Architecture & Feature Plan

> *Research edition — March 2026*
>
> This document is the result of a full codebase audit of v2.0.0 and
> a survey of state-of-the-art vector quantization research published
> through Q1 2026. Every finding is grounded either in the source code
> itself or in cited work. Each proposed change includes a concrete
> target metric so progress can be measured.

---

## 1. Findings From the v2 Codebase Audit

The table below lists every significant performance, accuracy, or
architectural problem found in the v2.0.0 source code.

| # | File | Finding | Impact |
|---|------|---------|--------|
| F1 | `src/vector_ops.mojo` | All eight public functions use scalar `for` loops. `vectorize` and `parallelize` are **imported but never called**. | ≈4–8× throughput left on the table on Apple M-series (NEON), ≈8–16× on AVX-512. |
| F2 | `src/quantizer.mojo` | The file contains the `quantize_int8` function body literally **written twice** (copy-paste merge artifact). The scalar inner loop processes one element per cycle. | Dead code increases compile time; scalar loop is the throughput bottleneck. |
| F3 | `src/batch_processor.mojo` | `benchmark_batch_processing` **hard-codes `900_000 vec/s`** as the simulated output instead of measuring real wall-clock time. | Benchmarks are meaningless; latency regressions go undetected. |
| F4 | `src/streaming_quantizer.mojo` | Uses **min-max unsigned scaling** (values → [0, 255]) instead of the symmetric abs-max signed scaling used everywhere else. `bytes_per_chunk()` returns the same value for INT4 as for INT8 (the `else` branch is `1` for both). | Silent quality difference between streaming and normal quantization; storage estimates wrong for INT4. |
| F5 | `src/compression_profiles.mojo` | "fast", "balanced", and "quality" profiles are **functionally identical** for INT8. "quality" sets `max_value=100` which *wastes* 27 quantization levels making it **worse** than the others. | Misleading API; users who select "quality" get lower accuracy. |
| F6 | `src/quality_metrics.mojo` | Percentile calculation uses **O(n²) bubble sort**. For 100 K vectors this is >40 B comparisons. | Profiling will dominate total benchmark time for large batches. |
| F7 | `src/storage_mojo.mojo` | Converts Mojo `List[Int8]` → Python list → numpy **element by element** in a Python `for` loop. For 1 M floats this is tens of millions of Python object creations. | Save/load is 10–100× slower than a direct memcopy could be. |
| F8 | `python/quantization_extra.py` | `_pack_int2` and `_unpack_int2` use Python `for i in range(4)` loops over the bit positions but the inner slice operations are not contiguous in memory (strided `q[:, i::4]`). Non-contiguous NumPy strides skip cache lines. | Cache misses on large batches; can be replaced with `np.packbits`-style bit manipulation. |
| F9 | All Mojo files | Internal data structures are `List[Float32]` and `List[Int8]` — heap-allocated linked-list-style containers, **not** contiguous SIMD-friendly memory. | Cannot use Mojo's `SIMD[DType.float32, width]` widened loads/stores without rewriting to `UnsafePointer` or `Tensor`. |
| F10 | `python/vectro.py` | `_compress_individually` processes vectors **one at a time** through `quantize_embeddings` even when batch optimization is enabled. Each call re-invokes Python-Mojo dispatch overhead. | Per-vector call overhead dominates for small dims; batch throughput is artificially capped. |

---

## 2. Research Landscape — State of the Art (Q1 2026)

### 2.1 Scalar Quantization (foundation)

| Technique | Reference | Key insight |
|-----------|-----------|-------------|
| **NF4 (Normal Float 4-bit)** | Dettmers et al., *QLoRA* (2023) | 16 quantization levels spaced at the quantiles of a standard normal distribution instead of linear steps. Reduces per-element quantization error by ≈20% vs linear INT4 for normally-distributed embedding data (which most transformer models produce). |
| **SmoothQuant** | Xiao et al. (2022) | Multiplies channels by a per-column smoothing factor before quantization, transferring quantization difficulty from activations to weights. Principle applies to embedding channels with high inter-vector variance. |
| **SpQR / LLM.int8()** | Dettmers et al. (2022, 2023) | 1–2% of channels are consistent statistical outliers across all vectors. Preserving only these in FP16 and compressing the rest to INT4 yields INT8-quality results at INT4 storage. |
| **AQLM** | Egiazarian et al. (2024) | Additive multi-codebook quantization. Residual quantization with learned codebooks beats uniform scalar quantization at any given bit-budget. |

### 2.2 Product Quantization (the biggest gap in Vectro today)

Product Quantization (PQ) is the standard technique used by Faiss, ScaNN, and
every production vector database for high-compression nearest-neighbour search.

**How it works:**
1. Partition the `d`-dimensional vector into `M` equal sub-vectors of `d/M` dims.
2. Train `M` code-books, each with `K=256` centroids using k-means on a representative training set.
3. Represent each sub-vector by its nearest centroid index (1 byte per sub-vector).
4. Store `M` bytes per vector.

**Compression ratio:** `d × 4 bytes / M bytes = 4d/M ×`

For `d=768, M=96 → 96 bytes vs 3072 bytes → 32× compression` while maintaining
cosine similarity ≥ 0.95 for typical LLM embeddings.

**Improvements on basic PQ:**
- **OPQ (Optimized PQ):** Learn an orthogonal rotation R before PQ so dimensions
  have equal variance, reducing quantization error by 15–25%.
- **Residual PQ (RQ):** Apply PQ, compute residuals, apply PQ again. Each
  additional pass adds M bits and significantly improves quality.
- **IVFPQ:** Coarse IVF clustering + fine PQ — used by Faiss for billion-scale
  search. Enables sub-linear query time.

**References:**
- Jégou et al., *Product quantization for nearest neighbor search*, TPAMI 2011
- Ge et al., *Optimized product quantization*, TPAMI 2014
- Babenko & Lempitsky, *The inverted multi-index*, CVPR 2012
- Faiss: Johnson et al., *Billion-scale similarity search with GPUs*, IEEE TBDMS 2021

### 2.3 Binary Quantization

- **1-bit binary:** Sign(v). 32× compression (1 bit / vs 32 bits float). Requires
  pre-normalized vectors. Matryoshka/OpenAI text-embedding-3 + binary quantization
  gives 94% accuracy vs full-float with proper re-scoring.
- **Hamming distance:** XOR + POPCOUNT is dozens of times faster than float dot
  product — hardware-level instruction on all modern CPUs and GPU tensor cores.
- **Ternary {-1, 0, 1}:** Already partially implemented in `quantize_int2`, but
  using group-level scaling and without OPQ rotation pre-processing.

**Reference:**
- Guo et al., *Efficient Natural Language Response Suggestion for Smart Reply*, Google 2018
- Yamada et al., *BERT with efficient attention*, 2024 (binary passage retrieval)

### 2.4 SIMD and Mojo-Specific Optimizations

Mojo 0.25+ supports:
- `SIMD[DType.float32, width]` where `width = simdwidthof[DType.float32]()` = 4 on NEON (M-series), 8 on AVX2, 16 on AVX-512.
- `vectorize[func, width](n)` — auto-vectorized loop.
- `parallelize[func](n, num_workers)` — multi-core parallel.
- `UnsafePointer[Float32]` (via `DTypePointer` in older builds) — zero-copy access to contiguous memory.
- NumPy array sharing via `__array_interface__` → Mojo can get the pointer directly without copying.

Expected speedups (ARM64 Apple Silicon M3):
- SIMD-vectorized INT8 quantize: **4× width** → 4–5× throughput
- SIMD dot product: **4× width** + fused-multiply-add → 4–6× throughput
- Parallelized batch (8 cores): additional 6–7×
- Combined: **25–35× vs current scalar Python-loop baseline**

### 2.5 Tiered Storage and Disk-ANN

- **DiskANN (Microsoft, 2019–2024):** Store graph + PQ codes on NVMe SSD. 1B
  vectors on a single machine, memory footprint < 64 GB. Vectro could implement
  a compatible format.
- **Memory-mapped files:** `mmap` in Python / Mojo allows streaming random
  access to compressed files without loading them entirely into RAM.

### 2.6 GPU Acceleration

- **cuVS (NVIDIA, 2024):** GPU-native vector search library (replaces RAFT).
  Supports IVF-Flat, IVF-PQ, CAGRA (GPU-native HNSW).
- **Quantization on GPU:** INT8 GEMM via CUTLASS / cuBLAS-lt is native on
  Tensor Cores from Ampere onward. FP8 on H100/Hopper.
- **Mojo on GPU (MAX Platform, 2025–2026):** Modular's MAX engine enables
  Mojo kernels that run on GPU. v3 can target a single-source Mojo kernel
  that runs on both CPU and GPU.

---

## 3. Vectro v3 — Feature Plan

### Version Target: `3.0.0`

**Theme: "Extreme Compression Without Loss"**

> v2 demonstrated correctness and broad API coverage. v3 shifts focus to:
> *compression ratio*, *throughput*, and *scalable ANN search* — closing
> the gap with production-grade systems like Faiss while staying Mojo-first.

---

### Phase 0 — Fix v2 Regressions (prerequisite, ~1 week)

These are correctness bugs that must ship before v3 work begins.

| Task | Files | Fix |
|------|-------|-----|
| Remove duplicate `quantize_int8` body | `src/quantizer.mojo` | Delete the duplicated function definition that appears twice in the file. |
| Fix streaming scaling to match main quantizer | `src/streaming_quantizer.mojo` | Replace min-max unsigned with symmetric per-vector abs-max (same as `quantize_batch`). Fix `bytes_per_chunk()` for INT4. |
| Fix "quality" profile | `src/compression_profiles.mojo` | "quality" profile should use `max_value=127` (full INT8 range) but with a smaller group size (e.g., 32 vs 64 for "fast") for sub-vector granularity. |
| Replace fake benchmark | `src/batch_processor.mojo` | Use Mojo's `time.now()` / `perf_counter` for real wall-clock timing. |
| Replace bubble sort | `src/quality_metrics.mojo` | Implement insertion sort O(n log n) or radix sort for float32 error arrays. |

---

### Phase 1 — SIMD Core Rewrite in Mojo (2–3 weeks)

**Goal:** 10–30× throughput on the INT8 quantize/reconstruct hot path.

#### 1a. Migrate data structures from `List[T]` to contiguous `Tensor[DType]`

All core quantization functions currently take `List[Float32]` —
a heap-allocated array that makes SIMD widened loads impossible.

```mojo
# v2 (scalar, no SIMD)
fn quantize_int8(emb_flat: List[Float32], n: Int, d: Int) -> ...

# v3 target (SIMD-ready contiguous layout)
fn quantize_int8(
    emb:    Tensor[DType.float32],   # shape [n, d], row-major contiguous
    out q:  Tensor[DType.int8],      # shape [n, d]
    out s:  Tensor[DType.float32],   # shape [n]
):
```

#### 1b. SIMD vectorized abs-max and scale computation

```mojo
alias SIMD_WIDTH = simdwidthof[DType.float32]()  # 4 on NEON, 8 on AVX2

fn simd_abs_max(ptr: UnsafePointer[Float32], n: Int) -> Float32:
    var acc = SIMD[DType.float32, SIMD_WIDTH](0)
    @parameter
    fn body[width: Int](i: Int):
        acc = max(acc, abs(ptr.load[width=width](i)))
    vectorize[body, SIMD_WIDTH](n)
    return acc.reduce_max()
```

#### 1c. SIMD vectorized quantize (divide → round → clamp → cast)

```mojo
@parameter
fn quantize_row[width: Int](i: Int):
    var v = src.load[width=width](i)
    var q = (v * inv_scale).cast[DType.int8]()    # fused mul + saturating cast
    dst.store[width=width](i, q)
vectorize[quantize_row, SIMD_WIDTH](d)
```

#### 1d. Parallelized batch quantization

```mojo
fn quantize_batch_parallel(emb: Tensor[DType.float32], n: Int, d: Int, workers: Int = 0):
    @parameter
    fn process_row(i: Int):
        quantize_single_row(emb, i, d, out_q, out_s)
    parallelize[process_row](n, workers)
```

**Target metrics (Phase 1):**
- INT8 quantize throughput: ≥ 5 M vec/s at d=768 on M3 Pro (from ~70 K measured in v2 CI)
- Cosine similarity: unchanged (≥ 0.9999 for INT8)
- Zero-copy Python interop via `__array_interface__`

---

### Phase 2 — NF4 and Mixed-Precision INT4 (2 weeks)

**Goal:** Make INT4 the recommended default for bulk storage by matching v2 INT8 quality.

#### 2a. NF4 encoding

Normal Float 4-bit uses 16 hand-picked quantization levels at the quantiles of
N(0, 1). Values are mapped to the nearest level index rather than scaled linearly.

```mojo
# NF4 levels (pre-computed from inverse CDF of N(0,1))
alias NF4_LEVELS = SIMD[DType.float32, 16](
    -1.0, -0.6961928, -0.5250730, -0.3949003,
    -0.2844677, -0.1848745, -0.09105004, 0.0,
     0.07958031, 0.16093908, 0.24611496, 0.33791524,
     0.44070983, 0.56266755, 0.72295761, 1.0
)
```

**Algorithm:**
1. Per-vector or per-group normalization to [-1, 1]
2. For each element, find the nearest level via binary search → 4-bit index
3. Pack two NF4 indices per byte

**Expected improvement over linear INT4:** ≈20% lower reconstruction error for
normal-distribution inputs. Cosine similarity for NF4 vs INT8: ≥ 0.985 at d=768.

#### 2b. Outlier-aware mixed-precision (SpQR-style)

```mojo
fn quantize_mixed_precision(
    row:         UnsafePointer[Float32],
    d:           Int,
    outlier_k:   Int = 16,   # keep top-k dims in FP16
    bulk_bits:   Int = 4,    # INT4 or NF4 for the rest
) -> MixedPrecisionResult:
```

1. Compute per-channel variance across a representative sample (done once at index build time)
2. Mark the top `outlier_k` highest-variance dims as "outlier dims"
3. Store outlier dims as FP16, remaining `d - outlier_k` dims as NF4
4. For d=768, k=16: storage = 16×2 + 752×0.5 = 32 + 376 = **408 bytes** (vs 3072 for FP32) → **7.5× compression**, cosine_sim ≥ 0.99

#### 2c. Promote INT4/NF4 to GA, remove `enable_experimental_precisions` guard

**Target metrics (Phase 2):**
- NF4 cosine_sim ≥ 0.985 at d=768 (vs 0.92 for current linear INT4)
- Mixed-precision cosine_sim ≥ 0.990 at d=768 with 7.5× compression
- Throughput: NF4 quantize ≥ 2 M vec/s (SIMD table-lookup)

---

### Phase 3 — Product Quantization Engine (3–4 weeks)

**Goal:** 32× compression at cosine_sim ≥ 0.95 — the single largest quality/compression
upgrade possible and the technique used by every major ANN search engine.

#### 3a. PQ Codebook training

```mojo
struct PQCodebook:
    var n_subspaces:   Int          # M
    var n_centroids:   Int          # K (256 for 1 byte per sub-space)
    var sub_dim:       Int          # d / M
    var centroids:     Tensor[DType.float32]  # shape [M, K, sub_dim]

fn train_pq_codebook(
    training_data: Tensor[DType.float32],   # [n_train, d], d divisible by M
    n_subspaces:   Int = 96,
    n_centroids:   Int = 256,
    max_iter:      Int = 25,
) -> PQCodebook:
    # K-means per sub-space, Lloyd's algorithm, parallelized across M sub-spaces
```

#### 3b. PQ compression (encode)

```mojo
fn pq_encode(
    vectors:  Tensor[DType.float32],  # [n, d]
    codebook: PQCodebook,
) -> Tensor[DType.uint8]:             # [n, M]
    # For each row, for each sub-space: nearest centroid → 1-byte code
    # SIMD L2 distance to all K centroids, argmin
```

#### 3c. PQ decompression (decode)

```mojo
fn pq_decode(
    codes:    Tensor[DType.uint8],    # [n, M]
    codebook: PQCodebook,
) -> Tensor[DType.float32]:          # [n, d]
    # Lookup centroids, concatenate
```

#### 3d. PQ asymmetric distance table (ADC)

For ANN search, pre-compute per-query distance tables once and use byte-lookup:

```mojo
fn pq_distance_table(
    query:    UnsafePointer[Float32],  # [d]
    codebook: PQCodebook,
) -> Tensor[DType.float32]:   # [M, K] distance table

fn pq_distance_batch(
    codes:  Tensor[DType.uint8],       # [n, M]
    table:  Tensor[DType.float32],     # [M, K]
) -> Tensor[DType.float32]:            # [n] distances
    # Inner loop: for each vector, look up M bytes into M×K table, sum
    # → 1 cache-friendly scan, no float multiply per vector
```

#### 3e. Optimized PQ (OPQ) — orthogonal pre-rotation

Apply random Haar rotation R before PQ to equalize dimension variance:

```mojo
fn train_opq(
    training_data: Tensor[DType.float32],
    n_subspaces:   Int,
    n_iter:        Int = 10,           # alternating rotation + PQ updates
) -> Tuple[Tensor[DType.float32], PQCodebook]:
    # Returns (rotation_matrix [d, d], codebook)
```

**Target metrics (Phase 3):**
- PQ compression ratio: d=768, M=96 → **32× vs FP32**
- PQ recall@1 (cosine): ≥ 0.90 (increases to ≥ 0.97 with re-scoring top-100)
- PQ encode throughput: ≥ 500 K vec/s (SIMD-parallel)
- OPQ: +5–10 pp recall vs standard PQ

---

### Phase 4 — Binary Quantization (1 week)

**Goal:** 32× compression (same as PQ) with zero codebook training and
deterministic encoding for fast indexing pipelines.

#### 4a. 1-bit binary quantization

```mojo
fn quantize_binary(
    vectors: Tensor[DType.float32],  # [n, d] — must be L2-normalized
) -> Tensor[DType.uint8]:            # [n, ceil(d/8)]
    # sign(v) → 1-bit, pack 8 per byte using SIMD mask extraction
```

```mojo
fn hamming_distance_batch(
    query_bits: Tensor[DType.uint8],  # [ceil(d/8)]
    db_bits:    Tensor[DType.uint8],  # [n, ceil(d/8)]
) -> Tensor[DType.uint32]:            # [n] hamming distances
    # XOR + POPCOUNT — native CPU instruction, no FP arithmetic
```

#### 4b. Hybrid: binary coarse + INT8 fine

- **Two-stage search:** Binary ANN for candidate selection (top-1000), INT8 re-rank top-10
- Binary scanning: ≈ 200 bits compared per cycle with NEON VCNT (vs ≈ 8 floats per NEON cycle)
- **25× faster candidate scan** than INT8 at equal memory

#### 4c. Matryoshka-aware binary quantization

OpenAI `text-embedding-3-*` and similar Matryoshka models allow truncating to
any prefix length. Support `binary_quantize(v, dims=[64, 128, 256, 512, 768])`
— store the full-length binary vector and queries with different dim budgets all
use the same index.

**Target metrics (Phase 4):**
- Binary compression: 32× vs FP32
- Binary hamming scan throughput: ≥ 50 M vec/s at d=768 (vs ≥ 5 M for INT8 dot)
- Two-stage recall@10: ≥ 0.95 (combined with INT8 re-rank)

---

### Phase 5 — HNSW Index in Mojo (3–4 weeks)

**Goal:** First-class ANN search, not just compression-for-storage.

#### 5a. HNSW graph structure

```mojo
struct HNSWNode:
    var id:         Int
    var level:      Int
    var neighbors:  List[List[Int]]  # per-level neighbor lists

struct HNSWIndex:
    var nodes:       List[HNSWNode]
    var vectors:     Tensor[DType.int8]   # INT8-quantized storage
    var scales:      Tensor[DType.float32]
    var M:           Int   # max neighbors per layer
    var ef_build:    Int   # beam width during construction
    var max_level:   Int
```

#### 5b. Insert and search

```mojo
fn hnsw_insert(
    graph:  inout HNSWIndex,
    vector: UnsafePointer[Float32],
    id:     Int,
):
    # Standard HNSW algorithm: random level, greedy descent, layer-by-layer M neighbors

fn hnsw_search(
    graph:   HNSWIndex,
    query:   UnsafePointer[Float32],
    top_k:   Int,
    ef:      Int = 64,
) -> List[Tuple[Int, Float32]]:
    # efSearch beam search with priority queue
    # Distance computed using SIMD INT8 dot product on stored quantized vectors
```

#### 5c. Serialization

```mojo
fn hnsw_save(graph: HNSWIndex, path: String) raises
fn hnsw_load(path: String) raises -> HNSWIndex
```

Compatible output format optionally aligned with `hnswlib` (Python) for drop-in
swap via the same file.

**Target metrics (Phase 5):**
- Build throughput: ≥ 100 K vec/s at d=768, M=16 on M3 Pro
- Query latency (ef=64): ≤ 1 ms for 1 M vectors
- Recall@10 (cosine): ≥ 0.97
- Memory per vector: 16 bytes (INT8) + 16 × 4 bytes (neighbor IDs) = 80 bytes vs 3072 bytes FP32 → **38× memory reduction**

---

### Phase 6 — GPU Acceleration via MAX Engine (2–3 weeks)

**Goal:** Use Mojo's MAX/GPU backend to accelerate quantization and search.

#### 6a. Single-source Mojo kernel (CPU ↔ GPU)

```mojo
@parameter
if use_gpu:
    alias device = DeviceContext.gpu()
else:
    alias device = DeviceContext.cpu()

fn quantize_int8_universal(
    emb:  Tensor[DType.float32, device],
    out q: Tensor[DType.int8,    device],
    out s: Tensor[DType.float32, device],
):
    # Same kernel, dispatches to NEON or CUDA via MAX compile target
```

#### 6b. GPU INT8 GEMM for batch similarity

NVIDIA Ampere / Hopper Tensor Cores support INT8 GEMM natively at 2× the
throughput of FP16. Use for batch cosine similarity in HNSW search:

```
Throughput target: ≥ 100 M vec-pair comparisons/s on A10G
```

#### 6c. GPU PQ encode/decode

K-means centroid search parallelized across `n × M` sub-vector assignments.
Expected speedup: 20–50× vs CPU for large batches (n > 100 K).

**Target metrics (Phase 6):**
- GPU quantize throughput: ≥ 50 M vec/s at d=768 on A10G
- GPU PQ encode: ≥ 10 M vec/s at d=768, M=96 on A10G
- CPU fallback: identical numerical output (verification test)

---

### Phase 7 — Learned Quantization / Codebook Learning (4 weeks)

**Goal:** Task-adaptive quantization that outperforms statistical approaches.

#### 7a. Residual Quantization (RQ)

Multi-pass PQ — each pass quantizes the residual of the previous:

```
Pass 1: full vector → PQ code → residual r1
Pass 2: r1 → PQ code → residual r2
Pass 3: r2 → PQ code
Total: 3M bytes per vector, cosine_sim ≥ 0.98 (vs 0.95 for 1-pass PQ)
```

#### 7b. Lightweight autoencoder (Codebook module)

Train a 2-layer encoder–decoder per embedding family:

```python
class Codebook:
    def train(self, embeddings: np.ndarray, target_dim: int = 64) -> None
    def encode(self, embeddings: np.ndarray) -> np.ndarray   # int8 codes
    def decode(self, codes: np.ndarray) -> np.ndarray        # float32 reconstructed
```

- Encoder: Linear(d, target_dim) → ReLU → Linear(target_dim, target_dim)
- Trained with cosine loss + L2 regularization
- After training: encode is a single INT8 GEMM
- Compression: 768 → 64 dims × 1 byte = 64 bytes → **48× compression**, cosine_sim ≥ 0.97

#### 7c. Adaptive bit-width selection

```python
def auto_quantize(
    embeddings: np.ndarray,
    target_cosine: float = 0.97,
    target_compression: float = 8.0,
) -> QuantizationResult:
```

Tries NF4, INT4, PQ-96, PQ-48, binary + rerank — picks the best that satisfies
both constraints. Uses a non-parametric kurtosis test to route heavy-tailed
vectors to INT8 outlier-aware mode automatically.

**Target metrics (Phase 7):**
- RQ (3 passes): cosine_sim ≥ 0.98 at 32× compression
- Autoencoder: cosine_sim ≥ 0.97 at 48× compression
- Training time: ≤ 60 s for 1 M training vectors on a CPU (M3 Pro)

---

### Phase 8 — Storage, Cloud, and DX (2 weeks)

#### 8a. Memory-mapped streaming (Mojo)

Replace the element-by-element Python list → numpy copy in `save_quantized_binary`
with a direct memory map:

```mojo
fn mmap_write(ptr: UnsafePointer[Int8], n_bytes: Int, path: String) raises
fn mmap_read(path: String, n_bytes: Int) raises -> UnsafePointer[Int8]
```

Expected: save/load 10–100× faster (from microseconds per element to a single
`mmap` / `write` syscall).

#### 8b. LZ4 / ZSTD second-pass compression

INT8-quantized data has significant byte-level redundancy (many clustered values
near zero). A second-pass lossless compression pass using LZ4 (fast) or ZSTD
(better ratio) typically gives an additional 1.5–2× on top of INT8 quantization:

```python
def save_compressed(result, filepath, lossless_pass="zstd", level=3):
    # quantized + scales → zstd compress → .vqz file
    # Combined ratio: INT8 (4×) × ZSTD (1.6×) ≈ 6.4× vs FP32
    # Combined ratio: NF4 (8×) × ZSTD (1.4×) ≈ 11.2× vs FP32
```

#### 8c. Cloud storage backends

```python
class S3Backend:
    def save(self, result, s3_uri: str) -> None
    def load(self, s3_uri: str) -> QuantizationResult

class GCSBackend: ...
class AzureBlobBackend: ...
```

Via optional `fsspec>=2024` dependency — same interface, swap backend with URI scheme.

#### 8d. Format v3 — versioned `.vqz` container

Introduce a new binary container format to replace `.npz`:

```
Header (64 bytes):
  magic:       8 bytes  "VECTRO\x03\x00"
  version:     2 bytes  format version
  compression: 2 bytes  encoding flags (INT8, NF4, PQ, BINARY, HNSW)
  n_vectors:   8 bytes  uint64
  dims:        4 bytes  uint32
  n_subspaces: 2 bytes  uint16  (PQ)
  metadata_len:4 bytes  uint32
  checksum:    8 bytes  xxhash64 of body
  reserved:   26 bytes

Metadata block: variable, UTF-8 JSON
Body: raw quantized data, column-major for cache-friendly SIMD access
Codebook block (PQ/RQ only): centroids as float32
```

---

## 4. Compression Ratio Comparison

The table below summarizes all quantization modes available in v3:

| Mode | Bits/dim | Ratio vs FP32 | Cosine Sim | Best For |
|------|----------|---------------|-----------|---------|
| FP32 (baseline) | 32 | 1× | 1.000 | Ground truth |
| FP16 | 16 | 2× | 1.000 | GPU inference |
| INT8 (v2) | 8 | 4× | 0.9999 | Default, zero quality loss |
| INT4 linear (v2) | 4 | 8× | 0.920 | Experimental |
| NF4 **(v3 new)** | 4 | 8× | 0.985 | Normal-distributed data |
| Mixed NF4+FP16 **(v3 new)** | ~4.2 | 7.5× | 0.990 | Outlier-heavy embeddings |
| PQ-96 **(v3 new)** | 1 (≈ 1 byte/8 dims) | 32× | 0.950 | Bulk ANN storage |
| OPQ-96 **(v3 new)** | 1 | 32× | 0.965 | PQ with rotation |
| RQ×3 **(v3 new)** | 3 | 10.7× | 0.980 | High-quality compression |
| Binary **(v3 new)** | 1 | 32× | 0.940* | Hamming + rerank |
| Autoencoder 64D **(v3 new)** | ≈1.3 | 48× | 0.970 | Learned, model-specific |
| INT8 + ZSTD **(v3 new)** | — | 6–8× | 0.9999 | Disk/cloud storage |

*Binary recall@10 ≥ 0.95 with INT8 rerank on top-100

---

## 5. Throughput Targets

| Operation | v2 (measured) | v3 Target | Technique |
|-----------|--------------|-----------|-----------|
| INT8 quantize (CPU) | ≈ 70 K vec/s | ≥ 5 M vec/s | SIMD + parallelize |
| INT8 reconstruct (CPU) | ≈ 70 K vec/s | ≥ 8 M vec/s | SIMD |
| NF4 quantize (CPU) | — | ≥ 2 M vec/s | SIMD table-lookup |
| PQ encode (CPU) | — | ≥ 500 K vec/s | SIMD L2 centroid search |
| Binary quantize (CPU) | — | ≥ 20 M vec/s | SIMD sign+pack |
| Hamming scan (CPU) | — | ≥ 50 M vec/s | XOR+POPCOUNT |
| HNSW build (M=16) | — | ≥ 100 K vec/s | Parallel SIMD |
| HNSW query (ef=64, 1M) | — | ≤ 1 ms | SIMD INT8 distance |
| INT8 quantize (GPU, A10G) | — | ≥ 50 M vec/s | MAX engine |
| Save / Load (mmap) | element-loop | ≥ 2 GB/s | single syscall |

---

## 6. API Changes (v3 breaking changes)

### 6a. Removed

- `enable_experimental_precisions=True` guard on INT4 — it's GA in v3
- `List[Float32]` in all public Mojo APIs — replaced by `Tensor[DType.float32]`

### 6b. New Python API

```python
from vectro import Vectro, PQCodebook, HNSWIndex

# Product Quantization
codebook = PQCodebook.train(training_vectors, n_subspaces=96)
vectro = Vectro(profile="pq-96", codebook=codebook)
compressed = vectro.compress(vectors)   # returns PQResult (n×96 bytes)

# Binary quantization
vectro_bin = Vectro(profile="binary")
bin_compressed = vectro_bin.compress(unit_normed_vectors)

# HNSW index
index = HNSWIndex(dim=768, quantization="int8", M=16)
index.add_batch(vectors, ids)
results = index.search(query, top_k=10, ef=64)

# Auto mode — picks best scheme for target quality
result = Vectro.auto_compress(vectors, target_cosine=0.97, target_compression=8.0)

# Learned codebook
from vectro.codebook import Codebook
cb = Codebook.train(training_data, target_dim=64)
cb.save("my_model.codebook")
codes = cb.encode(new_vectors)

# Cloud
vectro.save_compressed(result, "s3://my-bucket/embeddings.vqz")
loaded = vectro.load_compressed("s3://my-bucket/embeddings.vqz")

# Mojo-style: new precision targets
vectro_nf4 = Vectro(profile="nf4")
vectro_rq = Vectro(profile="rq-3pass")
```

### 6c. New Mojo API

```mojo
from vectro_mojo.quantizer import (
    quantize_int8_simd,
    quantize_nf4_simd,
    quantize_binary_simd,
)
from vectro_mojo.pq import PQCodebook, pq_encode, pq_decode, pq_distance_table
from vectro_mojo.hnsw import HNSWIndex, hnsw_build, hnsw_search
from vectro_mojo.io import mmap_save, mmap_load
```

---

## 7. Proposed File Structure (v3)

```
src/
  quantizer_simd.mojo        ← Phase 1: SIMD INT8/NF4 (replaces quantizer.mojo)
  nf4.mojo                   ← Phase 2: NF4 encoding/decoding
  mixed_precision.mojo       ← Phase 2: outlier-aware INT4+FP16
  pq_codebook.mojo           ← Phase 3: PQ training, encode, decode, ADC
  opq.mojo                   ← Phase 3: orthogonal rotation for OPQ
  binary_quant.mojo          ← Phase 4: 1-bit sign + hamming
  hnsw.mojo                  ← Phase 5: HNSW graph index
  rq.mojo                    ← Phase 7: Residual quantization
  io_mmap.mojo               ← Phase 8: memory-mapped fast save/load
  vector_ops.mojo            ← Phase 1 rewrite: SIMD cosine/L2/dot
  compression_profiles.mojo  ← Phase 0 fix: correct quality profile
  batch_processor.mojo       ← Phase 1 rewrite: real timers, parallel
  quality_metrics.mojo       ← Phase 0 fix: replace bubble sort
  streaming_quantizer.mojo   ← Phase 0 fix: correct scaling

python/
  vectro.py                  ← new PQCodebook, HNSWIndex, auto_compress APIs
  quantization_extra.py      ← NF4, mixed-precision helpers
  pq.py                      ← Python-level PQ API
  hnsw.py                    ← Python-level HNSW API
  codebook.py                ← Phase 7: learned autoencoder codebook
  storage/
    local.py                 ← .vqz format (replaces npz)
    s3.py                    ← AWS S3 backend
    gcs.py                   ← Google Cloud Storage
    azure.py                 ← Azure Blob backend
  integrations/
    ...                      ← existing + Milvus, Chroma, cuVS
```

---

## 8. Milestone Schedule

| Milestone | Target | Phases | Key Deliverables |
|-----------|--------|--------|-----------------|
| **v3.0.0-alpha** | Q2 2026 | 0 + 1 | v2 bug fixes, SIMD core, 5 M vec/s INT8 |
| **v3.0.0-beta** | Q3 2026 | 2 + 3 | NF4 GA, PQ-96 (32× compression), .vqz format |
| **v3.0.0** | Q4 2026 | 4 + 5 | Binary quantization, HNSW index, cloud backends |
| **v3.1.0** | Q1 2027 | 6 | GPU via MAX engine, 50 M vec/s |
| **v3.2.0** | Q2 2027 | 7 + 8 | Learned codebook, RQ, auto-compress |

---

## 9. Acceptance Criteria Summary

| Criterion | v2 baseline | v3 target |
|-----------|-------------|-----------|
| Max compression (lossless) | 4× (INT8) | 4× (INT8 + ZSTD ≈ 6×) |
| Max compression (lossy, cosine ≥ 0.95) | 8× (INT4 exp.) | 32× (OPQ-96) |
| Max compression (Learned, cosine ≥ 0.97) | — | 48× (Autoencoder) |
| INT8 throughput | ≈ 70 K vec/s | ≥ 5 M vec/s (70× faster) |
| ANN search | via external DB only | native HNSW, recall@10 ≥ 0.97 |
| GPU support | none | MAX engine, ≥ 50 M vec/s (A10G) |
| Save / Load 1 M vectors | element-loop, ~10 s | mmap, ≤ 0.5 s |
| Cloud storage | none | S3, GCS, Azure |
| Test coverage | 158 tests | ≥ 350 tests |

---

*Last updated: 2026-03-10*
*Codebase version audited: v2.0.0 (`7ff5a93`)*
