# Vectro v2.1 Backlog

Items deferred from the v2.0 Overdrive milestone. Ordered roughly by expected
impact and implementation effort.

---

## 🥇 High Priority

### INT4 Stability (Experimental → General Availability)
- Remove the `enable_experimental_precisions=True` guard on INT4 mode
- Validate quality gates at INT4 across all standard embedding dimensions
  (128, 384, 768, 1536)
- Add INT4 to the profile selector in `ProfileManager`
- Update CI matrix with `precision=int4` regression lane
- **Acceptance:** `test_quantization_extra.py` INT4 path with cosine_sim ≥ 0.92

### AsyncIO Streaming Decompressor
- `AsyncStreamingDecompressor` — `async for chunk in AsyncStreamingDecompressor(result)`
- Compatible with `asyncio` event loops and `trio`
- Documents backpressure semantics and `chunk_size` tuning
- Useful for async RAG inference servers (AIOHTTP, FastAPI, Starlette)

### Milvus Connector
- `python/integrations/milvus_connector.py` — `MilvusConnector`
  - `store_batch(vectors, ids, metadata)` → collection upsert
  - `search(query_vec, top_k)` → ranked results
  - Optional dep: `pymilvus>=2.4`
  - Mirrors the interface of `QdrantConnector` and `WeaviateConnector`
- Exported from `python.integrations` and top-level `python` package

### `vectro info --benchmark` Quick Bench Flag
- Add `vectro info --benchmark` that runs a 5-second throughput estimation
  and prints compression ratio and cosine similarity on synthetic data
- Makes the CLI immediately useful for evaluating hardware performance

---

## 🥈 Medium Priority

### ONNX Export
- `python/onnx_export.py` — export the dequantization computation graph to ONNX
  so the decompression step can run inside ONNX Runtime (edge/embedded inference)
- Target: ONNX opset 17, float16 and float32 output modes
- Provides a `to_onnx_model(result)` function and a `vectro export-onnx` CLI command
- **Prerequisite:** requires a pure-Python dequantization path (no Mojo binary)

### Chromadb Connector
- `python/integrations/chroma_connector.py` — `ChromaConnector`
  - `store_batch`, `search`, `delete`, `collection_stats`
  - Optional dep: `chromadb>=0.4`
- Completes the "Big Four" open-source vector DB coverage
  (Qdrant ✅, Weaviate ✅, Milvus 🚧, Chroma 🚧)

### Adaptive Quantization Auto-Tuner
- Extend `quantize_adaptive` to automatically select `bits` (4, 6, 8) and
  `clip_ratio` based on the embedding's empirical kurtosis
- Expose as `CompressionOptimizer.auto_quantize(embeddings, target_cosine=0.97)`
- Gate: cosine_sim ≥ `target_cosine` with ratio ≥ 3.5×

### Cloud Storage Backends
Short-term targets for `vectro.save_compressed` / `load_compressed`:
- **AWS S3** — `vectro compress … --output s3://bucket/key`
- **GCS** — `vectro compress … --output gs://bucket/key`
- Implemented via optional `fsspec>=2024.2` dependency

---

## 🥉 Lower Priority / Research

### GPU Acceleration (CUDA)
- Investigate using CuPy or Numba CUDA kernels for the vectorised INT8 quantization
- Throughput target: ≥ 5M vec/s on an NVIDIA A10G (384D)
- Maintains identical output to the CPU path (numerical equivalence tests)

### Learned Quantization
- Train a lightweight (2-layer MLP) encoder–decoder per embedding family
  (e.g., text-embedding-3-large) to learn optimal codebook entries
- Package as `python/codebook.py` with `Codebook.train(embeddings)` and
  `Codebook.compress(embeddings)`
- Potential for >5× compression at cosine_sim > 0.98

### Federated Compression
- Explore privacy-preserving quantization where the codebook is learned via
  federated averaging across multiple data owners
- Research prototype only; no production code in v2.1 scope

### Kubernetes Operator
- Helm chart + operator for deploying Vectro as a sidecar compression service
  in vector database clusters
- Out of scope until Milvus + Cloud Storage backends are stable

---

## 🛠️ Developer Experience

| Item | Notes |
|---|---|
| `pytest-benchmark` integration | Replace manual `time.perf_counter` timings |
| Type stubs (`.pyi`) for all public modules | Improves IDE autocomplete |
| `mypy --strict` CI lane | Enforce type annotations added in v2. |
| Pre-commit hooks | `ruff`, `black`, `mypy --check` |
| Codecov badge | Wire `pytest-cov` output to codecov.io |
| Sphinx API docs | Auto-generate from docstrings; host on GitHub Pages |

---

## 📅 Tentative Timeline

| Milestone | Target | Contents |
|---|---|---|
| **v2.1.0** | Q2 2026 | INT4 GA, AsyncIO streaming, Milvus connector, Chroma connector |
| **v2.2.0** | Q3 2026 | ONNX export, cloud backends, adaptive auto-tuner |
| **v3.0.0** | Q1 2027 | GPU acceleration, learned quantization, federated compression |

---

*Last updated: 2026-03-10 (Phase 5 / v2.0.0 release)*
