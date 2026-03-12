# Vectro v2.1 Backlog [ARCHIVED]

> **This file is archived.** The items listed here were deferred from v2.0 and have since been completed
> and shipped in v3.x releases. See [PLAN.md](PLAN.md) for the current roadmap.

The following items have been completed:
- ✅ **INT4 Stability** → GA in v3.0.0
- ✅ **Milvus Connector** → v3.1.0
- ✅ **Chromadb Connector** → v3.1.0
- ✅ **ONNX Export** → v3.2.0
- ✅ **`vectro info --benchmark`** → v3.3.0
- ⏳ **AsyncIO Streaming Decompressor** → In progress

For a complete view of implementation status, see [PLAN.md](PLAN.md).
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
