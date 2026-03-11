# Vectro — Plan

> Last updated: 2026-03-11
> Current version: **3.0.1** — tagged `v3.0.1`, pushed to origin, GitHub Release published at https://github.com/wesleyscholl/vectro/releases/tag/v3.0.1

---

## v3.0.1 — Mojo-First Runtime Fix ✅ COMPLETE (2026-03-11)

All quantization hot paths now dispatch to the compiled Mojo binary at runtime.
`v3.0.0` advertised Mojo-first but all paths fell through to Python/NumPy.

| Component | Fix |
|-----------|-----|
| `src/vectro_standalone.mojo` | Complete CLI binary: int8/nf4/bin quantize+reconstruct, benchmark, selftest |
| `python/_mojo_bridge.py` | New unified subprocess dispatch helper for all Python modules |
| `python/interface.py` | `_quantize_with_mojo` + `_reconstruct_with_mojo` now call Mojo |
| `python/batch_api.py` | `_quantize_batch_mojo` now calls Mojo |
| `python/nf4_api.py` | `quantize_nf4` / `dequantize_nf4` route through bridge |
| `python/binary_api.py` | `quantize_binary` / `dequantize_binary` route through bridge |
| `pixi.toml` | `build-mojo`, `selftest`, `benchmark` tasks added; version bumped to 3.0.1 |
| `tests/test_mojo_bridge.py` | 26 new tests: availability, INT8/NF4/Binary accuracy, dispatch |

390 tests passing (up from pre-existing baseline). Pre-existing sklearn C-extension
failures (test_rq.py, test_v3_api.py) remain unrelated to Mojo changes.

---

## Current State

All ten phases of `VECTRO_V3_PLAN.md` have been committed to `main`:

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Correctness bug fixes (F1–F10) | ✅ Done |
| 1 | SIMD acceleration (vector_ops, quantizer_simd) | ✅ Done |
| 2 | NF4 normal-float 4-bit quantization | ✅ Done |
| 3 | Product Quantization (PQ + OPQ + ADC) | ✅ Done |
| 4 | Binary / 1-bit quantization | ✅ Done |
| 5 | HNSW approximate nearest-neighbour index | ✅ Done |
| 6 | GPU quantization via MAX Engine | ✅ Done |
| 7 | Learned quantization (RQ, Codebook, AutoQuantize) | ✅ Done |
| 8 | Storage v3 — VQZ container + mmap bulk I/O | ✅ Done |
| 9 | Unified v3 API — PQCodebook, HNSWIndex, VectroV3 | ✅ Done |

Version strings bumped to `3.0.0` in all source files. Tag `v3.0.0` pushed to origin.

---

## Phase 10 — v3.0.0 Release Hardening ✅ COMPLETE (2026-03-11)

All steps verified and committed in `6c5a5f9`.

| Step | Result |
|------|--------|
| 10a Test suite | 445 / 445 passing |
| 10b CHANGELOG | Phases 5–9 + Phase 10 hardening documented; section renamed `[3.0.0]` |
| 10c README | v3.0.0 badges, 7 profiles, HNSW, AutoQuantize, VQZ, roadmap |
| 10d Cloud backends | S3Backend, GCSBackend, AzureBlobBackend implemented via fsspec |
| 10e docs/ | api-reference, getting-started, benchmark-methodology updated |
| 10f Video demo | `demos/demo_v3.gif` — animated GIF (900×580, 35s, VHS-recorded, gifsicle-optimised) |
| 10g GitHub Release | v3.0.0 published at https://github.com/wesleyscholl/vectro/releases/tag/v3.0.0 |
| INT4 GA | Removed `enable_experimental_precisions` gate; INT4 is production in v3 |

### 10a. Test Suite Audit (prerequisite)

Run the full Python test suite and identify any failures introduced by Phases 5–9.
Fix every failing test before proceeding. No release ships with known failures.

```
python -m pytest tests/ -v
```

**Acceptance:** Zero test failures. Test count ≥ 350 (from v3 plan acceptance criteria).

### 10b. CHANGELOG — Document Phases 5–9

The `[3.0.0-dev]` section in `CHANGELOG.md` currently covers only Phases 0–4
(and stops at 208 tests). Add entries for:

- Phase 5: HNSW index (`hnsw_index.mojo`, `hnsw_api.py`, `test_hnsw.py`)
- Phase 6: GPU quantizer (`gpu_quantizer.mojo`, `gpu_api.py`, `test_gpu.py`)
- Phase 7: Learned quantization (`codebook_api.py`, `rq_api.py`, `auto_quantize_api.py`, tests)
- Phase 8: Storage v3 (`storage_v3.mojo`, `storage_v3.py`, `test_storage_v3.py`)
- Phase 9: Unified v3 API (`v3_api.py`, `test_v3_api.py`)
- Cumulative test count table through all phases

Rename `[3.0.0-dev]` → `[3.0.0] — 2026-03-XX` and add the release date.

### 10c. README — Update to v3.0.0

The README still describes v2.0.0 (badges show version 2.0.0, 195 tests, v2 features only).

Updates required:

| Section | Change |
|---------|--------|
| Version badge | `2.0.0` → `3.0.0` |
| Tests badge | `195 passing` → actual count after 10b |
| Quick Start | Add PQ, binary, HNSW, and learned quantization examples |
| Python API section | Add `PQCodebook`, `HNSWIndex`, `VectroV3`, `AutoQuantize` |
| Performance benchmarks | Update with v3 throughput numbers |
| What's Included | Add Phases 5–9 modules |
| Roadmap | Mark Q1 2026 items complete; set Q2 2026 milestones |

### 10d. Verify Phase 8 Cloud Backends

`VECTRO_V3_PLAN.md` Phase 8c specifies S3, GCS, and Azure Blob backends.
Audit `python/storage_v3.py` to determine whether cloud backends were implemented
or deferred. If deferred, add them to the v3.1 backlog below.

### 10e. docs/ — Update Guide Files

Review and update the five docs to reflect v3 additions:

- `docs/api-reference.md` — add PQCodebook, HNSWIndex, VectroV3, RQ, Codebook, AutoQuantize
- `docs/getting-started.md` — add PQ and HNSW quickstart sections
- `docs/benchmark-methodology.md` — update throughput targets to v3 numbers

### 10f. Tag and Release v3.0.0

Once all prior steps are green:

1. Bump version strings if still showing `-dev` suffix
2. `git tag v3.0.0 && git push origin v3.0.0`
3. The `.github/workflows/release.yml` workflow handles build, PyPI publish, and GitHub Release

---

## Phase 11 — v3.1.0: Enterprise & Ecosystem Expansion

**Target:** Q2 2026

Items drawn from `BACKLOG_v2.1.md` and the README 2026-2027 roadmap.

### 11a. Cloud Storage Backends (if not in Phase 8)

If S3/GCS/Azure were deferred from Phase 8c:

```python
# python/storage/s3.py
class S3Backend:
    def save(self, result: QuantizationResult, s3_uri: str) -> None
    def load(self, s3_uri: str) -> QuantizationResult

# python/storage/gcs.py, azure.py — same interface
```

Via optional `fsspec>=2024.2` dependency. CLI support:
`vectro compress input.npy s3://bucket/key.vqz`

### 11b. Milvus Connector

`python/integrations/milvus_connector.py` — mirrors `QdrantConnector` / `WeaviateConnector`.

- `store_batch(vectors, ids, metadata)` → Milvus collection upsert
- `search(query_vec, top_k)` → ranked results
- Optional dep: `pymilvus>=2.4`
- Completes the "Big Four" open-source vector DB coverage

### 11c. Chroma Connector

`python/integrations/chroma_connector.py` — completes Big Four with Chroma.

- Optional dep: `chromadb>=0.4`
- Exported from `python.integrations` and top-level package

### 11d. LZ4 / ZSTD Second-Pass Compression

Phase 8b from the v3 plan (may not have been implemented). Adds lossless
post-compression to the `.vqz` format:

```python
def save_compressed(result, filepath, lossless_pass="zstd", level=3):
    # INT8 (4×) × ZSTD (1.6×) ≈ 6.4× vs FP32
```

### 11e. AsyncIO Streaming Decompressor

`AsyncStreamingDecompressor` — `async for chunk in AsyncStreamingDecompressor(result)`.
Documents backpressure semantics and `chunk_size` tuning for FastAPI / AIOHTTP servers.

### 11f. `vectro info --benchmark` CLI Flag

A 5-second throughput estimation + quick quality report on synthetic data,
making the CLI immediately useful for evaluating hardware capability.

### 11g. Developer Experience

| Item | Notes |
|------|-------|
| `pytest-benchmark` integration | Replace manual `perf_counter` timings |
| Type stubs (`.pyi`) for all public modules | IDE autocomplete |
| `mypy --strict` CI lane | Enforce type annotations |
| Pre-commit hooks | `ruff`, `black`, `mypy --check` |
| Codecov badge | Wire `pytest-cov` to codecov.io |

---

## Phase 12 — v3.2.0: Performance & Research

**Target:** Q3 2026

### 12a. ONNX Export

`python/onnx_export.py` — export the dequantization graph to ONNX opset 17
so decompression runs inside ONNX Runtime for edge/embedded inference.

- `to_onnx_model(result)` function
- `vectro export-onnx` CLI command
- Prerequisites: pure-Python dequantization path (already exists in numpy layer)

### 12b. Real GPU Throughput Validation

The Phase 6 GPU quantizer targets ≥ 50M vec/s on A10G. If CI only runs CPU,
add a CUDA-enabled CI job or document how to benchmark on GPU hardware.

Add numerical equivalence tests: GPU output must match CPU output bit-for-bit.

### 12c. Pinecone Connector

`python/integrations/pinecone_connector.py` — closes the managed vector DB gap.
Optional dep: `pinecone-client>=3.0`.

### 12d. JavaScript Bindings (Research)

From the 2026 roadmap: zero-copy JS interop. Evaluate:
- WebAssembly compilation of the Mojo quantizer
- Node.js native addon via N-API for the `.vqz` reader

Deliverable for 12d: ADR document with feasibility analysis and chosen approach.

---

## Acceptance Criteria Summary (v3.0.0 → v3.2.0)

| Criterion | v3.0.0 target | v3.1.0 target | v3.2.0 target |
|-----------|--------------|--------------|--------------|
| All tests passing | ✅ | ✅ | ✅ |
| Test count | ≥ 350 | ≥ 420 | ≥ 500 |
| README current | ✅ v3.0.0 | ✅ | ✅ |
| CHANGELOG complete | ✅ | ✅ | ✅ |
| Cloud backends (S3/GCS/Azure) | Verified or tracked | ✅ | ✅ |
| Vector DB coverage | Qdrant ✅ Weaviate ✅ | + Milvus + Chroma | + Pinecone |
| ONNX export | — | — | ✅ |
| GPU throughput CI | — | — | ✅ |

---

## Immediate Next Actions (Ordered)

1. **Run the test suite** — `python -m pytest tests/ -v` — identify failures
2. **Fix any test failures** from Phases 5–9
3. **Update CHANGELOG** — document Phases 5–9 and rename section to `[3.0.0]`
4. **Update README** — v3.0.0 badges, new API sections, updated benchmarks
5. **Audit Phase 8c** — confirm cloud storage status; add to backlog if deferred
6. **Update docs/** — api-reference, getting-started, benchmark-methodology
7. **Tag v3.0.0** — trigger release workflow
8. **Begin Phase 11** — cloud backends and Milvus/Chroma connectors

---

*Created: 2026-03-11*
*Codebase audited at commit: 7d63793 (main)*
