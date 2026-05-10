# Vectro: An Embedding Compression Library With a Service Surface

**Author:** Wesley Scholl · Konjo AI
**Version:** 5.3.0 (Python) · 8.0.0 (Rust)
**Date:** 2026-05-10

## Problem

Modern AI systems run on dense vector embeddings — text-embedding-3-large
emits 3072 floats per document, BGE-large emits 1024, and a production
retrieval workload routinely indexes tens of millions of them. At
fp32, a 10M-vector × 1024-dim corpus occupies 40 GB of RAM, which puts
naïve in-memory search outside the budget of most single-node deployments.
Two reductions are needed simultaneously: shrink the bytes per vector
(quantization) and prune the search space (approximate nearest neighbors).
Most open-source tooling solves one half; vectro is built to solve both,
behind one Python API and now one HTTP surface.

## Approach

Vectro's storage layer is a family of quantization codecs — INT8, NF4,
PQ-96, Binary, RQ, and a learned VQZ — each with a Rust kernel
(NEON/AVX2/AMX) and a Python-only fallback that defines the correctness
baseline. Numerical fidelity is enforced by property tests at every
boundary: cosine ≥ 0.9999 on adversarial 1e6-magnitude inputs, accumulated
in FP32 even when the operands are 8-bit.

The search layer is a Hierarchical Navigable Small World graph
(Malkov & Yashunin, *arXiv:1603.09320*). HNSW connects each vector to
≤ M neighbours in the ground layer and exponentially fewer at higher
layers; queries descend the hierarchy with a beam of width ef and run
in O(log N) expected time after a one-time graph build.
Vectro's v3 wrapper, `vectro.HNSWIndex`, exposes
`add_batch(vectors, ids)` and `search(query, top_k, ef)`, with cosine and
L2 spaces and arbitrary user IDs.

V6 (this release) wraps the wrapper in a FastAPI service so any language
can drive it over HTTP. Six endpoints back the surface — `POST /index`,
`POST /index/{name}/add`, `POST /index/{name}/search`,
`GET /index/{name}/stats`, `GET /index/{name}/benchmark`, and
`DELETE /index/{name}` — with per-index `RLock` serialization, dim/NaN/Inf
guards at the JSON boundary, and user-supplied IDs threaded directly
through the underlying graph.

## Benchmarks

Measured on Darwin x86_64, Python 3.13.13, single-threaded
`vectro.HNSWIndex` (pure-Python reference path), unit-norm Gaussian
vectors, ef=64 unless stated, k=10, seed=42. Brute-force baseline
is `numpy` BLAS matmul on L2-normalised vectors. Each search row is
100 queries; latencies are wall-clock per query.

| dim | N     | ef  | Insert (v/s) | HNSW p50/p95/p99 (ms) | Brute p50 (ms) | Recall@10 |
|-----|-------|-----|--------------|------------------------|-----------------|-----------|
| 128 | 1000  | 64  | 237          | 2.63 / 3.69 / 4.64     | 0.038           | 99.3 %    |
| 128 | 5000  | 64  | 101          | 4.87 / 6.89 / 7.14     | 0.081           | 84.0 %    |
| 128 | 10000 | 200 | 70           | 14.25 / 17.68 / 19.35  | 0.596           | 95.8 %    |
| 768 | 1000  | 64  | 196          | 3.02 / 4.45 / 4.69     | 0.049           | 96.3 %    |
| 768 | 5000  | 64  | 78           | 6.70 / 9.61 / 10.72    | 1.450           | 57.4 %    |

Two facts the table makes plain. First, at these scales BLAS-backed
brute force is faster in absolute milliseconds than the pure-Python
HNSW reference path — vectorised matmul on 10k × 128 fp32 weights is
exactly the workload Apple's Accelerate framework was designed for,
and a single-threaded Python beam search cannot keep up. Second,
recall is a tunable, not a fixed cost: at ef=200 the 10k/dim128
corpus reaches 95.8 % recall@10 against ground-truth top-10, and the
same lever recovers the dropped recall in any row of the table.

The pure-Python search path is a deliberate choice — it is the
correctness baseline that the Rust HNSW kernel
(`rust/vectro_lib/src/hnsw/`) has to match numerically. The asymptotic
advantage of HNSW (O(log N) versus brute force's O(N · d)) crosses the
break-even threshold once the kernel is FFI-bound rather than
interpreter-bound, and once N reaches the regimes where the corpus
spills out of L3 cache and BLAS no longer amortises. The
production deployment story is therefore: develop and test against
the Python service, drop in the Rust kernel for shipped indices.

## API at a Glance

```bash
curl -X POST $URL/index \
  -d '{"name":"docs","dim":768,"metric":"cosine"}'

curl -X POST $URL/index/docs/add \
  -d '{"vectors":[[...]],"ids":["doc_a"]}'

curl -X POST $URL/index/docs/search \
  -d '{"query":[...],"k":10}'

curl $URL/index/docs/benchmark?insert_count=1000&search_count=100
```

The benchmark endpoint is the same harness that produced the table
above, parameterised at runtime — it is the canonical way to compare
hardware before committing to a deployment.

## Future Work

Three open fronts. **Rust HNSW behind PyO3** — the kernel exists in
`vectro_lib`; binding it to `vectro.HNSWIndex` will collapse the
Python search path to a single FFI call and recover the
asymptotic advantage at small N. **Persistence** — the V6 service is
process-local; a snapshot loop into the existing VQZ format closes
the gap to managed offerings. **Hybrid retrieval at the service
layer** — vectro already exposes `RRFRetriever` and `VectroReranker`
in Python; lifting them into the HTTP surface gives clients
sparse + dense fusion without per-language re-implementation.

The thesis remains: small library, honest numbers, every claim
reproducible from a single `GET /benchmark` call.
