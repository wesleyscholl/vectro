# ADR-002: v4.0 Architecture

**Status:** Accepted
**Date:** 2026-04-13
**Author:** Wesley Scholl
**Scope:** v4.0.0 — required gate before any v4.0 implementation

---

## Context

Vectro v3.x delivered all originally scoped quantization methods (INT8, INT4, NF4,
Binary, PQ, RQ, Codebook, AutoQuantize), the full Mojo SIMD hot-path, Python bridge,
ONNX export, vector DB integrations, JavaScript N-API bindings (ADR-001), and
distribution infrastructure (PyPI wheels, Homebrew, npm). The roadmap gate states:
**"ADR committed before first implementation line."**

This document commits explicit architectural decisions on four open questions
identified during v3.x:

1. LLM embedding pipeline — sub-1 ms single-vector encode latency
2. WASM target — browser / edge compatible distribution via `vectro_lib`
3. Model-type-aware AutoQuantize profiles (GTE, BGE, E5, BERT families)
4. Rust CLI fate — keep, archive, or supplement with a Python CLI

---

## Decision 1 — LLM Embedding Pipeline (Sub-1 ms Single-Vector Encode)

### Context

Current throughput benchmarks (batch mode, M3):

| Method | Throughput | Per-vector (compute only) |
|---|---|---|
| INT8 Mojo SIMD | 12.5 M vec/s | ≈ 80 ns |
| Python bridge warm | N/A | 1–3 ms round-trip |

The Python `_run_pipe()` bridge keeps a single Mojo subprocess alive and
communicates via JSON over stdin/stdout.  JSON serialization + pipe I/O dominates
single-vector latency (measured: 1–3 ms warm path, 8–30 ms cold start).  For
batch indexing this is acceptable; on the critical query path of a RAG or
semantic-search system it is not.

**Target:** p99 ≤ 1 ms encode latency for a single `d = 768 float32` vector on
Apple Silicon and x86-64 AVX2.

### Options Considered

**Option A — Improve pipe bridge (JSON → raw bytes)**
Replace JSON with a length-prefixed binary protocol over the existing stdio pipe.
Eliminates serialization overhead; pipe IPC overhead remains (~100–500 µs on macOS).
Does not guarantee sub-1 ms.

**Option B — CPython C-extension wrapping Mojo binary symbols**
Expose Mojo-compiled symbols via a thin `.so` (Cython or cffi).  Requires a stable
Mojo ABI, which the SDK does not guarantee before v1.0.  Build toolchain complexity
is high; risk of breakage on every Mojo SDK update.

**Option C — PyO3 path through `vectro_py` (Rust crate)**
`rust/vectro_py` already wraps `vectro_lib` via PyO3 and is distributed via PyPI
wheels (v3.9.0).  `vectro_lib` includes validated INT8 and NF4 implementations that
meet the same accuracy contracts as the Mojo path (INT8 ≥ 0.9999 cosine,
NF4 ≥ 0.9800).  PyO3 FFI overhead is 50–200 ns — comfortably within the 1 ms
budget.

**Option D — CFFI zero-copy shared-memory bridge**
Pin a NumPy buffer in shared memory; call Mojo via a shared-memory segment.
Eliminates JSON; still requires per-call coordination overhead (~50–150 µs).

### Decision: Option C

Use the PyO3 path in `vectro_py` as the single-vector latency champion.  The Mojo
path remains the throughput champion for batch workloads.  No new build toolchain.

**Python API contract (v4.1.0 target):**

```python
from vectro_plus import encode_int8_fast, encode_nf4_fast

# vec: np.ndarray shape (d,), dtype=float32
qvec                = encode_int8_fast(vec)    # returns uint8 ndarray, p99 < 1 ms
qvec, scale         = encode_nf4_fast(vec)     # returns (uint8, float32), p99 < 1 ms
```

**Benchmark gate:** `tests/test_latency_singleshot.py` — assert p99 < 1 ms for
`d = 768` on CI hardware (GitHub Actions `ubuntu-latest`).  Fail the PR if the
gate is missed.  Warmup: 1 000 calls discarded; measurement: 10 000 calls.

---

## Decision 2 — WASM Target

### Context

ADR-001 (v3.3.0) deferred WASM because Mojo→WASM was unsupported.  Rust has
mature, stable WASM support via `wasm-bindgen` / `wasm-pack`.  `vectro_lib` has
**no I/O dependencies** — the crate contains only pure algorithms — and compiles
to `wasm32-unknown-unknown` without modification.

Use cases:

1. **Browser-side `.vqz` reader** — decode compressed embeddings client-side for
   privacy-preserving semantic search without a server round-trip.
2. **Cloudflare Workers / Deno Deploy** — quantize/dequantize inside V8 isolates
   where native binaries are prohibited.
3. **Observable / Jupyter-lite notebooks** — interactive quantization demos without
   a Python kernel.

### Options Considered

**Option A — Mojo→WASM (deferred again)**
SDK 24.6 (April 2026) still does not declare a stable `wasm32` target.  Deferred
indefinitely until Mojo SDK v1.0.

**Option B — `wasm-pack` build of `vectro_lib`**
`wasm-pack build rust/vectro_lib --target web` produces a `.wasm` + `.js` +
`.d.ts` bundle.  WASM SIMD128 (`simd128` feature) is the natural mapping for
`std::simd` operations.  Maximum linear memory is 4 GB, which comfortably covers
in-memory embedding collections up to ~130 M vectors at `d = 768` INT8.

**Option C — Pyodide / PyScript (transpile Python)**
Ships a full CPython interpreter in WASM.  Binary size ≥ 10 MB after compression.
Acceptable for Jupyter-style notebooks; too large for Cloudflare Workers
(1 MB boot limit) and general browser distribution.

### Decision: Option B

Use `wasm-pack` to build `vectro_lib` as `@vectro/wasm` on npm.

**Scope for v4.0 / v4.1:** INT8 encode + decode only.  NF4, PQ, RQ are v4.2.

**Gates:**

- `wasm-pack test --headless --chrome rust/vectro_lib` passes in CI
  (ubuntu-latest, Chrome headless, Node 20).
- Output `.wasm` < 500 KB after brotli compression.
- `@vectro/wasm` published alongside `@vectro/core` in `npm-publish.yml`.

**SIMD guard pattern in `vectro_lib`:**

```rust
#[cfg(target_arch = "wasm32")]
use std::simd::f32x16;  // maps to WASM SIMD128 in wasm-bindgen
```

---

## Decision 3 — Model-Type-Aware AutoQuantize Profiles

### Context

`python/auto_quantize_api.py` currently selects the "best available" method via a
generic heuristic: try INT8 → check cosine similarity → fall back to NF4 → fall
back to Binary.  This ignores model-family-specific output distributions.

Well-known embedding model families have predictable distributions:

| Family | Normalized? | Current heuristic result | Optimal |
|---|---|---|---|
| GTE (Alibaba) | L2-normalized | NF4 (unnecessary overhead) | INT8 |
| BGE (BAAI) | Not normalized | INT8 (suboptimal) | NF4 |
| E5 (Microsoft) | L2-normalized | NF4 | INT8 |
| OpenAI text-embedding-3-* | Normalized | NF4 | INT8 |
| Cohere embed-v3 | Normalized | NF4 | INT8 |
| BERT-based general | Bounded, not normalized | NF4 | NF4 (correct) |

The heuristic is wrong for GTE and E5 (over-engineering to NF4 when INT8 is
lossless) and potentially wrong for BGE (under-engineering to INT8 when NF4 is
better).

### Options Considered

**Option A — Profile registry JSON**
A `profiles.json` file maps model name / family prefix → recommended method + group
size.  User-extendable via a custom JSON path.  Requires case-insensitive string
matching on arbitrary model names.

**Option B — Statistical fingerprint at encode time**
On first encode, compute `std(vec)`, `kurtosis(vec)`, `||vec||₂`.  Classify into a
profile bucket.  Fully automatic; no model name lookup.  Cost: one forward pass of
statistics per encode call.

**Option C — HuggingFace `config.json` registry lookup**
`detect_model_family(model_dir)` reads `config.json` → `architectures` field →
matches known `architectures` patterns.  This pattern is already established in the
Squish project for AWQ alpha selection.  Deterministic: same model always gets the
same profile.

### Decision: Option C

Use a config-based registry, consistent with the existing `detect_model_family()`
pattern.  No per-vector statistical overhead; no fragile name-matching.

**v4.0 profile table:**

| `architectures` value | Family | Default method | Rationale |
|---|---|---|---|
| `"NewModel"` (GTE) | GTE | INT8 | L2-normalized; scalar quant is lossless |
| `"BertModel"` + mean-pooling | BGE | NF4 | Unnormalized; NF4 recall +2–4 pp |
| `"XLMRobertaModel"` (E5) | E5 | INT8 | L2-normalized |
| `"BertModel"` (generic) | BERT | NF4 | Default NF4 for unbounded distributions |
| Unknown / missing | Generic | AutoQuantize | Existing heuristic unchanged |

**Implementation:** `python/profiles.py` — `get_profile(model_dir) -> QuantProfile`
where `QuantProfile` is a `dataclasses.dataclass` with fields `method: str`,
`group_size: int`, `notes: str`.

**Test gate:** `tests/test_auto_quantize_profiles.py` — one test per family
asserting `get_profile()` returns the expected `method`.  No external model weights
required (mock `config.json` files in `tests/fixtures/`).

---

## Decision 4 — Rust CLI Fate

### Context

`rust/vectro_cli` was shipped in v3.9.0 as the primary Vectro CLI binary,
distributed via:

- GitHub Releases (3-platform: Linux x86-64, macOS ARM64, macOS x86-64)
- `Formula/vectro.rb` (Homebrew)
- The `wheels.yml` `cli-binary` job

The Python library exposes the same algorithms.  Question: should `rust/vectro_cli`
remain the primary CLI, or should it be replaced or supplemented by a Python CLI
(Click-based entry point via `pyproject.toml` `[project.scripts]`)?

### Options Considered

**Option A — Keep Rust CLI as sole primary CLI**
Pro: single-binary distribution; no Python required; fast startup; already in
Homebrew and GitHub Releases.
Con: new Python API surface must be mirrored in Rust CLI to stay current.

**Option B — Replace with Python Click CLI**
Pro: always in sync with Python API; `pip install vectro` delivers CLI automatically.
Con: requires Python runtime on the operator's machine; Homebrew formula complexity
increases significantly.

**Option C — Keep both; Rust for ops, Python for library users**
Pro: zero-dependency binary for CI / ops pipelines; Python CLI for developers in a
venv.
Con: two surfaces to test; non-trivial synchronization burden.

### Decision: Option A

The Rust CLI remains the sole primary CLI.

Rationale:
- v3.9.0 already distributes it via Homebrew and GitHub Releases.  Deprecating it
  immediately would regress distribution.
- The crate structure (`vectro_lib ← vectro_cli`, `vectro_lib ← vectro_py`) cleanly
  separates concerns: CLI additions stay in `vectro_cli`, library additions stay in
  `vectro_lib`.
- A Python CLI would require maintaining subcommand parity with `vectro_cli`,
  doubling the test surface with no user-facing benefit.
- If scriptable Python access is needed, `python -m vectro` can delegate to the
  installed Rust binary via `subprocess`.

**Gate:** Any new `compress` / `search` / `bench` / `serve` subcommand added in
v4.x ships first in `rust/vectro_cli`.

---

## Summary

| Decision | Chosen option | One-line rationale |
|---|---|---|
| Sub-1 ms encode | PyO3 path (`vectro_py`) | Zero-overhead FFI; already distributed |
| WASM | `wasm-pack` on `vectro_lib` | Mature, no I/O deps, `@vectro/wasm` on npm |
| AutoQuantize profiles | Config registry (`profiles.py`) | Deterministic; matches squish pattern |
| Rust CLI | Keep — sole CLI binary | Stable; already shipped in v3.9.0 |

---

## v4.1.0 — First Implementation Sprint (derived from this ADR)

| Item | Crate / module | Gate |
|---|---|---|
| `encode_int8_fast` / `encode_nf4_fast` via PyO3 | `rust/vectro_py` | p99 < 1 ms on CI |
| `tests/test_latency_singleshot.py` | `tests/` | PR gate |
| WASM build of `vectro_lib` | `rust/vectro_lib` | `.wasm` < 500 KB (brotli) |
| `@vectro/wasm` INT8 encode + decode | `js/` | wasm-pack CI passes |
| `python/profiles.py` + registry | `python/` | all family tests pass |
| `tests/test_auto_quantize_profiles.py` | `tests/` | 5 family assertions |

*These items may proceed in any order; they are independent.*

---

*This ADR satisfies the v4.0.0 roadmap gate: "ADR committed before first
implementation line."  Implementation begins at v4.1.0.*
