# ADR-001: JavaScript / WASM Bindings

**Status:** Proposed
**Date:** 2026-Q3
**Author:** Wesley Scholl
**Scope:** v3.3.0 — JavaScript ecosystem interoperability

---

## Context

The v3.2.0 roadmap (Phase 12) identifies JavaScript/WASM bindings as a target
for the 2026 ecosystem expansion.  Concrete use cases driving this ADR:

1. **Browser-side `.vqz` reader** — allow a web client to decode a Vectro
   compressed artifact without a round-trip to a server.
2. **Node.js inference** — embed INT8 dequantization in server-side TypeScript /
   JavaScript applications (e.g. LLM inference pipelines built on LangChain.js).
3. **Edge inference** — run dequantization inside a Cloudflare Worker or similar
   V8-isolate environment where native binaries are prohibited.

The `.vqz` binary format (64-byte header + flat zstd/zlib-compressed body) is
deliberately simple: the body is a concatenation of an `int8` quantized array
and a `float32` scales array, both in C row-major order.  This means the
JavaScript runtime only needs to implement header parsing, decompression, and a
single multiply-broadcast to reconstruct float32 vectors — exactly the three-node
ONNX graph already exported by `python/onnx_export.py`.

---

## Options Considered

### Option 1 — WASM Compilation of the Mojo Quantizer

Compile `python/` Python logic or the Mojo quantizer to WebAssembly via Emscripten
or the forthcoming Mojo→WASM toolchain target.

**Pros:**
- Browser-compatible (runs inside any modern browser sandbox).
- Zero native binary distribution: ship a single `.wasm` file via npm alongside the
  TypeScript type definitions.
- Mojo's SIMD intrinsics map to WASM SIMD128 instructions.

**Cons:**
- Mojo→WASM is **not officially supported** in Mojo SDK ≤ 0.25.7; the toolchain
  target exists only as an experimental flag with no stability guarantees.
- WASM linear memory cap is 4 GB; large embedding databases (>500 M vectors) would
  need streaming from a server.
- The WASM execution model blocks the JavaScript event loop; callers must use Web
  Workers for non-blocking behaviour, adding significant glue code.
- Debugging across the Mojo→WASM boundary is immature.

**Overall:** Not viable until Mojo SDK v1.0 declares stable WASM support.  Deferred
to Phase 2 (tentatively v3.5.0 or later).

---

### Option 2 — Node.js N-API Native Addon

Write a C++ N-API addon that wraps the compiled Mojo binary (or a thin C shim over
the Python extension `.so`) and exposes the dequantize and `.vqz` parse functions
directly to Node.js.

**Pros:**
- Full native performance on Node.js (within ~5 % of the Mojo binary throughput on an
  x86-64 server).
- ABI-stable across Node.js v16–v22 via the N-API ABI guarantee — no recompilation
  per Node major version.
- `.vqz` header parsing is ≈ 20 lines of C++; the scalar dequantize kernel is one
  AVX2 loop easily authored by hand.
- `Int8Array` / `Float32Array` in JS share the same memory layout as NumPy `int8` /
  `float32` C-order arrays — zero-copy transfer from `.vqz` body to JS typed arrays
  is straightforward.
- Well-understood distribution path: `node-pre-gyp` / `node-gyp-build` with
  prebuilt platform tarballs for macOS-arm64, macOS-x64, linux-x64, linux-arm64,
  win32-x64.

**Cons:**
- Per-platform binary distribution: CI must build and upload 5 platform tarballs per
  release.
- Browser-incompatible: cannot be used in a `<script>` tag without a bundler shim.
- Requires maintaining a small C++ surface; any Vectro storage format change must be
  reflected here.

**Overall:** Recommended for Phase 1 (v3.3.0), server-side Node.js only.

---

### Option 3 — Pure JavaScript / TypeScript Reimplementation

Implement the `.vqz` reader and INT8 dequantize kernel entirely in TypeScript, using
`DataView` for header parsing, `pako` / native `DecompressionStream` for zstd/zlib,
and typed arrays for the dequant multiply.

**Pros:**
- Zero native binary; works in browser, Node.js, Deno, Cloudflare Workers, and Bun
  without any bundler configuration.
- No additional CI platform matrix.
- Simplest distribution: a single npm package with no optional native dependencies.

**Cons:**
- 10–20× slower than the N-API addon for large batches (no SIMD in JS without WASM).
- A second implementation of the dequantize logic that must be kept in sync with
  `python/interface.py` — an ongoing maintenance burden and a source of subtle
  numerical drift.
- `pako` does not support zstd; the native `DecompressionStream` API supports only
  gzip/deflate in current browser specs (zstd support is Chromium 123+, not yet in
  Firefox or Safari).

**Overall:** Rejected.  The maintenance burden of a second implementation outweighs
the simplicity benefit.  The pure-JS approach may be reconsidered as a thin shim
layer on top of the WASM build (Option 1) once the Mojo toolchain matures.

---

### Option 4 — Server-Only REST API (No Bindings)

Expose dequantize and `.vqz` read operations as a REST endpoint with a lightweight
HTTP server (e.g. FastAPI) and require JavaScript clients to call over the network.

**Pros:**
- No JS code to maintain.
- Language-agnostic: any HTTP client works.

**Cons:**
- Adds a network round-trip for every inference call — unacceptable latency for
  real-time applications.
- Introduces a stateful server dependency for what is fundamentally a stateless
  transformation.
- Out of scope for v3.x: Vectro is a library, not a service.

**Overall:** Out of scope.  Ruled out.

---

## Decision

**Adopt Option 2 (N-API native addon) as Phase 1 for v3.3.0.**

The N-API addon will:

1. Parse the 64-byte `.vqz` header (magic, version, comp_flag, n_vectors, dims).
2. Decompress the body using libzstd or zlib (statically linked).
3. Expose a `dequantize(quantizedInt8, scalesFloat32)` function that returns a
   `Float32Array` of shape `(n_vectors * dims,)`.
4. Be distributed as a prebuilt binary for the five priority targets:
   `darwin-arm64`, `darwin-x64`, `linux-x64`, `linux-arm64`, `win32-x64`.

TypeScript type definitions will be authored manually (not generated) to keep the
public API minimal and stable.

**Phase 2 (WASM, tentatively v3.5.0):** Revisit when Mojo SDK declares stable
WASM support.  The Phase 2 WASM build will target browser + edge environments and
should reuse the same TypeScript type definitions as the Phase 1 N-API addon.

---

## Consequences

- A new top-level directory `js/` is established in v3.3.0 containing the N-API
  addon source, `package.json`, and TypeScript definitions.
- CI gains a `node-addon` job matrix (macOS-arm64, ubuntu-x64) in `.github/workflows/ci.yml`.
- The `.vqz` binary format is now a cross-language public contract.  Breaking changes
  to `MAGIC`, `HEADER_SIZE`, or the body layout require a format version bump and a
  corresponding semver major bump in the `@vectro/core` npm package.
- Pure JS (Option 3) is explicitly rejected to avoid second-implementation drift.
  Any future pure-JS layer must be a generated wrapper over the WASM build, not a
  hand-authored reimplementation.

---

## References

- `python/storage_v3.py` — VQZ format specification and reference implementation.
- `python/onnx_export.py` — Three-node opset-17 ONNX graph that is the direct source
  of truth for the dequantize computation ported to N-API.
- [N-API documentation](https://nodejs.org/api/n-api.html) — ABI-stable Node.js addon API.
- [node-gyp-build](https://github.com/prebuild/node-gyp-build) — prebuilt binary loader for npm packages.
- [Mojo WASM RFC](https://github.com/modularml/mojo/issues/000) — upstream tracking issue (pending).
