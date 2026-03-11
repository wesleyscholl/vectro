# @vectro/core — Node.js N-API Addon

> **Status: ADR-001 Phase 1 scaffold** — the C++ implementation is stubbed.
> Full implementation is tracked as ADR-001 Phase 2 (v1.1.0).

Node.js native addon that exposes Vectro's INT8 dequantization and `.vqz` file
reader as a zero-copy N-API library for server-side JavaScript / TypeScript.

See [`docs/adr-001-javascript-bindings.md`](../docs/adr-001-javascript-bindings.md)
for the full decision record and design rationale.

---

## Requirements

- Node.js ≥ 16
- A C++17 compiler (`clang++` on macOS / Linux, MSVC 2019+ on Windows)
- `libzstd` development headers (static linking is preferred via `node-gyp`)

---

## Installation (from npm — Phase 2)

```bash
npm install @vectro/core
```

Pre-built binaries will be fetched automatically via `node-gyp-build` for:

| Platform | Arch |
|----------|------|
| macOS | arm64, x64 |
| Linux | x64, arm64 |
| Windows | x64 |

---

## Building from source

```bash
cd js/
npm install
npm run build
```

---

## API (TypeScript)

```typescript
import { dequantize, readVqz, VqzReader } from "@vectro/core";

// Read a .vqz artifact produced by python/storage_v3.py
const { quantized, scales, dims, n } = readVqz("/path/to/file.vqz");

// Dequantize: equivalent to quantized.astype(float32) * scales[:, None]
const floatVectors = dequantize(quantized, scales, dims);
// floatVectors: Float32Array of length n * dims

// Streaming reader for large files
const reader = new VqzReader("/path/to/large.vqz");
const data = reader.read();
reader.close();
```

Full TypeScript definitions are in [`index.d.ts`](./index.d.ts).

---

## Phase roadmap

| Phase | Version | Scope |
|-------|---------|-------|
| 1 (ADR-001) | v3.3.0 (Vectro) | Project scaffold, type definitions, C++ stub |
| 2 | v3.4.0 | Full N-API implementation: `.vqz` parser, zstd decompression, SIMD dequantize |
| 3 | v3.5.0 | WASM build for browser + edge (pending Mojo SDK WASM support) |
