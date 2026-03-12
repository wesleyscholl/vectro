# Migration Guide: Mojo Runtime → Rust Runtime (v3.x → v4.0)

This guide covers migrating application code from vectro v3.x (Mojo accelerated
runtime) to v4.0 (pure Rust).

---

## Why the change?

| | v3.x (Mojo) | v4.0 (Rust) |
|--|--|--|
| **Install** | `pixi install` + Mojo SDK | `pip install vectro` |
| **Platform** | macOS ARM64 only | macOS + Linux (x86_64 + ARM64) |
| **CI** | Manual wheel baked per release | GitHub Actions maturin matrix |
| **Build time** | ~4 min (Mojo compile) | ~30 s (cached Rust) |
| **Performance** | Mojo SIMD | Rust + NEON / AVX2 SIMD |

All public APIs are preserved; only import paths changed.

---

## Environment setup

### v3.x

```bash
pixi install
pixi shell
```

### v4.0

```bash
pip install vectro           # or: pip install vectro==4.0.0
```

No Mojo SDK, no pixi, no `vectro_quantizer` native extension required.

---

## Python API changes

### INT8 encoder

```python
# v3.x
from python._mojo_bridge import MojoInt8Encoder
enc = MojoInt8Encoder(dim=768)
codes, scales = enc.encode(vectors)          # list[ndarray] return

# v4.0
from vectro_py import PyInt8Encoder
enc = PyInt8Encoder(768)
codes  = enc.encode_np(vectors)              # zero-copy ndarray, dtype=int8
scales = ...                                 # returned together via encode()
codes, scales = enc.encode(vectors.tolist()) # list API still available
```

`encode_np` takes a `(n, d)` numpy `float32` array and returns a `(n, d)
int8` ndarray with **no copy** across the Python/Rust boundary.

### NF4 encoder

```python
# v3.x
from python.nf4 import Nf4Encoder
enc = Nf4Encoder(dim=768)
packed = enc.encode(vectors)

# v4.0
from vectro_py import PyNf4Encoder
enc = PyNf4Encoder(768)
packed = enc.encode(vectors.tolist())
```

### Binary encoder

```python
# v3.x
from python.binary_quant import BinaryEncoder

# v4.0
from vectro_py import PyBinaryEncoder
```

Constructor and `encode` / `decode` signature unchanged.

### Product Quantization

```python
# v3.x
from python.pq import PQCodebook
cb = PQCodebook(dim=768, n_subspaces=96, n_centroids=256)
cb.train(vectors)
codes = cb.encode(vectors)

# v4.0
from vectro_py import PyPQCodebook
cb = PyPQCodebook(768, 96, 256)
cb.train(vectors.tolist())
codes = cb.encode(vectors.tolist())
```

### HNSW index

```python
# v3.x
from python.hnsw import HnswIndex
idx = HnswIndex(dim=768, M=16, ef_construction=200)
for vec in vectors:
    idx.add(vec)
results = idx.search(query, k=10, ef=50)

# v4.0
from vectro_py import PyHnswIndex
idx = PyHnswIndex(768, 16, 200)
idx.add_np(vectors)           # bulk insert — zero-copy (n, d) float32
results = idx.search_np(query_row, k=10, ef=50)  # returns list[(dist, id)]

# Single-vector insert still works:
idx.add(vector.tolist())
```

---

## CLI changes

The CLI command name and sub-commands are identical.  The binary is now a
compiled Rust executable bundled inside the wheel; the Python shim dispatches
to it automatically.

```bash
# v3.x (required pixi shell)
vectro compress --input embeddings.npy --output compressed.npz --mode int8

# v4.0 (works in any venv after pip install)
vectro compress --input embeddings.npy --output compressed.npz --mode int8
```

Available `--mode` values: `int8` (default), `nf4`, `binary`, `pq`.

---

## Removed APIs

| v3.x symbol | Reason | v4.0 replacement |
|--|--|--|
| `vectro_quantizer` native extension | Replaced by `vectro_py` (Rust/PyO3) | `from vectro_py import ...` |
| `python._mojo_bridge` | Mojo runtime removed | `vectro_py.*` |
| `pixi.toml` tasks | pixi no longer required | `cargo` / `maturin` |

The `pixi.toml` file is retained in the repo as a historical artifact but is
not used by v4.0 builds.

---

## Data compatibility

Compressed files (`.npz`) produced by v3.x are fully compatible with v4.0.
The on-disk format (numpy arrays + metadata dict) did not change.

HNSW index serialisation uses `bincode` (Rust) in v4.0.  Indexes saved by v3.x
must be rebuilt; they cannot be loaded with `PyHnswIndex.load()`.

---

## Testing your migration

```python
import numpy as np
from vectro_py import PyInt8Encoder, PyHnswIndex

rng = np.random.default_rng(42)
vecs = rng.standard_normal((1000, 768)).astype(np.float32)
# normalise
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

enc = PyInt8Encoder(768)
codes, scales = enc.encode(vecs.tolist())

idx = PyHnswIndex(768, M=16, ef_construction=200)
idx.add_np(vecs)
results = idx.search_np(vecs[0], k=5, ef=50)
print("Top-5 neighbours:", results)
```

If this snippet runs without error, the migration is complete.
