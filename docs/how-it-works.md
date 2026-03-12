# How Vectro Works

> Mathematical explanations of all quantization algorithms and the HNSW index
> implemented in Vectro's Rust core (`rust/vectro_lib/`).

---

## Table of Contents

1. [INT8 Symmetric Abs-Max Quantization](#int8-symmetric-abs-max-quantization)
2. [NF4 Normal-Float 4-Bit Quantization](#nf4-normal-float-4-bit-quantization)
3. [Binary 1-Bit Quantization](#binary-1-bit-quantization)
4. [Product Quantization (PQ)](#product-quantization-pq)
5. [HNSW Approximate Nearest Neighbour Index](#hnsw-approximate-nearest-neighbour-index)
6. [Quality Metrics](#quality-metrics)

---

## INT8 Symmetric Abs-Max Quantization

**Source**: `rust/vectro_lib/src/quant/int8.rs`  
**Quality target**: cosine similarity ≥ 0.9999 vs original  
**Compression ratio**: 4× (float32 → int8)

### Algorithm

For each vector **v** ∈ ℝᵈ independently:

1. **Compute scale**:
   ```
   s = max(|v₀|, |v₁|, …, |v_{d-1}|)
   ```
   If `s = 0` (zero vector), set `s = 1`.

2. **Encode**:
   ```
   qᵢ = round(vᵢ / s × 127)    qᵢ ∈ [-127, 127]
   ```

3. **Decode** (approximate reconstruction):
   ```
   v̂ᵢ = qᵢ × s / 127
   ```

The scale `s` is stored alongside the 8-bit codes (4 bytes overhead per vector).

### NEON SIMD acceleration (AArch64)

On Apple Silicon and ARM64 Linux, `Int8Vector::encode_fast` dispatches to an
intrinsic path (`encode_neon`):

- **Pass 1 — abs-max**: `vmaxq_f32(vmax, vabsq_f32(vld1q_f32(...)))` in groups
  of 4, then `vmaxvq_f32` to horizontally reduce.
- **Pass 2 — quantize**: 16 elements per iteration using four `float32x4_t`
  registers `→ int32x4_t` via `vcvtq_s32_f32(vrndnq_f32(vmulq_f32(...)))`,
  then narrowed to `int16x8_t → int8x16_t` via `vmovn_s32 → vqmovn_s16`,
  stored with `vst1q_s8`.

Throughput on M3 (d=768, n=100K, rayon parallel):
`encode_batch` processes the full 100K×768 corpus in roughly the same wall
time as the Mojo SIMD kernel (≥ 12 M vec/s).

---

## NF4 Normal-Float 4-Bit Quantization

**Source**: `rust/vectro_lib/src/quant/nf4.rs`  
**Quality target**: cosine similarity ≥ 0.985  
**Compression ratio**: 8× (float32 → 4-bit, 2 values per byte)

### Codebook

NF4 uses 16 fixed quantisation levels designed so that the mapping preserves
data distributed according to **N(0, 1)**.  The levels are drawn from the
theoretical optimal quantiser for a normal distribution (Dettmers et al. 2023,
*QLoRA*):

```
NF4_LEVELS = [
  −1.0,      −0.6961928,  −0.5250730,  −0.3949951,
  −0.2844676, −0.1848719,  −0.0912280,   0.0,
   0.0794528,  0.1609292,   0.2461123,   0.3379152,
   0.4407289,  0.5626170,   0.7229568,   1.0
]
```

### Encoding

For each value `x` in the (optionally normalised) vector, find the nearest
codebook level by binary search on the midpoints array:

```
level_idx = upper_bound(NF4_MIDS, |x|)
```

Two 4-bit indices are packed into one byte: `byte = (hi_nibble << 4) | lo_nibble`.

### Decoding

Unpack nibbles, look up `NF4_LEVELS[nibble]`, multiply by the per-vector
abs-max scale.

---

## Binary 1-Bit Quantization

**Source**: `rust/vectro_lib/src/quant/binary.rs`  
**Quality target**: recall@10 ≥ 0.95 (with optional float re-rank)  
**Compression ratio**: 32× (float32 → 1 bit)

### Encoding

```
bᵢ = 1  if vᵢ > 0
bᵢ = 0  if vᵢ ≤ 0
```

Bits are packed LSB-first into `u8` bytes (`ceil(d/8)` bytes per vector).

### Search: Hamming distance

For query **q** and database vector **b**, the Hamming distance counts the
number of differing bits:

```
hamming(q, b) = popcount(q_packed XOR b_packed)
```

Retrieved candidates can be re-ranked with full float cosine similarity to boost
recall (the `binary_search` convenience function does this transparently when
`normalize = true`).

---

## Product Quantization (PQ)

**Source**: `rust/vectro_lib/src/quant/pq.rs`  
**Quality target**: cosine similarity ≥ 0.95  
**Compression ratio**: up to 96× with `M=96, K=256`

### Training

Divide each d-dimensional vector into **M** equal sub-vectors of dimension
`d/M`.  For each sub-space `m`, run Lloyd's k-means to learn **K** centroids:

```
k-means:
  initialise K centroids randomly (seeded)
  repeat (max_iter times):
    assign each sub-vector to nearest centroid
    update centroid = mean of assigned sub-vectors
```

### Encoding

Assign each sub-vector to its nearest centroid index: result is **M** bytes
(assuming K ≤ 256) per vector.

### Asymmetric Distance Computation (ADC)

For a query **q**, precompute a K×M distance table:

```
D[m][k] = || q_m − centroid[m][k] ||²
```

The approximate squared distance from **q** to database vector **x** is:

```
dist(q, x) ≈ Σₘ D[m][code_m(x)]
```

ADC allows scanning a million vectors in < 1 ms without decoding any stored vector.

---

## HNSW Approximate Nearest Neighbour Index

**Source**: `rust/vectro_lib/src/index/hnsw.rs`  
**Quality target**: recall@10 ≥ 0.97  
**Reference**: Malkov & Yashunin 2018, *arXiv:1603.09320*

### Structure

HNSW builds a multi-layer navigable small-world graph.

- **Layer 0** (bottom): all nodes, maximum `2*M` bi-directional links.
- **Layers 1–L** (upper): randomly selected subset of nodes, maximum `M` links.

Each node is inserted at a randomly selected maximum level:

```
level = floor(−ln(uniform(0, 1)) × mL)
mL = 1 / ln(M)
```

Vectro uses a deterministic LCG hash of the node id in place of a pseudo-RNG
so the index is reproducible without an RNG dependency:

```rust
fn random_level(&self) -> usize {
    let id = self.vectors.len() as u64;
    let r = lcg_hash(id);
    let frac = (r >> 11) as f64 / (1u64 << 53) as f64;
    ((-frac.ln()) * self.ml) as usize
}
```

### Insertion

1. Greedy descent from `max_level` to `node_level + 1` with `ef = 1`.
2. Beam search from `min(node_level, max_level)` to `0` with `ef_construction`.
3. Prune links at each layer to at most `M` (or `2M` at layer 0) neighbours
   using a simple nearest-first heuristic.
4. Add reverse links; prune neighbours of existing nodes if they exceed the cap.

### Search

1. Greedy descent from `max_level` to layer `1` with beam width `1`.
2. Beam search at layer `0` with beam width `ef` (≥ k).
3. Return top-k results sorted ascending by cosine distance.

### Distance metric

All stored vectors are pre-normalised to unit length.  Cosine distance is:

```
d(a, b) = 1 − a · b        (a, b unit vectors → dot product = cosine sim)
```

---

## Quality Metrics

| Metric | Definition |
|--------|-----------|
| **Cosine similarity** | `cos(v, v̂) = (v · v̂) / (‖v‖ ‖v̂‖)` — 1.0 = identical direction |
| **Recall@k** | Fraction of true top-k neighbours returned by ANN search |
| **NDCG@k** | Normalised Discounted Cumulative Gain — position-weighted recall |

Vectro's test suite enforces the quality thresholds documented in `PLAN.md` as
algorithmic correctness assertions in `cargo test --workspace`.
