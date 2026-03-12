# Vectro Benchmarking Guide

This document explains how to measure Vectro's performance and what conditions affect the numbers you'll see.

## Understanding Performance Modes

### Python/NumPy Fallback Mode (Always Available)

**When used:** If Mojo binary is not available or built  
**Throughput:** ~60–80K vec/s for INT8 at d=768 (depends on backend availability)

**Factors affecting throughput:**
- `squish_quant` Rust extension availability → ~3-4x speedup if installed
- NumPy BLAS configuration (OpenBLAS vs MKL → 2-3x difference)
- Batch size (larger batches → 10% faster, due to reduced Python overhead)
- Vector dimensionality (lower dims → higher throughput due to better cache locality)

**Measured (March 2026, codespace CPU, NumPy-only):**
```
d=768:   62K vec/s  (batch=10000)
d=384:   69K vec/s  (batch=10000)
d=128:   77K vec/s  (batch=10000)
```

### Mojo SIMD Mode (Requires Mojo Toolchain)

**When used:** After running `pixi run build-mojo`  
**Throughput:** 5M+ vec/s for INT8 at d=768 (Apple M-series / modern x86)

**Conditions for these numbers:**
- M-series Mac (Apple Silicon) or modern NVIDIA GPU
- Compiled Mojo binary with SIMD vectorization
- Batch size >= 1000 (smaller batches have dispatch overhead)

## Running Your Own Benchmarks

### Python/NumPy Fallback (This Environment)

Always available, no dependencies beyond numpy + pip:

```bash
python benchmarks/benchmark_python_fallback.py --output results/fallback_benchmark.json
```

Outputs JSON with:
- INT8 throughput at dimensions [128, 384, 768, 1536]
- Quality metrics (cosine similarity, compression ratio)
- Multiple batch sizes

### Mojo Binary (Requires Mojo Toolchain)

On a system with Mojo installed:

```bash
# Build the binary (one-time)
pixi run build-mojo

# Run benchmarks
pixi run benchmark
# Output format: Vec/s, cosine_sim, and compression_ratio for INT8/NF4/Binary
```

This runs: `./vectro_quantizer benchmark 10000 768`

## Interpreting Benchmark Results

### Reasonable Expectations

| Mode | d=768 | Condition |
|------|-------|-----------|
| NumPy (no squish) | 60-80K | Pure Python + NumPy BLAS |
| NumPy (+ squish) | 200-300K | With Rust INT8 fallback |
| Mojo SIMD | 5M+ | Apple Silicon or x86 SIMD |
| Mojo + GPU | 50M+ | NVIDIA A100 / Apple GPU |

### What Doesn't Matter

- Exact numbers will vary by ±20% due to system load, Python GIL contention, etc.
- Small batches (size=100) are slower due to dispatch overhead — use batch >= 1000 for fair comparison

### What Matters

- **Relative comparison:** Mojo should be 50-100x faster than NumPy on your hardware
- **Quality metrics:** cosine_sim >= 0.9999 for INT8, >= 0.98 for NF4
- **Reproducibility:** Run the same benchmark twice; results should match within ~5%

## Reporting Benchmark Results

If you publish results, include:

1. **Configuration:**
   - `Vectro version` (e.g., v3.4.0)
   - `Python version` (e.g., 3.11.9)
   - `NumPy version` (e.g., 1.24.3)
   - `squish_quant` presence (yes/no)
   - `Mojo version` (if using binary)

2. **Hardware:**
   - CPU model (e.g., Apple M3 Pro)
   - RAM
   - NUMA configuration (if relevant)

3. **Methodology:**
   - Vector data source (random, real embeddings, etc.)
   - Dimensions tested
   - Batch sizes used
   - Warm-up iterations
   - Number of trial runs

4. **Results:**
   - Raw JSON from benchmark script
   - Summary table
   - Comparison to Faiss/ONNX/etc. if available

## Known Issues / Caveats

- **NumPy throughput claims in README (200K–1.04M vec/s):** These were measured with squish_quant. Without it, expect 60–80K vec/s.
- **Mojo binary builds require Mojo >= 0.25.7** on Apple Silicon (Linux/x86 support in-progress).
- **GPU benchmarks require MAX Engine SDK** to enable GPU path. Falls back to CPU SIMD if unavailable.
- **Batch size very important:** OPT-in batches of 10000 for repeatability; don't compare batch=100 to batch=10000.

## Next: Faiss Comparison

To compare against Faiss on standard ANN benchmarks:

1. Install faiss: `pip install faiss-cpu` (or faiss-gpu)
2. Download SIFT1M or Glove-100 benchmark dataset
3. Run PQ compression at the same compression ratio on both libraries
4. Compare recall@10, throughput, and memory footprint

Vectro's HNSW index should yield recall@10 >= 0.97 on 1M vectors with INT8 storage.
