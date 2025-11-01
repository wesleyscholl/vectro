# Vectro Demos

This directory contains comprehensive demos and benchmarks for Vectro.

## 🎬 Real-World Benchmark (Recommended for Videos)

### Quick Start: All Datasets at Once! 🚀

Run this single command to download all datasets and benchmark them:

```bash
./demos/run_complete_demo.sh
```

This will:
1. Download SIFT1M (Learn set - 100K vectors, 128D)
2. Download GloVe (400K words, 100D)
3. Create SBERT sample (10K vectors, 384D)
4. Run comprehensive benchmarks on all three
5. Generate comparison analysis

**Perfect for demo videos!**

---

### Step 1: Download Individual Datasets

**Option A: SIFT1M - Classic ANN Benchmark (Recommended for credibility)**

```bash
python demos/download_public_dataset.py --dataset sift1m --sift-subset learn
```

**Option B: GloVe - Stanford Word Embeddings (Recommended for demos)**

```bash
python demos/download_public_dataset.py --dataset glove --dim 100
```

**Option C: SBERT - Sentence Transformers (Recommended for NLP)**

```bash
python demos/download_public_dataset.py --dataset sbert --num 10000
```

**Available datasets:**
- `sift1m` - INRIA SIFT1M (128D) - Gold standard for vector similarity
  - `learn`: 100K vectors, `base`: 1M vectors, `query`: 10K vectors
- `glove` - Stanford GloVe pre-trained word embeddings (50D-300D)
- `sbert` - Sentence-BERT style embeddings (384D) 
- `openai` - OpenAI text-embedding-ada-002 style (1536D)

**Options:**
```bash
# Download specific dimension
python demos/download_public_dataset.py --dataset glove --dim 300

# Limit vocabulary size (faster)
python demos/download_public_dataset.py --dataset glove --dim 100 --vocab 50000

# Create synthetic OpenAI-style embeddings
python demos/download_public_dataset.py --dataset openai --dim 1536 --num 10000
```

### Step 2: Run Benchmarks

**Option A: Single Dataset Benchmark**

```bash
# SIFT1M
python demos/benchmark_public_data.py --embeddings data/sift1m_learn.npy

# GloVe
python demos/benchmark_public_data.py --embeddings data/glove.6B.100d.npy

# SBERT
python demos/benchmark_public_data.py --embeddings data/sbert_msmarco_sample_10000.npy
```

**Option B: Multi-Dataset Comparison (Best for demos!)**

```bash
# Benchmark all datasets and compare
python demos/benchmark_all_datasets.py --sample 10000

# Benchmark specific datasets
python demos/benchmark_all_datasets.py --datasets sift1m glove
```

**What it benchmarks:**
- ✅ Quantization speed & throughput
- ✅ Quality & accuracy (MAE, RMSE, cosine similarity)
- ✅ Compression ratios & space savings
- ✅ Similarity search preservation
- ✅ Cross-dataset consistency analysis

**Quick Options:**
```bash
# Fast test (1000 vectors)
python demos/benchmark_public_data.py --embeddings data/glove.6B.100d.npy --sample 1000 --skip-search

# Medium test (5000 vectors)
python demos/benchmark_public_data.py --embeddings data/sift1m_learn.npy --sample 5000
```

### Step 3: Record Your Video

Follow the complete recording guide:

```bash
cat demos/RECORDING_GUIDE.md
```

---

## 🔥 Quick Mojo Demo

For a fast native Mojo demo (no external data needed):

```bash
mojo run demos/quick_demo.mojo
```

---

## 📊 Mojo Benchmark Demo

Comprehensive Mojo-native benchmark:

```bash
mojo run demos/benchmark_demo.mojo
```

---

## 🎨 Python Visual Demos

### Animated Demo
```bash
python demos/demo_animated.py
```

### Visual Benchmark
```bash
python demos/visual_bench.py
```

---

## 📋 Requirements

### For Python Demos
```bash
pip install numpy tqdm
```

### For Mojo Demos
```bash
pixi install
pixi shell
```

---

## 🎯 Recommended Demo Flow for Videos

### Option 1: Complete Multi-Dataset Demo (Best!)

1. **Introduction** - "Testing Vectro with 3 different embedding types"
2. **Run complete demo** - `./demos/run_complete_demo.sh`
3. **Show results** - Point out consistency across datasets
4. **Wrap up** - Emphasize versatility and production-readiness

**Total time:** 5-6 minutes  
**Datasets:** SIFT1M + GloVe + SBERT  
**Key message:** Works consistently across vision and NLP embeddings

### Option 2: Single Dataset Deep Dive

1. **Introduction** - Explain Vectro's purpose
2. **Download real data** - `python demos/download_public_dataset.py --dataset glove --dim 100`
3. **Run benchmark** - `python demos/benchmark_public_data.py --embeddings data/glove.6B.100d.npy --sample 10000`
4. **Show results** - Point out key metrics as they appear
5. **Wrap up** - Summarize performance and use cases

**Total time:** 4-5 minutes  
**Dataset:** Real Stanford GloVe embeddings  
**Key message:** Proven performance on well-known data

---

## 📊 Expected Performance

### Single Dataset Performance

| Dataset | Dimensions | Throughput | Accuracy | Compression |
|---------|-----------|------------|----------|-------------|
| SIFT1M | 128D | 900K-1M vec/s | 99.95%+ | 3.88x |
| GloVe | 100D | 800K-1M vec/s | 99.97%+ | 3.85x |
| SBERT | 384D | 700K-900K vec/s | 99.96%+ | 3.96x |

### Cross-Dataset Consistency

| Metric | Average | Variation |
|--------|---------|-----------|
| Throughput | 850K vec/s | ±100K |
| Accuracy | 99.96% | ±0.02% |
| Compression | 3.9x ratio | ±0.05x |
| Space Saved | 74.5% | ±0.5% |
| Latency | <2ms/vec | <0.5ms |

---

## 🎬 Files

- `download_public_dataset.py` - Download real public embeddings
- `benchmark_public_data.py` - Comprehensive benchmark script
- `RECORDING_GUIDE.md` - Complete video recording guide
- `VIDEO_SCRIPT.md` - Original video script template
- `benchmark_demo.mojo` - Mojo-native benchmark
- `quick_demo.mojo` - Fast Mojo demo
- Other demo files for various visualization styles

---

## 💡 Tips

- **For best results:** Use GloVe 100D with 10K sample
- **For speed:** Use `--skip-search` flag
- **For credibility:** Always mention "Stanford GloVe" dataset
- **For visuals:** Screen record at 1080p/60fps with high-contrast terminal theme

---

Happy benchmarking! 🚀
