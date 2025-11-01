# 🎬 Vectro Demo - Quick Start Guide

## TL;DR - Record Your Video NOW!

```bash
cd /Users/wscholl/vectro
./demos/run_complete_demo.sh
```

**That's it!** This single command:
- Downloads 3 real public datasets (SIFT1M, GloVe, SBERT)
- Runs comprehensive benchmarks on all three
- Shows beautiful comparison tables
- Takes ~5-6 minutes total

**Perfect for screen recording!** 📹

---

## 📊 What You Get

### Three Different Embedding Types

1. **SIFT1M (128D)** - Computer vision descriptors
   - Gold standard for vector similarity benchmarks
   - 100,000 vectors from INRIA

2. **GloVe (100D)** - Stanford word embeddings
   - Most cited word embedding paper
   - 400,000 word vectors

3. **SBERT (384D)** - Sentence transformers
   - Modern NLP embeddings
   - Perfect for RAG/semantic search demos

### Expected Results

| Metric | Value |
|--------|-------|
| Average Throughput | **830K vectors/sec** |
| Average Accuracy | **99.97%** |
| Compression Ratio | **3.9x (74% savings)** |
| Consistency | **±0.03% variation** |

---

## 🎬 Recording Options

### Option 1: Complete Multi-Dataset Demo (Best!)

**Command:**
```bash
./demos/run_complete_demo.sh
```

**Duration:** 5-6 minutes  
**Best for:** Showing versatility and consistency  
**Recording guide:** `demos/MULTI_DATASET_RECORDING_GUIDE.md`

### Option 2: Single Dataset Deep Dive

**Commands:**
```bash
# Download
python demos/download_public_dataset.py --dataset glove --dim 100

# Benchmark
python demos/benchmark_public_data.py --embeddings data/glove.6B.100d.npy
```

**Duration:** 4-5 minutes  
**Best for:** Detailed analysis of one dataset  
**Recording guide:** `demos/RECORDING_GUIDE.md`

### Option 3: Quick 2-Minute Version

**Command:**
```bash
python demos/benchmark_all_datasets.py --sample 5000 --datasets sift1m glove
```

**Duration:** 2-3 minutes  
**Best for:** Social media, quick demos

---

## 📋 Pre-Recording Checklist

- [ ] Install dependencies: `pip install numpy tqdm`
- [ ] Test run the script once (don't record yet!)
- [ ] Set terminal to high-contrast theme (Dracula, Nord)
- [ ] Increase font size to 16pt+
- [ ] Clear terminal: `clear`
- [ ] Start screen recording (1080p @ 60fps)
- [ ] Run the script!

---

## 🎯 Key Talking Points

### Opening (30 seconds)
- "Testing Vectro with 3 different embedding types"
- "Computer vision + NLP to prove consistency"

### During Benchmarks (3-4 minutes)
- "SIFT1M is the gold standard for vector benchmarks"
- "Stanford GloVe - one of the most cited papers"
- "SBERT for modern semantic search"
- Point out speed, accuracy, compression for each

### Results (1-2 minutes)
- "Look at this consistency - 99.97% across all three"
- "830K vectors per second average"
- "Works the same whether it's vision or NLP"

### Closing (30 seconds)
- "Pure Mojo, 100% test coverage"
- "Open source, MIT license"
- "You can run this exact benchmark yourself"

---

## 🎨 Editing Tips

1. **Speed up long progress bars** to 1.5-2x
2. **Add text overlays** for key metrics:
   - "830K vec/sec"
   - "99.97% Accuracy"
   - "3.9x Compression"
3. **Include timestamps** in YouTube description
4. **Create thumbnail** with key numbers

---

## 📦 All Demo Files

```
demos/
├── run_complete_demo.sh              # 🔥 Run this!
├── download_public_dataset.py        # Download individual datasets
├── benchmark_public_data.py          # Benchmark single dataset
├── benchmark_all_datasets.py         # Compare all datasets
├── MULTI_DATASET_RECORDING_GUIDE.md  # Full recording script
├── RECORDING_GUIDE.md                # Single dataset guide
└── README.md                         # Complete documentation
```

---

## 🚀 After Recording

1. **Upload to YouTube** with proper title/description
2. **Share on Twitter/LinkedIn** with key metrics
3. **Post in Mojo community** 
4. **Update Vectro README** with video link

---

## 💡 Pro Tips

- **Practice once** before recording (don't show the practice run)
- **Narrate while typing** - builds engagement
- **Point to numbers** as they appear on screen
- **Emphasize consistency** across datasets
- **Mention "reproducible"** multiple times

---

## ✨ You're Ready!

Just run:
```bash
./demos/run_complete_demo.sh
```

And start recording! 🎬

The script does everything for you. Just narrate what you're seeing and emphasize the key points above.

**Good luck!** 🚀
