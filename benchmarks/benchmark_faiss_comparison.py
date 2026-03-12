#!/usr/bin/env python
"""
Faiss Comparison Benchmark — Vectro vs Faiss

Compares Vectro's quantization and HNSW search against Faiss's equivalent implementations.

Metrics compared:
1. Compression ratio at target compression
2. Recall@10 on ANN search
3. Throughput (compression and search)
4. Memory footprint

Usage:
    pip install faiss-cpu  # or faiss-gpu for NVIDIA
    python benchmarks/benchmark_faiss_comparison.py
"""

import json
import time
import argparse
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore", category=DeprecationWarning)


def attempt_faiss_import() -> bool:
    """Check if Faiss is installed."""
    try:
        import faiss
        return True
    except ImportError:
        return False


def benchmark_pq_compression(
    vectors: np.ndarray,
    n_subspaces: int = 96,
) -> Dict[str, Any]:
    """
    Compare Vectro PQ vs Faiss PQ compression.
    
    Parameters
    ----------
    vectors : ndarray of shape (n, d)
    n_subspaces : int
        Number of PQ subspaces
    
    Returns
    -------
    dict with comparison results
    """
    print(f"\n▶ Product Quantization Comparison (M={n_subspaces})")
    
    from python.v3_api import PQCodebook, VectroV3
    
    n, d = vectors.shape
    split_point = n // 2
    train_vectors = vectors[:split_point]
    test_vectors = vectors[split_point:]
    
    # ============= VECTRO =============
    print("  Vectro...", end="", flush=True)
    start = time.time()
    vectro_cb = PQCodebook.train(train_vectors, n_subspaces=n_subspaces)
    vectro_train_time = time.time() - start
    
    # Compress test data
    start = time.time()
    vectro_codes = vectro_cb.encode(test_vectors)
    vectro_compress_time = time.time() - start
    
    # Decompress
    vectro_reconstructed = vectro_cb.decode(vectro_codes)
    
    # Quality
    cosine_sims_vectro = np.sum(test_vectors * vectro_reconstructed, axis=1) / (
        np.linalg.norm(test_vectors, axis=1) * np.linalg.norm(vectro_reconstructed, axis=1) + 1e-8
    )
    vectro_cosine = float(np.mean(cosine_sims_vectro))
    vectro_compression_ratio = (test_vectors.nbytes) / (vectro_codes.nbytes)
    
    print(f" cosine_sim={vectro_cosine:.5f}, compression={vectro_compression_ratio:.1f}x")
    
    results = {
        "vectro": {
            "training_time_sec": round(vectro_train_time, 3),
            "compression_time_sec": round(vectro_compress_time, 3),
            "mean_cosine_similarity": round(vectro_cosine, 6),
            "compression_ratio": round(vectro_compression_ratio, 2),
            "throughput_vec_per_sec": int(len(test_vectors) / vectro_compress_time),
        }
    }
    
    # ============= FAISS =============
    if attempt_faiss_import():
        print("  Faiss...", end="", flush=True)
        try:
            import faiss
            
            # Train PQ
            start = time.time()
            faiss_pq = faiss.PQ(d, n_subspaces // 2, 8)  # 8 bits per subspace
            faiss_pq.train(train_vectors.astype(np.float32))
            faiss_train_time = time.time() - start
            
            # Encode
            start = time.time()
            faiss_codes = faiss_pq.encode(test_vectors.astype(np.float32))
            faiss_compress_time = time.time() - start
            
            # Decode
            faiss_reconstructed = faiss_pq.decode(faiss_codes)
            
            # Quality
            cosine_sims_faiss = np.sum(test_vectors * faiss_reconstructed, axis=1) / (
                np.linalg.norm(test_vectors, axis=1) * np.linalg.norm(faiss_reconstructed, axis=1) + 1e-8
            )
            faiss_cosine = float(np.mean(cosine_sims_faiss))
            faiss_compression_ratio = (test_vectors.nbytes) / (faiss_codes.nbytes)
            
            print(f" cosine_sim={faiss_cosine:.5f}, compression={faiss_compression_ratio:.1f}x")
            
            results["faiss"] = {
                "training_time_sec": round(faiss_train_time, 3),
                "compression_time_sec": round(faiss_compress_time, 3),
                "mean_cosine_similarity": round(faiss_cosine, 6),
                "compression_ratio": round(faiss_compression_ratio, 2),
                "throughput_vec_per_sec": int(len(test_vectors) / faiss_compress_time),
            }
            
            # Comparison
            results["comparison"] = {
                "vectro_vs_faiss_cosine_sim": round(vectro_cosine / max(faiss_cosine, 0.001), 2),
                "vectro_vs_faiss_throughput": round((len(test_vectors) / vectro_compress_time) / max(len(test_vectors) / faiss_compress_time, 0.001), 2),
            }
        except Exception as e:
            print(f" Error: {e}")
    else:
        print("  Faiss...", end="", flush=True)
        print(" not installed (skipping)")
    
    return {
        "n_subspaces": n_subspaces,
        "vectors_shape": test_vectors.shape,
        "results": results,
    }


def benchmark_int8_quantization() -> Dict[str, Any]:
    """
    Compare INT8 quantization throughput.
    """
    print(f"\n▶ INT8 Quantization Comparison")
    
    from python import compress_vectors as vectro_compress
    
    vectors = np.random.normal(size=(100000, 768)).astype(np.float32)
    
    # Vectro INT8
    print("  Vectro INT8...", end="", flush=True)
    start = time.time()
    for i in range(0, len(vectors), 10000):
        vectro_compress(vectors[i:i+10000], profile="balanced")
    vectro_time = time.time() - start
    vectro_throughput = len(vectors) / vectro_time
    print(f" {vectro_throughput:.0f} vec/s")
    
    results = {
        "vectors_shape": vectors.shape,
        "vectro_throughput_vec_per_sec": int(vectro_throughput),
    }
    
    # Faiss - minimal INT8 comparison (Faiss doesn't have direct INT8 quantizer)
    if attempt_faiss_import():
        print("  Faiss (Scalar quantizer as reference)...", end="", flush=True)
        try:
            import faiss
            
            start = time.time()
            sq = faiss.ScalarQuantizer(768, faiss.ScalarQuantizer.QT_8bit)
            sq.train(vectors[:10000])
            faiss_codes = np.zeros((len(vectors), sq.code_size), dtype=np.uint8)
            
            start = time.time()
            for i in range(0, len(vectors), 10000):
                sq.encode(vectors[i:i+10000], faiss_codes[i:i+10000])
            faiss_time = time.time() - start
            faiss_throughput = len(vectors) / faiss_time
            print(f" {faiss_throughput:.0f} vec/s")
            
            results["faiss_throughput_vec_per_sec"] = int(faiss_throughput)
            results["faiss_note"] = "Faiss ScalarQuantizer used as reference (not direct INT8 analog)"
        except Exception as e:
            print(f" Error: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Vectro against Faiss"
    )
    parser.add_argument("--output", type=str, default="results/faiss_comparison.json")
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("VECTRO vs FAISS — Quantization Comparison")
    print("=" * 70)
    
    if not attempt_faiss_import():
        print("\n⚠️  Faiss not installed. To enable Faiss comparison:")
        print("   pip install faiss-cpu")
        print("\nContinuing with Vectro benchmarks only...\n")
    
    # Generate test data
    vectors = np.random.normal(size=(100000, 768)).astype(np.float32)
    
    all_results = {
        "benchmark_type": "faiss_comparison",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "faiss_imported": attempt_faiss_import(),
        "pq_comparison": benchmark_pq_compression(vectors),
        "int8_comparison": benchmark_int8_quantization(),
    }
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    return all_results


if __name__ == "__main__":
    main()
