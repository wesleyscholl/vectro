"""Benchmark harness for Vectro MVP.

Measures:
 - original vs compressed byte sizes
 - mean cosine similarity between original and reconstructed embeddings
 - recall@k on a small retrieval task (brute-force cosine)
 - throughput (vectors/sec) for quantize and reconstruct

Usage:
  python python/bench.py --n 2000 --d 128 --queries 100 --k 10
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import numpy as np

# Ensure project root is importable when running this script directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from python import interface


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # normalized dot product rows of a vs rows of b -> matrix (a_rows, b_rows)
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_n @ b_n.T


def topk_indices(sim_mat: np.ndarray, k: int) -> np.ndarray:
    # return top-k indices along axis 1 (largest cosine)
    # sim_mat shape (q, n)
    return np.argpartition(-sim_mat, kth=k-1, axis=1)[:, :k]


def recall_at_k(orig: np.ndarray, recon: np.ndarray, queries_idx: np.ndarray, k: int) -> float:
    # orig: (n,d), recon: (n,d), queries_idx: indices of queries in dataset
    qv = orig[queries_idx]
    # compute ground truth top-1 (excluding self) using orig
    sim_orig = cosine_sim_matrix(qv, orig)
    # if self present at same index, mask it
    for i, idx in enumerate(queries_idx):
        sim_orig[i, idx] = -np.inf
    true_top1 = np.argmax(sim_orig, axis=1)

    # compute top-k from recon
    sim_recon = cosine_sim_matrix(qv, recon)
    for i, idx in enumerate(queries_idx):
        sim_recon[i, idx] = -np.inf
    topk = topk_indices(sim_recon, k)

    # recall: fraction where true_top1 is in topk
    hits = 0
    for i in range(len(queries_idx)):
        if true_top1[i] in topk[i]:
            hits += 1
    return hits / len(queries_idx)


def run_once(embeddings: np.ndarray, k: int, queries: int, force_python=False):
    n, d = embeddings.shape
    # optionally force Python fallback
    mojo_present = interface._mojo_quant is not None
    if force_python:
        saved = interface._mojo_quant
        interface._mojo_quant = None

    t0 = time.time()
    out = interface.quantize_embeddings(embeddings)
    t1 = time.time()
    q = out['q']
    scales = out['scales']
    dims = out['dims']
    nvecs = out['n']
    quant_time = t1 - t0

    t0 = time.time()
    recon = interface.reconstruct_embeddings(q, scales, dims)
    t1 = time.time()
    recon_time = t1 - t0

    mean_cos = interface.mean_cosine_similarity(embeddings, recon)
    orig_bytes = embeddings.nbytes
    comp_bytes = np.asarray(q).nbytes + np.asarray(scales).nbytes

    # pick queries indices
    rng = np.random.default_rng(1)
    queries_idx = rng.choice(n, size=min(queries, n), replace=False)
    recall = recall_at_k(embeddings, recon, queries_idx, k)

    if force_python:
        interface._mojo_quant = saved

    return {
        'quant_time': quant_time,
        'recon_time': recon_time,
        'mean_cos': mean_cos,
        'orig_bytes': orig_bytes,
        'comp_bytes': comp_bytes,
        'recall@k': recall,
        'mojo_present': mojo_present and not force_python,
        'n': n,
        'd': d,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=2000, help='number of dataset vectors')
    parser.add_argument('--d', type=int, default=128, help='dimension')
    parser.add_argument('--queries', type=int, default=100, help='number of queries')
    parser.add_argument('--k', type=int, default=10, help='recall@k')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    emb = rng.standard_normal((args.n, args.d)).astype(np.float32)

    print(f"Dataset: n={args.n}, d={args.d}, queries={args.queries}, k={args.k}")

    results = {}

    # If Mojo is present, run it; also run Python fallback for comparison
    mojo_available = interface._mojo_quant is not None
    if mojo_available:
        print("Running Mojo-backed quantization (if available)...")
        r_mojo = run_once(emb, args.k, args.queries, force_python=False)
        results['mojo'] = r_mojo

    print("Running Python fallback quantization...")
    r_py = run_once(emb, args.k, args.queries, force_python=True)
    results['python'] = r_py

    # Print formatted results
    for k, v in results.items():
        print('\nBackend:', k)
        print(f"  mojo_present: {v['mojo_present']}")
        print(f"  quant_time: {v['quant_time']:.4f}s ({v['n']/max(v['quant_time'],1e-9):.0f} vec/s)")
        print(f"  recon_time: {v['recon_time']:.4f}s ({v['n']/max(v['recon_time'],1e-9):.0f} vec/s)")
        print(f"  orig_bytes: {v['orig_bytes']:,}")
        print(f"  comp_bytes: {v['comp_bytes']:,}")
        print(f"  mean_cosine: {v['mean_cos']:.6f}")
        print(f"  recall@{args.k}: {v['recall@k']:.4f}")


if __name__ == '__main__':
    main()
