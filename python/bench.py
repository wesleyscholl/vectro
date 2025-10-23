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
import time
import os
try:
    import psutil
    _PSUTIL = True
except Exception:
    _PSUTIL = False

# Ensure project root is importable when running this script directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from python import interface
from python.cli import read_vectro_header, iter_vectro_chunks
import json
import tempfile


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
    cython_present = interface._cython_quant is not None
    if force_python:
        saved_mojo = interface._mojo_quant
        saved_cython = interface._cython_quant
        interface._mojo_quant = None
        interface._cython_quant = None

    def rss_mb():
        if _PSUTIL:
            p = psutil.Process(os.getpid())
            return p.memory_info().rss / (1024 * 1024)
        else:
            return float('nan')

    t0 = time.time()
    mem_before = rss_mb()
    out = interface.quantize_embeddings(embeddings)
    mem_after = rss_mb()
    t1 = time.time()
    q = out['q']
    scales = out['scales']
    dims = out['dims']
    nvecs = out['n']
    quant_time = t1 - t0
    quant_mem_delta_mb = mem_after - mem_before

    t0 = time.time()
    mem_before = rss_mb()
    recon = interface.reconstruct_embeddings(q, scales, dims)
    mem_after = rss_mb()
    t1 = time.time()
    recon_time = t1 - t0
    recon_mem_delta_mb = mem_after - mem_before

    mean_cos = interface.mean_cosine_similarity(embeddings, recon)
    orig_bytes = embeddings.nbytes
    comp_bytes = np.asarray(q).nbytes + np.asarray(scales).nbytes

    # pick queries indices
    rng = np.random.default_rng(1)
    queries_idx = rng.choice(n, size=min(queries, n), replace=False)
    recall = recall_at_k(embeddings, recon, queries_idx, k)

    if force_python:
        interface._mojo_quant = saved_mojo
        interface._cython_quant = saved_cython

    return {
        'quant_time': quant_time,
        'recon_time': recon_time,
        'quant_mem_mb': quant_mem_delta_mb,
        'recon_mem_mb': recon_mem_delta_mb,
        'mean_cos': mean_cos,
        'orig_bytes': orig_bytes,
        'comp_bytes': comp_bytes,
        'recall@k': recall,
        'mojo_present': mojo_present and not force_python,
        'cython_present': cython_present and not force_python,
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
    parser.add_argument('--chunk-size', type=int, default=0, help='Chunk size for streaming quantize (rows). 0 = no streaming')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    emb = rng.standard_normal((args.n, args.d)).astype(np.float32)

    print(f"Dataset: n={args.n}, d={args.d}, queries={args.queries}, k={args.k}")

    results = {}

    # If chunk_size specified, run stream-like chunked quantization to measure throughput
    if args.chunk_size and args.chunk_size > 0:
        print(f"Running chunked quantization with chunk_size={args.chunk_size} (VECTRO2 streaming)...")
        # write a VECTRO2 file using streaming quantization and then reconstruct by streaming reads
        tmpfd, out_path = tempfile.mkstemp(prefix='vectro_bench_', suffix='.v2')
        os.close(tmpfd)
        data_tmp = out_path + '.data'
        t0 = time.time()
        chunk_metas = []
        offset = 0
        with open(data_tmp, 'wb') as dataf:
            for start in range(0, args.n, args.chunk_size):
                end = min(args.n, start + args.chunk_size)
                chunk = emb[start:end]
                out = interface.quantize_embeddings(chunk)
                q_arr = np.asarray(out['q'], dtype=np.int8)
                scales_arr = np.asarray(out['scales'], dtype=np.float32)
                q_bytes = q_arr.tobytes()
                scales_bytes = scales_arr.tobytes()
                meta = {'start': int(start), 'end': int(end), 'offset': int(offset), 'q_len': len(q_bytes), 'scales_len': len(scales_bytes)}
                chunk_metas.append(meta)
                dataf.write(q_bytes)
                dataf.write(scales_bytes)
                offset += len(q_bytes) + len(scales_bytes)
        header = {'version': 2, 'n': args.n, 'd': args.d, 'q_dtype': 'int8', 'scales_dtype': 'float32', 'chunk_size': int(args.chunk_size), 'chunks': chunk_metas}
        header_bytes = json.dumps(header).encode('utf-8')
        with open(out_path, 'wb') as out_f:
            out_f.write(b'VECTRO2')
            out_f.write(len(header_bytes).to_bytes(4, 'little'))
            out_f.write(header_bytes)
            with open(data_tmp, 'rb') as df:
                while True:
                    data = df.read(1 << 20)
                    if not data:
                        break
                    out_f.write(data)
        os.remove(data_tmp)
        quant_time = time.time() - t0

        # streaming reconstruct: do not load all scales/q into memory at once
        t0 = time.time()
        recon = np.empty((args.n, args.d), dtype=np.float32)
        with open(out_path, 'rb') as fp:
            header = read_vectro_header(fp)
            for meta, q_bytes, scales_bytes in iter_vectro_chunks(fp, header):
                n_chunk = meta['end'] - meta['start']
                q_arr = np.frombuffer(q_bytes, dtype=np.int8).reshape((n_chunk, args.d))
                scales_arr = np.frombuffer(scales_bytes, dtype=np.float32)
                recon_chunk = interface.reconstruct_embeddings(q_arr.ravel(), scales_arr, args.d)
                recon[meta['start']:meta['end']] = recon_chunk
        recon_time = time.time() - t0

        mean_cos = interface.mean_cosine_similarity(emb, recon)
        orig_bytes = emb.nbytes
        comp_bytes = sum(m['q_len'] + m['scales_len'] for m in chunk_metas)
        results['chunked_python'] = {
            'quant_time': quant_time,
            'recon_time': recon_time,
            'mean_cos': mean_cos,
            'orig_bytes': orig_bytes,
            'comp_bytes': comp_bytes,
            'recall@k': recall_at_k(emb, recon, np.random.default_rng(1).choice(args.n, size=min(args.queries,args.n), replace=False), args.k),
            'n': args.n,
            'd': args.d,
            'mojo_present': interface._mojo_quant is not None,
        }
        os.remove(out_path)

    # If Mojo is present, run it; also run Python fallback for comparison
    mojo_available = interface._mojo_quant is not None
    cython_available = interface._cython_quant is not None
    if mojo_available:
        print("Running Mojo-backed quantization (if available)...")
        r_mojo = run_once(emb, args.k, args.queries, force_python=False)
        results['mojo'] = r_mojo

    if cython_available:
        print("Running Cython-backed quantization...")
        r_cython = run_once(emb, args.k, args.queries, force_python=False)
        results['cython'] = r_cython

    print("Running Python fallback quantization...")
    r_py = run_once(emb, args.k, args.queries, force_python=True)
    results['python'] = r_py

    # Print formatted results
    for k, v in results.items():
        print('\nBackend:', k)
        print(f"  mojo_present: {v['mojo_present']}")
        print(f"  cython_present: {v.get('cython_present', False)}")
        quant_vps = v['n'] / max(v['quant_time'], 1e-9)
        recon_vps = v['n'] / max(v['recon_time'], 1e-9)
        ratio = v['comp_bytes'] / max(v['orig_bytes'], 1)
        print(f"  quant_time: {v['quant_time']:.4f}s ({quant_vps:.0f} vec/s)")
        print(f"  recon_time: {v['recon_time']:.4f}s ({recon_vps:.0f} vec/s)")
        print(f"  quant_mem_delta_MB: {v.get('quant_mem_mb', float('nan')):.2f}")
        print(f"  recon_mem_delta_MB: {v.get('recon_mem_mb', float('nan')):.2f}")
        print(f"  orig_bytes: {v['orig_bytes']:,}")
        print(f"  comp_bytes: {v['comp_bytes']:,}")
        print(f"  compression_ratio (comp/orig): {ratio:.4f}")
        print(f"  mean_cosine: {v['mean_cos']:.6f}")
        print(f"  recall@{args.k}: {v['recall@k']:.4f}")
        print(f"  storage_model: {k}")


if __name__ == '__main__':
    main()
