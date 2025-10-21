"""Command-line interface for Vectro MVP.

Provides two minimal commands:
 - compress --in embeddings.npy --out compressed.npz
 - eval --orig embeddings.npy --comp compressed.npz

This CLI uses the Python quantizer (or the Mojo backend if available) via
`python.interface`.
"""
from __future__ import annotations
import argparse
import numpy as np
from python.interface import quantize_embeddings, reconstruct_embeddings, mean_cosine_similarity


def cmd_compress(args: argparse.Namespace):
    emb = np.load(args.infile)
    out = quantize_embeddings(emb)
    # Save compressed data as a small npz file (q as int8 bytes, scales as float32, dims, n)
    q = np.asarray(out['q'], dtype=np.int8)
    scales = np.asarray(out['scales'], dtype=np.float32)
    np.savez_compressed(args.outfile, q=q, scales=scales, dims=out['dims'], n=out['n'])
    print(f"Wrote compressed file: {args.outfile}")


def cmd_eval(args: argparse.Namespace):
    orig = np.load(args.orig)
    npz = np.load(args.comp)
    q = npz['q']
    scales = npz['scales']
    dims = int(npz['dims'])
    recon = reconstruct_embeddings(q, scales, dims)
    mcos = mean_cosine_similarity(orig, recon)
    orig_size = orig.nbytes
    comp_size = q.nbytes + scales.nbytes
    print(f"Original bytes: {orig_size}")
    print(f"Compressed bytes (raw arrays): {comp_size}")
    print(f"Mean cosine similarity: {mcos:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='vectro', description='Vectro embedding compressor (MVP)')
    sub = parser.add_subparsers(dest='cmd')

    p_compress = sub.add_parser('compress', help='Compress embeddings (.npy)')
    p_compress.add_argument('--in', dest='infile', required=True, help='Input embeddings (.npy)')
    p_compress.add_argument('--out', dest='outfile', required=True, help='Output compressed (.npz)')
    p_compress.set_defaults(func=cmd_compress)

    p_eval = sub.add_parser('eval', help='Evaluate compression quality')
    p_eval.add_argument('--orig', required=True, help='Original embeddings (.npy)')
    p_eval.add_argument('--comp', required=True, help='Compressed file (.npz)')
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    args.func(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
