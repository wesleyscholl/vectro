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
    import os
    import json

    # chunked streaming mode when chunk_size > 0
    if args.chunk_size and args.chunk_size > 0:
        # streaming format: VECTRO1 | uint32 header_len | header_json
        # header_json contains {n, d, q_dtype, scales_dtype}
        # followed by raw q bytes (n*d int8) and raw scales (n float32)
        mmap = np.load(args.infile, mmap_mode='r')
        n, d = mmap.shape
        header = {"n": int(n), "d": int(d), "q_dtype": "int8", "scales_dtype": "float32"}
        header_bytes = json.dumps(header).encode('utf-8')
        with open(args.outfile, 'wb') as f:
            f.write(b'VECTRO1')
            f.write(len(header_bytes).to_bytes(4, 'little'))
            f.write(header_bytes)
            # write q chunks then scales chunks
            # we'll write q first (n*d bytes), then scales (n*4 bytes)
            # write q in chunks
            scales_file_pos = f.tell() + n * d  # where scales will begin (not used here)
            # quantize and write q per-chunk
            for start in range(0, n, args.chunk_size):
                end = min(n, start + args.chunk_size)
                chunk = np.asarray(mmap[start:end], dtype=np.float32)
                # use backend if available
                if hasattr(__import__('python.interface').interface, '_mojo_quant') and __import__('python.interface').interface._mojo_quant is not None:
                    mojo = __import__('python.interface').interface._mojo_quant
                    q_flat, scales = mojo.quantize_int8(chunk.ravel().tolist(), int(end - start), int(d))
                    q_arr = np.asarray(q_flat, dtype=np.int8)
                else:
                    out = quantize_embeddings(chunk)
                    q_arr = np.asarray(out['q'], dtype=np.int8)
                    scales = out['scales']
                f.write(q_arr.tobytes())
                # append scales to a temporary file to write later
                if start == 0:
                    tmp_scales_path = args.outfile + '.scales.tmp'
                    sf = open(tmp_scales_path, 'wb')
                for s in (np.asarray(scales, dtype=np.float32).tolist()):
                    sf.write(np.float32(s).tobytes())
            if 'sf' in locals():
                sf.close()
                # append scales file
                with open(tmp_scales_path, 'rb') as sf2:
                    f.write(sf2.read())
                os.remove(tmp_scales_path)
        print(f"Wrote streaming compressed file: {args.outfile}")
        return

    # non-streaming (legacy) behavior
    emb = np.load(args.infile)
    out = quantize_embeddings(emb)
    # Save compressed data as a small npz file (q as int8 bytes, scales as float32, dims, n)
    q = np.asarray(out['q'], dtype=np.int8)
    scales = np.asarray(out['scales'], dtype=np.float32)
    np.savez_compressed(args.outfile, q=q, scales=scales, dims=out['dims'], n=out['n'])
    print(f"Wrote compressed file: {args.outfile}")


def cmd_eval(args: argparse.Namespace):
    import json
    orig = np.load(args.orig)
    # detect streaming format by header
    with open(args.comp, 'rb') as f:
        magic = f.read(7)
        if magic == b'VECTRO1':
            header_len = int.from_bytes(f.read(4), 'little')
            header = json.loads(f.read(header_len).decode('utf-8'))
            n = header['n']
            d = header['d']
            # read q bytes
            q_bytes = f.read(n * d)
            q = np.frombuffer(q_bytes, dtype=np.int8)
            scales_bytes = f.read(n * 4)
            scales = np.frombuffer(scales_bytes, dtype=np.float32)
            dims = d
            recon = reconstruct_embeddings(q, scales, dims)
            mcos = mean_cosine_similarity(orig, recon)
            orig_size = orig.nbytes
            comp_size = len(q_bytes) + len(scales_bytes)
        else:
            f.seek(0)
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
    p_compress.add_argument('--chunk-size', dest='chunk_size', type=int, default=0, help='Stream compress chunk size (rows). 0 = no streaming')
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
