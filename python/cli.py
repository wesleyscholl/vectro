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
import json
import struct


def read_vectro_header(fp):
    # fp is an open file object at start
    magic = fp.read(7)
    if not magic or not magic.startswith(b'VECTRO'):
        raise ValueError('Not a VECTRO file')
    version = magic[6:7]
    # back-compat: VECTRO1 has magic b'VECTRO1' (7 bytes); we read next 4 bytes for header len
    header_len_bytes = fp.read(4)
    if len(header_len_bytes) < 4:
        raise EOFError('Truncated header length')
    header_len = int.from_bytes(header_len_bytes, 'little')
    header_json = fp.read(header_len)
    header = json.loads(header_json.decode('utf-8'))
    # compute data_start position
    data_start = fp.tell()
    header['data_start'] = data_start
    return header


def iter_vectro_chunks(fp, header):
    # Yields tuples (meta, q_bytes, scales_bytes) for each chunk by reading sequentially
    fp.seek(header['data_start'])
    if header.get('version', 1) == 1:
        # legacy: all q bytes then all scales bytes; yield a single pseudo-chunk
        n = header['n']
        d = header['d']
        q_bytes = fp.read(n * d)
        scales_bytes = fp.read(n * 4)
        meta = {'start': 0, 'end': n, 'offset': 0, 'q_len': len(q_bytes), 'scales_len': len(scales_bytes)}
        yield meta, q_bytes, scales_bytes
        return
    # VECTRO2: header contains 'chunks' with per-chunk q_len and scales_len and offsets relative to data_start
    for meta in header['chunks']:
        q_len = meta['q_len']
        scales_len = meta['scales_len']
        # seek and read q + scales
        fp.seek(header['data_start'] + meta['offset'])
        q_bytes = fp.read(q_len)
        scales_bytes = fp.read(scales_len)
        yield meta, q_bytes, scales_bytes


def reconstruct_slice_from_file(path, slice_start, slice_end):
    # Reconstruct only vectors in [slice_start, slice_end) by reading per-chunk bytes
    with open(path, 'rb') as fp:
        header = read_vectro_header(fp)
        d = header['d']
        result_parts = []
        for meta, q_bytes, scales_bytes in iter_vectro_chunks(fp, header):
            cstart = meta['start']
            cend = meta['end']
            # no overlap
            if cend <= slice_start or cstart >= slice_end:
                continue
            # reconstruct only overlapping portion
            rel_start = max(slice_start, cstart) - cstart
            rel_end = min(slice_end, cend) - cstart
            n_chunk = cend - cstart
            # reshape q and scales for full chunk then slice
            q_arr = np.frombuffer(q_bytes, dtype=np.int8).reshape((n_chunk, d))
            scales_arr = np.frombuffer(scales_bytes, dtype=np.float32).reshape((n_chunk,))
            sub_q = q_arr[rel_start:rel_end]
            sub_scales = scales_arr[rel_start:rel_end]
            # reconstruct sub-chunk vectors (pass flattened q)
            recon = reconstruct_embeddings(sub_q.ravel(), sub_scales, int(d))
            result_parts.append(recon)
        if result_parts:
            return np.vstack(result_parts)
        else:
            return np.empty((0, d), dtype=np.float32)


def cmd_compress(args: argparse.Namespace):
    import os
    import json

    # chunked streaming mode when chunk_size > 0 -> VECTRO2 format
    if args.chunk_size and args.chunk_size > 0:
        import tempfile
        mmap = np.load(args.infile, mmap_mode='r')
        n, d = mmap.shape
        # write chunk bodies to a temp file and record per-chunk metadata
        tmpfd, tmp_path = tempfile.mkstemp(prefix='vectro_chunks_', suffix='.bin')
        os.close(tmpfd)
        chunk_metas = []
        offset = 0
        with open(tmp_path, 'wb') as tmpf:
            for start in range(0, n, args.chunk_size):
                end = min(n, start + args.chunk_size)
                chunk = np.asarray(mmap[start:end], dtype=np.float32)
                out = quantize_embeddings(chunk)
                q_arr = np.asarray(out['q'], dtype=np.int8)
                scales_arr = np.asarray(out['scales'], dtype=np.float32)
                q_bytes = q_arr.tobytes()
                scales_bytes = scales_arr.tobytes()
                # record metadata relative to chunk_data start
                meta = {
                    'start': int(start),
                    'end': int(end),
                    'offset': int(offset),
                    'q_len': len(q_bytes),
                    'scales_len': len(scales_bytes),
                }
                chunk_metas.append(meta)
                tmpf.write(q_bytes)
                tmpf.write(scales_bytes)
                offset += len(q_bytes) + len(scales_bytes)

        header = {
            'version': 2,
            'n': int(n),
            'd': int(d),
            'q_dtype': 'int8',
            'scales_dtype': 'float32',
            'chunk_size': int(args.chunk_size),
            'chunks': chunk_metas,
        }
        header_bytes = json.dumps(header).encode('utf-8')
        with open(args.outfile, 'wb') as out_f:
            out_f.write(b'VECTRO2')
            out_f.write(len(header_bytes).to_bytes(4, 'little'))
            out_f.write(header_bytes)
            # copy chunk data
            with open(tmp_path, 'rb') as tmpf:
                while True:
                    data = tmpf.read(1 << 20)
                    if not data:
                        break
                    out_f.write(data)
        os.remove(tmp_path)
        print(f"Wrote streaming compressed file (VECTRO2): {args.outfile}")
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
