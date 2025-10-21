"""Streaming writer + compact binary container for Vectro.

Features added:
- Header-last layout with footer pointer (8-byte header offset after magic).
- Per-chunk zstd compression if `zstandard` is available, otherwise zlib fallback.
- Per-chunk checksum using xxhash if available, otherwise zlib.crc32.
- Stores PQ codebooks in header if provided; chunks can contain PQ codes or int8+scales.
- Provides `VectroStreamingWriter`, `read_header`, `peek`, and `read_chunk_blob` helpers.
"""
from __future__ import annotations
import json
import os
import tempfile
import shutil
import struct
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

try:
    import zstandard as zstd
    _HAS_ZSTD = True
except Exception:
    import zlib as _zlib
    _HAS_ZSTD = False

try:
    import xxhash
    _HAS_XXHASH = True
except Exception:
    import zlib as _zlibchk
    _HAS_XXHASH = False

MAGIC = b'VTRB02'  # Vectro binary container v0.2 (header-last)


def _compress_bytes(data: bytes, codec: str = 'zstd:3') -> Tuple[bytes, str]:
    if codec.startswith('zstd') and _HAS_ZSTD:
        level = int(codec.split(':')[1]) if ':' in codec else 3
        cctx = zstd.ZstdCompressor(level=level)
        return cctx.compress(data), 'zstd'
    else:
        # fallback to zlib
        return _zlib.compress(data), 'zlib'


def _checksum_bytes(data: bytes) -> str:
    if _HAS_XXHASH:
        return format(xxhash.xxh64(data).intdigest(), '016x')
    else:
        return format((_zlibchk.crc32(data) & 0xFFFFFFFF), '08x')


class VectroStreamingWriter:
    """Streaming writer that appends chunk blobs then writes header at end.

    Usage:
      w = VectroStreamingWriter(out_path, codec='zstd:3')
      w.set_codebooks(codebooks)   # optional (PQ)
      for chunk in chunks:
          w.write_chunk(start, end, q_bytes=q_b, scales_bytes=s_b, backend='int8')
      w.finalize()
    """

    def __init__(self, path: str, codec: str = 'zstd:3', tmp_dir: Optional[str] = None):
        self.path = path
        self.codec = codec
        self.tmp_dir = tmp_dir or os.path.dirname(path) or '.'
        self.tmp_fd, self.tmp_path = tempfile.mkstemp(prefix='vectro_write_', suffix='.tmp', dir=self.tmp_dir)
        os.close(self.tmp_fd)
        self.f = open(self.tmp_path, 'wb')
        # write magic + placeholder for header_offset (8 bytes)
        self.f.write(MAGIC)
        self.f.write((0).to_bytes(8, 'little'))
        # blob area begins here
        self.blob_start = self.f.tell()
        self.chunk_index: List[Dict[str, Any]] = []
        self.codebooks: Optional[np.ndarray] = None

    def set_codebooks(self, codebooks: np.ndarray) -> None:
        # codebooks: (m, ks, dsub) float32
        self.codebooks = np.asarray(codebooks, dtype=np.float32)

    def write_chunk(self, start: int, end: int, *, q_bytes: bytes = b'', scales_bytes: Optional[bytes] = None, codes_bytes: Optional[bytes] = None, backend: str = 'int8') -> Dict[str, Any]:
        """Write a single chunk. Provide either q_bytes+scales_bytes (int8) or codes_bytes (pq).

        Returns metadata for the chunk (with offsets relative to blob start).
        """
        offset = self.f.tell() - self.blob_start
        meta: Dict[str, Any] = {'start': int(start), 'end': int(end), 'backend': backend}
        # write q_bytes then scales_bytes if present
        total_len = 0
        if codes_bytes is not None:
            # compress codes
            comp, codec_used = _compress_bytes(codes_bytes, self.codec)
            self.f.write(comp)
            clen = len(comp)
            meta.update({'codes_len': clen, 'codes_compressed': True, 'codes_codec': codec_used})
            meta['codes_checksum'] = _checksum_bytes(comp)
            total_len += clen
        else:
            # write q
            comp_q, codec_q = _compress_bytes(q_bytes, self.codec)
            self.f.write(comp_q)
            qlen = len(comp_q)
            meta.update({'q_len': qlen, 'q_compressed': True, 'q_codec': codec_q})
            meta['q_checksum'] = _checksum_bytes(comp_q)
            total_len += qlen
            if scales_bytes is not None:
                comp_s, codec_s = _compress_bytes(scales_bytes, self.codec)
                self.f.write(comp_s)
                slen = len(comp_s)
                meta.update({'scales_len': slen, 'scales_compressed': True, 'scales_codec': codec_s})
                meta['scales_checksum'] = _checksum_bytes(comp_s)
                total_len += slen

        meta['offset'] = int(offset)
        meta['length'] = int(total_len)
        self.chunk_index.append(meta)
        return meta

    def finalize(self) -> None:
        # build header
        header: Dict[str, Any] = {
            'version': 2,
            'codec': self.codec,
            'codebooks': None,
            'chunk_index': self.chunk_index,
        }
        if self.codebooks is not None:
            header['codebooks'] = {
                'm': int(self.codebooks.shape[0]),
                'ks': int(self.codebooks.shape[1]),
                'dsub': int(self.codebooks.shape[2]),
                'bytes': self.codebooks.tobytes().hex(),
                'dtype': str(self.codebooks.dtype),
            }
        header_bytes = json.dumps(header).encode('utf-8')
        # header_offset = current file position
        header_offset = self.f.tell()
        # write header bytes
        self.f.write(header_bytes)
        self.f.flush()
        self.f.close()
        # write header offset into placeholder (byte pos 6..13)
        with open(self.tmp_path, 'r+b') as fh:
            fh.seek(len(MAGIC))
            fh.write(struct.pack('<Q', header_offset))
            fh.flush()
        # atomic move
        shutil.move(self.tmp_path, self.path)


def read_header(path: str) -> Dict[str, Any]:
    """Read header for header-last format. Returns parsed header with absolute offsets computed."""
    with open(path, 'rb') as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError('Not a VTRB02 file')
        header_off_bytes = f.read(8)
        if len(header_off_bytes) < 8:
            raise EOFError('Truncated file (no header offset)')
        header_offset = struct.unpack('<Q', header_off_bytes)[0]
        f.seek(header_offset)
        header_json = f.read()
        header = json.loads(header_json.decode('utf-8'))
        # compute absolute offsets for chunks
        for meta in header.get('chunk_index', []):
            meta['offset_abs'] = header_offset + int(meta['offset'])
        return header


def peek(path: str) -> None:
    h = read_header(path)
    print('VTRB02 header summary:')
    print(' version:', h.get('version'))
    print(' codec:', h.get('codec'))
    if h.get('codebooks'):
        cb = h['codebooks']
        print(' codebooks: m=%d ks=%d dsub=%d dtype=%s' % (cb['m'], cb['ks'], cb['dsub'], cb.get('dtype')))
    print(' chunks:', len(h.get('chunk_index', [])))
    for i,meta in enumerate(h.get('chunk_index', [])):
        print(f"  chunk[{i}] start={meta['start']} end={meta['end']} offset={meta['offset']} length={meta['length']}")


def read_chunk_blob(path: str, meta: Dict[str, Any]) -> bytes:
    with open(path, 'rb') as f:
        header_off = struct.unpack('<Q', f.read(len(MAGIC) + 8)[len(MAGIC):])[0]
        f.seek(header_off + int(meta['offset']))
        return f.read(int(meta['length']))

