"""Binary container for Vectro arrays with optional compression.

Format (VTRB01):
 - magic: b'VTRB01' (6 bytes)
 - header_len: uint32 little-endian
 - header_json: JSON containing metadata and per-array offsets
 - blob area: concatenated blobs (each may be zlib-compressed)

Header example:
 {
   "arrays": {
       "q": {"dtype":"int8","shape":[n*d],"offset":123,"length":456,"compressed":true},
       "scales": {...}
   },
   "order": ["q","scales","dims","n"]
 }

The API supports writing arrays and reading individual arrays by name without loading the whole file.
"""
from __future__ import annotations
import json
import zlib
import numpy as np
from typing import Dict, Any, Optional


MAGIC = b'VTRB01'


def write_arrays(path: str, arrays: Dict[str, np.ndarray], compress: bool = True) -> None:
    # Build header metadata and write blobs
    header = {"arrays": {}, "order": list(arrays.keys())}
    with open(path, 'wb') as f:
        f.write(MAGIC)
        # reserve header_len
        f.write((0).to_bytes(4, 'little'))
        # placeholder for header; we'll write real header after collecting blob offsets
        header_pos = f.tell()
        # we'll write header later; for now move to blob start
        # write blobs and collect offsets
        for name, arr in arrays.items():
            offset = f.tell()
            raw = np.ascontiguousarray(arr).tobytes()
            if compress:
                blob = zlib.compress(raw)
                compressed = True
            else:
                blob = raw
                compressed = False
            f.write(blob)
            length = len(blob)
            header['arrays'][name] = {
                'dtype': str(arr.dtype),
                'shape': list(arr.shape),
                'offset': offset,
                'length': length,
                'compressed': compressed,
            }
        # write header at header_pos by seeking back; but header must go after magic+4 bytes,
        # so we'll build header bytes and rewrite the file: read blobs, then prepend header is complex.
        # Simpler: write header after magic+4 at the start by rewriting the file: create temp file approach.


def write_arrays(path: str, arrays: Dict[str, np.ndarray], compress: bool = True) -> None:
    # Safer implementation: build header in memory then write header followed by blobs
    header = {"arrays": {}, "order": list(arrays.keys())}
    blobs = []
    # prepare blobs and metadata
    pos = 0
    for name, arr in arrays.items():
        raw = np.ascontiguousarray(arr).tobytes()
        if compress:
            blob = zlib.compress(raw)
            compressed = True
        else:
            blob = raw
            compressed = False
        length = len(blob)
        header['arrays'][name] = {
            'dtype': str(arr.dtype),
            'shape': list(arr.shape),
            'offset': pos,
            'length': length,
            'compressed': compressed,
        }
        blobs.append(blob)
        pos += length

    header_bytes = json.dumps(header).encode('utf-8')
    with open(path, 'wb') as f:
        f.write(MAGIC)
        f.write(len(header_bytes).to_bytes(4, 'little'))
        f.write(header_bytes)
        # write blobs sequentially
        for blob in blobs:
            f.write(blob)


def read_header(path: str) -> Dict[str, Any]:
    with open(path, 'rb') as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError('Not a VTRB01 file')
        header_len_bytes = f.read(4)
        if len(header_len_bytes) < 4:
            raise EOFError('Truncated header')
        header_len = int.from_bytes(header_len_bytes, 'little')
        header_json = f.read(header_len)
        header = json.loads(header_json.decode('utf-8'))
        # compute absolute offsets for blobs (header_end offset)
        header_end = f.tell()
        for meta in header['arrays'].values():
            meta['offset_abs'] = header_end + int(meta['offset'])
        return header


def read_array(path: str, name: str) -> np.ndarray:
    header = read_header(path)
    if name not in header['arrays']:
        raise KeyError(name)
    meta = header['arrays'][name]
    with open(path, 'rb') as f:
        f.seek(meta['offset_abs'])
        blob = f.read(meta['length'])
        if meta.get('compressed', False):
            raw = zlib.decompress(blob)
        else:
            raw = blob
        dtype = np.dtype(meta['dtype'])
        arr = np.frombuffer(raw, dtype=dtype).reshape(meta['shape'])
        return arr.copy()


def read_arrays(path: str, names: Optional[list[str]] = None) -> Dict[str, np.ndarray]:
    header = read_header(path)
    out = {}
    if names is None:
        names = header['order']
    for name in names:
        out[name] = read_array(path, name)
    return out
