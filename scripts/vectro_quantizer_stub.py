#!/usr/bin/env python3
"""
vectro_quantizer_stub — Python implementation of the Mojo binary pipe protocol.

Used exclusively by CI to smoke-test _mojo_bridge._run_pipe without requiring
a real Mojo toolchain on the runner.  Implements the exact same stdin/stdout
protocol as the compiled vectro_quantizer binary.

Protocol (all integers as decimal strings in argv):
  pipe int8 quantize <n> <d>  stdin: n*d*4 bytes f32   stdout: n*d int8 + n*4 f32 scales
  pipe int8 recon    <n> <d>  stdin: n*d int8 + n*4 f32 scales  stdout: n*d*4 f32
  pipe nf4  encode   <n> <d>  stdin: n*d*4 bytes f32   stdout: n*ceil(d/2) u8 + n*4 f32
  pipe nf4  decode   <n> <d>  stdin: n*ceil(d/2) u8 + n*4 f32  stdout: n*d*4 f32
  pipe bin  encode   <n> <d>  stdin: n*d*4 bytes f32   stdout: n*ceil(d/8) u8
  pipe bin  decode   <n> <d>  stdin: n*ceil(d/8) u8    stdout: n*d*4 f32

Exit 0 on success, 1 on unknown subcommand.
"""
from __future__ import annotations

import sys
import struct
import numpy as np

# NF4 lookup table — 16 normal-float-4 values in [-1, 1].
# Identical to the compile-time table in src/quantizer_simd.mojo.
_NF4_TABLE = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=np.float32)


def _int8_quantize(vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales = np.abs(vecs).max(axis=1).astype(np.float32)  # (n,)
    safe_scales = np.where(scales > 0, scales, 1.0)
    q = np.clip(np.round(vecs / safe_scales[:, None] * 127), -127, 127).astype(np.int8)
    return q, scales


def _int8_recon(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) / 127.0) * scales[:, None]


def _nf4_encode(vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, d = vecs.shape
    scales = np.abs(vecs).max(axis=1).astype(np.float32)
    safe_scales = np.where(scales > 0, scales, 1.0)
    norm = vecs / safe_scales[:, None]  # (n, d) in [-1, 1]

    # Find nearest NF4 code for every element via broadcast distance
    codes = np.argmin(np.abs(norm[:, :, None] - _NF4_TABLE[None, None, :]), axis=2).astype(np.uint8)

    # Pack two nibbles per byte: low nibble = even index, high nibble = odd
    half_d = (d + 1) // 2
    packed = np.zeros((n, half_d), dtype=np.uint8)
    for i in range(0, d, 2):
        low = codes[:, i]
        high = codes[:, i + 1] if i + 1 < d else np.zeros(n, dtype=np.uint8)
        packed[:, i // 2] = low | (high << 4)

    return packed, scales


def _nf4_decode(packed: np.ndarray, scales: np.ndarray, d: int) -> np.ndarray:
    n = len(scales)
    half_d = (d + 1) // 2
    codes = np.zeros((n, d), dtype=np.uint8)
    for i in range(half_d):
        byte = packed[:, i]
        col0 = 2 * i
        col1 = 2 * i + 1
        codes[:, col0] = byte & 0x0F
        if col1 < d:
            codes[:, col1] = (byte >> 4) & 0x0F
    norm = _NF4_TABLE[codes]  # (n, d) float32
    return norm * scales[:, None]


def _bin_encode(vecs: np.ndarray) -> np.ndarray:
    n, d = vecs.shape
    bpv = (d + 7) // 8
    packed = np.zeros((n, bpv), dtype=np.uint8)
    for bit_idx in range(d):
        byte_idx = bit_idx // 8
        bit_pos = bit_idx % 8
        positive = (vecs[:, bit_idx] > 0).astype(np.uint8)
        packed[:, byte_idx] |= positive << bit_pos
    return packed


def _bin_decode(packed: np.ndarray, d: int) -> np.ndarray:
    n = packed.shape[0]
    out = np.full((n, d), -1.0, dtype=np.float32)
    for bit_idx in range(d):
        byte_idx = bit_idx // 8
        bit_pos = bit_idx % 8
        bit = (packed[:, byte_idx] >> bit_pos) & 1
        out[:, bit_idx] = np.where(bit == 1, 1.0, -1.0)
    return out


def _read_stdin(n_bytes: int) -> bytes:
    data = sys.stdin.buffer.read(n_bytes)
    if len(data) != n_bytes:
        raise ValueError(f"Expected {n_bytes} bytes, got {len(data)}")
    return data


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) < 3 or argv[0] != "pipe":
        sys.stderr.write(f"usage: vectro_quantizer pipe <op> <cmd> <n> <d>\n")
        return 1

    _, op, cmd, *rest = argv
    n = int(rest[0])
    d = int(rest[1])

    if op == "int8" and cmd == "quantize":
        raw = _read_stdin(n * d * 4)
        vecs = np.frombuffer(raw, dtype="<f4").reshape(n, d)
        q, scales = _int8_quantize(vecs)
        sys.stdout.buffer.write(q.tobytes())
        sys.stdout.buffer.write(scales.astype("<f4").tobytes())

    elif op == "int8" and cmd == "recon":
        raw_q = _read_stdin(n * d)
        raw_s = _read_stdin(n * 4)
        q = np.frombuffer(raw_q, dtype=np.int8).reshape(n, d)
        scales = np.frombuffer(raw_s, dtype="<f4")
        recon = _int8_recon(q, scales)
        sys.stdout.buffer.write(recon.astype("<f4").tobytes())

    elif op == "nf4" and cmd == "encode":
        raw = _read_stdin(n * d * 4)
        vecs = np.frombuffer(raw, dtype="<f4").reshape(n, d)
        packed, scales = _nf4_encode(vecs)
        sys.stdout.buffer.write(packed.tobytes())
        sys.stdout.buffer.write(scales.astype("<f4").tobytes())

    elif op == "nf4" and cmd == "decode":
        half_d = (d + 1) // 2
        raw_p = _read_stdin(n * half_d)
        raw_s = _read_stdin(n * 4)
        packed = np.frombuffer(raw_p, dtype=np.uint8).reshape(n, half_d)
        scales = np.frombuffer(raw_s, dtype="<f4")
        recon = _nf4_decode(packed, scales, d)
        sys.stdout.buffer.write(recon.astype("<f4").tobytes())

    elif op == "bin" and cmd == "encode":
        raw = _read_stdin(n * d * 4)
        vecs = np.frombuffer(raw, dtype="<f4").reshape(n, d)
        packed = _bin_encode(vecs)
        sys.stdout.buffer.write(packed.tobytes())

    elif op == "bin" and cmd == "decode":
        bpv = (d + 7) // 8
        raw = _read_stdin(n * bpv)
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(n, bpv)
        recon = _bin_decode(packed, d)
        sys.stdout.buffer.write(recon.astype("<f4").tobytes())

    else:
        sys.stderr.write(f"Unknown op/cmd: {op}/{cmd}\n")
        return 1

    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
