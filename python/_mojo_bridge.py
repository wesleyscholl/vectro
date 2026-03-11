"""
Mojo bridge — subprocess dispatch to the compiled ``vectro_quantizer`` binary.

All compute-intensive hot paths (INT8, NF4, Binary) are routed through this
module so that Python is a thin orchestration layer and Mojo performs every
quantization operation.

Data exchange protocol
----------------------
- float32 files : n × d × 4 bytes, little-endian IEEE 754
- int8 files    : n × d × 1 bytes, signed two's complement
- uint8 files   : n × k × 1 bytes (k = ceil(d/2) for NF4, ceil(d/8) for binary)
- scales files  : n × 4 bytes, little-endian IEEE 754 float32

All exposed functions raise ``RuntimeError`` if the binary is unavailable so
callers can fall back gracefully.
"""

from __future__ import annotations

import math
import os
import pathlib
import subprocess
import tempfile
from typing import Tuple

import numpy as np

# ── binary discovery ─────────────────────────────────────────────────────────

_BINARY_NAME = "vectro_quantizer"

def _find_binary() -> str | None:
    """Return the absolute path to the compiled Mojo binary, or None."""
    candidates = [
        pathlib.Path(__file__).parent.parent / _BINARY_NAME,
        pathlib.Path(_BINARY_NAME),
        pathlib.Path(os.getcwd()) / _BINARY_NAME,
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return str(p.resolve())
    return None


_binary_path: str | None = _find_binary()


def is_available() -> bool:
    """Return True if the ``vectro_quantizer`` binary is reachable."""
    return _binary_path is not None


def binary_path() -> str:
    """Return the path to the binary, raising RuntimeError if absent."""
    if _binary_path is None:
        raise RuntimeError(
            "vectro_quantizer binary not found.  "
            "Build it with:  pixi run build-mojo"
        )
    return _binary_path


# ── low-level I/O helpers ─────────────────────────────────────────────────────

def _write_f32(arr: np.ndarray, path: str) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.tofile(path)


def _read_f32(path: str, count: int) -> np.ndarray:
    return np.fromfile(path, dtype="<f4", count=count)


def _write_i8(arr: np.ndarray, path: str) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.int8)
    arr.tofile(path)


def _read_i8(path: str, count: int) -> np.ndarray:
    return np.fromfile(path, dtype=np.int8, count=count)


def _write_u8(arr: np.ndarray, path: str) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    arr.tofile(path)


def _read_u8(path: str, count: int) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint8, count=count)


def _run(args: list[str]) -> None:
    """Execute the Mojo binary with ``args``.  Raises on non-zero exit."""
    cmd = [binary_path()] + args
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"vectro_quantizer exited with code {result.returncode}: "
            + " ".join(cmd)
        )


# ── INT8 ──────────────────────────────────────────────────────────────────────

def int8_quantize(
    vectors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize float32 vectors to INT8 using the Mojo binary.

    Args:
        vectors: Shape (n, d), float32.

    Returns:
        q:      Shape (n, d), int8.
        scales: Shape (n,),   float32.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis]
    n, d = vectors.shape

    with tempfile.TemporaryDirectory() as tmp:
        f_in   = os.path.join(tmp, "in.f32")
        f_out  = os.path.join(tmp, "out.i8")
        f_sc   = os.path.join(tmp, "scales.f32")

        _write_f32(vectors, f_in)
        _run(["int8", "quantize", f_in, f_out, f_sc, str(n), str(d)])

        q      = _read_i8(f_out, n * d).reshape(n, d)
        scales = _read_f32(f_sc, n)

    return q, scales


def int8_reconstruct(
    q: np.ndarray,
    scales: np.ndarray,
    d: int | None = None,
) -> np.ndarray:
    """Reconstruct float32 vectors from INT8 + scales using the Mojo binary.

    Args:
        q:      Shape (n, d) or (n*d,), int8.
        scales: Shape (n,), float32.
        d:      Dimension (inferred from q if 2-D).

    Returns:
        Reconstructed float32 array of shape (n, d).
    """
    q = np.ascontiguousarray(q, dtype=np.int8)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    n = len(scales)
    if d is None:
        d = q.size // n
    q = q.reshape(n, d)

    with tempfile.TemporaryDirectory() as tmp:
        f_q    = os.path.join(tmp, "in.i8")
        f_sc   = os.path.join(tmp, "in_scales.f32")
        f_out  = os.path.join(tmp, "out.f32")

        _write_i8(q, f_q)
        _write_f32(scales, f_sc)
        _run(["int8", "recon", f_q, f_sc, f_out, str(n), str(d)])

        recon = _read_f32(f_out, n * d).reshape(n, d)

    return recon


# ── NF4 ───────────────────────────────────────────────────────────────────────

def nf4_encode(
    vectors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """NF4-encode float32 vectors using the Mojo binary.

    Args:
        vectors: Shape (n, d), float32.

    Returns:
        packed: Shape (n, ceil(d/2)), uint8.
        scales: Shape (n,), float32.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis]
    n, d = vectors.shape
    half_d = (d + 1) // 2

    with tempfile.TemporaryDirectory() as tmp:
        f_in  = os.path.join(tmp, "in.f32")
        f_out = os.path.join(tmp, "out.u8")
        f_sc  = os.path.join(tmp, "scales.f32")

        _write_f32(vectors, f_in)
        _run(["nf4", "encode", f_in, f_out, f_sc, str(n), str(d)])

        packed = _read_u8(f_out, n * half_d).reshape(n, half_d)
        scales = _read_f32(f_sc, n)

    return packed, scales


def nf4_decode(
    packed: np.ndarray,
    scales: np.ndarray,
    d: int,
) -> np.ndarray:
    """Decode NF4-packed bytes to float32 using the Mojo binary.

    Args:
        packed: Shape (n, ceil(d/2)), uint8.
        scales: Shape (n,), float32.
        d:      Original vector dimension.

    Returns:
        Reconstructed float32 array of shape (n, d).
    """
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    n = len(scales)

    with tempfile.TemporaryDirectory() as tmp:
        f_p   = os.path.join(tmp, "in.u8")
        f_sc  = os.path.join(tmp, "in_scales.f32")
        f_out = os.path.join(tmp, "out.f32")

        _write_u8(packed, f_p)
        _write_f32(scales, f_sc)
        _run(["nf4", "decode", f_p, f_sc, f_out, str(n), str(d)])

        recon = _read_f32(f_out, n * d).reshape(n, d)

    return recon


# ── Binary ────────────────────────────────────────────────────────────────────

def bin_encode(
    vectors: np.ndarray,
) -> np.ndarray:
    """Binary (sign-bit) encode float32 vectors using the Mojo binary.

    Args:
        vectors: Shape (n, d), float32.  Positive values → bit 1.

    Returns:
        packed: Shape (n, ceil(d/8)), uint8.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis]
    n, d = vectors.shape
    bpv = (d + 7) // 8

    with tempfile.TemporaryDirectory() as tmp:
        f_in  = os.path.join(tmp, "in.f32")
        f_out = os.path.join(tmp, "out.u8")

        _write_f32(vectors, f_in)
        _run(["bin", "encode", f_in, f_out, str(n), str(d)])

        packed = _read_u8(f_out, n * bpv).reshape(n, bpv)

    return packed


def bin_decode(
    packed: np.ndarray,
    d: int,
) -> np.ndarray:
    """Decode binary-packed bytes to {-1, +1} float32 using the Mojo binary.

    Args:
        packed: Shape (n, ceil(d/8)), uint8.
        d:      Original vector dimension.

    Returns:
        Float32 array of shape (n, d) with values in {-1.0, +1.0}.
    """
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    n = packed.shape[0]

    with tempfile.TemporaryDirectory() as tmp:
        f_p   = os.path.join(tmp, "in.u8")
        f_out = os.path.join(tmp, "out.f32")

        _write_u8(packed, f_p)
        _run(["bin", "decode", f_p, f_out, str(n), str(d)])

        recon = _read_f32(f_out, n * d).reshape(n, d)

    return recon
