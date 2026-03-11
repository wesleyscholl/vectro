"""tests/test_storage_v3.py — Phase 8: VQZ container and cloud backend stubs."""

import hashlib
import io
import os
import struct
import tempfile
import zlib

import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from storage_v3 import (
    MAGIC,
    HEADER_SIZE,
    save_vqz,
    load_vqz,
    S3Backend,
    GCSBackend,
    AzureBlobBackend,
    _compress,
    _decompress,
    _parse_header,
    _body_checksum,
    _build_header,
    _FLAG_NONE,
    _FLAG_ZSTD,
    _FLAG_ZLIB,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_data():
    rng = np.random.default_rng(42)
    n, d = 32, 64
    quantized = rng.integers(-128, 128, size=(n, d), dtype=np.int8)
    scales    = rng.random(n).astype(np.float32)
    return quantized, scales, d


@pytest.fixture()
def tmp_vqz(tmp_path, small_data):
    quantized, scales, d = small_data
    path = str(tmp_path / "test.vqz")
    save_vqz(quantized, scales, d, path)
    return path, quantized, scales, d


# ---------------------------------------------------------------------------
# 1. Header construction and parsing
# ---------------------------------------------------------------------------

class TestHeader:
    def test_magic_length(self):
        assert len(MAGIC) == 8

    def test_header_size_constant(self):
        assert HEADER_SIZE == 64

    def test_build_and_parse_round_trip(self):
        checksum = b"\xab\xcd\xef\x01\x23\x45\x67\x89"
        raw = _build_header(
            comp_flag=_FLAG_ZSTD,
            n_vectors=1000,
            dims=768,
            n_subspaces=96,
            metadata_len=42,
            checksum=checksum,
        )
        assert len(raw) == HEADER_SIZE
        hdr = _parse_header(raw)
        assert hdr["version"] == 1
        assert hdr["comp_flag"] == _FLAG_ZSTD
        assert hdr["n_vectors"] == 1000
        assert hdr["dims"] == 768
        assert hdr["n_subspaces"] == 96
        assert hdr["metadata_len"] == 42
        assert hdr["checksum"] == checksum[:8]

    def test_parse_rejects_bad_magic(self):
        bad = b"INVALID\x00" + b"\x00" * (HEADER_SIZE - 8)
        with pytest.raises(ValueError, match="magic"):
            _parse_header(bad)

    def test_reserved_bytes_are_zero(self):
        checksum = b"\x00" * 8
        raw = _build_header(_FLAG_NONE, 1, 4, 0, 0, checksum)
        assert raw[38:64] == b"\x00" * 26


# ---------------------------------------------------------------------------
# 2. Compression helpers
# ---------------------------------------------------------------------------

class TestCompression:
    def test_zlib_round_trip(self):
        data = b"hello world " * 1000
        compressed, flag = _compress(data, "zlib", 6)
        assert flag == _FLAG_ZLIB
        assert len(compressed) < len(data)
        assert _decompress(compressed, flag) == data

    def test_none_round_trip(self):
        data = b"\x00\x01\x02\x03" * 100
        out, flag = _compress(data, "none", 0)
        assert flag == _FLAG_NONE
        assert out == data
        assert _decompress(out, _FLAG_NONE) == data

    def test_zstd_round_trip(self):
        pytest.importorskip("zstandard")
        data = b"compress me " * 500
        compressed, flag = _compress(data, "zstd", 3)
        assert flag == _FLAG_ZSTD
        assert _decompress(compressed, flag) == data

    def test_zstd_fallback_to_zlib_when_missing(self, monkeypatch):
        import storage_v3
        monkeypatch.setattr(storage_v3, "_HAVE_ZSTD", False)
        data = b"fallback test " * 200
        compressed, flag = storage_v3._compress(data, "zstd", 3)
        assert flag == _FLAG_ZLIB
        assert storage_v3._decompress(compressed, flag) == data

    def test_body_checksum_deterministic(self):
        data = b"checksum me"
        c1 = _body_checksum(data)
        c2 = _body_checksum(data)
        assert c1 == c2
        assert len(c1) == 8

    def test_body_checksum_sensitive(self):
        assert _body_checksum(b"abc") != _body_checksum(b"abd")


# ---------------------------------------------------------------------------
# 3. save_vqz / load_vqz round-trips
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_basic_round_trip_zstd(self, small_data, tmp_path):
        pytest.importorskip("zstandard")
        quantized, scales, d = small_data
        path = str(tmp_path / "out.vqz")
        save_vqz(quantized, scales, d, path, compression="zstd")
        result = load_vqz(path)
        np.testing.assert_array_equal(result["quantized"], quantized)
        np.testing.assert_array_almost_equal(result["scales"], scales)

    def test_basic_round_trip_zlib(self, small_data, tmp_path):
        quantized, scales, d = small_data
        path = str(tmp_path / "out_zlib.vqz")
        save_vqz(quantized, scales, d, path, compression="zlib")
        result = load_vqz(path)
        np.testing.assert_array_equal(result["quantized"], quantized)
        np.testing.assert_array_almost_equal(result["scales"], scales)

    def test_basic_round_trip_no_compression(self, small_data, tmp_path):
        quantized, scales, d = small_data
        path = str(tmp_path / "out_raw.vqz")
        save_vqz(quantized, scales, d, path, compression="none")
        result = load_vqz(path)
        np.testing.assert_array_equal(result["quantized"], quantized)
        np.testing.assert_array_almost_equal(result["scales"], scales)

    def test_quantized_dtype_preserved(self, tmp_vqz):
        path, _, _, _ = tmp_vqz
        result = load_vqz(path)
        assert result["quantized"].dtype == np.int8

    def test_scales_dtype_preserved(self, tmp_vqz):
        path, _, _, _ = tmp_vqz
        result = load_vqz(path)
        assert result["scales"].dtype == np.float32

    def test_shape_preserved(self, small_data, tmp_vqz):
        path, quantized, scales, d = tmp_vqz
        result = load_vqz(path)
        assert result["quantized"].shape == quantized.shape
        assert result["scales"].shape == scales.shape

    def test_dims_in_result(self, tmp_vqz):
        path, _, _, d = tmp_vqz
        result = load_vqz(path)
        assert result["dims"] == d

    def test_n_vectors_in_result(self, small_data, tmp_vqz):
        path, quantized, _, _ = tmp_vqz
        result = load_vqz(path)
        assert result["n_vectors"] == quantized.shape[0]

    def test_version_in_result(self, tmp_vqz):
        path, _, _, _ = tmp_vqz
        result = load_vqz(path)
        assert result["version"] == 1

    def test_metadata_round_trip_bytes(self, small_data, tmp_path):
        quantized, scales, d = small_data
        path = str(tmp_path / "meta.vqz")
        meta = b"\xde\xad\xbe\xef app-specific blob"
        save_vqz(quantized, scales, d, path, metadata=meta)
        result = load_vqz(path)
        assert result["metadata"] == meta

    def test_metadata_round_trip_string(self, small_data, tmp_path):
        quantized, scales, d = small_data
        path = str(tmp_path / "meta_str.vqz")
        save_vqz(quantized, scales, d, path, metadata="hello vectro")
        result = load_vqz(path)
        assert result["metadata"] == b"hello vectro"

    def test_empty_metadata(self, small_data, tmp_path):
        quantized, scales, d = small_data
        path = str(tmp_path / "nometa.vqz")
        save_vqz(quantized, scales, d, path, metadata=b"")
        result = load_vqz(path)
        assert result["metadata"] == b""

    def test_n_subspaces_stored(self, small_data, tmp_path):
        quantized, scales, d = small_data
        path = str(tmp_path / "pq.vqz")
        save_vqz(quantized, scales, d, path, n_subspaces=8)
        result = load_vqz(path)
        assert result["n_subspaces"] == 8

    def test_compressed_file_smaller_than_raw(self, tmp_path):
        rng = np.random.default_rng(7)
        n, d = 512, 256
        # Use non-random (compressible) data
        quantized = np.zeros((n, d), dtype=np.int8)
        scales    = np.ones(n, dtype=np.float32)
        raw_path  = str(tmp_path / "raw.vqz")
        zlib_path = str(tmp_path / "zlib.vqz")
        save_vqz(quantized, scales, d, raw_path,  compression="none")
        save_vqz(quantized, scales, d, zlib_path, compression="zlib")
        assert os.path.getsize(zlib_path) < os.path.getsize(raw_path)

    def test_checksum_detects_corruption(self, tmp_vqz):
        path, _, _, _ = tmp_vqz
        with open(path, "r+b") as fh:
            # Flip a byte in the body (after header)
            fh.seek(HEADER_SIZE + 5)
            orig = fh.read(1)
            fh.seek(HEADER_SIZE + 5)
            fh.write(bytes([orig[0] ^ 0xFF]))
        with pytest.raises(ValueError, match="checksum"):
            load_vqz(path)

    def test_rejects_truncated_file(self, tmp_path):
        path = str(tmp_path / "truncated.vqz")
        with open(path, "wb") as fh:
            fh.write(b"\x00" * 10)
        with pytest.raises(ValueError):
            load_vqz(path)

    def test_large_batch(self, tmp_path):
        rng = np.random.default_rng(99)
        n, d = 4096, 384
        quantized = rng.integers(-128, 128, size=(n, d), dtype=np.int8)
        scales    = rng.random(n).astype(np.float32)
        path = str(tmp_path / "large.vqz")
        save_vqz(quantized, scales, d, path, compression="zlib")
        result = load_vqz(path)
        np.testing.assert_array_equal(result["quantized"], quantized)
        np.testing.assert_array_almost_equal(result["scales"], scales)


# ---------------------------------------------------------------------------
# 4. Cloud backend stubs (instantiation-only; no real I/O)
# ---------------------------------------------------------------------------

class TestCloudBackendStubs:
    def test_s3_backend_requires_fsspec(self, monkeypatch):
        import storage_v3
        monkeypatch.setattr(storage_v3, "_HAVE_FSSPEC", False)
        with pytest.raises(ImportError, match="fsspec"):
            S3Backend("my-bucket")

    def test_gcs_backend_requires_fsspec(self, monkeypatch):
        import storage_v3
        monkeypatch.setattr(storage_v3, "_HAVE_FSSPEC", False)
        with pytest.raises(ImportError, match="fsspec"):
            GCSBackend("my-bucket")

    def test_azure_backend_requires_fsspec(self, monkeypatch):
        import storage_v3
        monkeypatch.setattr(storage_v3, "_HAVE_FSSPEC", False)
        with pytest.raises(ImportError, match="fsspec"):
            AzureBlobBackend("my-container")

    def test_backends_have_expected_attributes(self, monkeypatch):
        import storage_v3
        # Patch _open_fs so we don't need real cloud credentials.
        monkeypatch.setattr(storage_v3, "_HAVE_FSSPEC", True)

        class _FakeFS:
            pass

        for BackendCls in (S3Backend, GCSBackend, AzureBlobBackend):
            monkeypatch.setattr(
                BackendCls, "_open_fs", lambda self: _FakeFS()
            )
            backend = BackendCls("test-bucket", "prefix/v1")
            assert backend.bucket == "test-bucket"
            assert backend.prefix == "prefix/v1"
            assert hasattr(backend, "upload")
            assert hasattr(backend, "download")
            assert hasattr(backend, "save_vqz")
            assert hasattr(backend, "load_vqz")
