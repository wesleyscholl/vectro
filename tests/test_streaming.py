"""Tests for python/streaming.py — StreamingDecompressor."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.batch_api import BatchQuantizationResult
from python.interface import QuantizationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_int8_batch(n: int = 50, dim: int = 32) -> BatchQuantizationResult:
    """Create a simple INT8 BatchQuantizationResult for testing."""
    rng = np.random.default_rng(7)
    floats = rng.standard_normal((n, dim)).astype(np.float32)
    scales = np.max(np.abs(floats), axis=1) / 127.0
    q = np.clip(np.round(floats / scales[:, None]), -127, 127).astype(np.int8)
    original = n * dim * 4
    compressed = q.nbytes + scales.nbytes
    return BatchQuantizationResult(
        quantized_vectors=list(q),
        scales=scales,
        batch_size=n,
        vector_dim=dim,
        compression_ratio=original / compressed,
        total_original_bytes=original,
        total_compressed_bytes=compressed,
    )


def _make_int8_quant_result(n: int = 50, dim: int = 32) -> QuantizationResult:
    """Create a QuantizationResult (single-block) for testing."""
    rng = np.random.default_rng(13)
    floats = rng.standard_normal((n, dim)).astype(np.float32)
    scales = np.max(np.abs(floats), axis=1) / 127.0
    q = np.clip(np.round(floats / scales[:, None]), -127, 127).astype(np.int8)
    return QuantizationResult(quantized=q, scales=scales, dims=dim, n=n)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreamingDecompressorInit(unittest.TestCase):
    def test_invalid_chunk_size_raises(self):
        from python.streaming import StreamingDecompressor

        result = _make_int8_batch()
        with self.assertRaises(ValueError):
            StreamingDecompressor(result, chunk_size=0)

    def test_chunk_size_one_allowed(self):
        from python.streaming import StreamingDecompressor

        result = _make_int8_batch(n=5)
        sd = StreamingDecompressor(result, chunk_size=1)
        self.assertIsNotNone(sd)


class TestStreamingDecompressorLen(unittest.TestCase):
    def test_len_batch_result(self):
        from python.streaming import StreamingDecompressor

        result = _make_int8_batch(n=37)
        sd = StreamingDecompressor(result, chunk_size=10)
        self.assertEqual(len(sd), 37)

    def test_len_quant_result(self):
        from python.streaming import StreamingDecompressor

        result = _make_int8_quant_result(n=22)
        sd = StreamingDecompressor(result, chunk_size=10)
        self.assertEqual(len(sd), 22)


class TestStreamingDecompressorChunks(unittest.TestCase):
    def test_total_vectors_match(self):
        """All chunks together reconstruct exactly n vectors."""
        from python.streaming import StreamingDecompressor

        n, dim, chunk = 50, 32, 13
        result = _make_int8_batch(n=n, dim=dim)
        sd = StreamingDecompressor(result, chunk_size=chunk)
        total = sum(len(c) for c in sd)
        self.assertEqual(total, n)

    def test_chunk_shape(self):
        """Each chunk has shape (≤chunk_size, dim)."""
        from python.streaming import StreamingDecompressor

        n, dim, chunk_size = 27, 16, 10
        result = _make_int8_batch(n=n, dim=dim)
        for chunk in StreamingDecompressor(result, chunk_size=chunk_size):
            self.assertEqual(chunk.ndim, 2)
            self.assertEqual(chunk.shape[1], dim)
            self.assertLessEqual(chunk.shape[0], chunk_size)

    def test_last_chunk_correct_size(self):
        """Last chunk contains the remainder (n % chunk_size)."""
        from python.streaming import StreamingDecompressor

        n, dim, chunk_size = 25, 16, 7   # remainder = 4
        result = _make_int8_batch(n=n, dim=dim)
        chunks = list(StreamingDecompressor(result, chunk_size=chunk_size))
        self.assertEqual(len(chunks[-1]), n % chunk_size)

    def test_single_chunk_when_n_lt_chunk_size(self):
        """When n < chunk_size we get exactly one chunk."""
        from python.streaming import StreamingDecompressor

        n, dim = 5, 8
        result = _make_int8_batch(n=n, dim=dim)
        chunks = list(StreamingDecompressor(result, chunk_size=100))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].shape[0], n)

    def test_chunk_dtype_float32(self):
        """Reconstructed chunks are float32."""
        from python.streaming import StreamingDecompressor

        result = _make_int8_batch(n=10, dim=8)
        for chunk in StreamingDecompressor(result, chunk_size=5):
            self.assertEqual(chunk.dtype, np.float32)

    def test_reconstruction_accuracy(self):
        """Reconstructed values are approximately correct (INT8 quantization error)."""
        from python.streaming import StreamingDecompressor

        n, dim = 20, 32
        rng = np.random.default_rng(42)
        floats = rng.standard_normal((n, dim)).astype(np.float32)
        scales = np.max(np.abs(floats), axis=1) / 127.0
        q = np.clip(np.round(floats / scales[:, None]), -127, 127).astype(np.int8)
        original = n * dim * 4
        compressed = q.nbytes + scales.nbytes
        result = BatchQuantizationResult(
            quantized_vectors=list(q),
            scales=scales,
            batch_size=n,
            vector_dim=dim,
            compression_ratio=original / compressed,
            total_original_bytes=original,
            total_compressed_bytes=compressed,
        )
        recon = np.vstack(list(StreamingDecompressor(result, chunk_size=7)))

        dot = np.sum(floats * recon, axis=1)
        norms = np.linalg.norm(floats, axis=1) * np.linalg.norm(recon, axis=1)
        cos_sim = np.mean(dot / (norms + 1e-10))
        self.assertGreater(cos_sim, 0.99)

    def test_iterator_reuse(self):
        """Iterating twice produces the same total count."""
        from python.streaming import StreamingDecompressor

        result = _make_int8_batch(n=20, dim=8)
        sd = StreamingDecompressor(result, chunk_size=7)
        first_total = sum(len(c) for c in sd)
        second_total = sum(len(c) for c in sd)
        self.assertEqual(first_total, second_total)


class TestStreamingDecompressorQuantResult(unittest.TestCase):
    """Test the QuantizationResult path."""

    def test_len_quant_result(self):
        from python.streaming import StreamingDecompressor

        result = _make_int8_quant_result(n=15, dim=8)
        self.assertEqual(len(StreamingDecompressor(result)), 15)

    def test_total_vectors_quant_result(self):
        from python.streaming import StreamingDecompressor

        n = 15
        result = _make_int8_quant_result(n=n, dim=8)
        total = sum(len(c) for c in StreamingDecompressor(result, chunk_size=6))
        self.assertEqual(total, n)

    def test_chunk_dtype_quant_result(self):
        from python.streaming import StreamingDecompressor

        result = _make_int8_quant_result(n=10, dim=8)
        for chunk in StreamingDecompressor(result, chunk_size=4):
            self.assertEqual(chunk.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
