"""GPU equivalence tests — verify python/gpu_api.py matches python/interface.py.

All tests are CPU-safe: gpu_api.py falls back to NumPy when MAX Engine / CUDA
is unavailable.  The intent is to certify numerical correctness of the GPU path
so that a real GPU runner can run these same tests with a hardware target.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.gpu_api import (
    batch_cosine_similarity,
    gpu_available,
    gpu_benchmark,
    quantize_int8_batch,
    reconstruct_int8_batch,
)
from python.interface import quantize_embeddings, reconstruct_embeddings

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_VECS = _RNG.standard_normal((64, 32)).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGPUEquivalence(unittest.TestCase):
    """Numerical equivalence between gpu_api and interface reference paths."""

    def test_quantize_int8_batch_scales_match_interface(self):
        """gpu_api scales ≈ interface scales (atol=1e-5)."""
        q_gpu, s_gpu = quantize_int8_batch(_VECS)
        result_ref = quantize_embeddings(_VECS)

        np.testing.assert_allclose(
            s_gpu,
            np.asarray(result_ref.scales, dtype=np.float32),
            atol=1e-5,
            err_msg="Per-vector scales differ between gpu_api and interface",
        )

    def test_quantize_int8_batch_codes_match_interface(self):
        """Quantised int8 codes are byte-for-byte equal to the interface path."""
        q_gpu, _ = quantize_int8_batch(_VECS)
        result_ref = quantize_embeddings(_VECS)

        np.testing.assert_array_equal(
            q_gpu,
            np.asarray(result_ref.quantized, dtype=np.int8),
        )

    def test_reconstruct_int8_batch_matches_interface(self):
        """Reconstructed float32 values ≈ reference path (atol=1e-6)."""
        q_gpu, s_gpu = quantize_int8_batch(_VECS)
        recon_gpu = reconstruct_int8_batch(q_gpu, s_gpu)

        result_ref = quantize_embeddings(_VECS)
        recon_ref = reconstruct_embeddings(result_ref)

        np.testing.assert_allclose(
            recon_gpu,
            recon_ref.astype(np.float32),
            atol=1e-6,
            err_msg="Reconstructed vectors differ between gpu_api and interface",
        )

    def test_round_trip_cosine_similarity_above_threshold(self):
        """INT8 round-trip cosine similarity > 0.999 on 256×64 Gaussian data."""
        rng = np.random.default_rng(0)
        vecs = rng.standard_normal((256, 64)).astype(np.float32)

        q, s = quantize_int8_batch(vecs)
        recon = reconstruct_int8_batch(q, s)

        # Per-row cosine sim
        a_norm = np.linalg.norm(vecs, axis=1, keepdims=True)
        b_norm = np.linalg.norm(recon, axis=1, keepdims=True)
        a_hat = vecs / np.where(a_norm == 0, 1.0, a_norm)
        b_hat = recon / np.where(b_norm == 0, 1.0, b_norm)
        cos_sim = float(np.mean(np.sum(a_hat * b_hat, axis=1)))

        self.assertGreater(cos_sim, 0.999, f"Mean cosine similarity {cos_sim:.6f} < 0.999")

    def test_zero_vector_produces_no_nan(self):
        """A zero input vector must not produce NaN in quantisation or reconstruction."""
        zero = np.zeros((4, 16), dtype=np.float32)
        q, s = quantize_int8_batch(zero)
        recon = reconstruct_int8_batch(q, s)

        self.assertFalse(np.any(np.isnan(q.astype(np.float32))), "NaN in quantised codes")
        self.assertFalse(np.any(np.isnan(s)), "NaN in scales")
        self.assertFalse(np.any(np.isnan(recon)), "NaN in reconstructed vectors")

    def test_single_vector_equivalence(self):
        """Single-row input produces the same result as the interface path."""
        v = _VECS[[0]]  # shape (1, 32)
        q_gpu, s_gpu = quantize_int8_batch(v)
        recon_gpu = reconstruct_int8_batch(q_gpu, s_gpu)

        result_ref = quantize_embeddings(v)
        recon_ref = reconstruct_embeddings(result_ref)

        np.testing.assert_allclose(recon_gpu, recon_ref.astype(np.float32), atol=1e-6)

    def test_self_similarity_diagonal_is_one(self):
        """batch_cosine_similarity(v, v) diagonal ≈ 1.0 for unit-norm input."""
        rng = np.random.default_rng(1)
        v = rng.standard_normal((16, 32)).astype(np.float32)

        sim = batch_cosine_similarity(v, v)  # (16, 16)
        diag = np.diag(sim)

        np.testing.assert_allclose(
            diag,
            np.ones(16, dtype=np.float32),
            atol=1e-5,
            err_msg="Self-similarity diagonal is not 1.0",
        )

    def test_gpu_benchmark_keys_present(self):
        """gpu_benchmark() returns a dict with all required keys."""
        result = gpu_benchmark(n=64, d=32)

        required_keys = {"backend", "device_name", "quantize_vecs_per_sec", "cosine_pairs_per_sec"}
        self.assertTrue(
            required_keys.issubset(result.keys()),
            f"Missing keys: {required_keys - set(result.keys())}",
        )

    def test_gpu_benchmark_throughput_positive(self):
        """Reported quantize throughput must be a positive finite float."""
        result = gpu_benchmark(n=64, d=32)
        vps = result["quantize_vecs_per_sec"]

        self.assertIsInstance(vps, float)
        self.assertGreater(vps, 0.0)
        self.assertTrue(np.isfinite(vps), f"quantize_vecs_per_sec is not finite: {vps}")

    def test_gpu_available_returns_bool(self):
        """gpu_available() must return a Python bool (not a truthy/falsy object)."""
        result = gpu_available()
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
