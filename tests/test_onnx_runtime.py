"""ONNX Runtime integration tests for the INT8 dequantization graph.

All tests in this file require **both** ``onnx`` and ``onnxruntime`` to be
installed.  When either is absent the entire class is skipped gracefully so
the test suite still passes in environments without these optional packages.

To enable these tests::

    pip install "onnx>=1.14" "onnxruntime>=1.17"

"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from typing import NamedTuple

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

_ONNX_AVAILABLE = importlib.util.find_spec("onnx") is not None
_RT_AVAILABLE = importlib.util.find_spec("onnxruntime") is not None
_BOTH_AVAILABLE = _ONNX_AVAILABLE and _RT_AVAILABLE


# ---------------------------------------------------------------------------
# Minimal result stub
# ---------------------------------------------------------------------------


class _QResult(NamedTuple):
    quantized: object
    scales: object
    dims: int
    n: int
    precision_mode: str = "int8"
    group_size: int = 0


def _make_result(n: int = 8, d: int = 16) -> _QResult:
    rng = np.random.default_rng(42)
    q = rng.integers(-127, 127, size=(n, d), dtype=np.int8)
    s = rng.uniform(0.01, 1.0, size=(n,)).astype(np.float32)
    return _QResult(quantized=q, scales=s, dims=d, n=n)


def _numpy_reconstruct(result: _QResult) -> np.ndarray:
    """Mirror of interface.py's NumPy dequantization path."""
    return result.quantized.astype(np.float32) * result.scales[:, np.newaxis]


# ---------------------------------------------------------------------------
# Tests — all skipped unless both onnx + onnxruntime are installed
# ---------------------------------------------------------------------------


@unittest.skipUnless(_BOTH_AVAILABLE, "onnx and onnxruntime packages not installed")
class TestOnnxRuntimeIntegration(unittest.TestCase):
    """Round-trip tests through to_onnx_model → onnxruntime.InferenceSession."""

    @classmethod
    def setUpClass(cls):
        import onnxruntime as ort
        from python.onnx_export import to_onnx_model

        cls._ort = ort
        cls._to_onnx_model = staticmethod(to_onnx_model)

    def _session(self, result: _QResult):
        model = self._to_onnx_model(result)
        return self._ort.InferenceSession(model.SerializeToString())

    def _infer(self, result: _QResult) -> np.ndarray:
        sess = self._session(result)
        outputs = sess.run(
            None,
            {
                "quantized": result.quantized,
                "scales": result.scales,
            },
        )
        return outputs[0]

    def test_to_session_from_model_proto(self):
        """InferenceSession can be created directly from a serialised ModelProto."""
        result = _make_result()
        sess = self._session(result)
        self.assertIsNotNone(sess)

    def test_inference_output_shape(self):
        """Inference output shape must equal (n, d)."""
        n, d = 8, 16
        result = _make_result(n=n, d=d)
        output = self._infer(result)
        self.assertEqual(output.shape, (n, d))

    def test_inference_output_dtype_float32(self):
        """Output dtype must be float32."""
        result = _make_result()
        output = self._infer(result)
        self.assertEqual(output.dtype, np.float32)

    def test_inference_matches_numpy_reconstruction(self):
        """ONNX Runtime output must match the NumPy reference path (atol=1e-5)."""
        result = _make_result()
        ort_out = self._infer(result)
        np_out = _numpy_reconstruct(result)
        np.testing.assert_allclose(ort_out, np_out, atol=1e-5)

    def test_single_vector_inference(self):
        """Single-vector batches (n=1) must work correctly."""
        result = _make_result(n=1, d=32)
        output = self._infer(result)
        expected = _numpy_reconstruct(result)
        self.assertEqual(output.shape, (1, 32))
        np.testing.assert_allclose(output, expected, atol=1e-5)

    def test_large_batch_inference(self):
        """Verify correctness on a larger batch (n=512, d=128)."""
        result = _make_result(n=512, d=128)
        output = self._infer(result)
        expected = _numpy_reconstruct(result)
        self.assertEqual(output.shape, (512, 128))
        np.testing.assert_allclose(output, expected, atol=1e-5)

    def test_inference_with_all_zeros_quantized(self):
        """All-zero quantized array should produce all-zero float output."""
        rng = np.random.default_rng(9)
        n, d = 4, 8
        q = np.zeros((n, d), dtype=np.int8)
        s = rng.uniform(0.01, 1.0, size=(n,)).astype(np.float32)
        result = _QResult(quantized=q, scales=s, dims=d, n=n)
        output = self._infer(result)
        np.testing.assert_array_equal(output, np.zeros((n, d), dtype=np.float32))

    def test_inference_with_max_values(self):
        """All-127 quantized array multiplied by scale should equal 127 * scale."""
        n, d = 4, 8
        q = np.full((n, d), 127, dtype=np.int8)
        s = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        result = _QResult(quantized=q, scales=s, dims=d, n=n)
        output = self._infer(result)
        expected = 127.0 * s[:, np.newaxis] * np.ones((n, d), dtype=np.float32)
        np.testing.assert_allclose(output, expected, atol=1e-5)

    def test_export_to_file_then_load_session(self):
        """export_onnx writes a valid .onnx file that InferenceSession can load."""
        from python.onnx_export import export_onnx

        result = _make_result()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            export_onnx(result, tmp_path)
            sess = self._ort.InferenceSession(tmp_path)
            outputs = sess.run(
                None,
                {"quantized": result.quantized, "scales": result.scales},
            )
            self.assertEqual(outputs[0].shape, (result.n, result.dims))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_onnx_session_input_names_match(self):
        """The session's input names must be 'quantized' and 'scales'."""
        result = _make_result()
        sess = self._session(result)
        input_names = {inp.name for inp in sess.get_inputs()}
        self.assertIn("quantized", input_names)
        self.assertIn("scales", input_names)


if __name__ == "__main__":
    unittest.main()
