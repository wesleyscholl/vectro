"""Tests for PyTorch bridge using a lightweight Tensor mock (no torch required)."""

import sys
import types
import unittest

import numpy as np

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()


# ---------------------------------------------------------------------------
# Minimal torch mock — avoids requiring a real torch installation in CI
# ---------------------------------------------------------------------------


class _MockTensor:
    """Minimal stand-in for a torch.Tensor."""

    def __init__(self, arr: np.ndarray):
        self._arr = np.asarray(arr, dtype=np.float32)

    def detach(self) -> "_MockTensor":
        return self

    def float(self) -> "_MockTensor":
        return self

    def cpu(self) -> "_MockTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr

    # Expose shape for assertions
    @property
    def shape(self):
        return self._arr.shape


def _build_torch_mock() -> types.ModuleType:
    torch_mock = types.ModuleType("torch")

    def from_numpy(arr: np.ndarray) -> _MockTensor:
        return _MockTensor(arr)

    cuda_mock = types.SimpleNamespace(is_available=lambda: False)
    mps_mock = types.SimpleNamespace(is_available=lambda: False)
    backends_mock = types.SimpleNamespace(mps=mps_mock)

    torch_mock.from_numpy = from_numpy
    torch_mock.cuda = cuda_mock
    torch_mock.backends = backends_mock

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    torch_mock.no_grad = _NoGrad

    return torch_mock


# Install the mock before importing the bridge module
_torch_mock = _build_torch_mock()
sys.modules.setdefault("torch", _torch_mock)


from python.integrations.torch_bridge import compress_tensor, reconstruct_tensor  # noqa: E402


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompressTensor(unittest.TestCase):
    def _make_tensor(self, n: int = 8, dim: int = 16) -> _MockTensor:
        rng = np.random.default_rng(42)
        return _MockTensor(rng.standard_normal((n, dim)).astype(np.float32))

    def test_compress_returns_quantization_result(self):
        from python.interface import QuantizationResult

        tensor = self._make_tensor(8, 16)
        result = compress_tensor(tensor)
        self.assertIsInstance(result, QuantizationResult)

    def test_compress_shape(self):
        tensor = self._make_tensor(5, 32)
        result = compress_tensor(tensor)
        self.assertEqual(result.quantized.shape, (5, 32))
        self.assertEqual(result.scales.shape, (5,))

    def test_compress_single_vec_promoted(self):
        """1-D tensor (single vector) should be auto-promoted to 2-D."""
        arr = np.random.randn(32).astype(np.float32)
        single = _MockTensor(arr)
        result = compress_tensor(single)
        self.assertEqual(result.quantized.shape, (1, 32))

    def test_precision_mode_field(self):
        tensor = self._make_tensor(4, 8)
        result = compress_tensor(tensor, precision_mode="int8")
        self.assertEqual(result.precision_mode, "int8")

    def test_reconstruct_returns_mock_tensor(self):
        tensor = self._make_tensor(4, 16)
        result = compress_tensor(tensor)
        out = reconstruct_tensor(result)
        # mock from_numpy returns _MockTensor
        self.assertIsInstance(out, _MockTensor)
        self.assertEqual(out.shape, (4, 16))

    def test_reconstruct_fidelity(self):
        """Reconstructed values should be close to the originals (int8 round-trip)."""
        rng = np.random.default_rng(7)
        arr = rng.standard_normal((10, 64)).astype(np.float32)
        tensor = _MockTensor(arr)
        result = compress_tensor(tensor)
        out = reconstruct_tensor(result)
        # _MockTensor.numpy() returns the reconstructed array; compute cosine sim
        reconstructed = out.numpy()
        dot = np.sum(arr * reconstructed, axis=1)
        norm_orig = np.linalg.norm(arr, axis=1)
        norm_rec = np.linalg.norm(reconstructed, axis=1)
        cosine_sim = np.mean(dot / (norm_orig * norm_rec + 1e-10))
        self.assertGreater(float(cosine_sim), 0.995)


if __name__ == "__main__":
    unittest.main()
