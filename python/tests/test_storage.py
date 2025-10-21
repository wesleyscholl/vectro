import numpy as np
from python import storage
import os


def test_storage_roundtrip(tmp_path):
    q = (np.arange(48, dtype=np.int8)).reshape((12, 4))
    scales = np.linspace(0.1, 1.2, 12).astype(np.float32)
    dims = np.array([4], dtype=np.int32)
    n = np.array([12], dtype=np.int32)
    path = str(tmp_path / 'test.vtrb')
    storage.write_arrays(path, {'q': q, 'scales': scales, 'dims': dims, 'n': n}, compress=True)
    # read back individual arrays
    q2 = storage.read_array(path, 'q')
    scales2 = storage.read_array(path, 'scales')
    dims2 = storage.read_array(path, 'dims')
    n2 = storage.read_array(path, 'n')
    assert np.array_equal(q, q2)
    assert np.allclose(scales, scales2)
    assert dims2[0] == dims[0]
    assert n2[0] == n[0]
