import numpy as np
from python import pq


def test_pq_roundtrip_random():
    rng = np.random.default_rng(0)
    n = 100
    d = 32
    emb = rng.standard_normal((n, d)).astype(np.float32)
    m = 8
    ks = 16
    codebooks = pq.train_pq(emb, m, ks, iters=5)
    codes = pq.encode_pq(emb, codebooks)
    recon = pq.decode_pq(codes, codebooks)
    assert recon.shape == emb.shape
    # check average MSE is reasonable (not exact)
    mse = np.mean((emb - recon) ** 2)
    assert mse < 10.0


def test_pq_zero_vectors():
    emb = np.zeros((10, 16), dtype=np.float32)
    m = 4
    ks = 4
    codebooks = pq.train_pq(emb, m, ks, iters=3)
    codes = pq.encode_pq(emb, codebooks)
    recon = pq.decode_pq(codes, codebooks)
    assert np.allclose(recon, 0.0)


def test_pq_bytes_roundtrip():
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((50, 16)).astype(np.float32)
    m = 4
    ks = 8
    codebooks = pq.train_pq(emb, m, ks, iters=3)
    codes = pq.encode_pq(emb, codebooks)
    codes_b, cbs_b = pq.encode_pq_bytes(codes, codebooks, compress=True)
    dec = pq.decode_pq_bytes(codes_b, cbs_b, m=m, ks=ks, dsub=16//m, compressed=True)
    assert dec.shape == emb.shape
