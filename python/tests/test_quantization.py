import os
import sys
import numpy as np

# Ensure project root is on sys.path so tests can import local package
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from interface import quantize_embeddings, reconstruct_embeddings, mean_cosine_similarity


def test_quantize_reconstruct_random():
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((50, 128)).astype(np.float32)
    out = quantize_embeddings(emb)
    recon = reconstruct_embeddings(out['q'], out['scales'], out['dims'])
    mcos = mean_cosine_similarity(emb, recon)
    print("mean cosine", mcos)
    assert mcos > 0.95


def test_zero_vector():
    emb = np.zeros((5, 64), dtype=np.float32)
    out = quantize_embeddings(emb)
    recon = reconstruct_embeddings(out['q'], out['scales'], out['dims'])
    assert np.allclose(recon, 0)
    assert mean_cosine_similarity(emb, recon) == 1.0
