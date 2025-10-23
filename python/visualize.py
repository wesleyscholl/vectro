"""Visualization utilities for Vectro compression quality.

Provides functions to visualize embedding distributions before/after compression
using PCA/t-SNE and cosine similarity heatmaps.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from python.interface import quantize_embeddings, reconstruct_embeddings, mean_cosine_similarity


def plot_embedding_scatter(orig: np.ndarray, recon: np.ndarray, title: str = "Embeddings", save_path: str = None):
    """Plot 2D scatter of original vs reconstructed embeddings using PCA."""
    # Reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    orig_2d = pca.fit_transform(orig)
    recon_2d = pca.transform(recon)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(orig_2d[:, 0], orig_2d[:, 1], alpha=0.6, s=10, c='blue', label='Original')
    plt.title(f'{title} - Original')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(recon_2d[:, 0], recon_2d[:, 1], alpha=0.6, s=10, c='red', label='Reconstructed')
    plt.title(f'{title} - Reconstructed')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_cosine_similarity_distribution(orig: np.ndarray, recon: np.ndarray, save_path: str = None):
    """Plot distribution of cosine similarities between original and reconstructed vectors."""
    # Compute pairwise cosine similarities (for small datasets)
    n = len(orig)
    if n > 1000:
        # Sample for large datasets
        idx = np.random.choice(n, 1000, replace=False)
        orig_sample = orig[idx]
        recon_sample = recon[idx]
    else:
        orig_sample = orig
        recon_sample = recon

    # Cosine similarity for each vector
    norms_orig = np.linalg.norm(orig_sample, axis=1, keepdims=True)
    norms_recon = np.linalg.norm(recon_sample, axis=1, keepdims=True)
    cos_sim = np.sum(orig_sample * recon_sample, axis=1) / (norms_orig.ravel() * norms_recon.ravel())

    plt.figure(figsize=(8, 6))
    plt.hist(cos_sim, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(cos_sim), color='red', linestyle='--', label=f'Mean: {np.mean(cos_sim):.6f}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Cosine Similarity Distribution: Original vs Reconstructed')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def visualize_compression_quality(embeddings_path: str, compressed_path: str = None, backend: str = 'python'):
    """Load embeddings, compress if needed, and create visualizations."""
    # Load original
    orig = np.load(embeddings_path)
    print(f"Loaded embeddings: shape {orig.shape}")

    if compressed_path:
        # Load compressed and reconstruct
        if compressed_path.endswith('.npz'):
            npz = np.load(compressed_path)
            q = npz['q']
            scales = npz['scales']
            dims = int(npz['dims'][0])
        else:
            # For VTRB02, need to implement loading
            raise NotImplementedError("VTRB02 loading not implemented yet")
        recon = reconstruct_embeddings(q, scales, dims)
    else:
        # Compress on the fly
        out = quantize_embeddings(orig)
        recon = reconstruct_embeddings(out['q'], out['scales'], out['dims'])

    # Compute metrics
    mcos = mean_cosine_similarity(orig, recon)
    orig_bytes = orig.nbytes
    comp_bytes = out['q'].nbytes + out['scales'].nbytes
    ratio = comp_bytes / orig_bytes

    print(f"Mean cosine similarity: {mcos:.6f}")
    print(f"Compression ratio: {ratio:.4f} ({comp_bytes:,} / {orig_bytes:,} bytes)")

    # Create plots
    plot_embedding_scatter(orig, recon, f"Vectro {backend.upper()} Compression")
    plot_cosine_similarity_distribution(orig, recon)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', required=True, help='Path to .npy embeddings')
    parser.add_argument('--compressed', help='Path to compressed file (optional)')
    parser.add_argument('--backend', default='python', choices=['python', 'pq'], help='Compression backend')
    args = parser.parse_args()

    visualize_compression_quality(args.embeddings, args.compressed, args.backend)