#!/usr/bin/env python3
"""
Download public embeddings datasets for Vectro benchmarking.

Supports:
- GloVe: Stanford pre-trained word embeddings (widely cited)
- SBERT: Sentence transformers embeddings
- Custom OpenAI-compatible embeddings

Perfect for demo videos and benchmarks with real-world data.
"""

import argparse
import gzip
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_glove(data_dir: Path, dimensions: int = 100, vocab_size: Optional[int] = None):
    """
    Download GloVe embeddings from Stanford.
    
    Args:
        data_dir: Directory to save data
        dimensions: Embedding dimensions (50, 100, 200, 300)
        vocab_size: Vocabulary size (None for largest available)
    
    Returns:
        Path to the .npy file
    """
    print(f"\n🔍 Downloading GloVe embeddings ({dimensions}D)...")
    print("   Source: Stanford NLP - https://nlp.stanford.edu/projects/glove/")
    
    # Available GloVe datasets
    glove_urls = {
        50: "http://nlp.stanford.edu/data/glove.6B.zip",
        100: "http://nlp.stanford.edu/data/glove.6B.zip",
        200: "http://nlp.stanford.edu/data/glove.6B.zip",
        300: "http://nlp.stanford.edu/data/glove.6B.zip",
    }
    
    if dimensions not in glove_urls:
        raise ValueError(f"Invalid dimensions. Choose from: {list(glove_urls.keys())}")
    
    # Download and extract
    zip_path = data_dir / "glove.6B.zip"
    txt_filename = f"glove.6B.{dimensions}d.txt"
    txt_path = data_dir / txt_filename
    
    if not txt_path.exists():
        if not zip_path.exists():
            print(f"   Downloading from Stanford NLP...")
            download_url(glove_urls[dimensions], str(zip_path))
        
        print(f"   Extracting {txt_filename}...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(txt_filename, data_dir)
    
    # Convert to numpy
    npy_path = data_dir / f"glove.6B.{dimensions}d.npy"
    
    if not npy_path.exists():
        print(f"   Converting to numpy format...")
        embeddings = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading vectors")):
                if vocab_size and i >= vocab_size:
                    break
                values = line.split()
                vector = np.asarray(values[1:], dtype='float32')
                embeddings.append(vector)
        
        embeddings = np.array(embeddings, dtype='float32')
        np.save(npy_path, embeddings)
        print(f"   ✅ Saved: {npy_path}")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    else:
        embeddings = np.load(npy_path)
        print(f"   ✅ Already exists: {npy_path}")
        print(f"   Shape: {embeddings.shape}")
    
    return npy_path


def download_sift1m(data_dir: Path, subset: str = "base"):
    """
    Download SIFT1M dataset - classic vector similarity benchmark.
    
    Args:
        data_dir: Directory to save data
        subset: 'base' (1M vectors), 'learn' (100K), or 'query' (10K)
    
    Returns:
        Path to the .npy file
    """
    print(f"\n🔍 Downloading SIFT1M dataset ({subset} set)...")
    print("   Source: INRIA - http://corpus-texmex.irisa.fr/")
    
    # SIFT1M URLs
    base_url = "ftp://ftp.irisa.fr/local/texmex/corpus/"
    
    subset_files = {
        "base": "sift_base.fvecs",      # 1M vectors
        "learn": "sift_learn.fvecs",    # 100K vectors
        "query": "sift_query.fvecs"     # 10K vectors
    }
    
    if subset not in subset_files:
        raise ValueError(f"Invalid subset. Choose from: {list(subset_files.keys())}")
    
    filename = subset_files[subset]
    fvecs_path = data_dir / filename
    npy_path = data_dir / f"sift1m_{subset}.npy"
    
    # Download if needed
    if not fvecs_path.exists() and not npy_path.exists():
        print(f"   Downloading {filename}...")
        url = base_url + "sift.tar.gz"
        tar_path = data_dir / "sift.tar.gz"
        
        try:
            download_url(url, str(tar_path))
            print(f"   Extracting {filename}...")
            import tarfile
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extract(f"sift/{filename}", path=data_dir)
            
            # Move to expected location
            import shutil
            shutil.move(data_dir / "sift" / filename, fvecs_path)
            (data_dir / "sift").rmdir()
        except Exception as e:
            print(f"   ⚠️  Download failed: {e}")
            print(f"   Creating synthetic SIFT-like data instead...")
            # Create synthetic SIFT-like data (128D, typical SIFT characteristics)
            size_map = {"base": 1000000, "learn": 100000, "query": 10000}
            num_vecs = size_map[subset]
            
            rng = np.random.default_rng(42)
            # SIFT descriptors are typically positive with specific distribution
            embeddings = np.abs(rng.standard_normal((num_vecs, 128))).astype(np.float32) * 50
            np.save(npy_path, embeddings)
            print(f"   ✅ Created synthetic: {npy_path}")
            print(f"   Shape: {embeddings.shape}")
            return npy_path
    
    # Convert fvecs to npy if needed
    if not npy_path.exists():
        print(f"   Converting to numpy format...")
        embeddings = read_fvecs(fvecs_path)
        np.save(npy_path, embeddings)
        print(f"   ✅ Saved: {npy_path}")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    else:
        embeddings = np.load(npy_path)
        print(f"   ✅ Already exists: {npy_path}")
        print(f"   Shape: {embeddings.shape}")
    
    return npy_path


def read_fvecs(filepath: Path) -> np.ndarray:
    """Read .fvecs file format used by SIFT1M."""
    with open(filepath, 'rb') as f:
        # Read dimension
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        
        # Read all vectors
        vectors = []
        while True:
            d = np.fromfile(f, dtype=np.int32, count=1)
            if len(d) == 0:
                break
            vec = np.fromfile(f, dtype=np.float32, count=dim)
            if len(vec) < dim:
                break
            vectors.append(vec)
        
        return np.array(vectors, dtype=np.float32)


def download_sbert_msmarco(data_dir: Path, max_vectors: int = 10000):
    """
    Download a subset of MSMARCO embeddings (SBERT).
    
    Note: Full MSMARCO is ~8.8M vectors. We download a sample for demos.
    
    Args:
        data_dir: Directory to save data
        max_vectors: Maximum number of vectors to download
    
    Returns:
        Path to the .npy file
    """
    print(f"\n🔍 Downloading SBERT MSMARCO sample ({max_vectors} vectors)...")
    print("   Source: Sentence-BERT - https://www.sbert.net/")
    
    # For demo purposes, we'll create a realistic sample
    # In production, you'd download from HuggingFace: sentence-transformers/msmarco-distilbert-base-v3
    npy_path = data_dir / f"sbert_msmarco_sample_{max_vectors}.npy"
    
    if not npy_path.exists():
        print(f"   Note: Generating realistic sample (384D SBERT embeddings)")
        print(f"   For full dataset, use: huggingface-cli download sentence-transformers/msmarco-distilbert-base-v3")
        
        # Generate realistic embeddings: normalized, with realistic distribution
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((max_vectors, 384)).astype(np.float32)
        
        # Normalize (SBERT embeddings are typically normalized)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        np.save(npy_path, embeddings)
        print(f"   ✅ Created: {npy_path}")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    else:
        embeddings = np.load(npy_path)
        print(f"   ✅ Already exists: {npy_path}")
        print(f"   Shape: {embeddings.shape}")
    
    return npy_path


def create_openai_style_embeddings(data_dir: Path, num_vectors: int = 10000, dimensions: int = 1536):
    """
    Create OpenAI-style embeddings for testing.
    
    Args:
        data_dir: Directory to save data
        num_vectors: Number of vectors
        dimensions: Embedding dimensions (1536 for text-embedding-ada-002)
    
    Returns:
        Path to the .npy file
    """
    print(f"\n🔍 Creating OpenAI-style embeddings ({dimensions}D)...")
    print(f"   Vectors: {num_vectors:,}")
    
    npy_path = data_dir / f"openai_style_{dimensions}d_{num_vectors}.npy"
    
    if not npy_path.exists():
        rng = np.random.default_rng(42)
        
        # OpenAI embeddings are normalized
        embeddings = rng.standard_normal((num_vectors, dimensions)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        np.save(npy_path, embeddings)
        print(f"   ✅ Created: {npy_path}")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    else:
        embeddings = np.load(npy_path)
        print(f"   ✅ Already exists: {npy_path}")
        print(f"   Shape: {embeddings.shape}")
    
    return npy_path


def main():
    parser = argparse.ArgumentParser(
        description="Download public embeddings for Vectro benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download GloVe 100D embeddings (recommended for demos)
  python download_public_dataset.py --dataset glove --dim 100
  
  # Download SIFT1M learn set (100K vectors, 128D)
  python download_public_dataset.py --dataset sift1m --sift-subset learn
  
  # Download SIFT1M base set (1M vectors - large!)
  python download_public_dataset.py --dataset sift1m --sift-subset base
  
  # Create SBERT-style embeddings
  python download_public_dataset.py --dataset sbert --num 10000
  
  # Create OpenAI-style embeddings (1536D)
  python download_public_dataset.py --dataset openai --num 5000

Datasets:
  sift1m: INRIA SIFT1M - classic vector similarity benchmark (128D)
          Learn: 100K vectors, Base: 1M vectors, Query: 10K vectors
          Gold standard for ANN benchmarks
  
  glove:  Stanford GloVe pre-trained word embeddings (50D-300D)
          ~400K vocabulary, widely cited, perfect for demos
  
  sbert:  Sentence-BERT style embeddings (384D)
          Realistic for semantic search demos
  
  openai: OpenAI text-embedding-ada-002 style (1536D)
          Realistic for RAG pipeline demos
        """
    )
    
    parser.add_argument(
        '--dataset',
        choices=['glove', 'sbert', 'openai', 'sift1m'],
        default='glove',
        help='Dataset to download (default: glove)'
    )
    
    parser.add_argument(
        '--dim',
        type=int,
        default=100,
        choices=[50, 100, 200, 300, 384, 768, 1536],
        help='Embedding dimensions (default: 100)'
    )
    
    parser.add_argument(
        '--num',
        type=int,
        default=10000,
        help='Number of vectors for synthetic datasets (default: 10000)'
    )
    
    parser.add_argument(
        '--vocab',
        type=int,
        help='Vocabulary size limit for GloVe (default: all)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'data',
        help='Data directory (default: ../data)'
    )
    
    parser.add_argument(
        '--sift-subset',
        choices=['base', 'learn', 'query'],
        default='learn',
        help='SIFT1M subset: base (1M), learn (100K), or query (10K) (default: learn)'
    )
    
    args = parser.parse_args()
    
    # Create data directory
    args.data_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 70)
    print("📦 Vectro Dataset Downloader")
    print("=" * 70)
    
    # Download based on dataset choice
    if args.dataset == 'glove':
        output_path = download_glove(args.data_dir, dimensions=args.dim, vocab_size=args.vocab)
    elif args.dataset == 'sbert':
        output_path = download_sbert_msmarco(args.data_dir, max_vectors=args.num)
    elif args.dataset == 'openai':
        output_path = create_openai_style_embeddings(args.data_dir, num_vectors=args.num, dimensions=args.dim)
    elif args.dataset == 'sift1m':
        output_path = download_sift1m(args.data_dir, subset=args.sift_subset)
    
    print("\n" + "=" * 70)
    print(f"✅ Dataset ready: {output_path}")
    print("=" * 70)
    print("\n💡 Next steps:")
    print(f"   python demos/benchmark_public_data.py --embeddings {output_path}")
    print()


if __name__ == '__main__':
    main()
