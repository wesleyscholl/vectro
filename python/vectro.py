"""
Vectro - Ultra-High-Performance LLM Embedding Compressor
Unified Python API providing access to all Mojo-powered compression capabilities.

This module provides a clean, Pythonic interface to Vectro's high-performance
vector quantization capabilities implemented in Mojo.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import warnings

# Import Vectro modules
from .interface import (
    QuantizationResult, 
    quantize_embeddings, 
    reconstruct_embeddings,
    mean_cosine_similarity,
    get_backend_info
)
from .batch_api import (
    VectroBatchProcessor,
    BatchQuantizationResult,
    BatchCompressionAnalyzer,
    quantize_embeddings_batch,
    benchmark_batch_compression
)
from .quality_api import (
    VectroQualityAnalyzer,
    QualityMetrics,
    QualityBenchmark,
    QualityReport,
    evaluate_quantization_quality,
    generate_quality_report
)
from .profiles_api import (
    ProfileManager,
    CompressionProfile,
    CompressionStrategy,
    CompressionOptimizer,
    ProfileComparison,
    get_compression_profile,
    create_custom_profile
)

__version__ = "1.2.0"
__author__ = "Wesley Scholl"
__license__ = "MIT"
__description__ = "Ultra-High-Performance LLM Embedding Compressor"


class Vectro:
    """
    Main Vectro API class providing unified access to all compression capabilities.
    
    This class serves as the primary interface for vector quantization operations,
    combining high-performance Mojo backends with convenient Python APIs.
    """
    
    def __init__(
        self, 
        backend: str = "auto",
        profile: str = "balanced",
        enable_batch_optimization: bool = True
    ):
        """Initialize Vectro compressor.
        
        Args:
            backend: Processing backend ("auto", "mojo", "cython", "python")
            profile: Default compression profile ("fast", "balanced", "quality")  
            enable_batch_optimization: Enable batch processing optimizations
        """
        self.backend = backend
        self.default_profile = profile
        self.enable_batch_optimization = enable_batch_optimization
        
        # Initialize components
        self.batch_processor = VectroBatchProcessor(backend=backend)
        self.quality_analyzer = VectroQualityAnalyzer()
        
        # Initialize profiles
        ProfileManager.initialize_builtin_profiles()
        
        # Cache for performance metrics
        self._performance_cache = {}
        
    def compress(
        self,
        vectors: Union[np.ndarray, List[np.ndarray]],
        profile: Optional[str] = None,
        return_quality_metrics: bool = False
    ) -> Union[QuantizationResult, BatchQuantizationResult, Tuple[Any, QualityMetrics]]:
        """Compress vectors using quantization.
        
        Args:
            vectors: Input vectors to compress (single vector or batch)
            profile: Compression profile to use (None = use default)
            return_quality_metrics: Return quality analysis alongside results
            
        Returns:
            Quantization result(s) and optionally quality metrics
        """
        if profile is None:
            profile = self.default_profile
            
        # Convert input to numpy array
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float32)
        elif not isinstance(vectors, np.ndarray):
            raise ValueError("vectors must be numpy array or list")
            
        vectors = vectors.astype(np.float32)
        
        # Determine if single vector or batch
        if vectors.ndim == 1:
            # Single vector
            vectors = vectors.reshape(1, -1)
            result = self._compress_single_batch(vectors, profile)
            
            # Extract single result
            single_result = QuantizationResult(
                quantized=result.quantized_vectors[0],
                scales=result.scales[0:1],
                dims=result.vector_dim,
                n=1
            )
            
            if return_quality_metrics:
                original = vectors
                reconstructed = result.reconstruct_batch()
                quality = self.quality_analyzer.evaluate_quality(original, reconstructed)
                return single_result, quality
            else:
                return single_result
                
        elif vectors.ndim == 2:
            # Batch of vectors
            if self.enable_batch_optimization and len(vectors) > 1:
                result = self._compress_single_batch(vectors, profile)
            else:
                # Process individually for small batches
                result = self._compress_individually(vectors, profile)
            
            if return_quality_metrics:
                original = vectors
                reconstructed = result.reconstruct_batch()
                quality = self.quality_analyzer.evaluate_quality(original, reconstructed)
                return result, quality
            else:
                return result
        else:
            raise ValueError("vectors must be 1D or 2D array")
    
    def decompress(
        self, 
        result: Union[QuantizationResult, BatchQuantizationResult]
    ) -> np.ndarray:
        """Decompress quantized vectors back to float32.
        
        Args:
            result: Quantization result to decompress
            
        Returns:
            Reconstructed vectors as float32 array
        """
        if isinstance(result, QuantizationResult):
            return reconstruct_embeddings(result, backend=self.backend)
        elif isinstance(result, BatchQuantizationResult):
            return result.reconstruct_batch()
        else:
            raise ValueError("result must be QuantizationResult or BatchQuantizationResult")
    
    def analyze_quality(
        self,
        original: np.ndarray,
        compressed_result: Union[QuantizationResult, BatchQuantizationResult]
    ) -> QualityMetrics:
        """Analyze compression quality metrics.
        
        Args:
            original: Original vectors
            compressed_result: Compression result
            
        Returns:
            Comprehensive quality metrics
        """
        reconstructed = self.decompress(compressed_result)
        
        if isinstance(compressed_result, BatchQuantizationResult):
            compression_ratio = compressed_result.compression_ratio
        else:
            # Estimate compression ratio for single vector
            orig_bytes = original.size * 4
            comp_bytes = len(compressed_result.quantized) * 1 + 4
            compression_ratio = orig_bytes / comp_bytes
            
        return self.quality_analyzer.evaluate_quality(
            original, reconstructed, compression_ratio
        )
    
    def benchmark_performance(
        self,
        vector_dims: List[int] = None,
        batch_sizes: List[int] = None,
        profiles: List[str] = None,
        num_trials: int = 3
    ) -> Dict[str, Any]:
        """Comprehensive performance benchmarking.
        
        Args:
            vector_dims: Dimensions to benchmark
            batch_sizes: Batch sizes to test
            profiles: Profiles to compare
            num_trials: Number of trials per configuration
            
        Returns:
            Detailed performance metrics
        """
        if vector_dims is None:
            vector_dims = [128, 384, 768]
        if batch_sizes is None:
            batch_sizes = [100, 1000]
        if profiles is None:
            profiles = ["fast", "balanced", "quality"]
        
        results = {
            "throughput_benchmarks": {},
            "profile_comparisons": {},
            "quality_benchmarks": {},
            "system_info": get_backend_info()
        }
        
        # Throughput benchmarking
        results["throughput_benchmarks"] = self.batch_processor.benchmark_batch_performance(
            batch_sizes, vector_dims, num_trials
        )
        
        # Profile comparisons
        for dim in vector_dims:
            test_vectors = np.random.randn(500, dim).astype(np.float32)
            comparison = ProfileComparison.compare_profiles(test_vectors, profiles)
            results["profile_comparisons"][f"dim_{dim}"] = comparison
        
        # Quality benchmarking across dimensions
        quality_results = QualityBenchmark.benchmark_dimensions(
            vector_dims, num_vectors=200
        )
        results["quality_benchmarks"] = {
            dim: metrics.to_dict() for dim, metrics in quality_results.items()
        }
        
        return results
    
    def optimize_profile(
        self,
        sample_vectors: np.ndarray,
        target_similarity: float = 0.995,
        target_compression: float = 3.0
    ) -> CompressionProfile:
        """Automatically optimize a compression profile for specific data.
        
        Args:
            sample_vectors: Representative sample of your data
            target_similarity: Desired cosine similarity threshold
            target_compression: Desired compression ratio
            
        Returns:
            Optimized CompressionProfile
        """
        return CompressionOptimizer.auto_optimize_profile(
            sample_vectors, target_similarity, target_compression
        )
    
    def compare_profiles(
        self,
        vectors: np.ndarray,
        profile_names: List[str] = None
    ) -> str:
        """Compare compression profiles and generate a report.
        
        Args:
            vectors: Test vectors for comparison
            profile_names: Profiles to compare (None = default set)
            
        Returns:
            Formatted comparison report
        """
        comparison = ProfileComparison.compare_profiles(vectors, profile_names)
        return ProfileComparison.generate_comparison_report(comparison)
    
    def save_compressed(
        self,
        result: Union[QuantizationResult, BatchQuantizationResult],
        filepath: str
    ):
        """Save compressed vectors to file.
        
        Args:
            result: Compression result to save
            filepath: Output file path (.npz format)
        """
        if isinstance(result, QuantizationResult):
            np.savez_compressed(
                filepath,
                quantized=result.quantized,
                scales=result.scales,
                dims=result.dims,
                n=result.n
            )
        elif isinstance(result, BatchQuantizationResult):
            # Convert list of arrays to single array for storage
            quantized_array = np.array(result.quantized_vectors)
            np.savez_compressed(
                filepath,
                quantized=quantized_array,
                scales=result.scales,
                batch_size=result.batch_size,
                vector_dim=result.vector_dim,
                compression_ratio=result.compression_ratio,
                total_original_bytes=result.total_original_bytes,
                total_compressed_bytes=result.total_compressed_bytes
            )
    
    def load_compressed(self, filepath: str) -> Union[QuantizationResult, BatchQuantizationResult]:
        """Load compressed vectors from file.
        
        Args:
            filepath: Input file path (.npz format)
            
        Returns:
            Loaded compression result
        """
        data = np.load(filepath)
        
        if 'batch_size' in data:
            # BatchQuantizationResult
            quantized_list = [data['quantized'][i] for i in range(len(data['quantized']))]
            
            return BatchQuantizationResult(
                quantized_vectors=quantized_list,
                scales=data['scales'],
                batch_size=int(data['batch_size']),
                vector_dim=int(data['vector_dim']),
                compression_ratio=float(data['compression_ratio']),
                total_original_bytes=int(data['total_original_bytes']),
                total_compressed_bytes=int(data['total_compressed_bytes'])
            )
        else:
            # QuantizationResult
            return QuantizationResult(
                quantized=data['quantized'],
                scales=data['scales'],
                dims=int(data['dims']),
                n=int(data['n'])
            )
    
    def _compress_single_batch(
        self, 
        vectors: np.ndarray, 
        profile: str
    ) -> BatchQuantizationResult:
        """Compress vectors as a single batch."""
        return self.batch_processor.quantize_batch(vectors, profile)
    
    def _compress_individually(
        self, 
        vectors: np.ndarray, 
        profile: str
    ) -> BatchQuantizationResult:
        """Compress vectors individually and combine results."""
        batch_size, vector_dim = vectors.shape
        quantized_vectors = []
        scales = np.zeros(batch_size, dtype=np.float32)
        
        for i in range(batch_size):
            result = quantize_embeddings(vectors[i:i+1], backend=self.backend)
            quantized_vectors.append(result.quantized[0])
            scales[i] = result.scales[0]
        
        # Calculate compression metrics
        original_bytes = batch_size * vector_dim * 4
        compressed_bytes = batch_size * vector_dim * 1 + batch_size * 4
        compression_ratio = original_bytes / compressed_bytes
        
        return BatchQuantizationResult(
            quantized_vectors=quantized_vectors,
            scales=scales,
            batch_size=batch_size,
            vector_dim=vector_dim,
            compression_ratio=compression_ratio,
            total_original_bytes=original_bytes,
            total_compressed_bytes=compressed_bytes
        )
    
    @property
    def available_profiles(self) -> List[str]:
        """Get list of available compression profiles."""
        return ProfileManager.list_profiles()
    
    @property
    def backend_info(self) -> Dict[str, Any]:
        """Get information about available backends."""
        return get_backend_info()


# Convenience functions for quick access
def compress_vectors(
    vectors: Union[np.ndarray, List[np.ndarray]],
    profile: str = "balanced",
    backend: str = "auto"
) -> Union[QuantizationResult, BatchQuantizationResult]:
    """Quick vector compression with default settings."""
    compressor = Vectro(backend=backend, profile=profile)
    return compressor.compress(vectors)


def decompress_vectors(
    result: Union[QuantizationResult, BatchQuantizationResult]
) -> np.ndarray:
    """Quick vector decompression."""
    compressor = Vectro()
    return compressor.decompress(result)


def analyze_compression_quality(
    original: np.ndarray,
    result: Union[QuantizationResult, BatchQuantizationResult]
) -> QualityMetrics:
    """Quick quality analysis."""
    compressor = Vectro()
    return compressor.analyze_quality(original, result)


def generate_compression_report(
    original: np.ndarray,
    result: Union[QuantizationResult, BatchQuantizationResult]
) -> str:
    """Generate a comprehensive compression report."""
    quality = analyze_compression_quality(original, result)
    return QualityReport.generate_report(quality, "Compression Quality Report")


# Module-level configuration
def set_default_backend(backend: str):
    """Set the default backend for all operations."""
    import sys
    module = sys.modules[__name__]
    module._default_backend = backend


def get_version_info() -> Dict[str, str]:
    """Get version information."""
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "description": __description__
    }


# Example usage demonstration
if __name__ == "__main__":
    print(f"Vectro {__version__} - {__description__}")
    print("=" * 50)
    
    # Create sample embeddings
    np.random.seed(42)
    embeddings = np.random.randn(1000, 384).astype(np.float32)
    
    print(f"Sample embeddings: {embeddings.shape}")
    print(f"Original size: {embeddings.nbytes / 1024:.1f} KB")
    
    # Initialize Vectro compressor
    vectro = Vectro(profile="balanced")
    
    # Compress vectors
    print("\nCompressing vectors...")
    compressed_result = vectro.compress(embeddings)
    
    # Display compression results
    print(f"Compression ratio: {compressed_result.compression_ratio:.2f}x")
    print(f"Compressed size: {compressed_result.total_compressed_bytes / 1024:.1f} KB")
    print(f"Space savings: {(1 - 1/compressed_result.compression_ratio)*100:.1f}%")
    
    # Analyze quality
    quality = vectro.analyze_quality(embeddings, compressed_result)
    print(f"\nQuality Analysis:")
    print(f"  Cosine similarity: {quality.mean_cosine_similarity:.6f}")
    print(f"  Quality grade: {quality.quality_grade()}")
    print(f"  Mean Absolute Error: {quality.mean_absolute_error:.6f}")
    
    # Compare profiles
    print(f"\nProfile Comparison:")
    sample_vectors = embeddings[:100]  # Small sample for demo
    comparison_report = vectro.compare_profiles(sample_vectors)
    print(comparison_report)
    
    print(f"\nAvailable backends: {vectro.backend_info}")
    print(f"Available profiles: {vectro.available_profiles}")
    
    # Demonstrate round-trip compression
    decompressed = vectro.decompress(compressed_result)
    reconstruction_error = np.mean(np.abs(embeddings - decompressed))
    print(f"\nRound-trip reconstruction error: {reconstruction_error:.6f}")
    
    print("\n✅ Vectro API demonstration completed successfully!")