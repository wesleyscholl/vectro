"""
Unified Mojo API for Vectro quantization library.
Provides high-level interface to all Mojo-accelerated functionality.
"""
from .batch_processor import BatchProcessor, quantize_batch_simple, reconstruct_batch_simple
from .vector_ops import (
    cosine_similarity,
    euclidean_distance,
    manhattan_distance,
    dot_product,
    vector_norm,
    normalize_vector,
    VectorOps
)
from .compression_profiles import (
    CompressionProfile,
    create_fast_profile,
    create_balanced_profile,
    create_quality_profile,
    quantize_with_profile,
    reconstruct_with_profile,
    ProfileManager
)
from .vectro_standalone import quantize_vector, reconstruct_vector


struct VectroAPI:
    """Unified API for all Vectro Mojo functionality."""
    
    @staticmethod
    fn version() -> String:
        """Get Vectro Mojo version."""
        return "0.2.0"
    
    @staticmethod
    fn info() -> None:
        """Print information about available functionality."""
        print("=" * 70)
        print("Vectro Mojo API v0.2.0")
        print("=" * 70)
        print("\nCore Quantization:")
        print("  - quantize_vector()")
        print("  - reconstruct_vector()")
        print("\nBatch Processing:")
        print("  - quantize_batch_simple()")
        print("  - reconstruct_batch_simple()")
        print("  - BatchProcessor struct")
        print("\nVector Operations:")
        print("  - cosine_similarity()")
        print("  - euclidean_distance()")
        print("  - manhattan_distance()")
        print("  - dot_product()")
        print("  - vector_norm()")
        print("  - normalize_vector()")
        print("\nCompression Profiles:")
        print("  - fast (maximum speed)")
        print("  - balanced (speed/quality tradeoff)")
        print("  - quality (maximum accuracy)")
        print("\nPerformance:")
        print("  - SIMD optimized")
        print("  - Parallel processing")
        print("  - ~900K+ vectors/sec quantization")
        print("=" * 70)


fn main():
    """Display API information."""
    VectroAPI.info()
