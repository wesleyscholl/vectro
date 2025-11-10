"""
Vectro - Ultra-High-Performance LLM Embedding Compressor

Python API for high-performance vector quantization powered by Mojo.
Provides comprehensive compression, quality analysis, and benchmarking capabilities.
"""

from .vectro import (
    Vectro,
    compress_vectors,
    decompress_vectors,
    analyze_compression_quality,
    generate_compression_report,
    get_version_info
)

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

__all__ = [
    # Main API
    "Vectro",
    "compress_vectors",
    "decompress_vectors", 
    "analyze_compression_quality",
    "generate_compression_report",
    
    # Core interface
    "QuantizationResult",
    "quantize_embeddings",
    "reconstruct_embeddings",
    "mean_cosine_similarity",
    "get_backend_info",
    
    # Batch processing
    "VectroBatchProcessor",
    "BatchQuantizationResult",
    "BatchCompressionAnalyzer",
    "quantize_embeddings_batch",
    "benchmark_batch_compression",
    
    # Quality analysis
    "VectroQualityAnalyzer",
    "QualityMetrics", 
    "QualityBenchmark",
    "QualityReport",
    "evaluate_quantization_quality",
    "generate_quality_report",
    
    # Compression profiles
    "ProfileManager",
    "CompressionProfile",
    "CompressionStrategy", 
    "CompressionOptimizer",
    "ProfileComparison",
    "get_compression_profile",
    "create_custom_profile",
    
    # Utilities
    "get_version_info"
]
