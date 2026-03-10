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

from .integrations import (
    StoredVectorBatch,
    VectorDBConnector,
    InMemoryVectorDBConnector,
    QdrantConnector,
    WeaviateConnector,
    compress_tensor,
    reconstruct_tensor,
    HuggingFaceCompressor,
    result_to_table,
    table_to_result,
    write_parquet,
    read_parquet,
    to_arrow_bytes,
    from_arrow_bytes,
)

from .streaming import StreamingDecompressor
from .quantization_extra import quantize_int2, dequantize_int2, quantize_adaptive
from .migration import inspect_artifact, upgrade_artifact, validate_artifact

__version__ = "2.0.0"
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

    # Integrations
    "StoredVectorBatch",
    "VectorDBConnector",
    "InMemoryVectorDBConnector",
    "QdrantConnector",
    "WeaviateConnector",
    "compress_tensor",
    "reconstruct_tensor",
    "HuggingFaceCompressor",
    # Arrow / Parquet bridge
    "result_to_table",
    "table_to_result",
    "write_parquet",
    "read_parquet",
    "to_arrow_bytes",
    "from_arrow_bytes",
    # Streaming
    "StreamingDecompressor",
    # Advanced quantization
    "quantize_int2",
    "dequantize_int2",
    "quantize_adaptive",
    # Migration tooling
    "inspect_artifact",
    "upgrade_artifact",
    "validate_artifact",

    # Utilities
    "get_version_info"
]
