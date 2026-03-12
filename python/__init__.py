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
    MilvusConnector,
    ChromaConnector,
    PineconeConnector,
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

from .streaming import StreamingDecompressor, AsyncStreamingDecompressor
from .quantization_extra import quantize_int2, dequantize_int2, quantize_adaptive
from .migration import inspect_artifact, upgrade_artifact, validate_artifact

# v3 unified API (Phase 9)
from .v3_api import (
    PQCodebook,
    HNSWIndex,
    V3Result,
    VectroV3,
)
from .nf4_api import (
    quantize_nf4,
    dequantize_nf4,
    quantize_mixed,
    dequantize_mixed,
    select_outlier_dims,
)
from .binary_api import quantize_binary, dequantize_binary
from .rq_api import ResidualQuantizer
from .codebook_api import Codebook
from .auto_quantize_api import auto_quantize
from .storage_v3 import save_vqz, load_vqz, save_compressed, load_compressed, VQZResult
from .onnx_export import to_onnx_model, export_onnx

__version__ = "3.5.0"
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
    "MilvusConnector",
    "ChromaConnector",
    "PineconeConnector",
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
    "AsyncStreamingDecompressor",
    # Advanced quantization
    "quantize_int2",
    "dequantize_int2",
    "quantize_adaptive",
    # Migration tooling
    "inspect_artifact",
    "upgrade_artifact",
    "validate_artifact",

    # v3 unified API
    "PQCodebook",
    "HNSWIndex",
    "V3Result",
    "VectroV3",
    # v3 quantization
    "quantize_nf4",
    "dequantize_nf4",
    "quantize_mixed",
    "dequantize_mixed",
    "select_outlier_dims",
    "quantize_binary",
    "dequantize_binary",
    "ResidualQuantizer",
    "Codebook",
    "auto_quantize",
    # v3 storage
    "save_vqz",
    "load_vqz",
    "save_compressed",
    "load_compressed",
    "VQZResult",
    # ONNX export
    "to_onnx_model",
    "export_onnx",

    # Utilities
    "get_version_info"
]
