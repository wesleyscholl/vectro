"""Integration adapters for external vector and ML ecosystems."""

from .vector_db import (
    StoredVectorBatch,
    VectorDBConnector,
    InMemoryVectorDBConnector,
)
from .qdrant_connector import QdrantConnector
from .weaviate_connector import WeaviateConnector
from .torch_bridge import compress_tensor, reconstruct_tensor, HuggingFaceCompressor
from .arrow_bridge import (
    result_to_table,
    table_to_result,
    write_parquet,
    read_parquet,
    to_arrow_bytes,
    from_arrow_bytes,
)

__all__ = [
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
]
