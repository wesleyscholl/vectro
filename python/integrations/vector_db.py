"""Vector database adapter contracts for Vectro integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class StoredVectorBatch:
    """Container for compressed vectors and adapter-level metadata."""

    ids: List[str]
    quantized: np.ndarray
    scales: np.ndarray
    vector_dim: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorDBConnector(ABC):
    """Base contract for vector database integrations."""

    @abstractmethod
    def upsert_compressed(
        self,
        ids: Sequence[str],
        quantized: np.ndarray,
        scales: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store already-compressed vectors in the backing system."""

    @abstractmethod
    def fetch_compressed(self, ids: Sequence[str]) -> StoredVectorBatch:
        """Fetch compressed vectors by IDs."""

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> int:
        """Delete vectors by IDs and return number deleted."""


class InMemoryVectorDBConnector(VectorDBConnector):
    """Reference implementation used for local testing and adapter validation."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def upsert_compressed(
        self,
        ids: Sequence[str],
        quantized: np.ndarray,
        scales: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(ids) != len(quantized) or len(ids) != len(scales):
            raise ValueError("ids, quantized rows, and scales rows must have matching lengths")

        meta = metadata or {}
        vector_dim = int(quantized.shape[1])
        for idx, vector_id in enumerate(ids):
            self._store[vector_id] = {
                "quantized": np.asarray(quantized[idx], dtype=np.int8),
                "scales": np.asarray(scales[idx], dtype=np.float32),
                "vector_dim": vector_dim,
                "metadata": meta,
            }

    def fetch_compressed(self, ids: Sequence[str]) -> StoredVectorBatch:
        rows = []
        scales = []
        out_ids = []
        combined_metadata: Dict[str, Any] = {}

        for vector_id in ids:
            if vector_id not in self._store:
                continue
            row = self._store[vector_id]
            rows.append(row["quantized"])
            scales.append(row["scales"])
            out_ids.append(vector_id)
            combined_metadata.update(row.get("metadata", {}))

        if not rows:
            raise KeyError("No vectors found for the requested ids")

        quantized_arr = np.vstack(rows)
        scales_arr = np.asarray(scales, dtype=np.float32)
        return StoredVectorBatch(
            ids=out_ids,
            quantized=quantized_arr,
            scales=scales_arr,
            vector_dim=int(quantized_arr.shape[1]),
            metadata=combined_metadata,
        )

    def delete(self, ids: Sequence[str]) -> int:
        deleted = 0
        for vector_id in ids:
            if vector_id in self._store:
                del self._store[vector_id]
                deleted += 1
        return deleted
