"""Weaviate connector for storing and retrieving Vectro-compressed vectors."""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .vector_db import StoredVectorBatch, VectorDBConnector


def _make_data_object(uuid: str, properties: Dict[str, Any]) -> Any:
    """Create a Weaviate DataObject (or stub for testing without weaviate installed)."""
    try:
        wc_data = importlib.import_module("weaviate.classes.data")
        return wc_data.DataObject(uuid=uuid, properties=properties)
    except ImportError:
        from types import SimpleNamespace

        return SimpleNamespace(uuid=uuid, properties=properties)


class WeaviateConnector(VectorDBConnector):
    """Connector that stores compressed vectors as Weaviate object properties.

    Compressed payload (quantized bytes + scales + metadata) is stored in
    Weaviate object properties rather than as native Weaviate vectors, matching
    the Qdrant connector's payload-centric design.

    Requires ``weaviate-client>=4.0``.  Install with::

        pip install "vectro[integrations]"
    """

    def __init__(
        self,
        collection_name: str,
        client: Optional[Any] = None,
        host: str = "localhost",
        port: int = 8080,
    ):
        self.collection_name = collection_name
        self._host = host
        self._port = port

        if client is not None:
            self._client = client
        else:
            try:
                weaviate_mod = importlib.import_module("weaviate")
            except ImportError as exc:
                raise RuntimeError(
                    "weaviate-client>=4.0 is required for WeaviateConnector. "
                    "Install with: pip install 'weaviate-client>=4.0'"
                ) from exc
            self._client = weaviate_mod.connect_to_local(host=host, port=port)

        self._collection = self._client.collections.get(collection_name)

    # ------------------------------------------------------------------
    # VectorDBConnector interface
    # ------------------------------------------------------------------

    def upsert_compressed(
        self,
        ids: Sequence[str],
        quantized: np.ndarray,
        scales: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(ids) != len(quantized) or len(ids) != len(scales):
            raise ValueError("ids, quantized rows, and scales rows must have matching lengths")

        payload_meta = metadata or {}
        objects: List[Any] = []

        for idx, vector_id in enumerate(ids):
            q_row = np.asarray(quantized[idx])
            s_row = np.asarray(scales[idx], dtype=np.float32)

            # Infer precision from dtype: uint8 → int4 (nibble-packed), int8 → standard
            precision_mode = "int4" if q_row.dtype == np.uint8 else "int8"
            vector_dim = int(q_row.shape[0] * (2 if precision_mode == "int4" else 1))

            props: Dict[str, Any] = {
                "vectro_quantized": q_row.tolist(),
                "vectro_scales": s_row.tolist(),
                "vectro_vector_dim": vector_dim,
                "vectro_quantized_dtype": str(q_row.dtype),
                "vectro_precision_mode": precision_mode,
                "vectro_metadata": payload_meta,
            }

            objects.append(_make_data_object(str(vector_id), props))

        self._collection.data.insert_many(objects)

    def fetch_compressed(self, ids: Sequence[str]) -> StoredVectorBatch:
        out_ids: List[str] = []
        quantized_rows: List[np.ndarray] = []
        scales_rows: List[np.ndarray] = []
        vector_dim = 0
        metadata: Dict[str, Any] = {}

        for vector_id in ids:
            obj = self._collection.query.fetch_object_by_id(str(vector_id))
            if obj is None:
                continue

            props = obj.properties
            precision_mode = props.get("vectro_precision_mode", "int8")
            q_arr = np.asarray(
                props.get("vectro_quantized", []),
                dtype=np.uint8 if precision_mode == "int4" else np.int8,
            )
            s_arr = np.asarray(props.get("vectro_scales", []), dtype=np.float32)

            out_ids.append(str(vector_id))
            quantized_rows.append(q_arr)
            scales_rows.append(s_arr)
            vector_dim = int(props.get("vectro_vector_dim", q_arr.shape[0]))
            metadata.update(props.get("vectro_metadata", {}))

        if not out_ids:
            raise KeyError("No vectors found for the requested ids")

        quantized_arr = np.vstack(quantized_rows)
        scales_arr = np.asarray(scales_rows, dtype=np.float32)

        return StoredVectorBatch(
            ids=out_ids,
            quantized=quantized_arr,
            scales=scales_arr,
            vector_dim=vector_dim,
            metadata=metadata,
        )

    def delete(self, ids: Sequence[str]) -> int:
        deleted = 0
        for vector_id in ids:
            self._collection.data.delete_by_id(str(vector_id))
            deleted += 1
        return deleted
