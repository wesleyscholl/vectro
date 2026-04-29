"""LlamaIndex VectorStore adapter backed by Vectro compression.

Provides a drop-in ``VectorStore`` for LlamaIndex pipelines that transparently
compresses stored node embeddings to INT8 or NF4, reducing memory by 4–8×.

Usage::

    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.schema import TextNode
    from python.integrations.llamaindex_integration import VectroVectorStore

    # Build a store
    vector_store = VectroVectorStore(compression_profile="balanced")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=[], storage_context=storage_context)

    # Insert nodes (LlamaIndex calls vector_store.add() internally)
    node = TextNode(text="Paris is the capital of France", id_="node-1")
    node.embedding = [0.1, 0.2, ...]  # set by LlamaIndex embedding pipeline
    vector_store.add([node])

    # Query
    from llama_index.core.vector_stores.types import VectorStoreQuery
    result = vector_store.query(VectorStoreQuery(query_embedding=[...], similarity_top_k=2))

Memory comparison (768-dim, 1M vectors):
    float32 baseline  : 3 072 MB
    INT8  (balanced)  :   784 MB  (3.9× reduction)
"""
from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List, Optional, Sequence, cast

import numpy as np

_LLAMAINDEX_ERROR = (
    "llama-index-core is required for VectroVectorStore (LlamaIndex). "
    "Install with: pip install llama-index-core"
)


class VectroVectorStore:
    """LlamaIndex-compatible VectorStore with Vectro compression.

    Implements the ``BasePydanticVectorStore`` / ``VectorStore`` duck-typing
    protocol used by LlamaIndex's ``VectorStoreIndex``.  No hard import of
    llama-index at class-definition time; the dependency is checked lazily.

    Args:
        compression_profile: ``"fast"``, ``"balanced"`` (default),
            ``"quality"``, or ``"binary"``.
        model_dir: Optional HuggingFace model directory for family-aware
            profile selection via :func:`python.profiles.get_profile`.
    """

    # LlamaIndex protocol attribute: this store manages its own embeddings
    stores_text: bool = True
    is_embedding_query: bool = True
    flat_metadata: bool = False

    def __init__(
        self,
        compression_profile: str = "balanced",
        model_dir: Optional[str] = None,
    ) -> None:
        from python.vectro import Vectro

        self._profile = compression_profile
        self._model_dir = model_dir
        self._vectro = Vectro()

        self._lock = threading.Lock()
        # node_id → (text, metadata)
        self._node_store: Dict[str, tuple] = {}
        # Ordered list of node_ids matching rows in self._compressed
        self._node_ids: List[str] = []
        self._compressed: Any = None   # BatchQuantizationResult | None
        self._n_dims: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild(self, new_embs: np.ndarray) -> None:
        """Append *new_embs* to the compressed store (caller holds lock)."""
        if self._compressed is None:
            self._compressed = self._vectro.compress(
                new_embs, profile=self._profile, model_dir=self._model_dir
            )
            self._n_dims = new_embs.shape[1]
        else:
            existing = self._compressed.reconstruct_batch()
            combined = np.vstack([existing, new_embs])
            self._compressed = self._vectro.compress(
                combined, profile=self._profile, model_dir=self._model_dir
            )

    def _cosine_scores(self, query_emb: np.ndarray) -> np.ndarray:
        mat = self._compressed.reconstruct_batch()
        q = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
        return (mat / norms) @ q

    # ------------------------------------------------------------------
    # LlamaIndex VectorStore protocol
    # ------------------------------------------------------------------

    def add(
        self,
        nodes: List[Any],
        **add_kwargs: Any,
    ) -> List[str]:
        """Store a list of LlamaIndex ``BaseNode`` objects.

        Each node must have its ``.embedding`` field set (LlamaIndex does this
        automatically during the ingestion pipeline).
        """
        if not nodes:
            return []

        embeddings = []
        ids = []
        for node in nodes:
            emb = getattr(node, "embedding", None)
            if emb is None:
                raise ValueError(
                    f"Node {node.node_id!r} has no embedding. "
                    "Run an embedding pipeline before calling add()."
                )
            embeddings.append(np.asarray(emb, dtype=np.float32))
            node_id = getattr(node, "node_id", None) or str(uuid.uuid4())
            ids.append(node_id)

        emb_matrix = np.stack(embeddings, axis=0)

        with self._lock:
            for node, node_id in zip(nodes, ids):
                text = getattr(node, "text", "")
                meta = dict(getattr(node, "metadata", {}) or {})
                self._node_store[node_id] = (text, meta)
                self._node_ids.append(node_id)
            self._rebuild(emb_matrix)

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Remove all nodes with ``ref_doc_id`` in their metadata."""
        with self._lock:
            keep_idx = [
                i for i, nid in enumerate(self._node_ids)
                if self._node_store.get(nid, ({}, {}))[1].get("doc_id") != ref_doc_id
                and nid != ref_doc_id
            ]
            if len(keep_idx) == len(self._node_ids):
                return  # nothing to delete

            if not keep_idx:
                self._node_ids.clear()
                self._node_store.clear()
                self._compressed = None
                return

            kept_embs = self._compressed.reconstruct_batch()[keep_idx]
            removed_ids = {self._node_ids[i] for i in range(len(self._node_ids))
                           if i not in set(keep_idx)}
            for rid in removed_ids:
                self._node_store.pop(rid, None)
            self._node_ids = [self._node_ids[i] for i in keep_idx]
            self._compressed = self._vectro.compress(
                kept_embs, profile=self._profile, model_dir=self._model_dir
            )

    def query(
        self,
        query: Any,
        **kwargs: Any,
    ) -> Any:
        """Run ANN search against stored compressed vectors.

        Args:
            query: A ``VectorStoreQuery`` object (from llama-index-core).

        Returns:
            A ``VectorStoreQueryResult`` with nodes and similarities.
        """
        try:
            from llama_index.core.vector_stores.types import VectorStoreQueryResult
            from llama_index.core.schema import TextNode, NodeWithScore
        except ImportError as exc:
            raise ImportError(_LLAMAINDEX_ERROR) from exc

        q_emb = getattr(query, "query_embedding", None)
        if q_emb is None:
            raise ValueError("VectorStoreQuery.query_embedding must be set.")
        k = getattr(query, "similarity_top_k", 4)

        q_arr = np.asarray(q_emb, dtype=np.float32)

        with self._lock:
            if self._compressed is None or not self._node_ids:
                return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
            scores = self._cosine_scores(q_arr)
            node_ids = list(self._node_ids)
            node_store = dict(self._node_store)

        k = min(k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        result_nodes = []
        result_sims = []
        result_ids = []

        for i in top_idx:
            nid = node_ids[i]
            text, meta = node_store.get(nid, ("", {}))
            node = TextNode(text=text, id_=nid, metadata=meta)
            result_nodes.append(node)
            result_sims.append(float(scores[i]))
            result_ids.append(nid)

        return VectorStoreQueryResult(
            nodes=result_nodes,
            similarities=result_sims,
            ids=result_ids,
        )

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Any = None,
    ) -> List[Any]:
        """Retrieve stored nodes by id."""
        try:
            from llama_index.core.schema import TextNode
        except ImportError as exc:
            raise ImportError(_LLAMAINDEX_ERROR) from exc

        with self._lock:
            target = node_ids if node_ids is not None else list(self._node_ids)
            nodes = []
            for nid in target:
                if nid in self._node_store:
                    text, meta = self._node_store[nid]
                    nodes.append(TextNode(text=text, id_=nid, metadata=meta))
        return nodes

    # ------------------------------------------------------------------
    # Async variants
    # ------------------------------------------------------------------

    async def async_add(
        self,
        nodes: List[Any],
        **add_kwargs: Any,
    ) -> List[str]:
        """Non-blocking variant of :meth:`add` — delegates to a thread-pool."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.add(nodes))

    async def aquery(
        self,
        query: Any,
        **kwargs: Any,
    ) -> Any:
        """Non-blocking variant of :meth:`query` — delegates to a thread-pool."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.query(query))

    # ------------------------------------------------------------------
    # Persistence (save / load)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the store to *path* (a directory).

        Creates two files:
        - ``meta.json`` — node ids, text, metadata, profile, model_dir
        - ``vectors.npy`` — reconstructed float32 embeddings

        The directory is created automatically.
        """
        import json
        import os

        os.makedirs(path, exist_ok=True)

        with self._lock:
            n = len(self._node_ids)
            if n == 0:
                mat = np.zeros((0, max(self._n_dims, 1)), dtype=np.float32)
            else:
                mat = self._compressed.reconstruct_batch()
            node_ids = list(self._node_ids)
            node_store_serial = {
                nid: {"text": v[0], "meta": v[1]}
                for nid, v in self._node_store.items()
            }

        np.save(os.path.join(path, "vectors.npy"), mat)
        meta = {
            "version": 1,
            "store_type": "llamaindex",
            "profile": self._profile,
            "model_dir": self._model_dir,
            "n_dims": self._n_dims,
            "node_ids": node_ids,
            "node_store": node_store_serial,
        }
        with open(os.path.join(path, "meta.json"), "w") as fh:
            json.dump(meta, fh)

    @classmethod
    def load(cls, path: str) -> "VectroVectorStore":
        """Load a previously saved store from *path*.

        Returns:
            Fully restored ``VectroVectorStore`` (LlamaIndex adapter).
        """
        import json
        import os

        with open(os.path.join(path, "meta.json")) as fh:
            meta = json.load(fh)

        if meta.get("store_type") != "llamaindex":
            raise ValueError(
                f"meta.json store_type={meta.get('store_type')!r} is not 'llamaindex'."
            )

        store = cls(
            compression_profile=meta["profile"],
            model_dir=meta.get("model_dir"),
        )
        mat = np.load(os.path.join(path, "vectors.npy"))
        node_ids = meta["node_ids"]
        node_store_serial = meta["node_store"]

        if len(mat) > 0 and len(node_ids) > 0:
            with store._lock:
                for nid in node_ids:
                    entry = node_store_serial.get(nid, {"text": "", "meta": {}})
                    store._node_store[nid] = (entry["text"], entry["meta"])
                    store._node_ids.append(nid)
                store._n_dims = meta["n_dims"]
                store._compressed = store._vectro.compress(
                    mat, profile=meta["profile"], model_dir=meta.get("model_dir")
                )

        return store

    # ------------------------------------------------------------------
    # Vectro-specific helpers
    # ------------------------------------------------------------------

    @property
    def compression_stats(self) -> dict:
        with self._lock:
            n = len(self._node_ids)
            if n == 0 or self._compressed is None:
                return {"n_nodes": 0, "compression_ratio": 1.0}
            d = self._n_dims
            original_mb = n * d * 4 / (1024 ** 2)
            compressed_mb = self._compressed.total_compressed_bytes / (1024 ** 2)
            return {
                "n_nodes": n,
                "dimensions": d,
                "compression_profile": self._profile,
                "original_mb": round(original_mb, 3),
                "compressed_mb": round(compressed_mb, 3),
                "compression_ratio": round(self._compressed.compression_ratio, 2),
                "memory_saved_mb": round(original_mb - compressed_mb, 3),
            }

    def __len__(self) -> int:
        return len(self._node_ids)

    def __repr__(self) -> str:
        return (
            f"VectroVectorStore[LlamaIndex](n={len(self)}, "
            f"profile={self._profile!r}, dims={self._n_dims})"
        )
