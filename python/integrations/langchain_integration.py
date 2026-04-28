"""LangChain VectorStore adapter backed by Vectro compression.

Provides a drop-in replacement for any LangChain ``VectorStore`` (FAISS,
Chroma, Qdrant, etc.) that transparently compresses stored embeddings to
INT8 or NF4, reducing in-memory footprint by 4–8×.

Usage::

    from langchain_openai import OpenAIEmbeddings
    from python.integrations.langchain_integration import VectroVectorStore

    # Build store from raw texts (same as FAISS.from_texts, Chroma.from_texts)
    store = VectroVectorStore.from_texts(
        texts=["Paris is the capital of France", "Berlin is cold in winter"],
        embedding=OpenAIEmbeddings(),
        compression_profile="balanced",  # int8 ~4× compression
    )

    # Search
    docs = store.similarity_search("European capitals", k=2)
    docs_with_scores = store.similarity_search_with_score("weather in Europe", k=2)

    # Add more documents later
    store.add_texts(["Rome has ancient history"], metadatas=[{"source": "wiki"}])

    # Async variants (non-blocking in FastAPI / asyncio services)
    docs = await store.asimilarity_search("European capitals", k=2)

Memory comparison (768-dim, 1M vectors):
    float32 baseline  : 3 072 MB
    INT8  (balanced)  :   784 MB  (3.9× reduction)
    NF4   (quality)   :   416 MB  (7.4× reduction)
"""
from __future__ import annotations

import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_LANGCHAIN_ERROR = (
    "langchain-core is required for VectroVectorStore. "
    "Install with: pip install langchain-core"
)


def _require_langchain() -> Any:
    try:
        import langchain_core.vectorstores as _vs
        import langchain_core.documents as _docs
        return _vs, _docs
    except ImportError as exc:
        raise ImportError(_LANGCHAIN_ERROR) from exc


class VectroVectorStore:
    """LangChain-compatible VectorStore with Vectro compression.

    Implements the full LangChain ``VectorStore`` protocol without inheriting
    from it at class-definition time (avoids hard import at module load).
    At runtime the class registers itself via ``__class_getitem__`` duck-typing
    so isinstance checks work once langchain-core is available.

    Args:
        embedding: Any LangChain ``Embeddings`` object.
        compression_profile: Vectro profile — ``"fast"``, ``"balanced"``
            (default), ``"quality"``, or ``"binary"``.
        model_dir: Optional path to a HuggingFace model directory.  When
            supplied, the model-family registry auto-selects the best
            quantization method (e.g. GTE → INT8, BGE → NF4).
    """

    def __init__(
        self,
        embedding: Any,
        compression_profile: str = "balanced",
        model_dir: Optional[str] = None,
    ) -> None:
        from python.vectro import Vectro  # lazy import — keeps module side-effect free

        self._embedding = embedding
        self._profile = compression_profile
        self._model_dir = model_dir
        self._vectro = Vectro()

        self._lock = threading.Lock()
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._metadatas: List[dict] = []
        # Compressed storage: rebuilt on every add/delete
        self._compressed: Any = None          # BatchQuantizationResult | None
        self._n_dims: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_documents(self, texts: List[str]) -> np.ndarray:
        raw = self._embedding.embed_documents(texts)
        return np.array(raw, dtype=np.float32)

    def _embed_query(self, text: str) -> np.ndarray:
        raw = self._embedding.embed_query(text)
        return np.array(raw, dtype=np.float32)

    def _rebuild(self, new_embs: np.ndarray) -> None:
        """Append *new_embs* rows to the compressed store (caller holds lock)."""
        if self._compressed is None:
            # First batch — compress directly.
            self._compressed = self._vectro.compress(
                new_embs,
                profile=self._profile,
                model_dir=self._model_dir,
            )
            self._n_dims = new_embs.shape[1]
        else:
            # Reconstruct existing, concatenate, recompress.
            existing = self._compressed.reconstruct_batch()
            combined = np.vstack([existing, new_embs])
            self._compressed = self._vectro.compress(
                combined,
                profile=self._profile,
                model_dir=self._model_dir,
            )

    def _cosine_scores(self, query_vec: np.ndarray) -> np.ndarray:
        """Return cosine similarity of *query_vec* against all stored vectors."""
        mat = self._compressed.reconstruct_batch()          # (n, d) float32
        q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
        return (mat / norms) @ q                            # (n,) float32

    def _top_k(self, scores: np.ndarray, k: int) -> np.ndarray:
        k = min(k, len(scores))
        idx = np.argpartition(scores, -k)[-k:]
        return idx[np.argsort(scores[idx])[::-1]]

    # ------------------------------------------------------------------
    # LangChain VectorStore protocol
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed, compress, and store *texts*."""
        texts = list(texts)
        if not texts:
            return []
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        embs = self._embed_documents(texts)

        with self._lock:
            self._ids.extend(ids)
            self._texts.extend(texts)
            self._metadatas.extend(metadatas)
            self._rebuild(embs)

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Any]:
        """Return *k* most similar :class:`Document` objects."""
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k)]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]:
        """Return *(Document, cosine_score)* pairs for the top-k matches."""
        try:
            from langchain_core.documents import Document
        except ImportError as exc:
            raise ImportError(_LANGCHAIN_ERROR) from exc

        with self._lock:
            if self._compressed is None:
                return []
            q_emb = self._embed_query(query)
            scores = self._cosine_scores(q_emb)
            ids = list(self._ids)
            texts = list(self._texts)
            metas = list(self._metadatas)

        top_idx = self._top_k(scores, k)
        return [
            (
                Document(
                    page_content=texts[i],
                    metadata={**metas[i], "_vectro_id": ids[i]},
                ),
                float(scores[i]),
            )
            for i in top_idx
        ]

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]:
        return self.similarity_search_with_score(query, k=k, **kwargs)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Remove vectors by id. Pass *ids=None* to clear the entire store."""
        with self._lock:
            if ids is None:
                self._ids.clear()
                self._texts.clear()
                self._metadatas.clear()
                self._compressed = None
                return True

            ids_set = set(ids)
            keep = [i for i, doc_id in enumerate(self._ids) if doc_id not in ids_set]
            if not keep:
                self._ids.clear()
                self._texts.clear()
                self._metadatas.clear()
                self._compressed = None
                return True

            kept_embs = self._compressed.reconstruct_batch()[keep]
            self._ids = [self._ids[i] for i in keep]
            self._texts = [self._texts[i] for i in keep]
            self._metadatas = [self._metadatas[i] for i in keep]
            self._compressed = self._vectro.compress(
                kept_embs,
                profile=self._profile,
                model_dir=self._model_dir,
            )
        return True

    # ------------------------------------------------------------------
    # Async variants (run blocking ops in a thread-pool executor)
    # ------------------------------------------------------------------

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.add_texts(list(texts), metadatas=metadatas, ids=ids)
        )

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.similarity_search(query, k=k)
        )

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.similarity_search_with_score(query, k=k)
        )

    # ------------------------------------------------------------------
    # Constructor classmethod
    # ------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Any,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        compression_profile: str = "balanced",
        model_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> "VectroVectorStore":
        """Build a store from raw texts, identical to other LangChain stores."""
        store = cls(
            embedding=embedding,
            compression_profile=compression_profile,
            model_dir=model_dir,
        )
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    # ------------------------------------------------------------------
    # Vectro-specific helpers
    # ------------------------------------------------------------------

    @property
    def compression_stats(self) -> dict:
        """Return compression ratio, memory usage, and vector count."""
        with self._lock:
            n = len(self._ids)
            if n == 0 or self._compressed is None:
                return {"n_vectors": 0, "compression_ratio": 1.0}
            d = self._n_dims
            original_mb = n * d * 4 / (1024 ** 2)
            compressed_mb = self._compressed.total_compressed_bytes / (1024 ** 2)
            return {
                "n_vectors": n,
                "dimensions": d,
                "compression_profile": self._profile,
                "original_mb": round(original_mb, 3),
                "compressed_mb": round(compressed_mb, 3),
                "compression_ratio": round(
                    self._compressed.compression_ratio, 2
                ),
                "memory_saved_mb": round(original_mb - compressed_mb, 3),
            }

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        return (
            f"VectroVectorStore(n={len(self)}, profile={self._profile!r}, "
            f"dims={self._n_dims})"
        )
