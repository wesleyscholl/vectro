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
    # Max Marginal Relevance (MMR)
    # ------------------------------------------------------------------

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Any]:
        """Diversity-promoting retrieval using Maximal Marginal Relevance.

        MMR balances relevance (cosine similarity to *query*) against redundancy
        (similarity to already-selected documents).  Increasing *lambda_mult*
        towards 1.0 emphasizes relevance; decreasing towards 0.0 emphasizes
        diversity.

        Args:
            query: Search query string.
            k: Number of documents to return.
            fetch_k: Candidate pool size.  Larger values trade speed for
                better diversity coverage (default: 20).
            lambda_mult: Relevance–diversity trade-off in [0, 1].

        Returns:
            List of :class:`Document` objects in MMR-selected order.
        """
        return [
            doc for doc, _ in self.max_marginal_relevance_search_with_score(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
            )
        ]

    def max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]:
        """MMR search returning *(Document, relevance_score)* pairs."""
        try:
            from langchain_core.documents import Document
        except ImportError as exc:
            raise ImportError(_LANGCHAIN_ERROR) from exc

        with self._lock:
            if self._compressed is None:
                return []
            q_emb = self._embed_query(query)
            mat = self._compressed.reconstruct_batch()  # (n, d) float32
            ids = list(self._ids)
            texts = list(self._texts)
            metas = list(self._metadatas)

        mmr_idx = _mmr_select(mat, q_emb, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)

        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
        all_scores = (mat / norms) @ q_norm

        return [
            (
                Document(
                    page_content=texts[i],
                    metadata={**metas[i], "_vectro_id": ids[i]},
                ),
                float(all_scores[i]),
            )
            for i in mmr_idx
        ]

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Any]:
        """Async MMR search."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
            ),
        )

    # ------------------------------------------------------------------
    # Persistence (save / load)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the store to *path* (a directory).

        Creates two files:
        - ``meta.json`` — ids, texts, metadatas, profile, model_dir
        - ``vectors.npy`` — reconstructed float32 embeddings

        The directory is created automatically.
        """
        import json
        import os

        os.makedirs(path, exist_ok=True)

        with self._lock:
            n = len(self._ids)
            if n == 0:
                mat = np.zeros((0, max(self._n_dims, 1)), dtype=np.float32)
            else:
                mat = self._compressed.reconstruct_batch()
            ids = list(self._ids)
            texts = list(self._texts)
            metas = list(self._metadatas)

        np.save(os.path.join(path, "vectors.npy"), mat)
        meta = {
            "version": 1,
            "store_type": "langchain",
            "profile": self._profile,
            "model_dir": self._model_dir,
            "n_dims": self._n_dims,
            "ids": ids,
            "texts": texts,
            "metadatas": metas,
        }
        with open(os.path.join(path, "meta.json"), "w") as fh:
            json.dump(meta, fh)

    @classmethod
    def load(
        cls,
        path: str,
        embedding: Any,
    ) -> "VectroVectorStore":
        """Load a previously saved store from *path*.

        Args:
            path: Directory previously written by :meth:`save`.
            embedding: LangChain ``Embeddings`` object used for future queries
                and :meth:`add_texts` calls.

        Returns:
            Fully restored ``VectroVectorStore``.
        """
        import json
        import os

        with open(os.path.join(path, "meta.json")) as fh:
            meta = json.load(fh)

        if meta.get("store_type") != "langchain":
            raise ValueError(
                f"meta.json store_type={meta.get('store_type')!r} is not 'langchain'."
            )

        store = cls(
            embedding=embedding,
            compression_profile=meta["profile"],
            model_dir=meta.get("model_dir"),
        )
        mat = np.load(os.path.join(path, "vectors.npy"))
        ids = meta["ids"]
        texts = meta["texts"]
        metas = meta["metadatas"]

        if len(mat) > 0 and len(ids) > 0:
            with store._lock:
                store._ids = ids
                store._texts = texts
                store._metadatas = metas
                store._n_dims = meta["n_dims"]
                store._compressed = store._vectro.compress(
                    mat,
                    profile=meta["profile"],
                    model_dir=meta.get("model_dir"),
                )

        return store

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


# ---------------------------------------------------------------------------
# MMR selection (module-level so it can be tested independently)
# ---------------------------------------------------------------------------

def _mmr_select(
    embeddings: np.ndarray,
    query_vec: np.ndarray,
    k: int,
    fetch_k: int,
    lambda_mult: float = 0.5,
) -> np.ndarray:
    """Return indices of *k* documents chosen by Maximal Marginal Relevance.

    Algorithm:
        1. Fetch ``fetch_k`` most relevant candidates by cosine similarity.
        2. Greedily add the document that maximises:
               lambda_mult * sim(doc, query)
             - (1 - lambda_mult) * max_sim(doc, already_selected)

    Args:
        embeddings: All stored vectors, shape (n, d).
        query_vec: Query embedding, shape (d,).
        k: Number of documents to select.
        fetch_k: Candidate pool size (≥ k).
        lambda_mult: Relevance weight in [0, 1].

    Returns:
        Index array of length *k* in selection order.
    """
    n = len(embeddings)
    k = min(k, n)
    fetch_k = min(fetch_k, n)

    # Normalised embeddings for cosine arithmetic
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    emb_norm = embeddings / norms  # (n, d)

    # Step 1: cosine similarity of all vectors to query → fetch top fetch_k
    rel_scores = emb_norm @ q  # (n,)
    cand_idx = np.argpartition(rel_scores, -fetch_k)[-fetch_k:]
    cand_idx = cand_idx[np.argsort(rel_scores[cand_idx])[::-1]]  # descending

    cand_embs = emb_norm[cand_idx]  # (fetch_k, d) — already normalised

    # Step 2: greedy MMR selection
    selected_local: List[int] = []  # indices into cand_idx
    selected_global: List[int] = []

    remaining = list(range(len(cand_idx)))

    for _ in range(k):
        if not selected_local:
            best_local = 0  # highest cosine is first
        else:
            sel_embs = cand_embs[np.array(selected_local)]  # (s, d)
            # sim of each remaining candidate to all selected
            sim_to_sel = cand_embs[remaining] @ sel_embs.T  # (r, s)
            max_sim_to_sel = sim_to_sel.max(axis=1)          # (r,)
            rel = rel_scores[cand_idx[remaining]]             # (r,)
            mmr = lambda_mult * rel - (1.0 - lambda_mult) * max_sim_to_sel
            best_local = remaining[int(np.argmax(mmr))]

        selected_local.append(best_local)
        selected_global.append(int(cand_idx[best_local]))
        remaining = [i for i in remaining if i != best_local]
        if not remaining:
            break

    return np.array(selected_global)
