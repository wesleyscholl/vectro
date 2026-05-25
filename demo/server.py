#!/usr/bin/env python3
"""demo/server.py — live HTTP backend for the Vectro demo.

Runs **real Vectro code** behind the demo page so every number you see
in the browser is measured on this machine, right now.  No simulations,
no precomputed numbers.

    python3 demo/server.py [--port 8765]

Then open the URL it prints (default http://127.0.0.1:8765/).

Endpoints
---------
GET  /                  → serves demo/index.html
POST /api/compress      → real Vectro.compress on a synthetic batch
POST /api/search        → real VectroDSPyRetriever search over a built-in corpus
POST /api/benchmark     → real INT8 encode throughput on this CPU
GET  /api/index-stats   → current corpus / compression stats
GET  /api/health        → liveness probe (used by the page banner)

Standard library only: ``http.server`` + ``socketserver``.  No Flask,
no FastAPI, no extra installs.  Vectro itself is the only third-party
dependency, and it's resolved from the repo root via ``sys.path``.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import socketserver
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# V7 viz — share the numpy PCA + k-means + cosine helpers with the
# FastAPI app in ``api/app.py`` so behaviour is identical regardless of
# which entrypoint the user is running.
from api.store import IndexStore, cosine_topk, kmeans, pca_2d  # noqa: E402

# DSPy isn't required to be installed — the retriever falls back to a
# plain object when it's missing.  Install a tiny stub so the demo
# server doesn't depend on the real SDK.
if "dspy" not in sys.modules:
    _stub = types.ModuleType("dspy")

    class _Prediction:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    _stub.Prediction = _Prediction
    sys.modules["dspy"] = _stub

import python as vectro  # noqa: E402  (sys.path manipulation above)
from python.integrations.dspy_integration import VectroDSPyRetriever  # noqa: E402
from python.hnsw_api import HNSWIndex  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# Demo corpus — small, hand-curated, identical to the static map in
# index.html so the search panel agrees with the in-page visualisation.
# ─────────────────────────────────────────────────────────────────────────

CORPUS: List[Dict[str, Any]] = [
    # Geography
    {"text": "Paris is the capital of France", "tags": ["paris", "france", "capital"]},
    {"text": "Berlin is the capital of Germany", "tags": ["berlin", "germany", "capital"]},
    {"text": "Tokyo is the capital of Japan", "tags": ["tokyo", "japan", "capital"]},
    {"text": "Athens is the capital of Greece", "tags": ["athens", "greece", "capital"]},
    {"text": "Rome is the capital of Italy", "tags": ["rome", "italy", "capital"]},
    {"text": "Madrid is the capital of Spain", "tags": ["madrid", "spain", "capital"]},
    {"text": "London is the capital of the UK", "tags": ["london", "uk", "capital"]},
    # Climate
    {"text": "Berlin gets cold and wet in winter", "tags": ["berlin", "weather", "cold"]},
    {"text": "Tokyo summers are humid and rainy", "tags": ["tokyo", "weather", "summer"]},
    {"text": "Athens has a mild Mediterranean climate", "tags": ["athens", "weather"]},
    {"text": "Cairo is hot and arid year round", "tags": ["cairo", "weather", "hot"]},
    {"text": "Moscow has long snowy winters", "tags": ["moscow", "weather", "snow"]},
    # Food
    {"text": "French cuisine emphasises butter and wine", "tags": ["france", "food"]},
    {"text": "Italian pasta carbonara comes from Rome", "tags": ["rome", "italy", "food"]},
    {"text": "Japanese ramen has many regional styles", "tags": ["japan", "food"]},
    {"text": "Greek salad is a Mediterranean staple", "tags": ["greece", "food"]},
    {"text": "Sushi originated in Japan", "tags": ["japan", "food", "history"]},
    # Transit
    {"text": "Trains in France connect Paris to Marseille at 320 km/h", "tags": ["france", "trains"]},
    {"text": "The Tokyo subway carries 8 million riders per day", "tags": ["tokyo", "transit"]},
    {"text": "Italy's high-speed rail spans Milan to Naples", "tags": ["italy", "trains"]},
    {"text": "Berlin's U-Bahn is one of Europe's oldest", "tags": ["berlin", "transit"]},
    # Landmarks
    {"text": "The Eiffel Tower is in Paris", "tags": ["paris", "landmark"]},
    {"text": "Mount Fuji is a volcano on Honshu", "tags": ["japan", "landmark"]},
    {"text": "Acropolis sits above Athens", "tags": ["athens", "landmark"]},
    {"text": "Colosseum is an amphitheatre in Rome", "tags": ["rome", "landmark"]},
    {"text": "Brandenburg Gate stands in central Berlin", "tags": ["berlin", "landmark"]},
]

EMBED_DIM = 128


# Deterministic embed_fn — concatenates per-token random projections so
# that "France" + "capital" land near "Paris is the capital of France".
# Fully real numpy arithmetic; no models downloaded.
_VOCAB_RNG = np.random.default_rng(seed=0xC0DEC0DE)
_VOCAB_BASIS: Dict[str, np.ndarray] = {}


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "they",
    "this",
    "to",
    "was",
    "were",
    "where",
    "which",
    "with",
    "what",
    "who",
    "how",
    "many",
    "much",
    "do",
    "does",
    "i",
    "you",
    "we",
    "us",
    "our",
    "your",
    "their",
    "would",
    "should",
    "could",
    "can",
    "will",
    "may",
    "one",
}


def _token_vec(tok: str) -> np.ndarray:
    if tok not in _VOCAB_BASIS:
        # Stable per-token random projection — same token always lands in
        # the same direction.  L2-normalised so dot products correspond
        # to cosine similarity.
        rng = np.random.default_rng(abs(hash(("vectro-demo-v1", tok))) & 0xFFFFFFFF)
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        _VOCAB_BASIS[tok] = v
    return _VOCAB_BASIS[tok]


def _content_tokens(text: str) -> List[str]:
    raw = [t for t in text.lower().replace("'", " ").replace("-", " ").split() if t]
    return [t for t in raw if t not in _STOPWORDS]


def _embed_text(text: str) -> np.ndarray:
    """Deterministic 'bag of content tokens' embedder.

    Stopwords are dropped so two-token queries like "trains europe" don't
    dilute against a 9-token sentence.  Each surviving token contributes
    its random-projection direction.  Because random projections of
    unrelated tokens are near-orthogonal in 128-D, the dot product
    between two embeddings approximates token-overlap divided by the
    geometric mean of token counts — the same intuition as TF-IDF
    cosine, just simpler and dependency-free.
    """
    toks = _content_tokens(text)
    if not toks:
        v = np.zeros(EMBED_DIM, dtype=np.float32)
        v[0] = 1.0
        return v
    acc = np.zeros(EMBED_DIM, dtype=np.float32)
    for t in toks:
        acc += _token_vec(t)
    n = float(np.linalg.norm(acc))
    if n > 0:
        acc = acc / n
    return acc


def embed_fn(x):
    if isinstance(x, str):
        return _embed_text(x)
    return np.stack([_embed_text(t) for t in x], axis=0)


# ─────────────────────────────────────────────────────────────────────────
# Build the live retriever once at startup
# ─────────────────────────────────────────────────────────────────────────

print("[vectro-demo] building live retriever ...", flush=True)
_t0 = time.perf_counter()
_corpus_texts = [item["text"] for item in CORPUS]
_corpus_embs = np.stack([_embed_text(t) for t in _corpus_texts], axis=0)

RETRIEVER = VectroDSPyRetriever(
    embed_fn=embed_fn,
    k=5,
    compression_profile="balanced",
)
RETRIEVER.add_texts(_corpus_texts, embeddings=_corpus_embs, metadatas=[{"i": i, **item} for i, item in enumerate(CORPUS)])
_t1 = time.perf_counter()
print(f"[vectro-demo] retriever ready: {len(CORPUS)} passages, dim={EMBED_DIM}, profile=balanced, {(_t1 - _t0) * 1000:.1f} ms", flush=True)


# ─────────────────────────────────────────────────────────────────────────
# Endpoint implementations — all hit real Vectro code paths
# ─────────────────────────────────────────────────────────────────────────

VECTRO = vectro.Vectro()


def _api_compress(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Allocate a fresh batch of synthetic vectors, compress with the
    requested mode, return *measured* memory + timing.
    """
    n_vecs = max(1, int(payload.get("n_vecs", 10_000)))
    dim = max(1, int(payload.get("dim", 128)))
    mode = str(payload.get("mode", "balanced"))

    # Cap to keep request latency reasonable on tiny machines.
    n_vecs = min(n_vecs, 200_000)
    dim = min(dim, 4096)

    profile_map = {
        "int8": "balanced",
        "fast": "fast",
        "balanced": "balanced",
        "quality": "quality",
        "nf4": "quality",  # NF4 ships under the "quality" profile
        "binary": "binary",
    }
    profile = profile_map.get(mode, "balanced")

    rng = np.random.default_rng(seed=42)
    data = rng.standard_normal((n_vecs, dim)).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    data = (data / norms).astype(np.float32)

    original_bytes = int(data.nbytes)

    t0 = time.perf_counter()
    result = VECTRO.compress(data, profile=profile)
    elapsed = time.perf_counter() - t0

    compressed_bytes = int(result.total_compressed_bytes)
    ratio = float(result.compression_ratio)
    throughput = (n_vecs / elapsed) if elapsed > 0 else float("inf")

    return {
        "n_vecs": n_vecs,
        "dim": dim,
        "mode": mode,
        "profile": profile,
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "original_mb": round(original_bytes / (1024**2), 4),
        "compressed_mb": round(compressed_bytes / (1024**2), 4),
        "ratio": round(ratio, 3),
        "saved_bytes": original_bytes - compressed_bytes,
        "timing_ms": round(elapsed * 1000, 3),
        "throughput_vec_per_s": round(throughput, 1),
        "throughput_M_vec_s": round(throughput / 1_000_000, 3),
    }


def _api_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Real VectroDSPyRetriever.forward against the built-in corpus."""
    query = str(payload.get("query_text", "")).strip() or "capital of France"
    k = max(1, min(int(payload.get("k", 5)), len(CORPUS)))

    t0 = time.perf_counter()
    out = RETRIEVER.forward(query, k=k)
    elapsed = time.perf_counter() - t0

    passages = list(getattr(out, "passages", []) or [])
    scores = list(getattr(out, "scores", []) or [])
    indices = list(getattr(out, "indices", []) or [])

    return {
        "query": query,
        "k": k,
        "passages": passages,
        "scores": [round(float(s), 4) for s in scores],
        "indices": [int(i) for i in indices],
        "tags": [list(CORPUS[i]["tags"]) for i in indices] if indices else [],
        "timing_ms": round(elapsed * 1000, 3),
    }


def _api_benchmark(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Measure real INT8 encode throughput at the requested shape."""
    n_vecs = max(1000, min(int(payload.get("n_vecs", 50_000)), 200_000))
    dim = max(64, min(int(payload.get("dim", 768)), 4096))
    reps = max(1, min(int(payload.get("reps", 3)), 8))

    rng = np.random.default_rng(seed=0xBEEF)
    data = rng.standard_normal((n_vecs, dim)).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    data = (data / norms).astype(np.float32)

    # Warmup
    _ = VECTRO.compress(data[:1000], profile="balanced")

    samples_ms: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        result = VECTRO.compress(data, profile="balanced")
        elapsed = time.perf_counter() - t0
        samples_ms.append(elapsed * 1000)

    best_ms = min(samples_ms)
    best_throughput = n_vecs / (best_ms / 1000.0)
    median_ms = sorted(samples_ms)[len(samples_ms) // 2]
    median_throughput = n_vecs / (median_ms / 1000.0)

    return {
        "n_vecs": n_vecs,
        "dim": dim,
        "reps": reps,
        "samples_ms": [round(s, 3) for s in samples_ms],
        "best_ms": round(best_ms, 3),
        "median_ms": round(median_ms, 3),
        "best_throughput_vec_s": round(best_throughput, 1),
        "median_throughput_vec_s": round(median_throughput, 1),
        "best_M_vec_s": round(best_throughput / 1_000_000, 3),
        "median_M_vec_s": round(median_throughput / 1_000_000, 3),
        "compressed_mb": round(result.total_compressed_bytes / (1024**2), 4),
        "ratio": round(float(result.compression_ratio), 3),
        "platform": f"{platform.system()} / {platform.machine()}",
        "python": platform.python_version(),
    }


def _api_index_stats(_payload: Dict[str, Any]) -> Dict[str, Any]:
    s = RETRIEVER.compression_stats
    return {
        "version": vectro.__version__,
        "platform": f"{platform.system()} / {platform.machine()}",
        "python": platform.python_version(),
        "n_passages": int(s.get("n_passages", 0)),
        "dimensions": int(s.get("dimensions", EMBED_DIM)),
        "profile": str(s.get("compression_profile", "balanced")),
        "original_mb": float(s.get("original_mb", 0.0)),
        "compressed_mb": float(s.get("compressed_mb", 0.0)),
        "memory_saved_mb": float(s.get("memory_saved_mb", 0.0)),
        "compression_ratio": float(s.get("compression_ratio", 1.0)),
    }


def _api_health(_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": True,
        "version": vectro.__version__,
        "platform": f"{platform.system()} / {platform.machine()}",
        "python": platform.python_version(),
        "uptime_s": round(time.monotonic(), 2),
    }


# ─────────────────────────────────────────────────────────────────────────
# v5.1.0 — P1 endpoints: recall estimator, compaction, metadata filter
# ─────────────────────────────────────────────────────────────────────────
#
# A small in-process HNSWIndex is seeded once at startup from the CORPUS
# embeddings so all three endpoints have a realistic index to operate on.
# The index is intentionally separate from RETRIEVER so the user can
# delete, compact, and re-estimate without affecting the search demo.

print("[vectro-demo] building demo HNSW index for P1 endpoints ...", flush=True)
_HNSW_LOCK = threading.Lock()
_DEMO_HNSW: HNSWIndex = HNSWIndex(M=8, ef_construction=80)
_t_hnsw0 = time.perf_counter()
_demo_vecs = np.stack([_embed_text(item["text"]) for item in CORPUS], axis=0)
_demo_meta = [{"category": item["tags"][0] if item["tags"] else "other", "tags": item["tags"], "text": item["text"]} for item in CORPUS]
_DEMO_HNSW.add(_demo_vecs, metadata=_demo_meta)
print(f"[vectro-demo] HNSW ready: {len(_DEMO_HNSW)} vectors, {(time.perf_counter() - _t_hnsw0) * 1000:.1f} ms", flush=True)


def _api_recall_estimate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate Recall@k against brute-force ground truth.

    Body: ``{sample_size?: int, k?: int, ef?: int}``
    Returns the full result dict from ``HNSWIndex.estimate_recall()``,
    plus a plain-English ``label`` for the demo gauge.
    """
    sample_size = max(1, min(int(payload.get("sample_size", 50)), len(_DEMO_HNSW)))
    k = max(1, min(int(payload.get("k", 5)), len(_DEMO_HNSW) - 1))
    ef = max(k, int(payload.get("ef", 32)))
    with _HNSW_LOCK:
        result = _DEMO_HNSW.estimate_recall(sample_size=sample_size, k=k, ef=ef)
    r = result["recall"]
    label = "Excellent" if r >= 0.95 else "Good" if r >= 0.85 else "Fair" if r >= 0.70 else "Poor"
    return {**result, "label": label}


def _api_compact(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run graph compaction — tombstone removal + orphan reconnection.

    Optionally delete a random subset first (for demo effect).
    Body: ``{delete_n?: int}`` — soft-delete up to *delete_n* random vectors
    before compacting (default 3).
    """
    delete_n = max(0, min(int(payload.get("delete_n", 3)), len(_DEMO_HNSW) - 2))
    with _HNSW_LOCK:
        # Soft-delete a few random live vectors
        alive = [i for i in range(len(_DEMO_HNSW._vectors)) if i not in _DEMO_HNSW._deleted]
        rng = np.random.default_rng()
        chosen = rng.choice(alive, size=min(delete_n, len(alive)), replace=False)
        for nid in chosen:
            _DEMO_HNSW.delete(int(nid))
        stats_before = _DEMO_HNSW.stats()
        t0 = time.perf_counter()
        result = _DEMO_HNSW.compact()
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        stats_after = _DEMO_HNSW.stats()
    return {
        "deleted_count_before": stats_before["n_deleted"],
        "orphan_count_before": stats_before["orphan_count"],
        "removed": result["removed"],
        "repaired": result["repaired"],
        "orphan_count_after": stats_after["orphan_count"],
        "timing_ms": elapsed_ms,
        "n_alive": stats_after["n_alive"],
    }


def _api_hnsw_stats(_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return health stats for the demo HNSW index."""
    with _HNSW_LOCK:
        s = _DEMO_HNSW.stats()
    return s


def _api_filtered_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    """HNSW pre-filtered nearest-neighbour search.

    Body: ``{query_text: str, k?: int, filter?: {field: value}}``
    Runs search on the demo HNSW index with optional metadata pre-filter.
    """
    query = str(payload.get("query_text", "")).strip() or "capital"
    k = max(1, min(int(payload.get("k", 5)), len(_DEMO_HNSW)))
    filt = payload.get("filter") or None

    # Validate filter dict
    if filt is not None:
        if not isinstance(filt, dict):
            filt = None
        else:
            # Only allow simple string/number equality filters
            filt = {str(fk): fv for fk, fv in filt.items() if isinstance(fv, (str, int, float, bool))}

    q_vec = _embed_text(query)
    t0 = time.perf_counter()
    with _HNSW_LOCK:
        indices, distances = _DEMO_HNSW.search(q_vec, k=k, ef=max(k, 32), filter=filt)
    elapsed = time.perf_counter() - t0

    results = []
    for nid, dist in zip(indices.tolist(), distances.tolist()):
        meta = _DEMO_HNSW._metadata[nid] if nid < len(_DEMO_HNSW._metadata) else {}
        results.append(
            {
                "id": int(nid),
                "distance": round(float(dist), 4),
                "similarity": round(max(0.0, 1.0 - float(dist)), 4),
                "text": (meta or {}).get("text", ""),
                "category": (meta or {}).get("category", ""),
                "tags": (meta or {}).get("tags", []),
            }
        )

    return {
        "query": query,
        "filter": filt,
        "k": k,
        "results": results,
        "timing_ms": round(elapsed * 1000, 3),
    }


ROUTES_POST: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "/api/compress": _api_compress,
    "/api/search": _api_search,
    "/api/benchmark": _api_benchmark,
    "/api/compact": _api_compact,
    "/api/filtered-search": _api_filtered_search,
    "/api/recall_estimate": _api_recall_estimate,
}
ROUTES_GET: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "/api/index-stats": _api_index_stats,
    "/api/hnsw-stats": _api_hnsw_stats,
    "/api/recall_estimate": lambda _: _api_recall_estimate({}),
    "/api/health": _api_health,
}


# ─────────────────────────────────────────────────────────────────────────
# V7 viz — live in-memory index for demo/viz.html
# ─────────────────────────────────────────────────────────────────────────
#
# The viz page POSTs to /index/{name}/{add,search,project,cluster}.  We
# keep its store separate from the main demo retriever (CORPUS) so the
# user can build, reset, and re-cluster vectors without touching the
# search demo above.
#
# Identical math to api/app.py — both share api.store.

VIZ_STORE = IndexStore()
_VIZ_PATH_RE = re.compile(r"^/index/(?P<name>[A-Za-z0-9_.\-]{1,64})(?:/(?P<action>[a-z]+))?$")


def _viz_summary(name: str) -> Dict[str, Any]:
    idx = VIZ_STORE.get(name)
    return {"name": idx.name, "dim": idx.dim, "n": len(idx)}


def _viz_create(name: str, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    dim = int(body.get("dim", 0))
    if dim < 1:
        return 400, {"error": "dim must be a positive integer"}
    try:
        VIZ_STORE.create(name, dim)
    except ValueError as exc:
        return 409, {"error": str(exc)}
    return 200, _viz_summary(name)


def _viz_add(name: str, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    vectors = body.get("vectors")
    if not vectors:
        return 400, {"error": "vectors must be non-empty"}
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        return 400, {"error": "vectors must be a 2-D list"}
    n_new, dim_new = arr.shape
    try:
        idx = VIZ_STORE.get(name)
    except KeyError:
        idx = VIZ_STORE.create(name, dim_new)
    if dim_new != idx.dim:
        return 400, {"error": f"dim mismatch: payload {dim_new} vs index {idx.dim}"}
    ids = body.get("ids") or []
    metadata = body.get("metadata") or []
    base = len(idx)
    for i in range(n_new):
        idx.vectors.append(arr[i].copy())
        idx.ids.append(str(ids[i]) if i < len(ids) else f"{name}:{base + i}")
        idx.metadata.append(dict(metadata[i]) if i < len(metadata) else {})
    return 200, {"name": name, "added": n_new, "total": len(idx)}


def _viz_search(name: str, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    try:
        idx = VIZ_STORE.get(name)
    except KeyError:
        return 404, {"error": f"no such index: {name}"}
    k = max(1, int(body.get("k", 5)))
    if len(idx) == 0:
        return 200, {"name": name, "k": k, "results": []}
    try:
        top = cosine_topk(idx.matrix(), np.asarray(body.get("query", []), dtype=np.float32), k)
    except ValueError as exc:
        return 400, {"error": str(exc)}
    return 200, {
        "name": name,
        "k": k,
        "results": [{"id": idx.ids[h["index"]], "score": h["score"], "index": h["index"]} for h in top],
    }


def _viz_project(name: str, _body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    try:
        idx = VIZ_STORE.get(name)
    except KeyError:
        return 404, {"error": f"no such index: {name}"}
    coords = pca_2d(idx.matrix()).tolist()
    return 200, {"name": name, "n": len(idx), "coords": coords, "ids": list(idx.ids)}


def _viz_cluster(name: str, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    try:
        idx = VIZ_STORE.get(name)
    except KeyError:
        return 404, {"error": f"no such index: {name}"}
    k = int(body.get("k", 3))
    seed = int(body.get("seed", 0))
    labels = kmeans(idx.matrix(), k, seed=seed).tolist()
    clamped = int(min(max(k, 1), max(len(idx), 1)))
    return 200, {"name": name, "k": clamped, "labels": labels, "ids": list(idx.ids)}


def _viz_delete(name: str) -> Tuple[int, Dict[str, Any]]:
    return 200, {"deleted": VIZ_STORE.delete(name), "name": name}


_VIZ_POST: Dict[str, Callable[[str, Dict[str, Any]], Tuple[int, Dict[str, Any]]]] = {
    "": _viz_create,  # POST /index/{name}
    "add": _viz_add,
    "search": _viz_search,
    "project": _viz_project,
    "cluster": _viz_cluster,
}


def _viz_dispatch(method: str, path: str, body: Dict[str, Any]) -> Optional[Tuple[int, Dict[str, Any]]]:
    """Handle /index/{name}/... routes.  Returns ``None`` if the path
    does not match — caller falls through to the legacy demo routes."""
    m = _VIZ_PATH_RE.match(path)
    if not m:
        return None
    name = m.group("name")
    action = m.group("action") or ""

    if method == "GET" and action == "":
        try:
            return 200, _viz_summary(name)
        except KeyError:
            return 404, {"error": f"no such index: {name}"}
    if method == "DELETE" and action == "":
        return _viz_delete(name)
    if method == "POST":
        handler = _VIZ_POST.get(action)
        if handler is not None:
            return handler(name, body)
        return 404, {"error": f"unknown action: {action!r}"}
    return 405, {"error": f"method not allowed: {method}"}


# ─────────────────────────────────────────────────────────────────────────
# HTTP handler
# ─────────────────────────────────────────────────────────────────────────

_DEMO_DIR = Path(__file__).resolve().parent
INDEX_HTML = (_DEMO_DIR / "index.html").read_bytes()


def _viz_html_bytes() -> bytes:
    """Read viz.html on every request — keeps dev iteration tight."""
    return (_DEMO_DIR / "viz.html").read_bytes()


class Handler(BaseHTTPRequestHandler):
    server_version = f"VectroDemo/{vectro.__version__}"

    # quiet down the default access log a bit
    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"  {self.address_string()}  {fmt % args}\n")

    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Don't cache so the live page picks up edits during dev.
        self.send_header("Cache-Control", "no-store")
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def _read_body(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            n = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(n) if n > 0 else b""
            return (json.loads(raw.decode("utf-8")) if raw else {}), None
        except Exception as exc:
            return None, f"bad json: {exc}"

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            self._send_html(INDEX_HTML)
            return
        if path == "/viz" or path == "/viz.html":
            try:
                self._send_html(_viz_html_bytes())
            except FileNotFoundError:
                self._send_json({"error": "viz.html not found"}, status=404)
            return
        if path in ROUTES_GET:
            try:
                self._send_json(ROUTES_GET[path]({}))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        viz = _viz_dispatch("GET", path, {})
        if viz is not None:
            status, payload = viz
            self._send_json(payload, status=status)
            return
        self._send_json({"error": f"not found: {path}"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        body, err = self._read_body()
        if err is not None:
            self._send_json({"error": err}, status=400)
            return

        handler = ROUTES_POST.get(path)
        if handler is not None:
            try:
                self._send_json(handler(body or {}))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return

        viz = _viz_dispatch("POST", path, body or {})
        if viz is not None:
            status, payload = viz
            self._send_json(payload, status=status)
            return

        self._send_json({"error": f"not found: {path}"}, status=404)

    def do_DELETE(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        viz = _viz_dispatch("DELETE", path, {})
        if viz is not None:
            status, payload = viz
            self._send_json(payload, status=status)
            return
        self._send_json({"error": f"not found: {path}"}, status=404)


class ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    httpd = ThreadingServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print()
    print("  ╔════════════════════════════════════════════════════╗")
    print(f"  ║  Vectro live demo — v{vectro.__version__:<8}                      ║")
    print("  ║                                                    ║")
    print(f"  ║  open: {url:<43}║")
    print(f"  ║  viz:  {url + 'viz':<43}║")
    print(f"  ║  api:  {url + 'api/health':<43}║")
    print("  ║                                                    ║")
    print("  ║  every number on the page is measured here, now    ║")
    print("  ╚════════════════════════════════════════════════════╝")
    print("  Ctrl-C to stop", flush=True)
    print()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[vectro-demo] shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
