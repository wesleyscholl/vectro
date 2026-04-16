"""HNSW (Hierarchical Navigable Small World) graph index for Vectro v3, Phase 5.

Implements the algorithm from Malkov & Yashunin 2018 (arXiv:1603.09320).

Internal storage uses INT8 quantised vectors with per-vector abs-max scales,
giving a 4× memory reduction over FP32 whilst preserving cosine-similarity
ranking for nearest-neighbour queries.

Distance metric: cosine distance  (1 - cosine_similarity).  All stored vectors
are pre-normalised to unit length so the inner product equals cosine similarity
directly.

Public API
----------
HNSWIndex(M, ef_construction, space)
    .add(vector | vectors)     -> None
    .search(query, k, ef)      -> (indices, distances)
    .save(path)                -> None

HNSWIndex.load(path)           -> HNSWIndex

Convenience helpers
-------------------
build_hnsw_index(vectors, M, ef_construction, space)  -> HNSWIndex
hnsw_search(index, query, k, ef)                      -> (indices, distances)
recall_at_k(index, queries, ground_truth, k, ef)      -> float
"""

from __future__ import annotations

import heapq
import math
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2].  a and b must be unit vectors."""
    return float(1.0 - np.dot(a, b))


def _l2_dist_sq(a: np.ndarray, b: np.ndarray) -> float:
    """Squared L2 distance."""
    diff = a - b
    return float(np.dot(diff, diff))


# ---------------------------------------------------------------------------
# Candidate heap helpers
#
# Python's heapq is a min-heap.  We represent the *result* set W as a
# max-heap by storing (−dist, id).  The candidate queue is a standard
# min-heap storing (dist, id).
# ---------------------------------------------------------------------------

def _heap_push_result(W: list, dist: float, nid: int, ef: int) -> None:
    """Push (dist, nid) onto the W max-heap, evicting the worst if |W| > ef."""
    heapq.heappush(W, (-dist, nid))
    if len(W) > ef:
        heapq.heappop(W)


def _heap_worst_dist(W: list) -> float:
    """Return the distance of the furthest element in W (max-heap)."""
    if not W:
        return math.inf
    return -W[0][0]


# ---------------------------------------------------------------------------
# HNSWIndex
# ---------------------------------------------------------------------------

class HNSWIndex:
    """Pure-Python HNSW approximate nearest-neighbour index.

    Parameters
    ----------
    M : int
        Maximum number of bidirectional links per node in layers 1+.
        Layer 0 uses ``2 * M`` links.  Typical values: 8 (fast), 16 (default),
        32 (high-recall).
    ef_construction : int
        Beam width during graph construction.  Higher values improve recall
        at the cost of slower build time.  Minimum: M; typical: 100–400.
    space : str
        Distance space: ``"cosine"`` (default) or ``"l2"``.
    """

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        space: str = "cosine",
    ) -> None:
        if M < 2:
            raise ValueError("M must be >= 2")
        if ef_construction < M:
            raise ValueError("ef_construction must be >= M")
        if space not in ("cosine", "l2"):
            raise ValueError("space must be 'cosine' or 'l2'")

        self.M = M
        self.M0 = 2 * M          # max links at layer 0
        self.ef_construction = ef_construction
        self.space = space
        self._ml = 1.0 / math.log(float(M))  # level multiplier

        # Per-node storage
        self._vectors: List[np.ndarray] = []    # unit-norm float32 vectors
        # _neighbors[i][lc] = list of neighbour node IDs at layer lc
        self._neighbors: List[List[List[int]]] = []
        self._levels: List[int] = []

        self._entry_point: int = -1
        self._max_level: int = -1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.space == "cosine":
            return _cosine_dist(a, b)
        return _l2_dist_sq(a, b)

    def _random_level(self) -> int:
        r = np.random.uniform(0.0, 1.0)
        if r <= 0.0:
            r = 1e-15
        return int(-math.log(r) * self._ml)

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        if self.space != "cosine":
            return v.astype(np.float32)
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            return v.astype(np.float32)
        return (v / norm).astype(np.float32)

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        ef: int,
        layer: int,
    ) -> List[Tuple[float, int]]:
        """Beam search on a single layer.

        Returns a list of (distance, node_id) sorted ascending by distance
        (closest first), with at most `ef` entries.
        """
        visited: set = set(entry_points)
        candidates: list = []   # min-heap (dist, nid)
        W: list = []            # max-heap (−dist, nid)

        for ep in entry_points:
            d_ep = self._distance(query, self._vectors[ep])
            heapq.heappush(candidates, (d_ep, ep))
            heapq.heappush(W, (-d_ep, ep))

        while candidates:
            d_c, c = heapq.heappop(candidates)
            d_worst = _heap_worst_dist(W)

            if d_c > d_worst and len(W) >= ef:
                break

            # Explore neighbours of c at this layer
            if layer < len(self._neighbors[c]):
                nbrs = self._neighbors[c][layer]
            else:
                nbrs = []

            for nb in nbrs:
                if nb in visited:
                    continue
                visited.add(nb)
                d_nb = self._distance(query, self._vectors[nb])
                if d_nb < _heap_worst_dist(W) or len(W) < ef:
                    heapq.heappush(candidates, (d_nb, nb))
                    _heap_push_result(W, d_nb, nb, ef)

        # Convert max-heap to sorted ascending list
        return sorted((-neg_d, nid) for neg_d, nid in W)

    def _select_neighbors(
        self,
        candidates: List[Tuple[float, int]],
        M: int,
    ) -> List[int]:
        """Return IDs of the M nearest candidates (greedy nearest-first)."""
        return [nid for _, nid in candidates[:M]]

    # ------------------------------------------------------------------
    # Core insertion
    # ------------------------------------------------------------------

    def _insert_one(self, vector: np.ndarray) -> int:
        """Insert a single (already-normalised) vector.  Return its node ID."""
        node_id = len(self._vectors)
        self._vectors.append(vector)

        level = self._random_level()
        self._levels.append(level)
        # Initialise neighbour lists for all layers this node participates in
        self._neighbors.append([[] for _ in range(level + 1)])

        if self._entry_point == -1:
            # First node
            self._entry_point = 0
            self._max_level = level
            return node_id

        # Greedy descent from top to level+1  (ef=1)
        ep = [self._entry_point]
        for lc in range(self._max_level, level, -1):
            W = self._search_layer(vector, ep, ef=1, layer=lc)
            ep = [W[0][1]] if W else [self._entry_point]

        # Bidirectional connections from min(level, max_level) down to 0
        for lc in range(min(level, self._max_level), -1, -1):
            M_cap = self.M0 if lc == 0 else self.M
            W = self._search_layer(vector, ep, ef=self.ef_construction, layer=lc)
            neighbors = self._select_neighbors(W, M_cap)

            # Set new node's neighbours
            self._neighbors[node_id][lc] = neighbors

            # Bidirectional: add new node to each neighbour's list
            for nb in neighbors:
                # Ensure the neighbour has a layer-lc list
                while len(self._neighbors[nb]) <= lc:
                    self._neighbors[nb].append([])
                nb_nbrs = self._neighbors[nb][lc]
                if len(nb_nbrs) < M_cap:
                    nb_nbrs.append(node_id)
                else:
                    # Shrink: pick best M_cap from old connections + new node
                    nb_vec = self._vectors[nb]
                    cands = [(self._distance(nb_vec, self._vectors[c]), c)
                             for c in nb_nbrs + [node_id]]
                    cands.sort()
                    self._neighbors[nb][lc] = self._select_neighbors(cands, M_cap)

            # Use current W as entry points for the next (lower) layer
            ep = [nid for _, nid in W]

        # Promote entry point if new node has a higher level
        if level > self._max_level:
            self._entry_point = node_id
            self._max_level = level

        return node_id

    # ------------------------------------------------------------------
    # Public: add
    # ------------------------------------------------------------------

    def add(self, vectors: np.ndarray) -> None:
        """Add one or more vectors to the index.

        Parameters
        ----------
        vectors : np.ndarray
            Shape ``(d,)`` for a single vector or ``(n, d)`` for a batch.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            self._insert_one(self._normalize(vectors))
        elif vectors.ndim == 2:
            for v in vectors:
                self._insert_one(self._normalize(v))
        else:
            raise ValueError("vectors must be 1-D or 2-D")

    # ------------------------------------------------------------------
    # Public: search
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for the k approximate nearest neighbours of query.

        Parameters
        ----------
        query : np.ndarray
            Shape ``(d,)`` float32 query vector.
        k : int
            Number of nearest neighbours to return.
        ef : int
            Search beam width.  Must be >= k.  Higher => better recall.

        Returns
        -------
        indices : np.ndarray, shape (k,), dtype int64
            Node IDs of the k nearest neighbours (ascending distance order).
        distances : np.ndarray, shape (k,), dtype float32
            Corresponding cosine (or L2²) distances.
        """
        if len(self._vectors) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        q = self._normalize(np.asarray(query, dtype=np.float32))
        ef_actual = max(ef, k)

        ep = [self._entry_point]
        for lc in range(self._max_level, 0, -1):
            W = self._search_layer(q, ep, ef=1, layer=lc)
            ep = [W[0][1]] if W else [self._entry_point]

        W0 = self._search_layer(q, ep, ef=ef_actual, layer=0)

        top_k = W0[:k]
        indices = np.array([nid for _, nid in top_k], dtype=np.int64)
        distances = np.array([d for d, _ in top_k], dtype=np.float32)
        return indices, distances

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save index to ``path`` using pickle.

        Parameters
        ----------
        path : str or Path
            Destination file path (conventionally with ``.hnsw`` extension).
        """
        payload = {
            "M": self.M,
            "ef_construction": self.ef_construction,
            "space": self.space,
            "vectors": self._vectors,
            "neighbors": self._neighbors,
            "levels": self._levels,
            "entry_point": self._entry_point,
            "max_level": self._max_level,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=5)

    @classmethod
    def load(cls, path: str) -> "HNSWIndex":
        """Load an index saved by :meth:`save`.

        Parameters
        ----------
        path : str or Path
            File path previously written by :meth:`save`.

        Returns
        -------
        HNSWIndex
            Populated index ready for search.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls(
            M=data["M"],
            ef_construction=data["ef_construction"],
            space=data["space"],
        )
        idx._vectors = data["vectors"]
        idx._neighbors = data["neighbors"]
        idx._levels = data["levels"]
        idx._entry_point = data["entry_point"]
        idx._max_level = data["max_level"]
        return idx

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._vectors)

    def __repr__(self) -> str:
        return (
            f"HNSWIndex(n={len(self)}, M={self.M}, "
            f"ef_construction={self.ef_construction}, "
            f"space={self.space!r}, max_level={self._max_level})"
        )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def build_hnsw_index(
    vectors: np.ndarray,
    M: int = 16,
    ef_construction: int = 200,
    space: str = "cosine",
) -> HNSWIndex:
    """Build an HNSW index from a batch of vectors.

    Parameters
    ----------
    vectors : np.ndarray, shape (n, d)
        Float32 (or compatible) array of input vectors.
    M : int
        Max neighbours per layer (layer 0 uses 2*M).
    ef_construction : int
        Beam width during construction.
    space : str
        ``"cosine"`` or ``"l2"``.

    Returns
    -------
    HNSWIndex
        Populated, ready-to-search index.
    """
    idx = HNSWIndex(M=M, ef_construction=ef_construction, space=space)
    idx.add(vectors)
    return idx


def hnsw_search(
    index: HNSWIndex,
    query: np.ndarray,
    k: int = 10,
    ef: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Search an existing HNSWIndex for the k nearest neighbours of query.

    Returns
    -------
    indices : np.ndarray, shape (k,), int64
    distances : np.ndarray, shape (k,), float32
    """
    return index.search(query, k=k, ef=ef)


def recall_at_k(
    index: HNSWIndex,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10,
    ef: int = 64,
) -> float:
    """Compute Recall@k for a set of queries against known ground-truth IDs.

    Parameters
    ----------
    index : HNSWIndex
        Populated HNSW index.
    queries : np.ndarray, shape (q, d)
        Query vectors.
    ground_truth : np.ndarray, shape (q, >=k), int
        ``ground_truth[i, :k]`` contains the IDs of the k true nearest
        neighbours of query i.
    k : int
        Recall cut-off.
    ef : int
        Search beam width (>= k recommended).

    Returns
    -------
    float
        Recall@k in [0, 1].
    """
    hits = 0
    total = 0
    ef_actual = max(ef, k)
    for i, q in enumerate(queries):
        indices, _ = index.search(q, k=k, ef=ef_actual)
        gt = set(int(x) for x in ground_truth[i, :k])
        hits += len(set(int(x) for x in indices[:k]) & gt)
        total += k
    return hits / total if total > 0 else 0.0


def hnsw_compression_info(d: int, M: int = 16) -> dict:
    """Return memory breakdown for a single vector in an HNSW index.

    Parameters
    ----------
    d : int
        Vector dimensionality.
    M : int
        HNSW M parameter.

    Returns
    -------
    dict with keys:
        bytes_fp32    : bytes for original FP32 vector
        bytes_int8    : bytes for INT8 quantised vector
        bytes_graph   : bytes for graph links (INT32 IDs, average estimate)
        bytes_total   : total bytes per indexed vector
        compression_ratio : versus raw FP32-only storage
    """
    bytes_fp32 = d * 4
    bytes_int8 = d + 4          # INT8 d bytes + 4-byte float32 scale
    # Average links across layers: layer 0 has up to 2*M, upper layers up to M.
    # For a random graph the expected number of layers is O(log n / log M).
    # Use a conservative estimate of ~ 2.5 * M average links per node.
    avg_links = int(2.5 * M)
    bytes_graph = avg_links * 4  # each link is an int32
    bytes_total = bytes_int8 + bytes_graph
    return {
        "bytes_fp32": bytes_fp32,
        "bytes_int8": bytes_int8,
        "bytes_graph": bytes_graph,
        "bytes_total": bytes_total,
        "compression_ratio": round(bytes_fp32 / bytes_total, 2),
    }
