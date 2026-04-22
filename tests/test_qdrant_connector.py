"""Tests for Qdrant connector behavior using a fake client."""

import unittest
import numpy as np
from types import SimpleNamespace

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python import QdrantConnector  # noqa: E402


class _FakeQdrantClient:
    def __init__(self):
        self.points = {}

    def upsert(self, collection_name, points):
        for p in points:
            self.points[str(p["id"])] = p

    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
        out = []
        for point_id in ids:
            p = self.points.get(str(point_id))
            if p is None:
                continue
            out.append(SimpleNamespace(id=str(point_id), payload=p["payload"]))
        return out

    def delete(self, collection_name, points_selector):
        ids = points_selector.get("points", [])
        for point_id in ids:
            self.points.pop(str(point_id), None)


class TestQdrantConnector(unittest.TestCase):
    def setUp(self):
        self.client = _FakeQdrantClient()
        self.connector = QdrantConnector(collection_name="test", client=self.client)

    def test_upsert_and_fetch_int8(self):
        ids = ["1", "2"]
        q = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
        s = np.array([0.1, 0.2], dtype=np.float32)

        self.connector.upsert_compressed(ids, q, s, metadata={"env": "unit"})
        batch = self.connector.fetch_compressed(["1", "2"])

        self.assertEqual(batch.ids, ["1", "2"])
        self.assertEqual(batch.quantized.dtype, np.int8)
        self.assertEqual(batch.quantized.shape, (2, 3))
        self.assertEqual(batch.scales.shape, (2,))
        self.assertEqual(batch.metadata["env"], "unit")

    def test_upsert_and_fetch_int4_payload_shape(self):
        ids = ["9"]
        packed = np.array([[11, 22, 33, 44]], dtype=np.uint8)
        scales = np.array([[0.5, 0.25]], dtype=np.float32)

        self.connector.upsert_compressed(ids, packed, scales)
        batch = self.connector.fetch_compressed(ids)

        self.assertEqual(batch.quantized.dtype, np.uint8)
        self.assertEqual(batch.quantized.shape, (1, 4))
        self.assertEqual(batch.scales.shape, (1, 2))
        self.assertEqual(batch.vector_dim, 8)

    def test_delete_returns_count(self):
        ids = ["a", "b"]
        q = np.array([[1, 2], [3, 4]], dtype=np.int8)
        s = np.array([0.1, 0.2], dtype=np.float32)
        self.connector.upsert_compressed(ids, q, s)

        deleted = self.connector.delete(["a", "x"])  # x absent
        self.assertEqual(deleted, 2)


if __name__ == "__main__":
    unittest.main()
