"""api/test_viz.py — TestClient coverage for the V7 visualization endpoints.

Six tests, no network: project + cluster shape, content, separation, and
the 404 path. We seed indices fresh inside each test (the in-memory store
is module-global) and use distinct names to avoid cross-test bleed.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.viz import INDICES

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clean_indices():
    INDICES.clear()
    yield
    INDICES.clear()


def _seed(name: str, vectors: list[list[float]]) -> None:
    r = client.post(f"/viz/index/{name}/add", json={"vectors": vectors})
    assert r.status_code == 200, r.text


def test_project_returns_2d_points_with_required_fields():
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((12, 8)).astype(np.float32).tolist()
    _seed("alpha", vecs)

    r = client.post("/viz/index/alpha/project")
    assert r.status_code == 200
    body = r.json()

    assert isinstance(body, list)
    assert len(body) == 12
    for i, pt in enumerate(body):
        assert set(pt.keys()) == {"id", "x", "y", "label"}
        assert pt["id"] == i
        assert isinstance(pt["x"], float)
        assert isinstance(pt["y"], float)
        assert pt["label"] == str(i)


def test_project_separates_two_well_separated_clusters():
    # Two tight blobs at +10 and -10 along axis 0 — PCA's first axis must
    # split them; the projected x-coordinate sign should match cluster.
    rng = np.random.default_rng(1)
    a = rng.standard_normal((6, 4)).astype(np.float32) + np.array([10, 0, 0, 0], dtype=np.float32)
    b = rng.standard_normal((6, 4)).astype(np.float32) + np.array([-10, 0, 0, 0], dtype=np.float32)
    _seed("blobs", np.vstack([a, b]).tolist())

    body = client.post("/viz/index/blobs/project").json()
    xs = [pt["x"] for pt in body]
    a_signs = [np.sign(x) for x in xs[:6]]
    b_signs = [np.sign(x) for x in xs[6:]]
    # All of cluster A on one side, all of cluster B on the other.
    assert len(set(a_signs)) == 1
    assert len(set(b_signs)) == 1
    assert a_signs[0] != b_signs[0]


def test_project_404_for_missing_index():
    r = client.post("/viz/index/does-not-exist/project")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


def test_cluster_default_k3_returns_assignments_for_every_vector():
    rng = np.random.default_rng(2)
    vecs = rng.standard_normal((30, 16)).astype(np.float32).tolist()
    _seed("cl", vecs)

    r = client.post("/viz/index/cl/cluster", json={"k": 3, "max_iter": 20, "seed": 7})
    assert r.status_code == 200
    body = r.json()

    assert len(body) == 30
    clusters = {pt["cluster"] for pt in body}
    # k=3 on 30 random points should produce >=2 non-empty clusters with
    # a stable seed; assert each label is in [0, k).
    assert all(0 <= pt["cluster"] < 3 for pt in body)
    assert len(clusters) >= 2
    assert [pt["id"] for pt in body] == list(range(30))


def test_cluster_custom_k_recovers_three_blobs():
    rng = np.random.default_rng(3)
    centers = np.array([[5, 0, 0], [-5, 0, 0], [0, 5, 0]], dtype=np.float32)
    blobs = np.vstack([
        rng.standard_normal((8, 3)).astype(np.float32) * 0.1 + c
        for c in centers
    ])
    _seed("three", blobs.tolist())

    body = client.post("/viz/index/three/cluster", json={"k": 3, "seed": 0}).json()
    labels = [pt["cluster"] for pt in body]

    # Each blob of 8 points should fall under a single label.
    blob_labels = [set(labels[i:i + 8]) for i in range(0, 24, 8)]
    assert all(len(s) == 1 for s in blob_labels), labels
    # And the three blob-labels must be distinct.
    assert len({next(iter(s)) for s in blob_labels}) == 3


def test_cluster_404_for_missing_index():
    r = client.post("/viz/index/missing/cluster", json={"k": 3})
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]
