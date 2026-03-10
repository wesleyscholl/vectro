"""Tests for python.migration — inspect, validate, and upgrade artifact versions."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from python.migration import (
    inspect_artifact,
    upgrade_artifact,
    validate_artifact,
    _CURRENT_FORMAT_VERSION,
    _STORAGE_FORMAT_NAME,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_v1_single(tmp_path: Path, n: int = 4, dim: int = 8) -> Path:
    """Create a v1-style single artifact (no storage_format_version field)."""
    path = tmp_path / "v1_single.npz"
    rng = np.random.default_rng(0)
    quantized = rng.integers(-128, 128, size=(n, dim), dtype=np.int8)
    scales = rng.random(n).astype(np.float32)
    np.savez_compressed(
        str(path),
        quantized=quantized,
        scales=scales,
        dims=np.array(dim),
        n=np.array(n),
        # NOTE: NO storage_format_version — this is the v1 signal
    )
    return path


def _make_v1_batch(tmp_path: Path, batch: int = 3, dim: int = 8) -> Path:
    """Create a v1-style batch artifact (has batch_size, no version field)."""
    path = tmp_path / "v1_batch.npz"
    rng = np.random.default_rng(1)
    quantized = rng.integers(-128, 128, size=(batch, dim), dtype=np.int8)
    scales = rng.random(batch).astype(np.float32)
    np.savez_compressed(
        str(path),
        quantized=quantized,
        scales=scales,
        batch_size=np.array(batch),
        vector_dim=np.array(dim),
        compression_ratio=np.array(4.0),
        total_original_bytes=np.array(batch * dim * 4),
        total_compressed_bytes=np.array(batch * dim),
    )
    return path


def _make_v2_single(tmp_path: Path, n: int = 4, dim: int = 8) -> Path:
    """Create a v2-style single artifact (all fields present)."""
    path = tmp_path / "v2_single.npz"
    rng = np.random.default_rng(2)
    quantized = rng.integers(-128, 128, size=(n, dim), dtype=np.int8)
    scales = rng.random(n).astype(np.float32)
    meta = json.dumps({"vectro_version": "1.2.0"})
    np.savez_compressed(
        str(path),
        quantized=quantized,
        scales=scales,
        dims=np.array(dim),
        n=np.array(n),
        artifact_type=np.array("single"),
        precision_mode=np.array("int8"),
        group_size=np.array(0),
        storage_format=np.array(_STORAGE_FORMAT_NAME),
        storage_format_version=np.array(_CURRENT_FORMAT_VERSION),
        metadata_json=np.array(meta),
    )
    return path


# ---------------------------------------------------------------------------
# inspect_artifact tests
# ---------------------------------------------------------------------------


class TestInspectArtifact:
    def test_detect_v1_single(self, tmp_path):
        path = _make_v1_single(tmp_path)
        info = inspect_artifact(path)
        assert info["format_version"] == 1
        assert info["needs_upgrade"] is True
        assert info["artifact_type"] == "single"
        assert info["n_vectors"] == 4
        assert info["vector_dim"] == 8

    def test_detect_v1_batch(self, tmp_path):
        path = _make_v1_batch(tmp_path)
        info = inspect_artifact(path)
        assert info["format_version"] == 1
        assert info["needs_upgrade"] is True
        assert info["artifact_type"] == "batch"
        assert info["n_vectors"] == 3

    def test_detect_v2_current(self, tmp_path):
        path = _make_v2_single(tmp_path)
        info = inspect_artifact(path)
        assert info["format_version"] == _CURRENT_FORMAT_VERSION
        assert info["needs_upgrade"] is False

    def test_file_size_populated(self, tmp_path):
        path = _make_v1_single(tmp_path)
        info = inspect_artifact(path)
        assert info["file_size_bytes"] > 0

    def test_compression_ratio_single(self, tmp_path):
        path = _make_v1_single(tmp_path)
        info = inspect_artifact(path)
        assert info["compression_ratio"] > 0.0

    def test_compression_ratio_batch(self, tmp_path):
        path = _make_v1_batch(tmp_path)
        info = inspect_artifact(path)
        assert info["compression_ratio"] == pytest.approx(4.0)

    def test_fields_list_returned(self, tmp_path):
        path = _make_v1_single(tmp_path)
        info = inspect_artifact(path)
        assert "quantized" in info["fields"]
        assert "scales" in info["fields"]

    def test_precision_mode_default_for_v1(self, tmp_path):
        path = _make_v1_single(tmp_path)
        info = inspect_artifact(path)
        assert info["precision_mode"] == "int8"

    def test_group_size_default_for_v1(self, tmp_path):
        path = _make_v1_single(tmp_path)
        info = inspect_artifact(path)
        assert info["group_size"] == 0

    def test_metadata_parsed_when_present(self, tmp_path):
        path = _make_v2_single(tmp_path)
        info = inspect_artifact(path)
        assert info["metadata"] is not None
        assert info["metadata"].get("vectro_version") == "1.2.0"

    def test_metadata_none_when_absent(self, tmp_path):
        path = _make_v1_single(tmp_path)
        info = inspect_artifact(path)
        assert info["metadata"] is None

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            inspect_artifact(tmp_path / "nonexistent.npz")

    def test_path_string_accepted(self, tmp_path):
        path = _make_v1_single(tmp_path)
        info = inspect_artifact(str(path))
        assert info["format_version"] == 1


# ---------------------------------------------------------------------------
# validate_artifact tests
# ---------------------------------------------------------------------------


class TestValidateArtifact:
    def test_valid_v1_single(self, tmp_path):
        path = _make_v1_single(tmp_path)
        result = validate_artifact(path)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_valid_v2_single(self, tmp_path):
        path = _make_v2_single(tmp_path)
        result = validate_artifact(path)
        assert result["valid"] is True

    def test_valid_v1_batch(self, tmp_path):
        path = _make_v1_batch(tmp_path)
        result = validate_artifact(path)
        assert result["valid"] is True

    def test_missing_quantized_field(self, tmp_path):
        path = tmp_path / "bad.npz"
        np.savez_compressed(str(path), scales=np.ones(4, dtype=np.float32))
        result = validate_artifact(path)
        assert result["valid"] is False
        assert any("quantized" in e for e in result["errors"])

    def test_missing_scales_field(self, tmp_path):
        path = tmp_path / "bad.npz"
        rng = np.random.default_rng(0)
        np.savez_compressed(
            str(path),
            quantized=rng.integers(-128, 128, size=(4, 8), dtype=np.int8),
            n=np.array(4),
            dims=np.array(8),
        )
        result = validate_artifact(path)
        assert result["valid"] is False
        assert any("scales" in e for e in result["errors"])

    def test_nonexistent_file(self, tmp_path):
        result = validate_artifact(tmp_path / "missing.npz")
        assert result["valid"] is False

    def test_shape_mismatch_single(self, tmp_path):
        path = tmp_path / "mismatch.npz"
        rng = np.random.default_rng(0)
        # n=4 but quantized has 5 rows
        np.savez_compressed(
            str(path),
            quantized=rng.integers(-128, 128, size=(5, 8), dtype=np.int8),
            scales=rng.random(4).astype(np.float32),
            n=np.array(4),
            dims=np.array(8),
        )
        result = validate_artifact(path)
        assert result["valid"] is False


# ---------------------------------------------------------------------------
# upgrade_artifact tests
# ---------------------------------------------------------------------------


class TestUpgradeArtifact:
    def test_upgrades_v1_single_to_v2(self, tmp_path):
        src = _make_v1_single(tmp_path)
        dst = tmp_path / "upgraded.npz"
        result = upgrade_artifact(src, dst)
        assert result["upgraded"] is True
        assert result["src_version"] == 1
        assert result["dst_version"] == _CURRENT_FORMAT_VERSION
        info = inspect_artifact(dst)
        assert info["format_version"] == _CURRENT_FORMAT_VERSION
        assert info["needs_upgrade"] is False

    def test_upgrades_v1_batch_to_v2(self, tmp_path):
        src = _make_v1_batch(tmp_path)
        dst = tmp_path / "upgraded_batch.npz"
        upgrade_artifact(src, dst)
        info = inspect_artifact(dst)
        assert info["format_version"] == _CURRENT_FORMAT_VERSION
        assert info["artifact_type"] == "batch"

    def test_upgrade_preserves_quantized_arrays(self, tmp_path):
        src = _make_v1_single(tmp_path)
        orig = np.load(src, allow_pickle=False)
        q_orig = orig["quantized"].copy()
        s_orig = orig["scales"].copy()

        dst = tmp_path / "upgraded.npz"
        upgrade_artifact(src, dst)

        upgraded = np.load(dst, allow_pickle=False)
        np.testing.assert_array_equal(upgraded["quantized"], q_orig)
        np.testing.assert_array_equal(upgraded["scales"], s_orig)

    def test_upgrade_adds_precision_mode(self, tmp_path):
        src = _make_v1_single(tmp_path)
        dst = tmp_path / "upgraded.npz"
        upgrade_artifact(src, dst)
        data = np.load(dst, allow_pickle=False)
        assert "precision_mode" in data.files
        assert str(data["precision_mode"]) == "int8"

    def test_upgrade_adds_group_size(self, tmp_path):
        src = _make_v1_single(tmp_path)
        dst = tmp_path / "upgraded.npz"
        upgrade_artifact(src, dst)
        data = np.load(dst, allow_pickle=False)
        assert "group_size" in data.files
        assert int(data["group_size"]) == 0

    def test_upgrade_adds_metadata_with_migration_record(self, tmp_path):
        src = _make_v1_single(tmp_path)
        dst = tmp_path / "upgraded.npz"
        upgrade_artifact(src, dst)
        data = np.load(dst, allow_pickle=False)
        assert "metadata_json" in data.files
        meta = json.loads(str(data["metadata_json"]))
        assert "migration" in meta
        assert meta["migration"]["migrated_from_version"] == 1
        assert meta["migration"]["migrated_to_version"] == _CURRENT_FORMAT_VERSION

    def test_no_upgrade_needed_for_v2(self, tmp_path):
        src = _make_v2_single(tmp_path)
        dst = tmp_path / "copy.npz"
        result = upgrade_artifact(src, dst)
        assert result["upgraded"] is False
        assert result["src_version"] == _CURRENT_FORMAT_VERSION
        assert dst.exists()

    def test_dry_run_does_not_write_file(self, tmp_path):
        src = _make_v1_single(tmp_path)
        dst = tmp_path / "should_not_exist.npz"
        result = upgrade_artifact(src, dst, dry_run=True)
        assert result["dry_run"] is True
        assert not dst.exists()

    def test_upgrade_artifact_type_single_inferred(self, tmp_path):
        src = _make_v1_single(tmp_path)
        dst = tmp_path / "upgraded.npz"
        upgrade_artifact(src, dst)
        data = np.load(dst, allow_pickle=False)
        assert str(data["artifact_type"]) == "single"

    def test_upgrade_artifact_type_batch_inferred(self, tmp_path):
        src = _make_v1_batch(tmp_path)
        dst = tmp_path / "upgraded.npz"
        upgrade_artifact(src, dst)
        data = np.load(dst, allow_pickle=False)
        assert str(data["artifact_type"]) == "batch"

    def test_upgrade_creates_parent_directory(self, tmp_path):
        src = _make_v1_single(tmp_path)
        dst = tmp_path / "subdir" / "nested" / "upgraded.npz"
        upgrade_artifact(src, dst)
        assert dst.exists()

    def test_upgraded_artifact_passes_validation(self, tmp_path):
        src = _make_v1_single(tmp_path)
        dst = tmp_path / "upgraded.npz"
        upgrade_artifact(src, dst)
        result = validate_artifact(dst)
        assert result["valid"] is True
        assert result["errors"] == []
