# Migration Guide: v1 → v2

Vectro 2.0 introduces a new versioned storage format (`vectro_npz` v2) that adds
precision-mode tracking, group-size metadata, and provenance records to every
artifact. All new artifacts written by `save_compressed` are v2.

**v1 artifacts are still readable** via `load_compressed` — they are handled
transparently. However, you should upgrade them to v2 to benefit from
migration tracking and to keep your tooling consistent.

---

## What Changed in v2

| Field | v1 | v2 |
|-------|----|----|
| `storage_format_version` | absent (treated as `1`) | `2` |
| `storage_format` | absent | `"vectro_npz"` |
| `artifact_type` | absent | `"single"` or `"batch"` |
| `precision_mode` | absent | `"int8"` / `"int4"` etc. |
| `group_size` | absent | integer (`0` = no grouping) |
| `metadata_json` | absent | JSON provenance blob |

The `quantized` and `scales` arrays are **byte-for-byte identical** in v2 —
no data is recomputed during upgrade.

---

## Detecting the Format Version

```python
from python.migration import inspect_artifact

info = inspect_artifact("embeddings.npz")
print(info["format_version"])   # 1 or 2
print(info["needs_upgrade"])    # True if version < 2
print(info["precision_mode"])   # "int8" (default for v1 files)
```

---

## Upgrading an Artifact

```python
from python.migration import upgrade_artifact

upgrade_artifact("old.npz", "old_v2.npz")
```

Or use the CLI:

```bash
python -m python.migration upgrade old.npz old_v2.npz
```

The upgrade tool writes a `migration` record inside `metadata_json`:

```json
{
  "migration": {
    "migrated_from_version": 1,
    "migrated_to_version": 2,
    "migrated_at_utc": "2025-01-15T12:00:00+00:00",
    "src_fields": ["quantized", "scales", "dims", "n"]
  }
}
```

### Dry-run mode

Preview what would happen without writing any files:

```bash
python -m python.migration upgrade old.npz new.npz --dry-run
```

---

## Validating an Artifact

```bash
python -m python.migration validate embeddings.npz
# ✓ embeddings.npz: valid

python -m python.migration validate corrupt.npz
# ✗ corrupt.npz: 1 error(s)
#   - quantized rows (5) != n (4)
```

---

## Bulk Upgrade Script

```python
from pathlib import Path
from python.migration import inspect_artifact, upgrade_artifact

for npz in Path("artifacts/").glob("**/*.npz"):
    info = inspect_artifact(npz)
    if info["needs_upgrade"]:
        dst = npz.with_suffix(".v2.npz")
        upgrade_artifact(npz, dst)
        print(f"Upgraded {npz.name} → {dst.name}")
    else:
        print(f"Already current: {npz.name}")
```

---

## API Compatibility Table

| v1.x API | v2.x API | Notes |
|----------|----------|-------|
| `Vectro.save_compressed(r, path)` | unchanged | now writes v2 |
| `Vectro.load_compressed(path)` | unchanged | reads v1 and v2 |
| `compress_vectors(vecs)` | unchanged | — |
| `decompress_vectors(result)` | unchanged | — |
| `quantize_embeddings(vecs)` | unchanged | — |
| `reconstruct_embeddings(result)` | unchanged | — |
| — | `inspect_artifact(path)` | **new** |
| — | `upgrade_artifact(src, dst)` | **new** |
| — | `validate_artifact(path)` | **new** |

All existing public symbols from v1.x are preserved.

---

## Breaking Changes

None for existing API consumers. The only breaking change is **format-level**:
tools that parse the raw `.npz` file and expect specific keys will need to
handle the new fields (`storage_format_version`, `precision_mode`, etc.).
`numpy.load` with `allow_pickle=False` will still work on both v1 and v2.
