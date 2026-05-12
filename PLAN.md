# Vectro — Plan

> Last updated: 2026-05-11
> Current version: **5.4.0** (Python) / **8.0.0** (Rust) — pipeline checkpointing, PipelineCheckpoint, save_pipeline, load_pipeline, checkpoint_info.

---

## v5.4.0 — Pipeline Checkpointing (2026-05-12)

### Summary

Adds save/load checkpointing for `CompressionPipeline`, enabling reproducible
pipeline configurations and experiment tracking without re-specifying stages.

**`python/pipeline_checkpoint.py` — new module.**
`PipelineCheckpoint` is a frozen dataclass capturing `version` (schema version
string), `created_at` (ISO-8601 UTC timestamp), `stage_configs` (ordered list
of per-stage dicts), and `metadata` (arbitrary user dict).
`save_pipeline(pipeline, path, *, metadata)` serialises a `CompressionPipeline`
to a human-readable JSON file using an atomic write (write to `.tmp`, then
`os.replace`) and creates parent directories automatically.
`load_pipeline(path)` deserialises the JSON and reconstructs a
`CompressionPipeline` with the same stage sequence; raises `FileNotFoundError`
on missing file and `ValueError` on invalid schema.
`checkpoint_info(path)` reads only the metadata without constructing a
pipeline, returning the raw dict — useful for inspecting checkpoints in scripts.

**`PipelineStage` introspection** — `to_config()` / `from_config()` added to
`PipelineStage` in `async_pipeline.py`.  `to_config()` returns a serialisable
dict with `name`, `mode`, and optional `profile`/`group_size`; `from_config()`
reconstructs the stage.

**`tests/test_pipeline_checkpoint.py` — 18 tests.** Covers file creation,
valid JSON output, stage-name round-trip, `CompressionPipeline` type,
`checkpoint_info` key set, version correctness, metadata round-trip, atomic
write, parent-dir creation, `TypeError` on wrong arg, `FileNotFoundError`,
`ValueError` on bad schema, zero-stage pipeline, three-stage pipeline,
`None` metadata, nested metadata, `to_config()` and `from_config()`.

**`python/__init__.py`** — exports `PipelineCheckpoint`, `save_pipeline`,
`load_pipeline`, `checkpoint_info`; version bumped to `5.4.0`.

**Version bump in all 4 version files:** `python/vectro.py`,
`python/__init__.py`, `pyproject.toml`, `pixi.toml`.

### Deliverables

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | `python/pipeline_checkpoint.py` — `PipelineCheckpoint`, `save_pipeline`, `load_pipeline`, `checkpoint_info` | ✅ |
| 2 | `python/async_pipeline.py` — `PipelineStage.to_config()` / `from_config()` | ✅ |
| 3 | `tests/test_pipeline_checkpoint.py` — 18 tests | ✅ |
| 4 | `python/__init__.py` — new exports + version `5.4.0` | ✅ |
| 5 | `python/__init__.pyi` — stubs for 4 new checkpoint symbols | ✅ |
| 6 | `python/vectro.py` — `__version__ = "5.4.0"` | ✅ |
| 7 | `pyproject.toml` — version `5.4.0` | ✅ |
| 8 | `pixi.toml` — version `5.4.0` | ✅ |

---

## v5.3.0 — Pipeline Telemetry & Observability (2026-05-07)

### Summary

Adds a structured, pluggable telemetry layer on top of v5.2.0's
`CompressionPipeline`, giving users per-stage metrics (throughput,
cosine fidelity, compression ratio, latency) emitted as
JSON-serialisable `TelemetryEvent` objects through pluggable
`TelemetryHook` callbacks.

**`python/telemetry.py` — new module.**
`TelemetryEvent` is a frozen dataclass capturing `stage_name`,
`stage_index`, `latency_ms`, `input_shape`, `output_shape`,
`input_dtype`, `output_dtype`, `compression_ratio`,
`throughput_vecs_per_sec`, `cosine_fidelity`, and an open-ended `extra`
dict for application metadata.  `TelemetryCollector` manages a list of
`TelemetryHook` callables and fans events out to all of them via
`emit()`.  `InMemoryTelemetryCollector` subclasses the base collector
and stores every event in a list; `export_json()` returns a JSON array
string.  `attach_telemetry()` monkey-patches a `CompressionPipeline`
instance's `run()` method to emit one event per stage, measuring
per-stage latency, throughput, compression ratio, and cosine fidelity
(computed in FP32 via `np.einsum` — accumulation-accurate) — all
transparent to the caller.

**`tests/test_telemetry.py` — 17 new tests.** Covers `TelemetryEvent`
construction and immutability, `to_dict()` key set and type coercions,
`to_json()` validity, `TelemetryCollector` attach/detach/clear/count,
duplicate-attach idempotency, `InMemoryTelemetryCollector` storage and
`export_json()`, and the `attach_telemetry()` pipeline integration:
event count per stage, stage-name/index correctness, throughput
positivity, compression-ratio positivity, the SIMD cosine-fidelity
property test (INT8 cosine similarity ≥ 0.9999 on L2-normalised
unit-vector inputs), multi-stage event ordering, latency non-negativity,
and `run()` return type unchanged.

**`python/__init__.py`** — exports `TelemetryEvent`, `TelemetryCollector`,
`TelemetryHook`, `InMemoryTelemetryCollector`, `attach_telemetry`;
version bumped to `5.3.0`.

**Version bump in all 4 version files:** `python/vectro.py`,
`python/__init__.py`, `pyproject.toml`, `pixi.toml`.

### Deliverables

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | `python/telemetry.py` — `TelemetryEvent`, `TelemetryCollector`, `TelemetryHook`, `InMemoryTelemetryCollector`, `attach_telemetry` | ✅ |
| 2 | `tests/test_telemetry.py` — 17 tests | ✅ |
| 3 | `python/__init__.py` — new exports + version `5.3.0` | ✅ |
| 4 | `python/__init__.pyi` — stubs for 5 new telemetry symbols | ✅ |
| 5 | `python/vectro.py` — `__version__ = "5.3.0"` | ✅ |
| 6 | `pyproject.toml` — version `5.3.0` | ✅ |
| 7 | `pixi.toml` — version `5.3.0` | ✅ |

---

## v5.2.0 — Async Compression Pipeline (2026-05-06)

### Summary

Adds a fully async-capable multi-stage compression pipeline to the Vectro Python API.

**`python/async_pipeline.py` — new module.** `CompressionPipeline` chains multiple
`PipelineStage` objects in sequence, feeding each stage's output as the next stage's
input. `PipelineResult` captures per-stage and total latency, input/output shapes and
dtypes, and overall compression ratio. `compress_async()` is a thin module-level helper
that wraps `Vectro.compress()` in an asyncio thread-pool executor, keeping the event
loop unblocked. `CompressionPipeline.run_async()` does the same for full pipeline runs.

**`tests/test_async_pipeline.py` — 15 new tests.** Covers stage validation,
empty-pipeline guard, 1-D input rejection, compression-ratio positivity, dtype
preservation, async round-trip, and `compress_async` basic smoke test.

**`python/__init__.py`** — exports `CompressionPipeline`, `PipelineStage`,
`PipelineResult`, `compress_async`; version bumped to `5.2.0`.

**Version bump in all 4 version files:** `python/vectro.py`, `python/__init__.py`,
`pyproject.toml`, `pixi.toml`.

### Deliverables

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | `python/async_pipeline.py` — `PipelineStage`, `PipelineResult`, `CompressionPipeline`, `compress_async` | ✅ |
| 2 | `tests/test_async_pipeline.py` — 15 tests | ✅ |
| 3 | `python/__init__.py` — new exports + version `5.2.0` | ✅ |
| 4 | `python/vectro.py` — `__version__ = "5.2.0"` | ✅ |
| 5 | `pyproject.toml` — version `5.2.0` | ✅ |
| 6 | `pixi.toml` — version `5.2.0` | ✅ |

---

## v5.1.0 — QuantizationConfig + Stub Completeness + Test Hardening ✅ COMPLETE (2026-05-05)

### Summary

Four parallel tracks closed in this sprint:

**Track 1 — `QuantizationConfig` dataclass (`python/vectro.py`).** A validated,
structured configuration container for `Vectro.compress()`. All parameters are
validated at construction time — unknown `precision_mode`, unknown `profile`,
non-power-of-2 `group_size`, bad `seed` type all raise `ValueError` immediately
instead of surfacing errors deep in the hot path. `from_profile(name, **overrides)`
class-method constructs a config from a named profile. `to_dict()` returns a
JSON-serialisable snapshot. `Vectro.compress(config=...)` wires it in as a clean
override of the individual kwargs. 36 new tests.

**Track 2 — Stub completeness.** `lora_api.pyi` (previously absent), `vectro.pyi`
rewritten to include `QuantizationConfig`, updated `compress(config=)` signature,
`compress_async`/`decompress_async`. `__init__.pyi` fully synced with `__init__.py`
— previously ~20 symbols behind the runtime (`lora_api`, `retriever`, `retrieval`,
`ivf_api`, `bf16_api`, `profiles`, `embeddings` all absent from the stub).

**Track 3 — Version string consistency.** `test_release_candidate.py`
`EXPECTED_VERSION` was hardcoded to `4.17.1` (3 minor versions stale). All 4
version files bumped: `pyproject.toml`, `pixi.toml`, `python/__init__.py`,
`python/vectro.py`.

**Track 4 — Test correctness gates.** Fixed 4 pre-existing failures in
`test_cross_platform_benchmarks.py`: p999 gate corrected for Python fallback path,
ADR-002 p99 `<1ms` and INT8 throughput floors guarded with `skipif not
_has_rust_ext()` (those floors are calibrated for the Rust SIMD path and should
not be enforced on Python NumPy).

### Deliverables

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | `python/vectro.py` — `QuantizationConfig` dataclass with `__post_init__` validation | ✅ |
| 2 | `python/vectro.py` — `Vectro.compress(config=...)` kwarg | ✅ |
| 3 | `python/lora_api.pyi` — type stubs (new file) | ✅ |
| 4 | `python/vectro.pyi` — full rewrite with `QuantizationConfig`, `compress_async` | ✅ |
| 5 | `python/__init__.pyi` — full sync: +`QuantizationConfig`, +`lora_api`, +`retriever`, +`retrieval`, +`ivf_api`, +`bf16_api`, +`profiles`, +`embeddings` | ✅ |
| 6 | `python/__init__.py` — `QuantizationConfig` exported in imports and `__all__` | ✅ |
| 7 | `tests/test_quantization_config.py` — 36 tests | ✅ |
| 8 | `tests/test_release_candidate.py` — `EXPECTED_VERSION` `4.17.1` → `5.1.0` | ✅ |
| 9 | `tests/test_cross_platform_benchmarks.py` — p999 gate, p99 skip guard, throughput skip guards | ✅ |
| 10 | Version bump `5.0.2` → `5.1.0` in all 4 version files | ✅ |
